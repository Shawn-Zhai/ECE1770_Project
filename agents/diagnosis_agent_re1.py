import json
from typing import Optional, Dict, Any
from utils.llm_client import create_llm_client

from config.setting import (
    LLM_DEFAULT_MODEL_NAME,
    BACKEND,
    DIAGNOSIS_AGENT_MODEL_NAME,
)
from utils.logger import RunLogger


class DiagnosisAgent:
    """
    Two-stage diagnosis agent:
    1. localize the faulty service
    2. determine the failure type from that service's state report
    """

    ALLOWED_FAILURE_TYPES = {"cpu", "mem", "disk", "delay", "loss"}

    def __init__(self, debug: bool = False):
        self.client = create_llm_client(BACKEND, "")
        self.model_name = (
            LLM_DEFAULT_MODEL_NAME
            if DIAGNOSIS_AGENT_MODEL_NAME == ""
            else DIAGNOSIS_AGENT_MODEL_NAME
        )
        self.debug = debug
        print(f"[DiagnosisAgent] Backend={BACKEND}, model='{self.model_name}'")

    def diagnose(
        self,
        state_report: dict,
        raw_telemetry: Optional[dict] = None,
        logger: Optional["RunLogger"] = None,
    ) -> Dict[str, Any]:
        faulty_service_result = self._diagnose_faulty_service(
            state_report=state_report,
            logger=logger,
        )
        faulty_service = faulty_service_result.get("faulty_service", "unknown")

        failure_type_result = self._diagnose_failure_type(
            state_report=state_report,
            faulty_service=faulty_service,
            logger=logger,
        )

        parsed = {
            "faulty_service": faulty_service,
            "failure_type": failure_type_result.get("failure_type", "unknown"),
            "failure_evidence": (
                "Stage 1 - faulty service: "
                f"{faulty_service_result.get('faulty_service_evidence', 'No explanation provided.')} "
                "Stage 2 - failure type: "
                f"{failure_type_result.get('failure_type_evidence', 'No explanation provided.')}"
            ),
        }

        if parsed.get("failure_type") not in self.ALLOWED_FAILURE_TYPES:
            parsed["failure_type"] = "unknown"

        if "faulty_service" not in parsed:
            parsed["faulty_service"] = "unknown"

        if "failure_evidence" not in parsed:
            parsed["failure_evidence"] = "No explanation provided."

        if logger:
            logger.log_step(
                "diagnose",
                {
                    "faulty_service_result": faulty_service_result,
                    "failure_type_result": failure_type_result,
                    "model_output": parsed,
                },
            )

        return parsed

    def _diagnose_faulty_service(
        self,
        state_report: dict,
        logger: Optional["RunLogger"] = None,
    ) -> Dict[str, Any]:
        compact_payload = build_compact_payload(state_report, top_k=20)

        system_msg = (
            "You are a diagnosis agent for root cause analysis.\n\n"
            "This is stage 1: identify the faulty service only.\n\n"
            "Your task is to analyze the structured state report and output:\n"
            "1. the faulty service\n"
            "2. the explanation for why that service is the most suspicious\n\n"

            "Output requirements:\n"
            "- Output JSON only.\n"
            '- Use exactly these keys: "faulty_service", "faulty_service_evidence".\n'
            "- Use only the provided metric facts.\n"
            "- Do not determine the failure type in this stage.\n"
            "- If evidence is insufficient or ambiguous, return unknown.\n\n"

            "Decision procedure:\n"
            "- Localize the faulty service first.\n"
            "- The faulty service should be the service with the strongest anomaly, especially the largest percentage change.\n"
            "- Percent change is more important than absolute change for localizing the faulty service.\n"
        )

        user_msg = (
            "Here is the structured state report. "
            "The metrics below are the top abnormal metrics sorted by absolute percentage change.\n\n"
            f"{json.dumps(compact_payload, indent=2, ensure_ascii=False)}\n"
        )

        resp = self.client.chat.completions.create(
            model=self.model_name,
            messages=[
                {"role": "system", "content": system_msg},
                {"role": "user", "content": user_msg},
            ],
            response_format={"type": "json_object"},
        )

        raw = resp.choices[0].message.content.strip()
        parsed = json.loads(raw)

        if "faulty_service" not in parsed:
            parsed["faulty_service"] = "unknown"

        if "faulty_service_evidence" not in parsed:
            parsed["faulty_service_evidence"] = "No explanation provided."

        if logger:
            logger.log_step(
                "diagnose_stage_1_faulty_service",
                {
                    "compact_payload": compact_payload,
                    "model_output": parsed,
                },
            )

        return parsed

    def _diagnose_failure_type(
        self,
        state_report: dict,
        faulty_service: str,
        logger: Optional["RunLogger"] = None,
    ) -> Dict[str, Any]:
        compact_payload = build_service_payload(
            state_report=state_report,
            service_name=faulty_service,
            top_k=20,
        )

        if faulty_service == "unknown" or not compact_payload["top_abnormal_metrics"]:
            return {
                "failure_type": "unknown",
                "failure_type_evidence": "Faulty service could not be localized, so failure type cannot be determined.",
            }

        system_msg = (
            "You are a diagnosis agent for root cause analysis.\n\n"
            "This is stage 2: determine the failure type for the already selected faulty service.\n\n"
            "Your task is to analyze the state report for the chosen faulty service and output:\n"
            "1. the failure type\n"
            "2. the explanation based only on the metric facts\n\n"

            "Output requirements:\n"
            "- Output JSON only.\n"
            '- Use exactly these keys: "failure_type", "failure_type_evidence".\n'
            '- "failure_type" must be one of: "cpu", "mem", "disk", "delay", "loss", "unknown".\n'
            "- Use only the provided metric facts.\n"
            "- If evidence is insufficient or ambiguous, return unknown.\n\n"

            "Decision procedure:\n"
            '1. cpu: only when CPU percentage change is the dominant anomaly and is unusually large.\n'
            '2. disk: when the 90th percentile latency change is at least over 7x the 50th percentile latency change.\n'
            '3. delay: when both 50th percentile latency and 90th percentile latency show percentage increases together, and the 90th percentile latency change is less than 7x the 50th percentile latency change.\n'
            '4. mem: memory increases unusually large or all the metrics increase, while workload tends to decrease.\n'
            '5. loss: when the pattern does not match cpu, mem, disk, or delay, but still indicates service degradation.\n\n'

            "Important interpretation notes:\n"
            "- CPU failure should not be selected unless CPU is clearly the main anomaly.\n"
            "- Disk failure is specifically a tail-latency-heavy pattern: p90 changes much more than p50, usually at least 7x.\n"
            "- Delay failure means both p50 and p90 latencies increase together.\n"
            "- Mem failure means multiple metrics are affected, cpu and mem both increasing, while workload is decreasing.\n"
            "- Loss is the fallback when the above patterns do not fit.\n"
        )

        user_msg = (
            f'The faulty service selected in stage 1 is "{faulty_service}". '
            "Analyze only this service's abnormal metrics from the state report.\n\n"
            f"{json.dumps(compact_payload, indent=2, ensure_ascii=False)}\n"
        )

        resp = self.client.chat.completions.create(
            model=self.model_name,
            messages=[
                {"role": "system", "content": system_msg},
                {"role": "user", "content": user_msg},
            ],
            response_format={"type": "json_object"},
        )

        raw = resp.choices[0].message.content.strip()
        parsed = json.loads(raw)

        if parsed.get("failure_type") not in self.ALLOWED_FAILURE_TYPES:
            parsed["failure_type"] = "unknown"

        if "failure_type_evidence" not in parsed:
            parsed["failure_type_evidence"] = "No explanation provided."

        if logger:
            logger.log_step(
                "diagnose_stage_2_failure_type",
                {
                    "faulty_service": faulty_service,
                    "compact_payload": compact_payload,
                    "model_output": parsed,
                },
            )

        return parsed


def build_compact_payload(state_report: dict, top_k: int = 8):
    metric_facts = state_report.get("metric_facts", [])
    simplified_metrics = []

    for metric in metric_facts:
        pct = metric.get("percent_change")
        if pct is None:
            continue

        simplified_metrics.append(
            {
                "service": metric.get("service"),
                "metric_name": metric.get("metric_name"),
                "metric_type": metric.get("metric_type"),
                "percent_change": pct,
            }
        )

    simplified_metrics.sort(
        key=lambda x: abs(x["percent_change"]),
        reverse=True
    )

    return {
        "top_abnormal_metrics": simplified_metrics[:top_k]
    }


def build_service_payload(state_report: dict, service_name: str, top_k: int = 8):
    metric_facts = state_report.get("metric_facts", [])
    simplified_metrics = []

    for metric in metric_facts:
        pct = metric.get("percent_change")
        if pct is None:
            continue

        if metric.get("service") != service_name:
            continue

        simplified_metrics.append(
            {
                "service": metric.get("service"),
                "metric_name": metric.get("metric_name"),
                "metric_type": metric.get("metric_type"),
                "percent_change": pct,
            }
        )

    simplified_metrics.sort(
        key=lambda x: abs(x["percent_change"]),
        reverse=True
    )

    return {
        "service": service_name,
        "top_abnormal_metrics": simplified_metrics[:top_k]
    }
