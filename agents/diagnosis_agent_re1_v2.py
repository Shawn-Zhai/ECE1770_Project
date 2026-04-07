import json
from typing import Optional, Dict, Any, List
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

    Expected input format:

    {
      "metrics": [
        {
          "service": "adservice",
          "metric_type": "cpu",
          "percent_change": 5160.49,
          "abs_percent_change": 5160.49,
          "vs_injection_sec": 15,
          "duration_sec": 585,
          "pattern": "sustained",
          "percent_rank": 1,
          "temporal_rank": 3
        },
        ...
      ]
    }
    """

    ALLOWED_FAILURE_TYPES = {"cpu", "mem", "disk", "delay", "loss", "unknown"}

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
        metrics = normalize_metrics(state_report)

        if not metrics:
            result = {
                "faulty_service": "unknown",
                "faulty_service_evidence": "No usable metrics were provided.",
            }
            if logger:
                logger.log_step(
                    "diagnose_stage_1_faulty_service",
                    {
                        "metrics": [],
                        "model_output": result,
                    },
                )
            return result

        compact_payload = {"metrics": metrics}

        system_msg = (
            "You are a diagnosis agent for root cause analysis.\n\n"
            "This is stage 1: identify the faulty service only.\n\n"
            "Your task is to analyze the structured anomaly table and output:\n"
            "1. the faulty_service\n"
            "2. the explanation for why that service is the most suspicious\n\n"

            "Output requirements:\n"
            "- Output JSON only.\n"
            '- Use exactly these keys: "faulty_service", "faulty_service_evidence".\n'
            "- Use only the provided metric facts.\n"
            "- Do not determine the failure type in this stage.\n"
            "- If evidence is insufficient or ambiguous, return unknown.\n\n"

            "Reasoning guidance:\n"
            "- Consider the evidence jointly rather than following a single fixed rule.\n"
            "- Important signals may include anomaly magnitude, temporal proximity to the injection time, anomaly duration, pattern, and relative ranking among metrics.\n"
            "- A larger percentage change can indicate a stronger anomaly.\n"
            "- An anomaly that begins earlier and closer to the injection time can be more causally relevant.\n"
            "- Sustained anomalies are usually more important than short spikes.\n"
            "- Prefer services whose evidence is strong, temporally relevant, and internally consistent across their metrics.\n"
            "- When a very large anomaly appears much later than the incident start, consider whether it may be a downstream effect rather than the root cause.\n"
            "- Do not rely on only one field when multiple signals disagree; weigh them together.\n"
            "- If one service has multiple mutually supporting anomalous metrics, that strengthens suspicion for that service.\n"
        )

        user_msg = (
            "Here is the structured anomaly table for stage 1 faulty service localization.\n"
            "Each item may include anomaly magnitude, timing relative to injection, duration, pattern, "
            "and ranking information.\n"
            "Use these fields jointly to decide which service is the most likely faulty service.\n\n"
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
        all_metrics = normalize_metrics(state_report)

        if faulty_service == "unknown":
            return {
                "failure_type": "unknown",
                "failure_type_evidence": "Faulty service could not be localized, so failure type cannot be determined.",
            }

        service_metrics = [
            metric for metric in all_metrics
            if metric.get("service") == faulty_service
        ]

        if not service_metrics:
            return {
                "failure_type": "unknown",
                "failure_type_evidence": f'No metrics were found for the selected service "{faulty_service}".',
            }

        compact_payload = {
            "service": faulty_service,
            "metrics": service_metrics,
        }

        system_msg = (
            "You are a diagnosis agent for root cause analysis.\n\n"
            "This is stage 2: determine the failure type for the already selected faulty service.\n\n"
            "Your task is to analyze the anomaly table for the chosen faulty service and output:\n"
            "1. the failure_type\n"
            "2. the explanation based only on the metric facts\n\n"

            "Output requirements:\n"
            "- Output JSON only.\n"
            '- Use exactly these keys: "failure_type", "failure_type_evidence".\n'
            '- "failure_type" must be one of: "cpu", "mem", "disk", "delay", "loss", "unknown".\n'
            "- Use only the provided metric facts.\n"
            "- If evidence is insufficient or ambiguous, return unknown.\n\n"

            "Reasoning guidance:\n"
            '- cpu: likely when CPU-related anomalies are the dominant and most causally relevant signals for the selected service.\n'
            '- mem: likely when memory-related anomalies are dominant, or memory rises together with broader degradation patterns.\n'
            '- disk: likely when tail latency is much worse than general latency, suggesting a tail-heavy latency pattern.\n'
            '- delay: likely when latency-related anomalies indicate general slowdown rather than a tail-only pattern.\n'
            '- loss: likely when the service shows degradation but the pattern does not fit cpu, mem, disk, or delay well.\n'
            "- Consider anomaly strength, timing, duration, pattern, and consistency across the selected service's metrics.\n"
            "- CPU failure should not be selected unless CPU is clearly the main and most relevant anomaly.\n"
            "- Disk failure is specifically a tail-latency-heavy pattern.\n"
            "- Delay failure usually means latency-related degradation that is not better explained by CPU, memory, or disk.\n"
            "- Loss is the fallback when the above patterns do not fit but degradation still exists.\n"
            "- Weigh the evidence jointly rather than following a single rigid rule.\n"
        )

        user_msg = (
            f'The faulty service selected in stage 1 is "{faulty_service}". '
            "Analyze only this service's anomaly metrics.\n"
            "The metric facts may include anomaly magnitude, timing relative to injection, duration, pattern, "
            "and ranking information. Use them jointly.\n\n"
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


def normalize_metrics(state_report: dict) -> List[Dict[str, Any]]:
    """
    Accept the new flat input format:
        {"metrics": [...]}

    Returns a cleaned, sorted metrics list for the LLM.
    """
    metrics = state_report.get("metrics", [])
    cleaned: List[Dict[str, Any]] = []

    for metric in metrics:
        if not isinstance(metric, dict):
            continue

        service = metric.get("service")
        metric_type = metric.get("metric_type")

        if not service or not metric_type:
            continue

        cleaned_metric = {
            "service": service,
            "metric_type": metric_type,
            "percent_change": metric.get("percent_change"),
            "abs_percent_change": metric.get("abs_percent_change"),
            "vs_injection_sec": metric.get("vs_injection_sec"),
            "duration_sec": metric.get("duration_sec"),
            "pattern": metric.get("pattern"),
            "percent_rank": metric.get("percent_rank"),
            "temporal_rank": metric.get("temporal_rank"),
        }

        cleaned.append(cleaned_metric)

    def percent_sort_key(m: Dict[str, Any]) -> float:
        value = m.get("abs_percent_change")
        if value is None:
            return float("-inf")
        return float(value)

    cleaned.sort(key=percent_sort_key, reverse=True)
    return cleaned