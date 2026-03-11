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
    Pure-LLM diagnosis agent.

    Input:
        structured state_report

    Output:
        {
          "faulty_service": str,
          "failure_type": str,
          "confidence": str
        }
    """

    ALLOWED_FAILURE_TYPES = {"cpu", "mem", "disk", "delay", "loss", "unknown"}
    ALLOWED_CONFIDENCE = {"high", "medium", "low"}

    def __init__(self):
        self.client = create_llm_client(BACKEND, "")
        self.model_name = (
            LLM_DEFAULT_MODEL_NAME
            if DIAGNOSIS_AGENT_MODEL_NAME == ""
            else DIAGNOSIS_AGENT_MODEL_NAME
        )
        print(f"[DiagnosisAgent] Backend={BACKEND}, model='{self.model_name}'")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def diagnose(
        self,
        state_report: dict,
        logger: Optional["RunLogger"] = None,
    ) -> Dict[str, Any]:
        result = self._llm_only_diagnose(state_report)

        if logger is not None:
            logger.log_json("diagnosis_input", state_report)
            logger.log_json("diagnosis_final_output", result)

        return result

    # ------------------------------------------------------------------
    # Pure LLM diagnosis
    # ------------------------------------------------------------------
    def _llm_only_diagnose(self, state_report: dict) -> Dict[str, Any]:
        compact_payload = {
            "observed_metric_count": state_report.get("observed_metric_count", 0),
            "data_quality": state_report.get("data_quality", {}),
            "metric_facts": state_report.get("metric_facts", [])[:120],
        }

        system_msg = (
            "You are a diagnosis agent for root cause analysis.\n\n"
            "Your job is ONLY to identify:\n"
            "1. faulty_service\n"
            "2. failure_type\n\n"
            "Rules:\n"
            "- Output JSON only.\n"
            '- Use exactly these keys: "faulty_service", "failure_type", "confidence"\n'
            '- "failure_type" must be one of: "cpu", "mem", "disk", "delay", "loss", "unknown"\n'
            '- "confidence" must be one of: "high", "medium", "low"\n'
            "- Read the structured state report carefully.\n"
            "- Prefer direct root-cause anomalies over downstream propagated symptoms.\n"
            "- CPU/mem/disk are resource faults.\n"
            "- Latency-related anomalies map to delay.\n"
            "- Error-related anomalies map to loss.\n"
            "- If evidence is conflicting or weak, output unknown.\n"
            "- Do not explain.\n"
            "- Do not include extra keys."
        )

        user_msg = (
            "Below is the structured state report.\n\n"
            f"{json.dumps(compact_payload, indent=2, ensure_ascii=False)}\n\n"
            "Return diagnosis JSON only."
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

        return {
            "faulty_service": str(parsed.get("faulty_service", "unknown")).strip(),
            "failure_type": self._normalize_failure_type(
                parsed.get("failure_type", "unknown")
            ),
            "confidence": self._normalize_confidence(
                parsed.get("confidence", "low")
            ),
        }

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _normalize_failure_type(self, value: Any) -> str:
        text = str(value).strip().lower()
        if text in self.ALLOWED_FAILURE_TYPES:
            return text
        if text == "latency":
            return "delay"
        if text == "error":
            return "loss"
        return "unknown"

    def _normalize_confidence(self, value: Any) -> str:
        text = str(value).strip().lower()
        if text in self.ALLOWED_CONFIDENCE:
            return text
        return "low"