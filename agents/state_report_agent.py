import json
from typing import Any, Dict, Optional
from utils.llm_client import create_llm_client
from config.setting import (
    LLM_DEFAULT_MODEL_NAME,
    BACKEND,
    STATE_REPORT_AGENT_MODEL_NAME,
)
from utils.logger import RunLogger


class State_report_agent:
    def __init__(self):
        self.client = create_llm_client()
        self.model_name = (
            LLM_DEFAULT_MODEL_NAME
            if STATE_REPORT_AGENT_MODEL_NAME == ""
            else STATE_REPORT_AGENT_MODEL_NAME
        )
        print(f"[StateReportAgent] Backend={BACKEND}, model='{self.model_name}'")

    def generate_state_report(
        self,
        raw_telemetry: Dict[str, Any],
        ground_truth: Dict[str, Any],
        logger: Optional["RunLogger"] = None,
    ) -> Dict[str, Any]:
        """
        Convert raw telemetry + ground truth into a normalized RCA state report.
        """

        system_msg = (
            "You are a state report generation agent for root cause analysis (RCA).\n\n"
            "Your job is to read raw telemetry data and ground truth, then produce a structured "
            "JSON state report for downstream RCA agents.\n\n"
            "The report should organize the incident into:\n"
            "- incident metadata\n"
            "- system overview\n"
            "- symptoms\n"
            "- telemetry summary\n"
            "- detected anomalies\n"
            "- causal candidates\n"
            "- ground truth\n"
            "- supporting raw references\n\n"
            "You MUST respond with ONLY valid JSON.\n\n"
            "Required JSON structure:\n"
            "{\n"
            "  \"incident_id\": \"...\",\n"
            "  \"timestamp\": \"...\",\n"
            "  \"system_overview\": {\n"
            "    \"services\": [...],\n"
            "    \"incident_window\": {\n"
            "      \"start\": \"...\",\n"
            "      \"end\": \"...\"\n"
            "    }\n"
            "  },\n"
            "  \"symptoms\": [...],\n"
            "  \"telemetry_summary\": {...},\n"
            "  \"detected_anomalies\": [\n"
            "    {\n"
            "      \"component\": \"...\",\n"
            "      \"metric\": \"...\",\n"
            "      \"observation\": \"...\"\n"
            "    }\n"
            "  ],\n"
            "  \"causal_candidates\": [\n"
            "    {\n"
            "      \"candidate\": \"...\",\n"
            "      \"confidence\": 0.0,\n"
            "      \"supporting_evidence\": [...]\n"
            "    }\n"
            "  ],\n"
            "  \"ground_truth\": {...},\n"
            "  \"supporting_raw_references\": {\n"
            "    \"logs\": [...],\n"
            "    \"metrics\": [...],\n"
            "    \"traces\": [...]\n"
            "  }\n"
            "}\n\n"
            "Rules:\n"
            "- Preserve facts from the input.\n"
            "- Do not invent unsupported data.\n"
            "- Summarize important telemetry clearly.\n"
            "- Include ground truth exactly if provided.\n"
            "- Output ONLY valid JSON."
        )

        user_msg = (
            "Raw Telemetry:\n"
            f"{json.dumps(raw_telemetry, indent=2, ensure_ascii=False)}\n\n"
            "Ground Truth:\n"
            f"{json.dumps(ground_truth, indent=2, ensure_ascii=False)}\n\n"
            "Generate the state report JSON."
        )

        resp = self.client.chat.completions.create(
            model=self.model_name,
            messages=[
                {"role": "system", "content": system_msg},
                {"role": "user", "content": user_msg},
            ],
        )

        content = resp.choices[0].message.content

        if logger is not None:
            logger.log_json("state_report_input_raw_telemetry", raw_telemetry)
            logger.log_json("state_report_input_ground_truth", ground_truth)
            logger.log_text("state_report_raw_output", content)

        try:
            state_report = json.loads(content)
        except json.JSONDecodeError:
            print("[StateReportAgent] JSON parsing failed. Using fallback report.")
            state_report = {
                "incident_id": "unknown",
                "timestamp": "",
                "system_overview": {
                    "services": [],
                    "incident_window": {
                        "start": "",
                        "end": ""
                    }
                },
                "symptoms": [],
                "telemetry_summary": {},
                "detected_anomalies": [],
                "causal_candidates": [],
                "ground_truth": ground_truth,
                "supporting_raw_references": {
                    "logs": [],
                    "metrics": [],
                    "traces": []
                }
            }

        if logger is not None:
            logger.log_json("state_report_result", state_report)

        return state_report