import json
from typing import Optional
from utils.llm_client import create_llm_client
from config.setting import (
    LLM_DEFAULT_MODEL_NAME,
    BACKEND,
    EXPLANATION_GENERATION_AGENT_MODEL_NAME,
)
from utils.logger import RunLogger


class ExplanationGenerationAgent:
    def __init__(self):
        self.client = create_llm_client()
        self.model_name = (
            LLM_DEFAULT_MODEL_NAME
            if EXPLANATION_GENERATION_AGENT_MODEL_NAME == ""
            else EXPLANATION_GENERATION_AGENT_MODEL_NAME
        )
        print(f"[ExplanationGenerationAgent] Backend={BACKEND}, model='{self.model_name}'")

    def generate_explanation(
        self,
        state_report: dict,
        logger: Optional["RunLogger"] = None,
    ) -> str:
        """
        Generate an RCA explanation from a structured state report.

        Input:
            state_report: dict
                A JSON-like dictionary containing telemetry, symptoms,
                anomalies, and optionally ground truth.

        Output:
            str
                A natural-language RCA explanation.
        """

        system_msg = (
            "You are an explanation generation agent for root cause analysis (RCA).\n\n"
            "Your task is to read a structured state report and produce a clear, "
            "concise, evidence-based RCA explanation.\n\n"
            "The state report may contain:\n"
            "- raw telemetry summaries\n"
            "- anomalies\n"
            "- service relationships\n"
            "- possible causes\n"
            "- ground truth root cause\n\n"
            "Instructions:\n"
            "- Explain the likely failure progression clearly.\n"
            "- Use evidence from the state report.\n"
            "- If ground truth is present, align the explanation with it.\n"
            "- Do not invent facts not supported by the state report.\n"
            "- Write in a concise but complete paragraph or short multi-paragraph explanation.\n"
            "- Focus on what happened, why it happened, and what evidence supports it.\n"
            "- Do NOT output JSON.\n"
            "- Output only the RCA explanation text."
        )

        user_msg = (
            "State Report JSON:\n\n"
            f"{json.dumps(state_report, indent=2, ensure_ascii=False)}\n\n"
            "Generate the RCA explanation."
        )

        resp = self.client.chat.completions.create(
            model=self.model_name,
            messages=[
                {"role": "system", "content": system_msg},
                {"role": "user", "content": user_msg},
            ],
        )

        explanation = resp.choices[0].message.content.strip()

        if logger is not None:
            logger.log_json("explanation_generation_input", state_report)
            logger.log_text("explanation_generation_output", explanation)

        return explanation