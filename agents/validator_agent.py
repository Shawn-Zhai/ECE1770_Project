import json
from typing import List, Optional

from utils.llm_client import create_llm_client
from config.setting import (
    LLM_DEFAULT_MODEL_NAME,
    BACKEND,
    VALIDATOR_AGENT_MODEL_NAME,
)
from utils.logger import RunLogger
from utils.data_structures import (
    ClaimStep,
    ValidationAction,
    EvidenceItem,
    ValidationResult,
)
from utils.validator_executor import ValidatorExecutor


class ValidatorAgent:
    def __init__(self):
        self.client = create_llm_client()
        self.executor = ValidatorExecutor()
        self.model_name = (
            LLM_DEFAULT_MODEL_NAME
            if VALIDATOR_AGENT_MODEL_NAME == ""
            else VALIDATOR_AGENT_MODEL_NAME
        )
        print(f"[ValidatorAgent] Backend={BACKEND}, model='{self.model_name}'")

    def validate_claim(
        self,
        claim: ClaimStep,
        state_report: dict,
        logger: Optional["RunLogger"] = None,
        max_steps: int = 5,
    ) -> ValidationResult:

        evidence_bank: List[EvidenceItem] = []

        for step_idx in range(max_steps):
            action = self._controller_next_action(
                claim=claim,
                state_report=state_report,
                evidence_bank=evidence_bank,
                logger=logger,
                step_idx=step_idx,
            )

            if action.source == "finish":
                result = self.controller_final_decision(
                    claim=claim,
                    evidence_bank=evidence_bank,
                    logger=logger,
                )
                return result

            observation = self.executor.execute(action, state_report)
            evidence_bank.append(observation)

            if logger is not None:
                logger.log_json(
                    f"validator_step_{step_idx+1}_action",
                    action.__dict__
                )
                logger.log_json(
                    f"validator_step_{step_idx+1}_observation",
                    {
                        "source": observation.source,
                        "operation": observation.operation,
                        "target": observation.target,
                        "summary": observation.summary,
                        "details": observation.details,
                    }
                )

        # Fallback if max steps reached
        return self.controller_final_decision(
            claim=claim,
            evidence_bank=evidence_bank,
            logger=logger,
        )

    def _controller_next_action(
        self,
        claim: ClaimStep,
        state_report: dict,
        evidence_bank: List[EvidenceItem],
        logger: Optional["RunLogger"],
        step_idx: int,
    ) -> ValidationAction:

        system_msg = (
            "You are the controller for an RCA validator agent.\n\n"
            "Your job is to decide what evidence to inspect next in order to validate a claim.\n\n"
            "Available evidence sources:\n"
            "- logs\n"
            "- metrics\n"
            "- traces\n\n"
            "You can also decide to finish if enough evidence has already been collected.\n\n"
            "Return ONLY valid JSON in one of these forms:\n\n"
            "{\n"
            "  \"source\": \"logs | metrics | traces\",\n"
            "  \"operation\": \"...\",\n"
            "  \"target\": \"...\",\n"
            "  \"reason\": \"...\"\n"
            "}\n\n"
            "or\n\n"
            "{\n"
            "  \"source\": \"finish\",\n"
            "  \"operation\": \"finish\",\n"
            "  \"target\": \"\",\n"
            "  \"reason\": \"...\"\n"
            "}\n\n"
            "Guidelines:\n"
            "- Choose the next most informative source.\n"
            "- Prefer direct evidence.\n"
            "- Avoid repeating the same weak query unless needed.\n"
            "- Stop when evidence is sufficient or clearly insufficient."
        )

        compact_evidence = [
            {
                "source": e.source,
                "operation": e.operation,
                "target": e.target,
                "summary": e.summary,
            }
            for e in evidence_bank
        ]

        user_msg = (
            f"Claim to validate:\n{claim.text}\n\n"
            f"Claim type:\n{claim.type}\n\n"
            f"State report summary fields available:\n"
            f"{json.dumps(self.compress_state_report_for_controller(state_report), indent=2, ensure_ascii=False)}\n\n"
            f"Evidence collected so far:\n"
            f"{json.dumps(compact_evidence, indent=2, ensure_ascii=False)}\n\n"
            "Decide the next action."
        )

        resp = self.client.chat.completions.create(
            model=self.model_name,
            messages=[
                {"role": "system", "content": system_msg},
                {"role": "user", "content": user_msg},
            ],
        )

        content = resp.choices[0].message.content.strip()

        if logger is not None:
            logger.log_text(f"validator_controller_raw_step_{step_idx+1}", content)

        try:
            data = json.loads(content)
            return ValidationAction(
                source=str(data.get("source", "finish")),
                operation=str(data.get("operation", "finish")),
                target=str(data.get("target", "")),
                reason=str(data.get("reason", "")),
            )
        except json.JSONDecodeError:
            return ValidationAction(
                source="finish",
                operation="finish",
                target="",
                reason="Controller output could not be parsed."
            )

    def controller_final_decision(
        self,
        claim: ClaimStep,
        evidence_bank: List[EvidenceItem],
        logger: Optional["RunLogger"] = None,
    ) -> ValidationResult:

        system_msg = (
            "You are the controller for an RCA validator agent.\n\n"
            "Based on collected evidence, decide whether the claim is:\n"
            "- supported\n"
            "- contradicted\n"
            "- insufficient\n\n"
            "Return ONLY valid JSON in this format:\n\n"
            "{\n"
            "  \"verdict\": \"supported | contradicted | insufficient\",\n"
            "  \"confidence\": 0.0,\n"
            "  \"reason\": \"...\"\n"
            "}\n\n"
            "Rules:\n"
            "- Be conservative.\n"
            "- Use only the evidence provided.\n"
            "- Confidence must be between 0 and 1."
        )

        evidence_payload = [
            {
                "source": e.source,
                "operation": e.operation,
                "target": e.target,
                "summary": e.summary,
                "details": e.details,
            }
            for e in evidence_bank
        ]

        user_msg = (
            f"Claim:\n{claim.text}\n\n"
            f"Claim type:\n{claim.type}\n\n"
            f"Collected evidence:\n{json.dumps(evidence_payload, indent=2, ensure_ascii=False)}\n\n"
            "Return the final validation decision."
        )

        resp = self.client.chat.completions.create(
            model=self.model_name,
            messages=[
                {"role": "system", "content": system_msg},
                {"role": "user", "content": user_msg},
            ],
        )

        content = resp.choices[0].message.content.strip()

        if logger is not None:
            logger.log_text("validator_final_raw", content)

        try:
            data = json.loads(content)
            result = ValidationResult(
                claim_id=claim.id,
                claim_text=claim.text,
                verdict=str(data.get("verdict", "insufficient")),
                confidence=float(data.get("confidence", 0.0)),
                evidence=evidence_bank,
                reason=str(data.get("reason", "")),
            )
        except Exception:
            result = ValidationResult(
                claim_id=claim.id,
                claim_text=claim.text,
                verdict="insufficient",
                confidence=0.0,
                evidence=evidence_bank,
                reason="Failed to parse final validation result."
            )

        if logger is not None:
            logger.log_json(
                "validator_final_result",
                {
                    "claim_id": result.claim_id,
                    "claim_text": result.claim_text,
                    "verdict": result.verdict,
                    "confidence": result.confidence,
                    "reason": result.reason,
                    "evidence": [
                        {
                            "source": e.source,
                            "operation": e.operation,
                            "target": e.target,
                            "summary": e.summary,
                            "details": e.details,
                        }
                        for e in result.evidence
                    ],
                }
            )

        return result

    def compress_state_report_for_controller(self, state_report: dict) -> dict:
        """
        Give controller enough structure to choose actions without dumping huge raw telemetry.
        """
        return {
            "incident_id": state_report.get("incident_id", ""),
            "symptoms": state_report.get("symptoms", []),
            "detected_anomalies": state_report.get("detected_anomalies", []),
            "telemetry_summary": state_report.get("telemetry_summary", {}),
            "ground_truth": state_report.get("ground_truth", {}),
            "available_raw_sources": list(state_report.get("supporting_raw_data", {}).keys()),
        }