import json
import os
from typing import Any, Dict, List

from agents.state_report_agent import State_report_agent
from agents.explanation_gen_agent import ExplanationGenerationAgent
from agents.claim_decomposition_agent import Claim_decomposition_agent
from agents.validator_agent import ValidatorAgent

from utils.logger import RunLogger


class RCAPipeline:
    def __init__(self, output_dir: str = "outputs"):
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)

        self.state_report_agent = State_report_agent()
        self.explanation_agent = ExplanationGenerationAgent()
        self.claim_agent = Claim_decomposition_agent()
        self.validator_agent = ValidatorAgent()

    def run(
        self,
        raw_telemetry: Dict[str, Any],
        ground_truth: Dict[str, Any],
        run_name: str = "default_run",
    ) -> Dict[str, Any]:
        """
        Full RCA pipeline:
        raw telemetry + ground truth
            -> state report
            -> explanation
            -> claims
            -> validation results
            -> groundedness score
        """

        logger = RunLogger(run_name=run_name, output_dir=self.output_dir)

        # 1) Generate state report
        state_report = self.state_report_agent.generate_state_report(
            raw_telemetry=raw_telemetry,
            ground_truth=ground_truth,
            logger=logger,
        )

        # compatibility normalization
        if "supporting_raw_data" not in state_report:
            state_report["supporting_raw_data"] = {
                "logs": raw_telemetry.get("logs", []),
                "metrics": raw_telemetry.get("metrics", {}),
                "traces": raw_telemetry.get("traces", []),
            }

        self._save_json(
            state_report,
            os.path.join(self.output_dir, "state_report.json")
        )

        # 2) Generate explanation
        explanation = self.explanation_agent.generate_explanation(
            state_report=state_report,
            logger=logger,
        )
        self._save_text(
            explanation,
            os.path.join(self.output_dir, "explanation.txt")
        )

        # 3) Decompose explanation into claims
        claims = self.claim_agent.decompose_claim(
            explanation=explanation,
            logger=logger,
        )

        claims_json = [self.obj_to_dict(c) for c in claims]
        self._save_json(
            claims_json,
            os.path.join(self.output_dir, "claims.json")
        )

        # 4) Validate each claim
        validation_results = []
        for claim in claims:
            result = self.validator_agent.validate_claim(
                claim=claim,
                state_report=state_report,
                logger=logger,
                max_steps=5,
            )
            validation_results.append(result)

        validation_json = [self.obj_to_dict(v) for v in validation_results]
        self._save_json(
            validation_json,
            os.path.join(self.output_dir, "validation_results.json")
        )

        # 5) Compute groundedness score
        groundedness_result = self._compute_groundedness_score(validation_results)
        self._save_json(
            groundedness_result,
            os.path.join(self.output_dir, "groundedness_score.json")
        )

        # 6) Final packaged result
        final_result = {
            "state_report": state_report,
            "explanation": explanation,
            "claims": claims_json,
            "validation_results": validation_json,
            "groundedness_result": groundedness_result,
        }

        self._save_json(
            final_result,
            os.path.join(self.output_dir, "final_result.json")
        )
        return final_result

    def _compute_groundedness_score(self, validation_results: List[Any]) -> Dict[str, Any]:
        """
        Groundedness score:
        supported -> 1
        otherwise -> 0
        G = sum(scores) / n
        grounded if G >= 0.5 else ungrounded
        """
        scores = []

        for result in validation_results:
            result_dict = self.obj_to_dict(result)
            verdict = str(result_dict.get("verdict", "")).lower()

            if verdict == "supported":
                scores.append(1)
            else:
                scores.append(0)

        n = len(scores)
        groundedness_score = sum(scores) / n if n > 0 else 0.0
        label = "grounded" if groundedness_score >= 0.5 else "ungrounded"

        return {
            "num_claims": n,
            "claim_scores": scores,
            "groundedness_score": groundedness_score,
            "label": label,
        }

    def _save_json(self, data: Any, path: str) -> None:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

    def _save_text(self, text: str, path: str) -> None:
        with open(path, "w", encoding="utf-8") as f:
            f.write(text)

    def obj_to_dict(self, obj: Any) -> Any:
        if isinstance(obj, list):
            return [self.obj_to_dict(x) for x in obj]
        if isinstance(obj, dict):
            return {k: self.obj_to_dict(v) for k, v in obj.items()}
        if hasattr(obj, "__dict__"):
            return {k: self.obj_to_dict(v) for k, v in obj.__dict__.items()}
        return obj