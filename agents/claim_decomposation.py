import json
import re
from typing import Any, Dict, List, Optional, Tuple

from utils.llm_client import create_llm_client
from config.setting import (
    LLM_DEFAULT_MODEL_NAME,
    BACKEND,
    DIAGNOSIS_AGENT_MODEL_NAME,
)


class ClaimDecomposer:
    """
    Decompose free-form RCA explanation text into verifier-friendly atomic claims.

    Intended input shape:
    {
        "case_id": "...",
        "service_correct": true,
        "fault_correct": true,
        "gt_failure_type": "cpu",
        "prediction": {
            "faulty_service": "adservice",
            "failure_type": "cpu",
            "failure_evidence": "Stage 1 - faulty service: ... Stage 2 - failure type: ..."
        }
    }

    Output shape:
    {
        "case_id": "...",
        "pred_service": "adservice",
        "gt_service": "adservice" or None,
        "service_correct": true,
        "pred_fault": "cpu",
        "gt_fault": "cpu",
        "fault_correct": true,
        "outcome": "both_right",
        "service_evidence_claims": [...],
        "failure_evidence_claims": [...]
    }
    """

    MAX_SERVICE_CLAIMS = 3
    MAX_FAILURE_CLAIMS = 3
    CLAIM_TYPES = {"observation", "inference"}

    def __init__(self, debug: bool = False, use_llm: bool = True):
        self.client = create_llm_client(BACKEND, "")
        self.model_name = (
            LLM_DEFAULT_MODEL_NAME
            if DIAGNOSIS_AGENT_MODEL_NAME == ""
            else DIAGNOSIS_AGENT_MODEL_NAME
        )
        self.debug = debug
        self.use_llm = use_llm

    def decompose_case(self, case_result: Dict[str, Any]) -> Dict[str, Any]:
        prediction = case_result.get("prediction", {}) or {}
        pred_service = prediction.get("faulty_service", "unknown")
        pred_fault = prediction.get("failure_type", "unknown")
        failure_evidence = prediction.get("failure_evidence", "") or ""

        service_text, failure_text = self.split_failure_evidence(failure_evidence)

        if self.use_llm:
            service_claims = self._decompose_text_with_llm(
                text=service_text,
                claim_scope="service",
                pred_service=pred_service,
                pred_fault=pred_fault,
                max_claims=self.MAX_SERVICE_CLAIMS,
                prefix="c",
            )
            failure_claims = self._decompose_text_with_llm(
                text=failure_text,
                claim_scope="failure",
                pred_service=pred_service,
                pred_fault=pred_fault,
                max_claims=self.MAX_FAILURE_CLAIMS,
                prefix="claim",
            )
        else:
            service_claims = self._decompose_text_rule_based(
                text=service_text,
                claim_scope="service",
                max_claims=self.MAX_SERVICE_CLAIMS,
                prefix="c",
            )
            failure_claims = self._decompose_text_rule_based(
                text=failure_text,
                claim_scope="failure",
                max_claims=self.MAX_FAILURE_CLAIMS,
                prefix="claim",
            )

        service_claims = self._normalize_claims(
            service_claims,
            prefix="c",
            max_claims=self.MAX_SERVICE_CLAIMS,
            fallback_text=service_text,
        )
        failure_claims = self._normalize_claims(
            failure_claims,
            prefix="claim",
            max_claims=self.MAX_FAILURE_CLAIMS,
            fallback_text=failure_text,
        )

        return {
            "case_id": case_result.get("case_id"),
            "pred_service": pred_service,
            "gt_service": case_result.get("gt_service"),
            "service_correct": bool(case_result.get("service_correct", False)),
            "pred_fault": pred_fault,
            "gt_fault": case_result.get("gt_failure_type") or case_result.get("gt_fault"),
            "fault_correct": bool(case_result.get("fault_correct", False)),
            "outcome": self._derive_outcome(case_result),
            "service_evidence_claims": service_claims,
            "failure_evidence_claims": failure_claims,
            "raw_explanation_split": {
                "service_explanation_text": service_text,
                "failure_explanation_text": failure_text,
            },
        }

    def split_failure_evidence(self, text: str) -> Tuple[str, str]:
        if not text or not isinstance(text, str):
            return "", ""

        stage1_pattern = re.compile(
            r"Stage\s*1\s*-\s*faulty\s*service\s*:\s*",
            flags=re.IGNORECASE,
        )
        stage2_pattern = re.compile(
            r"Stage\s*2\s*-\s*failure\s*type\s*:\s*",
            flags=re.IGNORECASE,
        )

        stage1_match = stage1_pattern.search(text)
        stage2_match = stage2_pattern.search(text)

        if stage1_match and stage2_match:
            service_start = stage1_match.end()
            failure_start = stage2_match.end()
            service_text = text[service_start:stage2_match.start()].strip()
            failure_text = text[failure_start:].strip()
            return service_text, failure_text

        if stage2_match:
            failure_text = text[stage2_match.end():].strip()
            prefix_text = text[:stage2_match.start()].strip()
            prefix_text = stage1_pattern.sub("", prefix_text).strip()
            return prefix_text, failure_text

        if stage1_match:
            service_text = text[stage1_match.end():].strip()
            return service_text, ""

        return text.strip(), ""

    def _decompose_text_with_llm(
        self,
        text: str,
        claim_scope: str,
        pred_service: str,
        pred_fault: str,
        max_claims: int,
        prefix: str,
    ) -> List[Dict[str, str]]:
        if not text.strip():
            return []

        system_msg = (
            "You are a claim decomposer for root cause analysis explanations.\n\n"
            "Your task is to convert free-form diagnostic explanation text into atomic claims.\n\n"
            "Rules:\n"
            "- Output JSON only.\n"
            '- Use exactly one top-level key: "claims".\n'
            '- "claims" must be a JSON array of claim objects.\n'
            '- Each claim object must use exactly these keys: "id", "text", "type".\n'
            '- "type" must be either "observation" or "inference".\n'
            '- observation = directly stated factual content from the input text, such as metric changes, rankings, comparisons, or absence/presence of patterns.\n'
            '- inference = conclusion, judgment, or interpretation stated in the input text.\n'
            '- Each claim must contain only one proposition.\n'
            '- Split mixed fact-and-conclusion sentences into separate claims when possible.\n'
            '- Keep metric names and numeric values when they appear in the input text.\n'
            '- Do not add new facts that are not present in the input text.\n'
            '- Prefer short, verifier-friendly statements.\n'
            f'- Produce at most {max_claims} claims.\n'
            "- Do not include extra keys or commentary.\n"
        )

        payload = {
            "claim_scope": claim_scope,
            "pred_service": pred_service,
            "pred_fault": pred_fault,
            "explanation_text": text,
        }

        user_msg = (
            "Convert the following RCA explanation text into atomic claims.\n\n"
            f"{json.dumps(payload, indent=2, ensure_ascii=False)}"
        )

        try:
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
            claims = parsed.get("claims", [])
            if not isinstance(claims, list):
                return []
            return self._normalize_claims(claims, prefix=prefix, max_claims=max_claims)
        except Exception:
            return self._decompose_text_rule_based(
                text=text,
                claim_scope=claim_scope,
                max_claims=max_claims,
                prefix=prefix,
            )

    def _decompose_text_rule_based(
        self,
        text: str,
        claim_scope: str,
        max_claims: int,
        prefix: str,
    ) -> List[Dict[str, str]]:
        if not text.strip():
            return []

        candidate_units = self._split_into_candidate_units(text)
        raw_claims: List[Dict[str, str]] = []

        for unit in candidate_units:
            atomic_units = self._split_mixed_unit(unit)
            for atomic in atomic_units:
                atomic = self._clean_claim_text(atomic)
                if not atomic:
                    continue
                claim_type = self._infer_claim_type(atomic, claim_scope=claim_scope)
                raw_claims.append(
                    {
                        "text": atomic,
                        "type": claim_type,
                    }
                )

        return self._normalize_claims(raw_claims, prefix=prefix, max_claims=max_claims)

    def _split_into_candidate_units(self, text: str) -> List[str]:
        text = text.replace("\n", " ").strip()
        if not text:
            return []

        pieces = re.split(r"(?<=[.!?])\s+", text)
        units: List[str] = []
        for piece in pieces:
            piece = piece.strip(" ;")
            if not piece:
                continue
            units.append(piece)
        return units

    def _split_mixed_unit(self, unit: str) -> List[str]:
        unit = unit.strip()
        if not unit:
            return []

        lowered = unit.lower()

        if " while " in lowered:
            left, right = re.split(r"\bwhile\b", unit, maxsplit=1, flags=re.IGNORECASE)
            result = [left.strip(" ,;" )]
            result.extend(self._split_comparison_list(right.strip(" ,;")))
            return [x for x in result if x]

        # Split parenthetical metric lists where possible.
        if "(" in unit and ")" in unit and "," not in unit:
            return [unit]

        return [unit]

    def _split_comparison_list(self, text: str) -> List[str]:
        text = text.strip().rstrip(".")
        if not text:
            return []

        parts = re.split(r",\s*|\s+and\s+", text)
        parts = [p.strip(" ,;.") for p in parts if p.strip(" ,;.")]

        if len(parts) <= 1:
            return [text]

        claims: List[str] = []
        first = parts[0]
        claims.append(first + ".")
        for item in parts[1:]:
            claims.append(item + ".")
        return claims

    def _infer_claim_type(self, text: str, claim_scope: str) -> str:
        lowered = text.lower()

        inference_patterns = [
            "most suspicious",
            "faulty service",
            "dominant anomaly",
            "this makes",
            "therefore",
            "thus",
            "based on the decision rule",
            "reinforcing that",
            "reinforces that",
            "indicates that",
            "suggests that",
        ]

        observation_patterns = [
            "increased",
            "decreased",
            "largest",
            "next largest",
            "only around",
            "%",
            "pattern is provided",
            "pattern is not provided",
            "larger than",
            "smaller than",
            "shows",
        ]

        if any(p in lowered for p in inference_patterns):
            return "inference"
        if any(p in lowered for p in observation_patterns):
            return "observation"

        return "inference" if claim_scope == "failure" and "is" in lowered else "observation"

    def _normalize_claims(
        self,
        raw_claims: Any,
        prefix: str,
        max_claims: int,
        fallback_text: str = "",
    ) -> List[Dict[str, str]]:
        normalized: List[Dict[str, str]] = []
        seen = set()

        if isinstance(raw_claims, list):
            for item in raw_claims:
                if len(normalized) >= max_claims:
                    break

                if isinstance(item, dict):
                    text = self._clean_claim_text(str(item.get("text", "")))
                    claim_type = self._normalize_claim_type(item.get("type", "observation"))
                    claim_id = str(item.get("id", "")).strip()
                else:
                    text = self._clean_claim_text(str(item))
                    claim_type = "observation"
                    claim_id = ""

                if not text:
                    continue

                key = text.lower()
                if key in seen:
                    continue
                seen.add(key)

                normalized.append(
                    {
                        "id": claim_id or self._make_claim_id(prefix, len(normalized) + 1),
                        "text": text,
                        "type": claim_type,
                    }
                )

        if not normalized and fallback_text.strip():
            normalized.append(
                {
                    "id": self._make_claim_id(prefix, 1),
                    "text": self._clean_claim_text(fallback_text),
                    "type": "observation",
                }
            )

        return normalized[:max_claims]

    def _clean_claim_text(self, text: str) -> str:
        text = re.sub(r"\s+", " ", text).strip()
        text = text.strip(" ;")
        if not text:
            return ""

        text = re.sub(r"^(because|and|but)\s+", "", text, flags=re.IGNORECASE)

        if text[-1] not in ".!?":
            text += "."

        return text

    def _normalize_claim_type(self, value: Any) -> str:
        text = str(value).strip().lower()
        return text if text in self.CLAIM_TYPES else "observation"

    def _make_claim_id(self, prefix: str, index: int) -> str:
        if prefix == "c":
            return f"c{index}"
        if prefix == "claim":
            return f"claim_{index}"
        return f"{prefix}_{index}"

    def _derive_outcome(self, case_result: Dict[str, Any]) -> str:
        service_correct = bool(case_result.get("service_correct", False))
        fault_correct = bool(case_result.get("fault_correct", False))

        if service_correct and fault_correct:
            return "both_right"
        if service_correct and not fault_correct:
            return "service_only_right"
        if not service_correct and fault_correct:
            return "fault_only_right"
        return "both_wrong"

def decompose_claims_for_case(case_result: Dict[str, Any], debug: bool = False) -> Dict[str, Any]:
    decomposer = ClaimDecomposer(debug=debug, use_llm=True)
    return decomposer.decompose_case(case_result)

def decompose_claims_for_cases(case_results: List[Dict[str, Any]], debug: bool = False) -> List[Dict[str, Any]]:
    decomposer = ClaimDecomposer(debug=debug, use_llm=True)
    outputs: List[Dict[str, Any]] = []
    for row in case_results:
        outputs.append(decomposer.decompose_case(row))
    return outputs
