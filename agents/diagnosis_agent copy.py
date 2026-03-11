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
    Diagnose:
    1. faulty_service
    2. failure_type

    This agent is designed for a structured state report like:
    {
      "observed_metric_count": ...,
      "data_quality": {...},
      "metric_facts": [...]
    }

    Strategy:
    - first: rule-based scoring for service and fault type
    - second: optional LLM refinement / tie-breaking
    - final: strict JSON output for easy evaluation against ground truth
    """

    ALLOWED_FAILURE_TYPES = {"cpu", "mem", "disk", "delay", "loss", "unknown"}

    def __init__(self, use_llm_refinement: bool = True):
        self.client = create_llm_client(BACKEND, "")
        self.model_name = (
            LLM_DEFAULT_MODEL_NAME
            if DIAGNOSIS_AGENT_MODEL_NAME == ""
            else DIAGNOSIS_AGENT_MODEL_NAME
        )
        self.use_llm_refinement = use_llm_refinement
        print(f"[DiagnosisAgent] Backend={BACKEND}, model='{self.model_name}'")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def diagnose(
        self,
        state_report: dict,
        logger: Optional["RunLogger"] = None,
    ) -> Dict[str, Any]:
        """
        Input:
            state_report: dict

        Output:
            {
              "faulty_service": str,
              "failure_type": str,
              "confidence": str,
              "service_scores": {...},
              "failure_type_scores": {...},
              "top_evidence": [...]
            }
        """

        rule_result = self._rule_based_diagnose(state_report)

        final_result = rule_result
        if self.use_llm_refinement:
            try:
                llm_result = self._llm_refine(state_report, rule_result)
                final_result = self._merge_rule_and_llm(rule_result, llm_result)
            except Exception as e:
                # Fall back to deterministic output if LLM fails.
                print(f"ERROR: cannot process with LLM: {e}")
                final_result = rule_result

        if logger is not None:
            logger.log_json("diagnosis_input", state_report)
            logger.log_json("diagnosis_rule_output", rule_result)
            logger.log_json("diagnosis_final_output", final_result)

        return final_result

    # ------------------------------------------------------------------
    # Rule-based algorithm
    # ------------------------------------------------------------------
    def _rule_based_diagnose(self, state_report: dict) -> Dict[str, Any]:
        metric_facts = state_report.get("metric_facts", [])
        if not isinstance(metric_facts, list) or not metric_facts:
            return {
                "faulty_service": "unknown",
                "failure_type": "unknown",
                "confidence": "low",
                "service_scores": {},
                "failure_type_scores": {},
                "top_evidence": [],
            }

        service_scores: Dict[str, float] = {}
        failure_type_scores: Dict[str, float] = {}
        evidence_rows: List[Dict[str, Any]] = []

        for fact in metric_facts:
            service = str(fact.get("service", "unknown"))
            metric_name = str(fact.get("metric_name", "unknown"))
            metric_type_raw = str(fact.get("metric_type", "unknown")).lower()

            canonical_failure_type = self._map_metric_type_to_failure_type(metric_type_raw)

            baseline_mean = self._to_float(fact.get("baseline_mean", 0.0))
            incident_mean = self._to_float(fact.get("incident_mean", 0.0))
            mean_change = self._to_float(
                fact.get("mean_change", incident_mean - baseline_mean)
            )
            absolute_change = self._to_float(fact.get("absolute_change", abs(mean_change)))
            percent_change = fact.get("percent_change", None)
            baseline_std = abs(self._to_float(fact.get("baseline_std", 0.0)))
            incident_peak = self._to_float(fact.get("incident_peak", 0.0))

            z_mean = self._safe_z(mean_change, baseline_std)
            peak_abs_delta = abs(incident_peak - baseline_mean)
            z_peak = self._safe_z(peak_abs_delta, baseline_std)

            anomaly_score = max(abs(z_mean), abs(z_peak))
            change_ratio = abs(mean_change) / (abs(baseline_mean) + 1e-9)

            direction = "up" if mean_change > 0 else "down" if mean_change < 0 else "flat"

            # Downweight uninformative metrics
            if (
                abs(baseline_mean) < 1e-12
                and abs(incident_mean) < 1e-12
                and abs(baseline_std) < 1e-12
                and abs(incident_peak) < 1e-12
            ):
                continue

            # Prefer upward anomalies for injected resource / delay / loss faults
            direction_weight = 1.0
            if direction == "down":
                direction_weight = 0.50
            elif direction == "flat":
                direction_weight = 0.10

            # Resource metrics are better root-cause indicators than propagated symptoms
            metric_type_weight = self._metric_type_weight(metric_type_raw)

            # Combine normalized anomaly + relative shift
            combined_score = metric_type_weight * direction_weight * (
                0.8 * anomaly_score + 0.2 * min(change_ratio, 50.0)
            )

            service_scores[service] = service_scores.get(service, 0.0) + combined_score

            if canonical_failure_type != "unknown":
                failure_type_scores[canonical_failure_type] = (
                    failure_type_scores.get(canonical_failure_type, 0.0) + combined_score
                )

            evidence_rows.append(
                {
                    "service": service,
                    "metric_name": metric_name,
                    "metric_type": metric_type_raw,
                    "mapped_failure_type": canonical_failure_type,
                    "direction": direction,
                    "anomaly_score": round(anomaly_score, 6),
                    "z_mean": round(z_mean, 6),
                    "z_peak": round(z_peak, 6),
                    "change_ratio": round(change_ratio, 6),
                    "combined_score": round(combined_score, 6),
                    "baseline_mean": baseline_mean,
                    "incident_mean": incident_mean,
                    "absolute_change": absolute_change,
                    "mean_change": mean_change,
                    "percent_change": percent_change,
                    "incident_peak": incident_peak,
                    "baseline_std": baseline_std,
                }
            )

        evidence_rows.sort(key=lambda x: float(x["combined_score"]), reverse=True)

        if not service_scores:
            return {
                "faulty_service": "unknown",
                "failure_type": "unknown",
                "confidence": "low",
                "service_scores": {},
                "failure_type_scores": {},
                "top_evidence": [],
            }

        ranked_services = sorted(service_scores.items(), key=lambda x: x[1], reverse=True)
        ranked_failures = sorted(failure_type_scores.items(), key=lambda x: x[1], reverse=True)

        best_service = ranked_services[0][0] if ranked_services else "unknown"
        best_failure = ranked_failures[0][0] if ranked_failures else "unknown"

        confidence = self._estimate_confidence(ranked_services, ranked_failures)

        return {
            "faulty_service": best_service,
            "failure_type": best_failure,
            "confidence": confidence,
            "service_scores": {k: round(v, 6) for k, v in ranked_services},
            "failure_type_scores": {k: round(v, 6) for k, v in ranked_failures},
            "top_evidence": evidence_rows[:8],
        }

    # ------------------------------------------------------------------
    # LLM refinement
    # ------------------------------------------------------------------
    def _llm_refine(
        self,
        state_report: dict,
        rule_result: dict,
    ) -> Dict[str, Any]:
        compact_payload = {
            "observed_metric_count": state_report.get("observed_metric_count", 0),
            "data_quality": state_report.get("data_quality", {}),
            "rule_based_guess": {
                "faulty_service": rule_result.get("faulty_service", "unknown"),
                "failure_type": rule_result.get("failure_type", "unknown"),
                "confidence": rule_result.get("confidence", "low"),
                "top_evidence": rule_result.get("top_evidence", [])[:6],
                "service_scores": rule_result.get("service_scores", {}),
                "failure_type_scores": rule_result.get("failure_type_scores", {}),
            },
            "metric_facts": state_report.get("metric_facts", [])[:80],
        }

        system_msg = (
            "You are a diagnosis agent for root cause analysis.\n\n"
            "Your job is ONLY to identify:\n"
            "1. faulty_service\n"
            "2. failure_type\n\n"
            "Rules:\n"
            "- Output JSON only.\n"
            "- Use exactly these keys:\n"
            '  "faulty_service", "failure_type", "confidence"\n'
            '- "failure_type" must be one of: "cpu", "mem", "disk", "delay", "loss", "unknown"\n'
            '- "confidence" must be one of: "high", "medium", "low"\n'
            "- Prefer direct root-cause resource anomalies over downstream symptoms.\n"
            "- latency maps to delay; error maps to loss.\n"
            "- If evidence is weak or conflicting, output unknown when necessary.\n"
            "- Do not explain. Do not include extra keys."
        )

        user_msg = (
            "Structured state report and rule-based pre-analysis:\n\n"
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
            "faulty_service": str(parsed.get("faulty_service", "unknown")),
            "failure_type": self._normalize_failure_type(
                parsed.get("failure_type", "unknown")
            ),
            "confidence": self._normalize_confidence(parsed.get("confidence", "low")),
        }

    def _merge_rule_and_llm(self, rule_result: dict, llm_result: dict) -> Dict[str, Any]:
        rule_service = str(rule_result.get("faulty_service", "unknown"))
        rule_failure = self._normalize_failure_type(rule_result.get("failure_type", "unknown"))

        llm_service = str(llm_result.get("faulty_service", "unknown"))
        llm_failure = self._normalize_failure_type(llm_result.get("failure_type", "unknown"))

        final_service = rule_service
        final_failure = rule_failure
        final_confidence = rule_result.get("confidence", "low")

        if llm_service != "unknown":
            final_service = llm_service
        if llm_failure != "unknown":
            final_failure = llm_failure
        if llm_result.get("confidence"):
            final_confidence = self._normalize_confidence(llm_result["confidence"])

        return {
            "faulty_service": final_service,
            "failure_type": final_failure,
            "confidence": final_confidence,
            "service_scores": rule_result.get("service_scores", {}),
            "failure_type_scores": rule_result.get("failure_type_scores", {}),
            "top_evidence": rule_result.get("top_evidence", []),
        }

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _map_metric_type_to_failure_type(self, metric_type: str) -> str:
        metric_type = str(metric_type).lower()

        if metric_type in {"cpu", "mem", "disk"}:
            return metric_type
        if metric_type == "latency":
            return "delay"
        if metric_type == "error":
            return "loss"

        # load is often supporting evidence, not a direct fault type
        if metric_type == "load":
            return "unknown"

        return "unknown"

    def _metric_type_weight(self, metric_type: str) -> float:
        metric_type = str(metric_type).lower()

        if metric_type == "cpu":
            return 1.00
        if metric_type == "mem":
            return 1.00
        if metric_type == "disk":
            return 1.00
        if metric_type == "latency":
            return 1.00
        if metric_type == "error":
            return 1.00
        if metric_type == "load":
            return 1.00
        return 0.30

    def _estimate_confidence(self, ranked_services, ranked_failures) -> str:
        if not ranked_services or not ranked_failures:
            return "low"

        s1 = ranked_services[0][1]
        s2 = ranked_services[1][1] if len(ranked_services) > 1 else 0.0
        f1 = ranked_failures[0][1]
        f2 = ranked_failures[1][1] if len(ranked_failures) > 1 else 0.0

        service_margin = s1 / (s2 + 1e-9)
        failure_margin = f1 / (f2 + 1e-9)

        if service_margin >= 2.5 and failure_margin >= 2.0:
            return "high"
        if service_margin >= 1.4 and failure_margin >= 1.3:
            return "medium"
        return "low"

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
        if text in {"high", "medium", "low"}:
            return text
        return "low"

    def _safe_z(self, delta: float, sigma: float) -> float:
        if sigma < 1e-9:
            if abs(delta) < 1e-9:
                return 0.0
            return 25.0 if delta > 0 else -25.0
        return delta / sigma

    def _to_float(self, value: Any) -> float:
        try:
            if value is None:
                return 0.0
            return float(value)
        except Exception:
            return 0.0