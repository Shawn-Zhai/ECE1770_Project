import json
from typing import Optional, Dict, Any, List

from utils.llm_client import create_llm_client
from config.setting import (
    LLM_DEFAULT_MODEL_NAME,
    BACKEND,
    DIAGNOSIS_AGENT_MODEL_NAME,
)
from utils.logger import RunLogger

from utils.raw_telemetry_utils import (
    filter_raw_telemetry_by_service,
    summarize_service_state_evidence,
    build_metric_fact_lookup,
    compress_filtered_raw_telemetry,
)


class DiagnosisAgent:
    """
    Two-stage diagnosis agent.

    Stage 1:
        - LLM service decision from the stage-1 input by default
        - Optional rule-based-only fallback path when LLM is disabled

    Stage 2:
        - Use stage-1 localized faulty service
        - Filter raw telemetry for that service
        - Compress filtered raw telemetry
        - LLM failure-type decision from the stage-2 input by default
        - Optional rule-based-only fallback path when LLM is disabled
    """

    ALLOWED_FAILURE_TYPES = {"cpu", "mem", "disk", "delay", "loss", "unknown"}
    MAX_EVIDENCE_CLAIMS = 3

    def __init__(
        self,
        use_llm_refinement: bool = True,
        debug: bool = False,
    ):
        self.client = create_llm_client(BACKEND, "")
        self.model_name = (
            LLM_DEFAULT_MODEL_NAME
            if DIAGNOSIS_AGENT_MODEL_NAME == ""
            else DIAGNOSIS_AGENT_MODEL_NAME
        )
        self.use_llm_refinement = use_llm_refinement
        self.debug = debug

        print(
            f"[DiagnosisAgent] Backend={BACKEND}, model='{self.model_name}', "
            f"use_llm_refinement={self.use_llm_refinement}, debug={self.debug}"
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def diagnose(
        self,
        state_report: dict,
        raw_telemetry: Optional[dict] = None,
        logger: Optional["RunLogger"] = None,
    ) -> Dict[str, Any]:
        # -----------------------------
        # Stage 1: localize faulty service
        # -----------------------------
        stage1_rule_result = self._empty_stage1_result()
        stage1_final_result = self._empty_stage1_result()

        if self.use_llm_refinement:
            try:
                llm_service_result = self._llm_refine_service(state_report=state_report)
                stage1_final_result = llm_service_result
            except Exception as e:
                print(f"ERROR: stage-1 LLM service decision failed: {e}")
                stage1_final_result = self._empty_stage1_result()
        else:
            stage1_rule_result = self._rule_based_localize_service(state_report)
            stage1_final_result = stage1_rule_result

        # -----------------------------
        # Stage 2: identify failure type
        # -----------------------------
        stage2_rule_result = self._empty_stage2_result()
        stage2_final_result = self._empty_stage2_result()
        filtered_raw_telemetry = None
        compressed_raw_telemetry = None

        try:
            faulty_service = stage1_final_result.get("faulty_service", "unknown")

            if faulty_service != "unknown":
                filtered_raw_telemetry = self._filter_raw_telemetry(
                    raw_telemetry=raw_telemetry,
                    faulty_service=faulty_service,
                )

                compressed_raw_telemetry = self._compress_raw_telemetry(
                    filtered_raw_telemetry=filtered_raw_telemetry,
                    state_report=state_report,
                    faulty_service=faulty_service,
                )

                service_evidence_claims = self._build_service_evidence_claims(
                    state_report=state_report,
                    faulty_service=faulty_service,
                    stage1_result=stage1_final_result,
                )

                if self.debug:
                    print("\n[Stage2] faulty_service =", faulty_service)
                    print(
                        "[Stage2] filtered_raw_telemetry keys =",
                        list(filtered_raw_telemetry.keys())[:10]
                        if isinstance(filtered_raw_telemetry, dict)
                        else [],
                    )
                    print("[Stage2] compressed_raw_telemetry =")
                    print(
                        json.dumps(
                            compressed_raw_telemetry,
                            indent=2,
                            ensure_ascii=False,
                        )[:5000]
                    )

                if self.use_llm_refinement:
                    try:
                        llm_stage2_result = self._llm_identify_failure_type(
                            faulty_service=faulty_service,
                            compressed_raw_telemetry=compressed_raw_telemetry,
                            service_evidence_claims=service_evidence_claims,
                        )
                        stage2_final_result = llm_stage2_result
                    except Exception as e:
                        print(f"ERROR: stage-2 LLM failure-type decision failed: {e}")
                        stage2_final_result = self._empty_stage2_result()
                else:
                    stage2_rule_result = self._rule_based_identify_failure_type(
                        faulty_service=faulty_service,
                        compressed_raw_telemetry=compressed_raw_telemetry,
                        service_evidence_claims=service_evidence_claims,
                    )
                    stage2_final_result = stage2_rule_result
            else:
                stage2_rule_result = self._empty_stage2_result()
                stage2_final_result = self._empty_stage2_result()

        except Exception as e:
            print(f"ERROR: stage-2 failure type identification failed: {e}")
            stage2_rule_result = self._empty_stage2_result()
            stage2_final_result = self._empty_stage2_result()

        final_result = {
            "faulty_service": stage1_final_result.get("faulty_service", "unknown"),
            "failure_type": stage2_final_result.get("failure_type", "unknown"),
            "service_scores": stage1_final_result.get("service_scores", {}),
            "top_evidence": stage1_final_result.get("top_evidence", []),
            "service_evidence_claims": stage2_final_result.get(
                "service_evidence_claims",
                stage1_final_result.get("service_evidence_claims", []),
            ),
            "failure_evidence_claims": stage2_final_result.get(
                "failure_evidence_claims", []
            ),
            "filtered_raw_telemetry_used": bool(
                isinstance(filtered_raw_telemetry, dict) and filtered_raw_telemetry
            ),
            "compressed_raw_telemetry": compressed_raw_telemetry,
        }

        if logger is not None:
            logger.log_json("diagnosis_input_state_report", state_report)
            logger.log_json("diagnosis_stage1_rule_output", stage1_rule_result)
            logger.log_json("diagnosis_stage1_final_output", stage1_final_result)
            logger.log_json("diagnosis_stage2_rule_output", stage2_rule_result)
            logger.log_json("diagnosis_stage2_final_output", stage2_final_result)
            logger.log_json("diagnosis_final_output", final_result)

            if filtered_raw_telemetry is not None:
                logger.log_json(
                    "diagnosis_stage2_filtered_raw_telemetry",
                    filtered_raw_telemetry,
                )

            if compressed_raw_telemetry is not None:
                logger.log_json(
                    "diagnosis_stage2_compressed_raw_telemetry",
                    compressed_raw_telemetry,
                )

        return final_result

    # ------------------------------------------------------------------
    # Stage 1: Rule-based service localization only
    # ------------------------------------------------------------------
    def _rule_based_localize_service(self, state_report: dict) -> Dict[str, Any]:
        metric_facts = state_report.get("metric_facts", [])
        if not isinstance(metric_facts, list) or not metric_facts:
            return self._empty_stage1_result()

        scored_rows = self._build_scored_rows(metric_facts)
        if not scored_rows:
            return self._empty_stage1_result()

        service_scores = self._aggregate_service_scores(scored_rows)
        ranked_services = sorted(service_scores.items(), key=lambda x: x[1], reverse=True)

        best_service = ranked_services[0][0] if ranked_services else "unknown"

        service_rows_sorted = sorted(
            [row for row in scored_rows if row["service"] == best_service],
            key=lambda x: float(x["combined_score"]),
            reverse=True,
        )
        other_rows_sorted = sorted(
            [row for row in scored_rows if row["service"] != best_service],
            key=lambda x: float(x["combined_score"]),
            reverse=True,
        )

        top_evidence = (service_rows_sorted[:6] + other_rows_sorted[:2])[:8]
        return {
            "faulty_service": best_service,
            "service_scores": {k: round(v, 6) for k, v in ranked_services},
            "top_evidence": top_evidence,
            "service_evidence_claims": self._build_rule_service_claims(
                best_service=best_service,
                service_rows=service_rows_sorted,
                ranked_services=ranked_services,
            ),
        }

    def _build_scored_rows(self, metric_facts: List[dict]) -> List[Dict[str, Any]]:
        rows: List[Dict[str, Any]] = []

        for fact in metric_facts:
            service = str(fact.get("service", "unknown"))
            metric_name = str(fact.get("metric_name", "unknown"))
            metric_type_raw = str(fact.get("metric_type", "unknown")).lower()

            baseline_mean = self._to_float(fact.get("baseline_mean", 0.0))
            incident_mean = self._to_float(fact.get("incident_mean", 0.0))
            mean_change = self._to_float(
                fact.get("mean_change", incident_mean - baseline_mean)
            )
            absolute_change = self._to_float(fact.get("absolute_change", abs(mean_change)))
            percent_change = fact.get("percent_change", None)
            baseline_std = abs(self._to_float(fact.get("baseline_std", 0.0)))
            incident_peak = self._to_float(fact.get("incident_peak", 0.0))

            if (
                abs(baseline_mean) < 1e-12
                and abs(incident_mean) < 1e-12
                and abs(baseline_std) < 1e-12
                and abs(incident_peak) < 1e-12
            ):
                continue

            z_mean = self._safe_z(mean_change, baseline_std)
            peak_abs_delta = abs(incident_peak - baseline_mean)
            z_peak = self._safe_z(peak_abs_delta, baseline_std)

            anomaly_score = max(abs(z_mean), abs(z_peak))
            change_ratio = abs(mean_change) / (abs(baseline_mean) + 1e-9)
            direction = "up" if mean_change > 0 else "down" if mean_change < 0 else "flat"

            direction_weight = 1.0
            if direction == "down":
                direction_weight = 0.50
            elif direction == "flat":
                direction_weight = 0.10

            metric_type_weight = self._metric_type_weight(metric_type_raw)

            combined_score = metric_type_weight * direction_weight * (
                0.8 * min(anomaly_score, 200.0) + 0.2 * min(change_ratio, 50.0)
            )

            rows.append(
                {
                    "service": service,
                    "metric_name": metric_name,
                    "metric_type": metric_type_raw,
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

        return rows

    def _aggregate_service_scores(
        self, scored_rows: List[Dict[str, Any]]
    ) -> Dict[str, float]:
        service_scores: Dict[str, float] = {}
        for row in scored_rows:
            service = row["service"]
            score = self._to_float(row["combined_score"])
            service_scores[service] = service_scores.get(service, 0.0) + score
        return service_scores

    # ------------------------------------------------------------------
    # Stage 1: LLM decision for faulty service
    # ------------------------------------------------------------------
    def _llm_refine_service(
        self,
        state_report: dict,
    ) -> Dict[str, Any]:
        candidate_services = sorted(
            {
                str(row.get("service", "")).strip()
                for row in state_report.get("metric_facts", [])
                if isinstance(row, dict) and str(row.get("service", "")).strip()
            }
        )
        compact_payload = {
            "observed_metric_count": state_report.get("observed_metric_count", 0),
            "data_quality": state_report.get("data_quality", {}),
            "candidate_services": candidate_services,
            "metric_facts": state_report.get("metric_facts", []),
        }

        system_msg = (
            "You are a diagnosis agent for root cause analysis.\n\n"
            "Your task in this stage is ONLY to decide the faulty service.\n\n"
            "Rules:\n"
            "- Output JSON only.\n"
            '- Use exactly these keys: "faulty_service", "service_evidence_claims".\n'
            '- "faulty_service" must be one of the candidate services from the input.\n'
            '- "service_evidence_claims" must be a JSON array with 1 to 3 atomic claim objects.\n'
            '- Each claim object must use exactly these keys: "id", "text", "type".\n'
            '- "type" must be "observation" or "inference".\n'
            '- Each "text" must be one short verifiable statement.\n'
            "- Do NOT predict failure type.\n"
            "- Prefer direct root-cause anomalies over downstream symptoms.\n"
            "- Claims must be non-overlapping and focused on the strongest evidence only.\n"
            "- Decide from the input evidence itself, not from a prior guess.\n"
            "- Do not include extra keys."
        )

        user_msg = (
            "Structured state report for stage-1 service decision:\n\n"
            f"{json.dumps(compact_payload, indent=2, ensure_ascii=False)}\n\n"
            "Return service decision JSON only."
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
            "service_evidence_claims": self._normalize_claims(
                parsed.get("service_evidence_claims", []),
                prefix="service_claim",
                default_type="observation",
            ),
        }

    # ------------------------------------------------------------------
    # Stage 2: Rule-based failure type identification
    # ------------------------------------------------------------------
    def _rule_based_identify_failure_type(
        self,
        faulty_service: str,
        compressed_raw_telemetry: Optional[dict],
        service_evidence_claims: List[Dict[str, str]],
    ) -> Dict[str, Any]:
        if not isinstance(compressed_raw_telemetry, dict):
            return {
                "failure_type": "unknown",
                "failure_evidence_claims": [],
                "service_evidence_claims": service_evidence_claims[: self.MAX_EVIDENCE_CLAIMS],
            }

        if compressed_raw_telemetry.get("status") != "ok":
            return {
                "failure_type": "unknown",
                "failure_evidence_claims": [],
                "service_evidence_claims": service_evidence_claims[: self.MAX_EVIDENCE_CLAIMS],
            }

        events = compressed_raw_telemetry.get("metric_order_by_first_anomaly", [])
        if not isinstance(events, list) or not events:
            return {
                "failure_type": "unknown",
                "failure_evidence_claims": [],
                "service_evidence_claims": service_evidence_claims[: self.MAX_EVIDENCE_CLAIMS],
            }

        type_scores: Dict[str, float] = {}
        evidence: List[str] = []

        for rank, row in enumerate(events):
            metric = str(row.get("metric", "unknown"))
            metric_type = str(row.get("metric_type", "unknown")).lower()
            direction = str(row.get("direction", "flat")).lower()
            onset_z = abs(self._to_float(row.get("onset_zscore", 0.0)))
            peak_z = abs(self._to_float(row.get("peak_zscore", 0.0)))
            first_ts = row.get("first_anomaly_ts", "unknown")

            mapped_type = self._map_metric_type_to_failure_type(metric_type)
            if mapped_type == "unknown":
                continue

            rank_weight = max(0.35, 1.0 - 0.15 * rank)
            signal_strength = 0.7 * min(onset_z, 100.0) + 0.3 * min(peak_z, 100.0)

            direction_weight = 1.0
            if mapped_type in {"cpu", "mem", "disk"}:
                direction_weight = 1.0 if direction == "up" else 0.35
            elif mapped_type == "delay":
                direction_weight = 1.0 if direction == "up" else 0.50
            elif mapped_type == "loss":
                direction_weight = 1.0 if direction in {"up", "down"} else 0.50

            metric_weight = self._failure_metric_weight(mapped_type)

            score = rank_weight * direction_weight * metric_weight * signal_strength
            type_scores[mapped_type] = type_scores.get(mapped_type, 0.0) + score

            evidence.append(
                f"{metric} first abnormal at {first_ts} "
                f"(mapped_type={mapped_type}, raw_metric_type={metric_type}, "
                f"direction={direction}, onset_z={onset_z:.2f}, peak_z={peak_z:.2f}, "
                f"score={score:.2f})"
            )

        if not type_scores:
            return {
                "failure_type": "unknown",
                "failure_evidence_claims": [],
                "service_evidence_claims": service_evidence_claims[: self.MAX_EVIDENCE_CLAIMS],
            }

        ranked = sorted(type_scores.items(), key=lambda x: x[1], reverse=True)
        best_type = ranked[0][0]
        return {
            "failure_type": best_type,
            "failure_evidence_claims": self._normalize_claims(
                evidence,
                prefix="failure_claim",
                default_type="inference",
            ),
            "service_evidence_claims": service_evidence_claims[: self.MAX_EVIDENCE_CLAIMS],
        }

    # ------------------------------------------------------------------
    # Stage 2: Optional LLM decision
    # ------------------------------------------------------------------
    def _llm_identify_failure_type(
        self,
        faulty_service: str,
        compressed_raw_telemetry: Optional[dict],
        service_evidence_claims: List[Dict[str, str]],
    ) -> Dict[str, Any]:
        compact_payload = {
            "localized_faulty_service": faulty_service,
            "service_evidence_claims": service_evidence_claims[
                : self.MAX_EVIDENCE_CLAIMS
            ],
            "compressed_raw_telemetry": compressed_raw_telemetry or {},
        }

        system_msg = (
            "You are a diagnosis agent for root cause analysis.\n\n"
            "The faulty service has already been localized.\n"
            "Your task in this stage is ONLY to decide the failure type for that chosen service.\n\n"
            "Rules:\n"
            "- Output JSON only.\n"
            '- Use exactly these keys: "failure_type", "failure_evidence_claims".\n'
            '- "failure_type" must be one of: "cpu", "mem", "disk", "delay", "loss", "unknown".\n'
            '- "failure_evidence_claims" must be a JSON array with 1 to 3 atomic claim objects.\n'
            '- Each claim object must use exactly these keys: "id", "text", "type".\n'
            '- "type" must be "observation" or "inference".\n'
            '- Each "text" must be one short verifiable statement.\n'
            "- Do NOT relocalize service.\n"
            "- Use timestamp order and earliest sustained anomalies as the primary signal.\n"
            "- CPU/mem/disk are direct resource faults.\n"
            "- Persistent latency-dominant degradation maps to delay.\n"
            "- Error-dominant failure or dropped-request symptoms map to loss.\n"
            "- If evidence is weak or ambiguous, output unknown instead of forcing a type.\n"
            "- Claims must be non-overlapping and limited to the strongest 3 pieces of evidence.\n"
            "- Decide from the provided stage-2 input itself, not from a prior guess.\n"
            "- Do not include extra keys."
        )

        user_msg = (
            "Narrowed-scope diagnosis input for stage-2 failure-type decision:\n\n"
            f"{json.dumps(compact_payload, indent=2, ensure_ascii=False)}\n\n"
            "Return failure type decision JSON only."
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
            "failure_type": self._normalize_failure_type(
                parsed.get("failure_type", "unknown")
            ),
            "failure_evidence_claims": self._normalize_claims(
                parsed.get("failure_evidence_claims", []),
                prefix="failure_claim",
                default_type="observation",
            ),
        }

    # ------------------------------------------------------------------
    # Utils for stage 2 inputs
    # ------------------------------------------------------------------
    def _filter_raw_telemetry(
        self,
        raw_telemetry: Optional[dict],
        faulty_service: str,
    ) -> Optional[dict]:
        if raw_telemetry is None:
            return None
        return filter_raw_telemetry_by_service(raw_telemetry, faulty_service)

    def _compress_raw_telemetry(
    self,
    filtered_raw_telemetry: Optional[dict],
    state_report: dict,
    faulty_service: str,
) -> Optional[dict]:
        if filtered_raw_telemetry is None:
            return None

        metric_fact_lookup = build_metric_fact_lookup(
            state_report=state_report,
            service=faulty_service,
        )

        return compress_filtered_raw_telemetry(
                filtered_raw_telemetry=filtered_raw_telemetry,
                metric_fact_lookup= metric_fact_lookup,
                max_metrics=6,
                max_points_per_metric=12,
                min_consecutive_points=10
            )

    def _build_service_evidence_claims(
        self,
        state_report: dict,
        faulty_service: str,
        stage1_result: dict,
    ) -> List[Dict[str, str]]:
        try:
            summary = summarize_service_state_evidence(
                state_report=state_report,
                service=faulty_service,
                top_k=6,
            )
            if isinstance(summary, list) and summary:
                return self._normalize_claims(
                    summary,
                    prefix="service_claim",
                    default_type="observation",
                )
        except Exception:
            pass

        existing = stage1_result.get("service_evidence_claims", [])
        if isinstance(existing, list) and existing:
            return self._normalize_claims(
                existing,
                prefix="service_claim",
                default_type="observation",
            )

        top_evidence = stage1_result.get("top_evidence", [])
        return self._normalize_claims(
            self._compact_evidence_summary(top_evidence),
            prefix="service_claim",
            default_type="observation",
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _empty_stage1_result(self) -> Dict[str, Any]:
        return {
            "faulty_service": "unknown",
            "service_scores": {},
            "top_evidence": [],
            "service_evidence_claims": [],
        }

    def _empty_stage2_result(self) -> Dict[str, Any]:
        return {
            "failure_type": "unknown",
            "failure_evidence_claims": [],
            "service_evidence_claims": [],
        }

    def _compact_evidence_summary(self, top_evidence: List[dict]) -> List[str]:
        summaries: List[str] = []

        for row in top_evidence[:8]:
            service = str(row.get("service", "unknown"))
            metric_name = str(row.get("metric_name", "unknown"))
            direction = str(row.get("direction", "flat"))
            anomaly_score = self._to_float(row.get("anomaly_score", 0.0))
            change_ratio = self._to_float(row.get("change_ratio", 0.0))

            summaries.append(
                f"{service}.{metric_name} changed {direction} "
                f"(anomaly={anomaly_score:.2f}, ratio={change_ratio:.2f})"
            )

        return summaries

    def _build_rule_service_claims(
        self,
        best_service: str,
        service_rows: List[Dict[str, Any]],
        ranked_services: List[Any],
    ) -> List[Dict[str, str]]:
        claims: List[Any] = []

        for row in service_rows[:2]:
            claims.append(
                {
                    "text": (
                        f"{best_service}.{row.get('metric_name', 'unknown')} changed "
                        f"{row.get('direction', 'flat')} with anomaly score "
                        f"{self._to_float(row.get('anomaly_score', 0.0)):.2f}."
                    ),
                    "type": "observation",
                }
            )

        if ranked_services:
            claims.append(
                {
                    "text": (
                        f"{best_service} has the strongest aggregate anomaly signal "
                        "among the candidate services."
                    ),
                    "type": "inference",
                }
            )

        return self._normalize_claims(
            claims,
            prefix="service_claim",
            default_type="observation",
        )

    def _normalize_claims(
        self,
        raw_claims: Any,
        prefix: str,
        default_type: str = "observation",
        limit: Optional[int] = None,
    ) -> List[Dict[str, str]]:
        if not isinstance(raw_claims, list):
            return []

        normalized: List[Dict[str, str]] = []
        seen_texts = set()
        max_items = self.MAX_EVIDENCE_CLAIMS if limit is None else max(0, int(limit))

        for item in raw_claims:
            if len(normalized) >= max_items:
                break

            if isinstance(item, dict):
                text = str(item.get("text", "")).strip()
                claim_id = str(item.get("id", "")).strip()
                claim_type = self._normalize_claim_type(
                    item.get("type", default_type),
                    default=default_type,
                )
            else:
                text = str(item).strip()
                claim_id = ""
                claim_type = default_type

            if not text:
                continue

            text_key = text.lower()
            if text_key in seen_texts:
                continue
            seen_texts.add(text_key)

            normalized.append(
                {
                    "id": claim_id or f"{prefix}_{len(normalized) + 1}",
                    "text": text,
                    "type": claim_type,
                }
            )

        return normalized

    def _normalize_claim_type(self, value: Any, default: str = "observation") -> str:
        text = str(value).strip().lower()
        if text in {"observation", "inference", "causal"}:
            return text
        return default

    def _metric_type_weight(self, metric_type: str) -> float:
        metric_type = str(metric_type).lower()

        if metric_type in {"cpu", "mem", "disk", "latency", "error", "load"}:
            return 1.00
        return 0.30

    def _failure_metric_weight(self, failure_type: str) -> float:
        if failure_type in {"cpu", "mem", "disk"}:
            return 1.00
        if failure_type == "delay":
            return 0.95
        if failure_type == "loss":
            return 0.90
        return 0.20

    def _map_metric_type_to_failure_type(self, metric_type: str) -> str:
        metric_type = str(metric_type).strip().lower()

        if metric_type == "cpu":
            return "cpu"
        if metric_type == "mem":
            return "mem"
        if metric_type == "disk":
            return "disk"
        if metric_type in {"latency", "delay"}:
            return "delay"
        if metric_type in {"error", "errors", "loss"}:
            return "loss"
        return "unknown"

    def _normalize_failure_type(self, value: Any) -> str:
        text = str(value).strip().lower()
        if text in self.ALLOWED_FAILURE_TYPES:
            return text
        if text == "latency":
            return "delay"
        if text == "error":
            return "loss"
        return "unknown"

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
