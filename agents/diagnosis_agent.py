import json
from typing import Optional, Dict, Any, List, Tuple

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
        - Rule-based faulty service localization
        - Optional LLM refinement for faulty service only

    Stage 2:
        - Use stage-1 localized faulty service
        - Filter raw telemetry for that service
        - Compress filtered raw telemetry
        - Rule-based failure-type scoring
        - Optional LLM refinement only when needed
    """

    ALLOWED_FAILURE_TYPES = {"cpu", "mem", "disk", "delay", "loss", "unknown"}

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
        stage1_rule_result = self._rule_based_localize_service(state_report)

        stage1_final_result = stage1_rule_result
        if self.use_llm_refinement:
            try:
                llm_service_result = self._llm_refine_service(
                    state_report=state_report,
                    rule_result=stage1_rule_result,
                )
                stage1_final_result = self._merge_service_rule_and_llm(
                    rule_result=stage1_rule_result,
                    llm_result=llm_service_result,
                )
            except Exception as e:
                print(f"ERROR: stage-1 LLM refinement failed: {e}")
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

                service_evidence_summary = self._build_service_evidence_summary(
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

                stage2_rule_result = self._rule_based_identify_failure_type(
                    faulty_service=faulty_service,
                    compressed_raw_telemetry=compressed_raw_telemetry,
                    service_evidence_summary=service_evidence_summary,
                )

                stage2_final_result = stage2_rule_result

                should_use_llm = (
                    self.use_llm_refinement
                    and stage2_rule_result.get("failure_confidence", "low") != "high"
                )

                if should_use_llm:
                    try:
                        llm_stage2_result = self._llm_identify_failure_type(
                            faulty_service=faulty_service,
                            service_confidence=stage1_final_result.get(
                                "service_confidence", "low"
                            ),
                            compressed_raw_telemetry=compressed_raw_telemetry,
                            service_evidence_summary=service_evidence_summary,
                            rule_based_failure_type=stage2_rule_result.get(
                                "failure_type", "unknown"
                            ),
                            rule_based_failure_confidence=stage2_rule_result.get(
                                "failure_confidence", "low"
                            ),
                            rule_based_failure_evidence_summary=stage2_rule_result.get(
                                "failure_evidence_summary", []
                            ),
                        )

                        stage2_final_result = self._merge_failure_rule_and_llm(
                            rule_result=stage2_rule_result,
                            llm_result=llm_stage2_result,
                            compressed_raw_telemetry=compressed_raw_telemetry,
                        )
                    except Exception as e:
                        print(f"ERROR: stage-2 LLM refinement failed: {e}")
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
            "service_confidence": stage1_final_result.get("service_confidence", "low"),
            "failure_type": stage2_final_result.get("failure_type", "unknown"),
            "failure_confidence": stage2_final_result.get("failure_confidence", "low"),
            "service_scores": stage1_final_result.get("service_scores", {}),
            "top_evidence": stage1_final_result.get("top_evidence", []),
            "service_evidence_summary": stage2_final_result.get(
                "service_evidence_summary",
                stage1_final_result.get("service_evidence_summary", []),
            ),
            "failure_evidence_summary": stage2_final_result.get(
                "failure_evidence_summary", []
            ),
            "filtered_raw_telemetry_used": bool(
                isinstance(filtered_raw_telemetry, dict) and filtered_raw_telemetry
            ),
            "compressed_raw_telemetry": compressed_raw_telemetry,
            "failure_type_scores": stage2_final_result.get("failure_type_scores", {}),
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
        confidence = self._estimate_service_confidence(ranked_services)

        return {
            "faulty_service": best_service,
            "service_confidence": confidence,
            "service_scores": {k: round(v, 6) for k, v in ranked_services},
            "top_evidence": top_evidence,
            "service_evidence_summary": self._compact_evidence_summary(top_evidence),
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
    # Stage 1: LLM refinement for faulty service only
    # ------------------------------------------------------------------
    def _llm_refine_service(
        self,
        state_report: dict,
        rule_result: dict,
    ) -> Dict[str, Any]:
        compact_payload = {
            "observed_metric_count": state_report.get("observed_metric_count", 0),
            "data_quality": state_report.get("data_quality", {}),
            "rule_based_guess": {
                "faulty_service": rule_result.get("faulty_service", "unknown"),
                "service_confidence": rule_result.get("service_confidence", "low"),
                "top_evidence": rule_result.get("top_evidence", [])[:6],
                "service_scores": rule_result.get("service_scores", {}),
            },
            "metric_facts": state_report.get("metric_facts", []),
        }

        system_msg = (
            "You are a diagnosis agent for root cause analysis.\n\n"
            "Your task in this stage is ONLY to refine the faulty service.\n\n"
            "Rules:\n"
            "- Output JSON only.\n"
            '- Based on the metric facts of the state report and rule_based_guess, refine the guessed faulty service.\n'
            '- "confidence" must be one of: "high", "medium", "low".\n'
            '- "service_evidence_summary" must be a JSON array of short strings.\n'
            "- Do NOT predict failure type.\n"
            "- Prefer direct root-cause anomalies over downstream symptoms.\n"
            "- If uncertain, keep the strongest service candidate from the rule-based result.\n"
            "- Do not include extra keys."
        )

        user_msg = (
            "Structured state report and rule-based service localization result:\n\n"
            f"{json.dumps(compact_payload, indent=2, ensure_ascii=False)}\n\n"
            "Return service refinement JSON only."
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

        summary = parsed.get("service_evidence_summary", [])
        if not isinstance(summary, list):
            summary = []

        return {
            "faulty_service": str(parsed.get("faulty_service", "unknown")),
            "service_confidence": self._normalize_confidence(
                parsed.get("confidence", "low")
            ),
            "service_evidence_summary": [str(x) for x in summary[:8]],
        }

    def _merge_service_rule_and_llm(
        self,
        rule_result: dict,
        llm_result: dict,
    ) -> Dict[str, Any]:
        rule_service = str(rule_result.get("faulty_service", "unknown"))
        llm_service = str(llm_result.get("faulty_service", "unknown"))

        final_service = rule_service
        rule_conf = self._normalize_confidence(
            rule_result.get("service_confidence", "low")
        )

        if llm_service == rule_service:
            final_service = rule_service
        elif rule_conf == "low" and llm_service != "unknown":
            final_service = llm_service

        final_confidence = rule_result.get("service_confidence", "low")
        if llm_service == rule_service:
            final_confidence = self._normalize_confidence(
                llm_result.get("service_confidence", final_confidence)
            )

        llm_summary = llm_result.get("service_evidence_summary", [])
        if not isinstance(llm_summary, list):
            llm_summary = []

        return {
            "faulty_service": final_service,
            "service_confidence": final_confidence,
            "service_scores": rule_result.get("service_scores", {}),
            "top_evidence": rule_result.get("top_evidence", []),
            "service_evidence_summary": llm_summary
            if llm_summary
            else rule_result.get("service_evidence_summary", []),
        }

    # ------------------------------------------------------------------
    # Stage 2: Rule-based failure type identification
    # ------------------------------------------------------------------
    def _rule_based_identify_failure_type(
        self,
        faulty_service: str,
        compressed_raw_telemetry: Optional[dict],
        service_evidence_summary: List[str],
    ) -> Dict[str, Any]:
        if not isinstance(compressed_raw_telemetry, dict):
            return {
                "failure_type": "unknown",
                "failure_confidence": "low",
                "failure_type_scores": {},
                "failure_evidence_summary": [
                    "No compressed raw telemetry available for failure-type diagnosis."
                ],
                "service_evidence_summary": service_evidence_summary[:8],
            }

        if compressed_raw_telemetry.get("status") != "ok":
            return {
                "failure_type": "unknown",
                "failure_confidence": "low",
                "failure_type_scores": {},
                "failure_evidence_summary": [
                    str(compressed_raw_telemetry.get("instruction_hint", "Weak telemetry evidence."))
                ],
                "service_evidence_summary": service_evidence_summary[:8],
            }

        events = compressed_raw_telemetry.get("metric_order_by_first_anomaly", [])
        if not isinstance(events, list) or not events:
            return {
                "failure_type": "unknown",
                "failure_confidence": "low",
                "failure_type_scores": {},
                "failure_evidence_summary": [
                    "No abnormal metric onset detected for the localized service."
                ],
                "service_evidence_summary": service_evidence_summary[:8],
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
                "failure_confidence": "low",
                "failure_type_scores": {},
                "failure_evidence_summary": [
                    "Telemetry exists, but no metric could be mapped to a known failure type."
                ],
                "service_evidence_summary": service_evidence_summary[:8],
            }

        ranked = sorted(type_scores.items(), key=lambda x: x[1], reverse=True)
        best_type = ranked[0][0]
        s1 = ranked[0][1]
        s2 = ranked[1][1] if len(ranked) > 1 else 0.0

        margin = s1 / (s2 + 1e-9)
        if margin >= 2.5:
            confidence = "high"
        elif margin >= 1.4:
            confidence = "medium"
        else:
            confidence = "low"

        return {
            "failure_type": best_type,
            "failure_confidence": confidence,
            "failure_type_scores": {k: round(v, 6) for k, v in ranked},
            "failure_evidence_summary": evidence[:8],
            "service_evidence_summary": service_evidence_summary[:8],
        }

    # ------------------------------------------------------------------
    # Stage 2: Optional LLM refinement
    # ------------------------------------------------------------------
    def _llm_identify_failure_type(
        self,
        faulty_service: str,
        service_confidence: str,
        compressed_raw_telemetry: Optional[dict],
        service_evidence_summary: List[str],
        rule_based_failure_type: str,
        rule_based_failure_confidence: str,
        rule_based_failure_evidence_summary: List[str],
    ) -> Dict[str, Any]:
        compact_payload = {
            "localized_faulty_service": faulty_service,
            "service_confidence": service_confidence,
            "service_evidence_summary": service_evidence_summary[:8],
            "rule_based_failure_guess": {
                "failure_type": rule_based_failure_type,
                "failure_confidence": rule_based_failure_confidence,
                "failure_evidence_summary": rule_based_failure_evidence_summary[:8],
            },
            "compressed_raw_telemetry": compressed_raw_telemetry or {},
        }

        system_msg = (
            "You are a diagnosis agent for root cause analysis.\n\n"
            "The faulty service has already been localized.\n"
            "Your task in this stage is ONLY to refine the failure type for that chosen service.\n\n"
            "Rules:\n"
            "- Output JSON only.\n"
            '- Use exactly these keys: "failure_type", "confidence", "failure_evidence_summary".\n'
            '- "failure_type" must be one of: "cpu", "mem", "disk", "delay", "loss", "unknown".\n'
            '- "confidence" must be one of: "high", "medium", "low".\n'
            '- "failure_evidence_summary" must be a JSON array of short strings.\n'
            "- Do NOT relocalize service.\n"
            "- Prefer the rule-based failure guess when the telemetry evidence is strong.\n"
            "- Use timestamp order and earliest sustained anomalies as the primary signal.\n"
            "- CPU/mem/disk are direct resource faults.\n"
            "- Persistent latency-dominant degradation maps to delay.\n"
            "- Error-dominant failure or dropped-request symptoms map to loss.\n"
            "- If evidence is weak or ambiguous, output unknown instead of forcing a type.\n"
            "- Do not include extra keys."
        )

        user_msg = (
            "Narrowed-scope diagnosis input:\n\n"
            f"{json.dumps(compact_payload, indent=2, ensure_ascii=False)}\n\n"
            "Return failure type diagnosis JSON only."
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

        failure_summary = parsed.get("failure_evidence_summary", [])
        if not isinstance(failure_summary, list):
            failure_summary = []

        return {
            "failure_type": self._normalize_failure_type(
                parsed.get("failure_type", "unknown")
            ),
            "failure_confidence": self._normalize_confidence(
                parsed.get("confidence", "low")
            ),
            "failure_evidence_summary": [str(x) for x in failure_summary[:8]],
        }

    def _merge_failure_rule_and_llm(
        self,
        rule_result: dict,
        llm_result: dict,
        compressed_raw_telemetry: Optional[dict],
    ) -> Dict[str, Any]:
        rule_type = self._normalize_failure_type(rule_result.get("failure_type", "unknown"))
        llm_type = self._normalize_failure_type(llm_result.get("failure_type", "unknown"))

        rule_conf = self._normalize_confidence(rule_result.get("failure_confidence", "low"))
        llm_conf = self._normalize_confidence(llm_result.get("failure_confidence", "low"))

        final_type = rule_type
        final_conf = rule_conf

        if llm_type == rule_type:
            final_type = rule_type
            final_conf = max(rule_conf, llm_conf, key=self._confidence_rank)
        elif rule_conf == "low" and llm_conf in {"medium", "high"} and llm_type != "unknown":
            final_type = llm_type
            final_conf = llm_conf
        else:
            final_type = rule_type
            final_conf = rule_conf

        llm_summary = llm_result.get("failure_evidence_summary", [])
        if not isinstance(llm_summary, list):
            llm_summary = []

        final_summary = llm_summary if llm_summary else rule_result.get("failure_evidence_summary", [])

        final_conf = self._cap_failure_confidence(
            confidence=final_conf,
            compressed_raw_telemetry=compressed_raw_telemetry,
            failure_evidence_summary=final_summary,
        )

        return {
            "failure_type": final_type,
            "failure_confidence": final_conf,
            "failure_type_scores": rule_result.get("failure_type_scores", {}),
            "failure_evidence_summary": final_summary[:8],
            "service_evidence_summary": rule_result.get("service_evidence_summary", []),
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
            metric_fact_lookup=metric_fact_lookup,
            max_metrics=6,
            max_points_per_metric=12,
        )

    def _build_service_evidence_summary(
        self,
        state_report: dict,
        faulty_service: str,
        stage1_result: dict,
    ) -> List[str]:
        try:
            summary = summarize_service_state_evidence(
                state_report=state_report,
                service=faulty_service,
                top_k=8,
            )
            if isinstance(summary, list) and summary:
                return [str(x) for x in summary[:8]]
        except Exception:
            pass

        existing = stage1_result.get("service_evidence_summary", [])
        if isinstance(existing, list) and existing:
            return [str(x) for x in existing[:8]]

        top_evidence = stage1_result.get("top_evidence", [])
        return self._compact_evidence_summary(top_evidence)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _empty_stage1_result(self) -> Dict[str, Any]:
        return {
            "faulty_service": "unknown",
            "service_confidence": "low",
            "service_scores": {},
            "top_evidence": [],
            "service_evidence_summary": [],
        }

    def _empty_stage2_result(self) -> Dict[str, Any]:
        return {
            "failure_type": "unknown",
            "failure_confidence": "low",
            "failure_type_scores": {},
            "failure_evidence_summary": [],
            "service_evidence_summary": [],
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

    def _estimate_service_confidence(self, ranked_services: List[Tuple[str, float]]) -> str:
        if not ranked_services:
            return "low"

        s1 = ranked_services[0][1]
        s2 = ranked_services[1][1] if len(ranked_services) > 1 else 0.0
        service_margin = s1 / (s2 + 1e-9)

        if service_margin >= 2.5:
            return "high"
        if service_margin >= 1.4:
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

    def _confidence_rank(self, conf: str) -> int:
        order = {"low": 0, "medium": 1, "high": 2}
        return order.get(str(conf).lower(), 0)

    def _cap_failure_confidence(
        self,
        confidence: str,
        compressed_raw_telemetry: Optional[dict],
        failure_evidence_summary: List[str],
    ) -> str:
        level = self._confidence_rank(confidence)
        max_level = 2

        if not isinstance(compressed_raw_telemetry, dict):
            max_level = min(max_level, 0)
        elif compressed_raw_telemetry.get("status") != "ok":
            max_level = min(max_level, 0)

        if not failure_evidence_summary:
            max_level = min(max_level, 1)

        rev = {0: "low", 1: "medium", 2: "high"}
        return rev[min(level, max_level)]

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