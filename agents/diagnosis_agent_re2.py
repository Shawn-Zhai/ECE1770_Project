import json
import math
from collections import defaultdict
from typing import Any, Dict, List, Optional, Set, Tuple

from agents.diagnosis_agent import DiagnosisAgent


class DiagnosisAgentRE2(DiagnosisAgent):
    """
    RE2 diagnosis agent.

    Stage 1 is inherited from the existing metrics-first service localization
    logic. Stage 2 is replaced with a service-localized multimodal classifier
    that consumes RE2 metric/log/trace summaries instead of RE1-style raw-metric
    compression only.
    """

    ALLOWED_FAILURE_TYPES = {
        "cpu",
        "mem",
        "disk",
        "delay",
        "loss",
        "socket",
    }
    FAILURE_TYPE_ORDER = ("cpu", "mem", "disk", "delay", "loss", "socket")

    TRACE_OPERATION_EXCLUDE = (
        "opentelemetry.proto.collector.trace.v1.traceservice/export",
        "traceservice/export",
        "grpc.health",
        "health/check",
    )

    SOCKET_KEYWORDS = (
        "socket",
        "connection refused",
        "connection reset",
        "broken pipe",
        "econnreset",
        "econnrefused",
        "peer closed",
        "transport is closing",
        "connection aborted",
    )

    DELAY_KEYWORDS = (
        "timeout",
        "timed out",
        "deadline exceeded",
        "latency",
        "slow",
        "waiting",
        "retrying",
    )

    LOSS_KEYWORDS = (
        "error",
        "failed",
        "failure",
        "unavailable",
        "panic",
        "503",
        "500",
        "internal",
    )

    CPU_LOG_KEYWORDS = (
        "cpu",
        "throttle",
        "throttled",
        "high load",
        "overloaded",
        "busy",
    )

    MEM_LOG_KEYWORDS = (
        "oom",
        "out of memory",
        "memory",
        "heap",
        "gc overhead",
        "allocation",
    )

    DISK_LOG_KEYWORDS = (
        "disk",
        "i/o",
        "io error",
        "no space left",
        "filesystem",
        "read-only file system",
    )

    REQUEST_ERROR_LOG_KEYWORDS = (
        "request error",
        "upstream request timeout",
        "rpc error",
        "statuscode.deadline_exceeded",
        "deadline exceeded",
        "unavailable",
        "connection refused",
        "connection reset",
    )

    RESTART_LOG_KEYWORDS = (
        "starting ",
        "started on port",
        "server started",
        "listening on port",
        "tracing enabled",
        "profiler disabled",
        "dummy mode",
    )

    OBSERVABILITY_LOG_KEYWORDS = (
        "traces export",
        "exporting traces",
        "jaeger-collector",
        "opentelemetry",
        "telemetry:",
        "telemetry/",
        "otlp",
        "traceservice/export",
    )

    def __init__(
        self,
        use_llm_refinement: bool = True,
        debug: bool = False,
    ):
        super().__init__(
            use_llm_refinement=use_llm_refinement,
            debug=debug,
        )
        print("[DiagnosisAgentRE2] Stage-2 mode = localized multimodal RE2 diagnosis")

    def _filter_raw_telemetry(
        self,
        raw_telemetry: Optional[dict],
        faulty_service: str,
    ) -> Optional[dict]:
        if not faulty_service:
            return None
        # RE2 stage-2 evidence is derived from the preprocessed multimodal
        # state report. Keep a small marker object so the inherited diagnose()
        # flow still records that a localized stage-2 context was built.
        return {
            "faulty_service": faulty_service,
            "raw_telemetry_provided": bool(raw_telemetry),
            "mode": "re2_multimodal_state_report",
        }

    def _compress_raw_telemetry(
        self,
        filtered_raw_telemetry: Optional[dict],
        state_report: dict,
        faulty_service: str,
    ) -> Optional[dict]:
        metric_rows = self._localized_metric_facts(state_report, faulty_service)
        log_rows = self._localized_log_facts(state_report, faulty_service)
        trace_rows = self._localized_trace_facts(state_report, faulty_service)
        timeline_rows = self._localized_timeline(state_report, faulty_service)
        snapshot = self._service_snapshot(state_report, faulty_service)
        global_behavior_context = self._build_global_behavior_context(
            state_report=state_report,
            faulty_service=faulty_service,
        )
        type_discriminator_summary = self._build_type_discriminator_summary(
            faulty_service=faulty_service,
            metric_rows=metric_rows,
            log_rows=log_rows,
            trace_rows=trace_rows,
            timeline_rows=timeline_rows,
            global_behavior_context=global_behavior_context,
        )

        if not metric_rows and not log_rows and not trace_rows:
            return {
                "status": "insufficient_signal",
                "faulty_service": faulty_service,
                "localized_metric_facts": [],
                "localized_log_facts": [],
                "localized_trace_facts": [],
                "localized_timeline": [],
                "service_snapshot": snapshot,
                "global_behavior_context": global_behavior_context,
                "type_discriminator_summary": type_discriminator_summary,
                "instruction_hint": (
                    "No localized metric/log/trace evidence found for the chosen service."
                ),
                "meta": {
                    "metric_count": 0,
                    "log_count": 0,
                    "trace_count": 0,
                    "timeline_count": len(timeline_rows),
                },
            }

        return {
            "status": "ok",
            "faulty_service": faulty_service,
            "localized_metric_facts": metric_rows[:8],
            "localized_log_facts": log_rows[:8],
            "localized_trace_facts": trace_rows[:8],
            "localized_timeline": timeline_rows[:8],
            "service_snapshot": snapshot,
            "global_behavior_context": global_behavior_context,
            "type_discriminator_summary": type_discriminator_summary,
            "instruction_hint": (
                "Stage-2 evidence is localized to the chosen service and blends "
                "direct metric anomalies with service-scoped log and trace facts. "
                "Use the type discriminator summary first when comparing fault types."
            ),
            "meta": {
                "metric_count": len(metric_rows),
                "log_count": len(log_rows),
                "trace_count": len(trace_rows),
                "timeline_count": len(timeline_rows),
            },
        }

    def _build_service_evidence_claims(
        self,
        state_report: dict,
        faulty_service: str,
        stage1_result: dict,
    ) -> List[Dict[str, str]]:
        summary: List[str] = []

        snapshot = self._service_snapshot(state_report, faulty_service)
        if isinstance(snapshot, dict):
            for line in snapshot.get("metric_highlights", [])[:2]:
                summary.append(f"metric: {str(line)}")
            for line in snapshot.get("log_highlights", [])[:2]:
                summary.append(f"log: {str(line)}")
            for line in snapshot.get("trace_highlights", [])[:2]:
                summary.append(f"trace: {str(line)}")

        for row in self._localized_timeline(state_report, faulty_service)[:3]:
            summary.append(
                f"{row['source']} +{int(row['offset_sec'])}s: {str(row['summary'])}"
            )

        if not summary:
            return super()._build_service_evidence_claims(
                state_report=state_report,
                faulty_service=faulty_service,
                stage1_result=stage1_result,
            )

        return self._normalize_claims(
            self._dedupe_strings(summary, limit=self.MAX_EVIDENCE_CLAIMS),
            prefix="service_claim",
            default_type="observation",
        )

    def _rule_based_identify_failure_type(
        self,
        faulty_service: str,
        compressed_raw_telemetry: Optional[dict],
        service_evidence_claims: List[Dict[str, str]],
    ) -> Dict[str, Any]:
        empty_scores = self._zero_failure_type_scores()
        if not isinstance(compressed_raw_telemetry, dict):
            return {
                "failure_type": self._select_best_failure_type(empty_scores),
                "failure_evidence_claims": [],
                "service_evidence_claims": service_evidence_claims[: self.MAX_EVIDENCE_CLAIMS],
            }

        if compressed_raw_telemetry.get("status") != "ok":
            return {
                "failure_type": self._select_best_failure_type(empty_scores),
                "failure_evidence_claims": [],
                "service_evidence_claims": service_evidence_claims[: self.MAX_EVIDENCE_CLAIMS],
            }

        metric_rows = compressed_raw_telemetry.get("localized_metric_facts", [])
        log_rows = compressed_raw_telemetry.get("localized_log_facts", [])
        trace_rows = compressed_raw_telemetry.get("localized_trace_facts", [])

        if not isinstance(metric_rows, list):
            metric_rows = []
        if not isinstance(log_rows, list):
            log_rows = []
        if not isinstance(trace_rows, list):
            trace_rows = []

        type_scores: Dict[str, float] = self._zero_failure_type_scores()
        modality_support: Dict[str, Set[str]] = defaultdict(set)
        evidence: List[str] = []

        direct_metric_scores = {
            "cpu": 0.0,
            "mem": 0.0,
            "disk": 0.0,
            "socket": 0.0,
        }
        delay_trace_support = 0
        loss_log_support = 0
        socket_log_support = 0

        for rank, row in enumerate(metric_rows):
            if not isinstance(row, dict):
                continue

            metric_type = str(row.get("metric_type", "unknown")).strip().lower()
            mapped_type = self._map_metric_type_to_failure_type(metric_type)
            if mapped_type not in self.ALLOWED_FAILURE_TYPES:
                continue

            signal = self._metric_signal_strength(row)
            if signal <= 0.0:
                continue

            direction = self._metric_direction(row)
            rank_weight = max(0.55, 1.0 - 0.12 * rank)
            direction_weight = self._direction_weight_for_failure(mapped_type, direction)
            score = rank_weight * direction_weight * self._failure_metric_weight(mapped_type) * signal

            if mapped_type in direct_metric_scores:
                score *= 1.15
                direct_metric_scores[mapped_type] += score

            type_scores[mapped_type] += score
            modality_support[mapped_type].add("metrics")

            evidence.append(
                f"metric {row.get('metric_name', 'unknown')} favors {mapped_type} "
                f"(direction={direction}, score={score:.2f})"
            )

        for rank, row in enumerate(log_rows):
            if not isinstance(row, dict):
                continue

            combined_text = self._log_text_blob(row)
            socket_hits = self._keyword_hit_count(combined_text, self.SOCKET_KEYWORDS)
            delay_hits = self._keyword_hit_count(combined_text, self.DELAY_KEYWORDS)
            loss_hits = self._keyword_hit_count(combined_text, self.LOSS_KEYWORDS)

            incident_5xx = int(self._to_float(row.get("incident_http_5xx_count", 0)))
            incident_4xx = int(self._to_float(row.get("incident_http_4xx_count", 0)))
            keyword_hits = int(self._to_float(row.get("incident_keyword_hits", 0)))
            delta_count = max(0.0, self._to_float(row.get("delta_count", 0)))
            incident_ratio = max(1.0, self._to_float(row.get("incident_ratio", 1.0)))
            latency_delta_ms = max(0.0, self._to_float(row.get("latency_delta_ms", 0.0)))
            onset_bonus = self._onset_bonus(row.get("first_incident_offset_sec"))
            rank_weight = max(0.60, 1.0 - 0.08 * rank)

            severity_hint = str(row.get("severity_hint", "info")).strip().lower()
            severity_bonus = 0.0
            if severity_hint == "error":
                severity_bonus = 1.5
            elif severity_hint == "warn":
                severity_bonus = 0.75

            socket_score = onset_bonus * rank_weight * (
                3.5 * socket_hits
                + 0.12 * delta_count
                + 0.40 * severity_bonus
            )

            delay_score = onset_bonus * rank_weight * (
                2.0 * delay_hits
                + 0.04 * latency_delta_ms
                + 0.35 * max(0.0, incident_ratio - 1.0)
            )

            loss_base = (
                2.2 * incident_5xx
                + 0.8 * incident_4xx
                + 1.0 * keyword_hits
                + 1.2 * loss_hits
                + 0.05 * delta_count
                + severity_bonus
            )
            if socket_hits > 0:
                loss_base *= 0.55
            loss_score = onset_bonus * rank_weight * loss_base

            if socket_score > 0.0:
                type_scores["socket"] += socket_score
                modality_support["socket"].add("logs")
                socket_log_support += 1
                evidence.append(
                    f"log template favors socket "
                    f"(socket_hits={socket_hits}, score={socket_score:.2f})"
                )

            if delay_score > 0.0 and (delay_hits > 0 or latency_delta_ms > 0.0):
                type_scores["delay"] += delay_score
                modality_support["delay"].add("logs")
                evidence.append(
                    f"log template favors delay "
                    f"(delay_hits={delay_hits}, latency_delta_ms={latency_delta_ms:.2f}, score={delay_score:.2f})"
                )

            if loss_score > 0.0 and (
                incident_5xx > 0 or incident_4xx > 0 or keyword_hits > 0 or loss_hits > 0
            ):
                type_scores["loss"] += loss_score
                modality_support["loss"].add("logs")
                loss_log_support += 1
                evidence.append(
                    f"log template favors loss "
                    f"(5xx={incident_5xx}, keywords={keyword_hits}, score={loss_score:.2f})"
                )

        for rank, row in enumerate(trace_rows):
            if not isinstance(row, dict):
                continue

            operation = str(row.get("operation", "unknown"))
            if self._is_trace_noise(operation):
                continue

            delta_p95_ms = max(0.0, self._to_float(row.get("delta_p95_ms", 0.0)))
            latency_ratio = max(1.0, self._to_float(row.get("latency_ratio", 1.0)))
            onset_bonus = self._onset_bonus(row.get("first_incident_offset_sec"))
            rank_weight = max(0.60, 1.0 - 0.08 * rank)

            delay_score = onset_bonus * rank_weight * (
                0.06 * delta_p95_ms
                + 2.4 * max(0.0, min(latency_ratio, 12.0) - 1.0)
            )

            if delay_score > 0.0:
                type_scores["delay"] += delay_score
                modality_support["delay"].add("traces")
                delay_trace_support += 1
                evidence.append(
                    f"trace {operation} favors delay "
                    f"(delta_p95_ms={delta_p95_ms:.2f}, ratio={latency_ratio:.2f}, score={delay_score:.2f})"
                )

        ranked_resource_metrics = sorted(
            (
                (failure_type, score)
                for failure_type, score in direct_metric_scores.items()
                if score > 0.0
            ),
            key=lambda item: item[1],
            reverse=True,
        )

        if ranked_resource_metrics:
            winner, winner_score = ranked_resource_metrics[0]
            runner_up_score = ranked_resource_metrics[1][1] if len(ranked_resource_metrics) > 1 else 0.0

            if winner_score >= max(10.0, 1.35 * runner_up_score):
                dominance_bonus = 0.25 * winner_score
                type_scores[winner] += dominance_bonus
                modality_support[winner].add("metric_dominance")
                evidence.append(
                    f"direct metric dominance favors {winner} "
                    f"(winner={winner_score:.2f}, runner_up={runner_up_score:.2f})"
                )

                if winner in {"cpu", "mem", "disk", "socket"} and type_scores["delay"] > 0.0:
                    propagation_bonus = min(type_scores["delay"] * 0.25, winner_score * 0.35)
                    type_scores[winner] += propagation_bonus
                    evidence.append(
                        f"direct {winner} signal retained despite downstream latency "
                        f"(bonus={propagation_bonus:.2f})"
                    )

        if delay_trace_support >= 2:
            breadth_bonus = min(8.0, 1.5 * float(delay_trace_support))
            type_scores["delay"] += breadth_bonus
            modality_support["delay"].add("trace_breadth")
            evidence.append(
                f"multiple direct trace operations favor delay (bonus={breadth_bonus:.2f})"
            )

        if loss_log_support >= 2:
            error_bonus = 1.5 * float(loss_log_support)
            type_scores["loss"] += error_bonus
            modality_support["loss"].add("log_breadth")
            evidence.append(
                f"multiple direct log templates favor loss (bonus={error_bonus:.2f})"
            )

        if socket_log_support >= 1 and direct_metric_scores["socket"] > 0.0:
            socket_bonus = 0.20 * direct_metric_scores["socket"]
            type_scores["socket"] += socket_bonus
            modality_support["socket"].add("metric_log_agreement")
            evidence.append(
                f"socket metric and socket-like logs agree (bonus={socket_bonus:.2f})"
            )

        ranked = self._rank_failure_types(type_scores)
        best_type = ranked[0][0]
        return {
            "failure_type": best_type,
            "failure_evidence_claims": self._normalize_claims(
                self._dedupe_strings(
                evidence
                if evidence
                else [
                    "Localized evidence did not strongly separate the six RE2 fault types."
                ],
                limit=self.MAX_EVIDENCE_CLAIMS,
                ),
                prefix="failure_claim",
                default_type="inference",
            ),
            "service_evidence_claims": service_evidence_claims[: self.MAX_EVIDENCE_CLAIMS],
        }

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
            "type_discriminator_summary": (
                (compressed_raw_telemetry or {}).get("type_discriminator_summary", {})
                if isinstance(compressed_raw_telemetry, dict)
                else {}
            ),
            "localized_multimodal_evidence": compressed_raw_telemetry or {},
        }

        system_msg = (
            "You are a diagnosis agent for root cause analysis.\n\n"
            "The faulty service has already been localized.\n"
            "Your task in this stage is ONLY to decide the failure type for that chosen service.\n\n"
            "Rules:\n"
            "- Output JSON only.\n"
            '- Use exactly these keys: "failure_type", "failure_evidence_claims".\n'
            '- "failure_type" must be one of: "cpu", "mem", "disk", "delay", "loss", "socket".\n'
            '- "failure_evidence_claims" must be a JSON array with 1 to 3 atomic claim objects.\n'
            '- Each claim object must use exactly these keys: "id", "text", "type".\n'
            '- "type" must be "observation" or "inference".\n'
            '- Each "text" must be one short verifiable statement.\n'
            "- Do NOT relocalize service.\n"
            "- Use type_discriminator_summary as the primary comparison table before reading raw facts.\n"
            "- Use type_discriminator_summary only for internal reasoning. Do not quote its scores, "
            "ranks, strongest_direct_resource_candidate field, top_hypotheses, or behavior override "
            "guidance in evidence claims.\n"
            "- Start from strongest_direct_resource_candidate, behavior_override_guidance, and the "
            "per-family decision_score values.\n"
            "- Evidence claims must cite telemetry facts or simple statistics that can be computed "
            "from raw telemetry, such as means, peaks, p95, counts, count drops, restart frequency, "
            "or onset offsets.\n"
            "- Do not mention internal reasoning artifacts in evidence claims, including decision "
            "scores, direct_metric_score, discriminators, rankings, top hypotheses, normalized scores, "
            "or behavior overrides.\n"
            "- Treat behavior labels (delay/loss) as override labels, not defaults.\n"
            "- Choose delay or loss only when behavior_override_guidance says behavior can override "
            "the direct resource candidate, or when direct resource evidence is weak.\n"
            "- Compare cpu vs mem vs disk vs socket using the strongest direct metric, onset timing, "
            "sustainedness, and corroborating logs/traces.\n"
            "- Compare delay vs loss using latency amplification, request preservation, and "
            "error/availability evidence, but do not let preserved-flow latency alone override a "
            "strong direct resource signal.\n"
            "- Loss does not require localized 5xx logs. Treat dropped successful request counts, "
            "restart-after-injection behavior, missing request paths, or downstream request-error bursts "
            "as valid loss evidence.\n"
            "- Prefer direct evidence on the localized service over downstream symptoms.\n"
            "- Do not choose based only on the largest raw magnitude spike.\n"
            "- CPU/mem/disk/socket are direct service-level fault types.\n"
            "- Delay means latency-dominant degradation with preserved request flow.\n"
            "- Loss means error-dominant or availability-loss behavior.\n"
            "- Socket means connection/socket saturation or transport-level failures.\n"
            "- CPU is a generic secondary symptom; if cpu co-occurs with a strong and early "
            "socket/mem/disk metric, prefer the more specific family unless the raw facts clearly "
            "contradict it.\n"
            "- Delay is often a downstream effect of cpu/mem/disk/socket faults; do not choose delay "
            "when a strong direct resource metric appears early on the localized service.\n"
            "- Ignore observability export spans and health checks as root-cause evidence.\n"
            "- Socket faults can create secondary CPU and latency spikes; do not default to cpu "
            "when direct socket evidence exists.\n"
            "- Memory faults can create secondary disk I/O spikes from swapping or pressure; do not "
            "default to disk only because disk I/O is larger.\n"
            "- Disk faults can create secondary CPU spikes via retries or blocking; choose the label "
            "with the most direct and earliest storage evidence.\n"
            "- If evidence is weak or mixed, still choose the best-supported label.\n"
            "- Claims must be non-overlapping and limited to the strongest 3 pieces of evidence.\n"
            "- At least one failure evidence claim must explicitly explain why the chosen label is "
            "better supported than the strongest alternative label.\n"
            "- Decide from the provided stage-2 input itself, not from a prior guess.\n"
            "- Do not include extra keys."
        )

        user_msg = (
            "Narrowed-scope RE2 diagnosis input for stage-2 failure-type decision:\n\n"
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

        llm_type = self._coerce_failure_type(
            value=parsed.get("failure_type", ""),
            fallback=self._select_best_failure_type(self._zero_failure_type_scores()),
        )

        return {
            "failure_type": llm_type,
            "failure_evidence_claims": self._normalize_claims(
                parsed.get("failure_evidence_claims", []),
                prefix="failure_claim",
                default_type="observation",
            ),
        }

    def _metric_type_weight(self, metric_type: str) -> float:
        metric_family = self._canonical_metric_family(metric_type)
        if metric_family in {"cpu", "mem", "disk", "socket", "loss"}:
            return 1.00
        if metric_family == "delay":
            return 1.00
        return 0.30

    def _failure_metric_weight(self, failure_type: str) -> float:
        if failure_type in {"cpu", "mem", "disk", "socket"}:
            return 1.00
        if failure_type == "delay":
            return 0.95
        if failure_type == "loss":
            return 0.90
        return 0.20

    def _map_metric_type_to_failure_type(self, metric_type: str) -> str:
        return self._canonical_metric_family(metric_type)

    def _normalize_failure_type(self, value: Any) -> str:
        text = str(value).strip().lower()
        if text in self.ALLOWED_FAILURE_TYPES:
            return text
        if text in {"latency", "slow", "slowdown"}:
            return "delay"
        if text in {"error", "errors", "availability"}:
            return "loss"
        if text in {"network", "connection", "transport"}:
            return "socket"
        return ""

    def _empty_stage2_result(self) -> Dict[str, Any]:
        empty_scores = self._zero_failure_type_scores()
        return {
            "failure_type": self._select_best_failure_type(empty_scores),
            "failure_evidence_claims": [],
            "service_evidence_claims": [],
        }

    def _localized_metric_facts(
        self,
        state_report: Dict[str, Any],
        faulty_service: str,
    ) -> List[Dict[str, Any]]:
        rows = []
        for row in state_report.get("metric_facts", []):
            if isinstance(row, dict) and self._matches_service(row.get("service"), faulty_service):
                rows.append(self._annotate_metric_row(row))
        rows.sort(key=self._metric_priority_score, reverse=True)
        return rows

    def _localized_log_facts(
        self,
        state_report: Dict[str, Any],
        faulty_service: str,
    ) -> List[Dict[str, Any]]:
        rows = [
            row for row in state_report.get("log_facts", [])
            if isinstance(row, dict) and self._matches_service(row.get("service"), faulty_service)
        ]
        rows.sort(key=self._log_priority_score, reverse=True)
        return rows

    def _localized_trace_facts(
        self,
        state_report: Dict[str, Any],
        faulty_service: str,
    ) -> List[Dict[str, Any]]:
        rows = [
            row for row in state_report.get("trace_facts", [])
            if isinstance(row, dict)
            and self._matches_service(row.get("service"), faulty_service)
            and not self._is_trace_noise(str(row.get("operation", "")))
        ]
        rows.sort(key=self._trace_priority_score, reverse=True)
        return rows

    def _localized_timeline(
        self,
        state_report: Dict[str, Any],
        faulty_service: str,
    ) -> List[Dict[str, Any]]:
        rows = [
            row for row in state_report.get("incident_timeline", [])
            if isinstance(row, dict) and self._matches_service(row.get("service"), faulty_service)
        ]
        rows.sort(key=lambda row: int(self._to_float(row.get("offset_sec", 10 ** 9))))
        return rows

    def _service_snapshot(
        self,
        state_report: Dict[str, Any],
        faulty_service: str,
    ) -> Dict[str, Any]:
        snapshots = state_report.get("service_snapshots", [])
        if not isinstance(snapshots, list):
            return {}

        for row in snapshots:
            if isinstance(row, dict) and self._matches_service(row.get("service"), faulty_service):
                return row
        return {}

    def _build_global_behavior_context(
        self,
        state_report: Dict[str, Any],
        faulty_service: str,
    ) -> Dict[str, Any]:
        signals: List[str] = []
        total_score = 0.0

        log_rows = [
            row for row in state_report.get("log_facts", [])
            if isinstance(row, dict) and not self._matches_service(row.get("service"), faulty_service)
        ]
        log_rows.sort(key=self._log_priority_score, reverse=True)

        for rank, row in enumerate(log_rows[:12]):
            combined_text = self._log_text_blob(row)
            baseline_count = max(0.0, self._to_float(row.get("baseline_count", 0.0)))
            incident_count = max(0.0, self._to_float(row.get("incident_count", 0.0)))
            count_drop_strength = self._count_drop_strength(baseline_count, incident_count)
            request_error_hits = self._keyword_hit_count(
                combined_text,
                self.REQUEST_ERROR_LOG_KEYWORDS,
            )
            incident_5xx = int(self._to_float(row.get("incident_http_5xx_count", 0)))
            incident_4xx = int(self._to_float(row.get("incident_http_4xx_count", 0)))
            severity_hint = str(row.get("severity_hint", "info")).strip().lower()
            severity_bonus = 1.0 if severity_hint == "error" else 0.5 if severity_hint == "warn" else 0.0
            rank_weight = max(0.55, 1.0 - 0.08 * rank)

            signal_score = rank_weight * (
                2.5 * request_error_hits
                + 1.8 * incident_5xx
                + 0.8 * incident_4xx
                + 3.0 * count_drop_strength
                + severity_bonus
            )
            if signal_score <= 0.0:
                continue

            total_score += signal_score
            service = str(row.get("service", "unknown"))
            if count_drop_strength >= 0.15 and baseline_count > 0.0:
                signals.append(
                    f"log count dropped on {service} ({int(baseline_count)} -> {int(incident_count)})"
                )
            elif request_error_hits > 0 or incident_5xx > 0 or incident_4xx > 0:
                signals.append(
                    f"log errors increased on {service} (5xx={incident_5xx}, 4xx={incident_4xx}, request_error_hits={request_error_hits})"
                )
            if len(signals) >= 3:
                break

        trace_rows = [
            row for row in state_report.get("trace_facts", [])
            if isinstance(row, dict)
            and not self._matches_service(row.get("service"), faulty_service)
            and not self._is_trace_noise(str(row.get("operation", "")))
        ]
        trace_rows.sort(key=self._trace_priority_score, reverse=True)

        for rank, row in enumerate(trace_rows[:12]):
            baseline_count = max(0.0, self._to_float(row.get("baseline_count", 0.0)))
            incident_count = max(0.0, self._to_float(row.get("incident_count", 0.0)))
            count_drop_strength = self._count_drop_strength(baseline_count, incident_count)
            if count_drop_strength < 0.15:
                continue

            latency_ratio = max(1.0, self._to_float(row.get("latency_ratio", 1.0)))
            rank_weight = max(0.55, 1.0 - 0.08 * rank)
            total_score += rank_weight * (
                3.0 * count_drop_strength
                + 0.3 * max(0.0, min(latency_ratio, 8.0) - 1.0)
            )
            signals.append(
                f"trace counts dropped on {row.get('service', 'unknown')}::{row.get('operation', 'unknown')} "
                f"({int(baseline_count)} -> {int(incident_count)})"
            )
            if len(signals) >= 3:
                break

        return {
            "supported": bool(signals),
            "score": round(total_score, 4),
            "signals": self._dedupe_strings(signals, limit=3),
        }

    def _matches_service(self, candidate: Any, target: str) -> bool:
        candidate_norm = self._normalize_service_name(candidate)
        if not candidate_norm:
            return False
        return candidate_norm in self._service_aliases(target)

    def _service_aliases(self, service: Any) -> Set[str]:
        base = self._normalize_service_name(service)
        aliases = {base} if base else set()
        if not base:
            return aliases

        if base.endswith("-service"):
            trimmed = base[:-8]
            if trimmed:
                aliases.add(trimmed)
        elif base.endswith("service") and "-" not in base:
            trimmed = base[:-7]
            if trimmed:
                aliases.add(trimmed)

        if "-" in base:
            aliases.add(f"{base}-service")
        else:
            aliases.add(f"{base}service")

        return {alias for alias in aliases if alias}

    def _normalize_service_name(self, value: Any) -> str:
        if value is None:
            return ""
        return str(value).strip().lower()

    def _canonical_metric_family(self, value: Any) -> str:
        text = str(value).strip().lower()
        if not text:
            return ""
        if text in self.ALLOWED_FAILURE_TYPES:
            return text
        if text == "cpu" or "cpu" in text:
            return "cpu"
        if text in {"mem", "memory"} or "mem" in text:
            return "mem"
        if text == "socket" or "socket" in text or "conn" in text:
            return "socket"
        if text == "disk" or "disk" in text or text.endswith("io") or text == "iops":
            return "disk"
        if text.startswith("latency") or text in {"delay", "duration", "response_time", "response-time"}:
            return "delay"
        if text in {"error", "errors", "loss", "load"} or "error" in text or "fail" in text:
            return "loss"
        return ""

    def _annotate_metric_row(self, row: Dict[str, Any]) -> Dict[str, Any]:
        annotated = dict(row)
        metric_family = self._canonical_metric_family(
            annotated.get("metric_family", annotated.get("metric_type", ""))
        )
        signal_score = self._metric_signal_strength(annotated)
        annotated["metric_family"] = metric_family
        annotated["direction"] = annotated.get("direction") or self._metric_direction(annotated)
        annotated["signal_score"] = round(signal_score, 6)
        annotated["normalized_signal_score"] = round(
            self._normalize_signal_score(signal_score),
            4,
        )
        annotated["first_incident_offset_sec"] = annotated.get("first_incident_offset_sec")
        annotated["peak_offset_sec"] = annotated.get("peak_offset_sec")
        annotated["sustained_after_inject"] = bool(
            annotated.get("sustained_after_inject", False)
        )
        return annotated

    def _normalize_signal_score(self, raw_score: float) -> float:
        score = max(0.0, float(raw_score))
        return 1.0 - math.exp(-score / 35.0)

    def _metric_temporal_bonus(self, row: Dict[str, Any]) -> float:
        bonus = 1.0
        offset = row.get("first_incident_offset_sec")
        if offset is None:
            bonus *= 0.85
        else:
            bonus *= self._onset_bonus(offset)

        if bool(row.get("sustained_after_inject", False)):
            bonus *= 1.10

        first_offset = row.get("first_incident_offset_sec")
        peak_offset = row.get("peak_offset_sec")
        if first_offset is not None and peak_offset is not None:
            if abs(int(self._to_float(peak_offset)) - int(self._to_float(first_offset))) <= 30:
                bonus *= 1.03
        return bonus

    def _metric_priority_score(self, row: Dict[str, Any]) -> float:
        signal = float(row.get("signal_score", self._metric_signal_strength(row)))
        family = self._canonical_metric_family(
            row.get("metric_family", row.get("metric_type", ""))
        )
        family_bonus = 1.08 if family in {"cpu", "mem", "disk", "socket"} else 1.0
        return signal * self._metric_temporal_bonus(row) * family_bonus

    def _metric_support_score(self, row: Optional[Dict[str, Any]]) -> float:
        if not isinstance(row, dict):
            return 0.0
        normalized_signal = self._to_float(row.get("normalized_signal_score", 0.0))
        percent_change = abs(self._to_float(row.get("percent_change", 0.0)))
        percent_bonus = min(percent_change / 250.0, 2.0)
        sustained_bonus = 1.0 if bool(row.get("sustained_after_inject", False)) else 0.0
        return (
            6.0 * normalized_signal + percent_bonus + sustained_bonus
        ) * self._metric_temporal_bonus(row)

    def _summarize_metric_for_discriminator(
        self,
        row: Optional[Dict[str, Any]],
    ) -> Optional[Dict[str, Any]]:
        if not isinstance(row, dict):
            return None
        return {
            "metric_name": str(row.get("metric_name", "unknown")),
            "metric_type": str(row.get("metric_type", "unknown")),
            "metric_family": self._canonical_metric_family(
                row.get("metric_family", row.get("metric_type", ""))
            ),
            "baseline_mean": round(self._to_float(row.get("baseline_mean", 0.0)), 6),
            "incident_mean": round(self._to_float(row.get("incident_mean", 0.0)), 6),
            "percent_change": round(self._to_float(row.get("percent_change", 0.0)), 4),
            "normalized_signal_score": round(
                self._to_float(row.get("normalized_signal_score", 0.0)),
                4,
            ),
            "first_incident_offset_sec": row.get("first_incident_offset_sec"),
            "peak_offset_sec": row.get("peak_offset_sec"),
            "sustained_after_inject": bool(row.get("sustained_after_inject", False)),
        }

    def _empty_family_support(self, support_kind: str = "direct") -> Dict[str, Any]:
        return {
            "supported": False,
            "support_kind": support_kind,
            "score": 0.0,
            "signals": [],
        }

    def _append_support_signal(
        self,
        bucket: Dict[str, Any],
        signal_text: str,
    ) -> None:
        text = str(signal_text).strip()
        if not text:
            return
        existing = {str(item).strip() for item in bucket["signals"]}
        if text in existing:
            return
        if len(bucket["signals"]) < 2:
            bucket["signals"].append(text)

    def _metric_signal_strength(self, row: Dict[str, Any]) -> float:
        baseline_mean = self._to_float(row.get("baseline_mean", 0.0))
        incident_mean = self._to_float(row.get("incident_mean", 0.0))
        baseline_std = abs(self._to_float(row.get("baseline_std", 0.0)))
        incident_peak = self._to_float(row.get("incident_peak", incident_mean))

        mean_change = incident_mean - baseline_mean
        z_mean = abs(self._safe_z(mean_change, baseline_std))
        z_peak = abs(self._safe_z(incident_peak - baseline_mean, baseline_std))
        change_ratio = abs(mean_change) / (abs(baseline_mean) + 1e-9)
        return 0.8 * min(max(z_mean, z_peak), 150.0) + 0.2 * min(change_ratio, 50.0)

    def _metric_direction(self, row: Dict[str, Any]) -> str:
        baseline_mean = self._to_float(row.get("baseline_mean", 0.0))
        incident_mean = self._to_float(row.get("incident_mean", baseline_mean))
        mean_change = self._to_float(row.get("mean_change", incident_mean - baseline_mean))
        if mean_change > 0:
            return "up"
        if mean_change < 0:
            return "down"
        return "flat"

    def _direction_weight_for_failure(self, failure_type: str, direction: str) -> float:
        if failure_type in {"cpu", "mem", "disk", "socket"}:
            return 1.0 if direction == "up" else 0.35
        if failure_type == "delay":
            return 1.0 if direction == "up" else 0.50
        if failure_type == "loss":
            return 1.0 if direction in {"up", "down"} else 0.50
        return 0.25

    def _log_priority_score(self, row: Dict[str, Any]) -> float:
        combined_text = self._log_text_blob(row)
        baseline_count = max(0.0, self._to_float(row.get("baseline_count", 0.0)))
        incident_count = max(0.0, self._to_float(row.get("incident_count", 0.0)))
        count_drop_strength = self._count_drop_strength(baseline_count, incident_count)
        restart_hits = self._keyword_hit_count(combined_text, self.RESTART_LOG_KEYWORDS)
        anomaly_like = 0.0
        anomaly_like += 2.2 * self._to_float(row.get("incident_http_5xx_count", 0))
        anomaly_like += 1.0 * self._to_float(row.get("incident_keyword_hits", 0))
        anomaly_like += 0.04 * max(0.0, self._to_float(row.get("latency_delta_ms", 0.0)))
        anomaly_like += 0.05 * max(0.0, self._to_float(row.get("delta_count", 0.0)))
        anomaly_like += 4.0 * count_drop_strength
        if restart_hits > 0 and bool(row.get("new_in_incident", False)):
            anomaly_like += 2.5 * restart_hits
        if bool(row.get("new_in_incident", False)):
            anomaly_like += 3.0
        if self._is_observability_log(combined_text) and restart_hits == 0:
            anomaly_like *= 0.35
        return anomaly_like

    def _trace_priority_score(self, row: Dict[str, Any]) -> float:
        delta_p95_ms = max(0.0, self._to_float(row.get("delta_p95_ms", 0.0)))
        latency_ratio = max(1.0, self._to_float(row.get("latency_ratio", 1.0)))
        baseline_count = max(0.0, self._to_float(row.get("baseline_count", 0.0)))
        incident_count = max(0.0, self._to_float(row.get("incident_count", 0.0)))
        count_drop_strength = self._count_drop_strength(baseline_count, incident_count)
        preserved_flow = self._preserved_flow_factor(baseline_count, incident_count)
        latency_strength = min(delta_p95_ms / 250.0, 8.0)
        return (
            preserved_flow * (
                1.2 * latency_strength
                + 1.8 * max(0.0, min(latency_ratio, 12.0) - 1.0)
            )
            + 5.0 * count_drop_strength
        )

    def _log_text_blob(self, row: Dict[str, Any]) -> str:
        parts = [str(row.get("template", ""))]
        for msg in row.get("representative_messages", [])[:3]:
            parts.append(str(msg))
        return " | ".join(parts).lower()

    def _keyword_hit_count(self, text: str, keywords: Tuple[str, ...]) -> int:
        lowered = str(text).lower()
        return sum(1 for keyword in keywords if keyword in lowered)

    def _is_observability_log(self, text: str) -> bool:
        lowered = str(text).lower()
        return any(keyword in lowered for keyword in self.OBSERVABILITY_LOG_KEYWORDS)

    def _count_drop_strength(
        self,
        baseline_count: float,
        incident_count: float,
    ) -> float:
        if baseline_count <= 0.0:
            return 0.0
        ratio = incident_count / max(baseline_count, 1.0)
        if ratio >= 0.90:
            return 0.0
        return min((0.90 - ratio) / 0.45, 1.0)

    def _preserved_flow_factor(
        self,
        baseline_count: float,
        incident_count: float,
    ) -> float:
        if baseline_count <= 0.0:
            return 1.0
        ratio = incident_count / max(baseline_count, 1.0)
        if ratio >= 0.90:
            return 1.0
        if ratio <= 0.65:
            return 0.08
        return 0.08 + 0.92 * ((ratio - 0.65) / 0.25)

    def _build_log_family_support(self, log_rows: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
        support = {
            failure_type: self._empty_family_support("direct")
            for failure_type in self.FAILURE_TYPE_ORDER
        }

        for rank, row in enumerate(log_rows[:8]):
            if not isinstance(row, dict):
                continue

            combined_text = self._log_text_blob(row)
            onset_bonus = self._onset_bonus(row.get("first_incident_offset_sec"))
            rank_weight = max(0.60, 1.0 - 0.08 * rank)
            severity_hint = str(row.get("severity_hint", "info")).strip().lower()
            severity_bonus = 1.5 if severity_hint == "error" else 0.75 if severity_hint == "warn" else 0.0

            baseline_count = max(0.0, self._to_float(row.get("baseline_count", 0.0)))
            incident_count = max(0.0, self._to_float(row.get("incident_count", 0.0)))
            incident_5xx = int(self._to_float(row.get("incident_http_5xx_count", 0)))
            incident_4xx = int(self._to_float(row.get("incident_http_4xx_count", 0)))
            keyword_hits = int(self._to_float(row.get("incident_keyword_hits", 0)))
            delta_count = max(0.0, self._to_float(row.get("delta_count", 0.0)))
            incident_ratio = max(1.0, self._to_float(row.get("incident_ratio", 1.0)))
            latency_delta_ms = max(0.0, self._to_float(row.get("latency_delta_ms", 0.0)))
            count_drop_strength = self._count_drop_strength(baseline_count, incident_count)
            preserved_flow = max(
                0.25,
                self._preserved_flow_factor(baseline_count, incident_count),
            )

            socket_hits = self._keyword_hit_count(combined_text, self.SOCKET_KEYWORDS)
            delay_hits = self._keyword_hit_count(combined_text, self.DELAY_KEYWORDS)
            loss_hits = self._keyword_hit_count(combined_text, self.LOSS_KEYWORDS)
            cpu_hits = self._keyword_hit_count(combined_text, self.CPU_LOG_KEYWORDS)
            mem_hits = self._keyword_hit_count(combined_text, self.MEM_LOG_KEYWORDS)
            disk_hits = self._keyword_hit_count(combined_text, self.DISK_LOG_KEYWORDS)
            request_error_hits = self._keyword_hit_count(combined_text, self.REQUEST_ERROR_LOG_KEYWORDS)
            restart_hits = self._keyword_hit_count(combined_text, self.RESTART_LOG_KEYWORDS)
            observability_only = self._is_observability_log(combined_text) and restart_hits == 0

            contributions = {
                "socket": onset_bonus * rank_weight * (
                    3.5 * socket_hits + 0.12 * delta_count + 0.40 * severity_bonus
                ),
                "delay": preserved_flow * onset_bonus * rank_weight * (
                    2.0 * delay_hits
                    + 0.04 * latency_delta_ms
                    + 0.35 * max(0.0, incident_ratio - 1.0)
                ),
                "loss": onset_bonus * rank_weight * (
                    2.2 * incident_5xx
                    + 0.8 * incident_4xx
                    + 1.0 * keyword_hits
                    + 1.2 * loss_hits
                    + 1.6 * request_error_hits
                    + 2.4 * count_drop_strength
                    + (1.8 * restart_hits if bool(row.get("new_in_incident", False)) else 0.0)
                    + severity_bonus
                ),
                "cpu": onset_bonus * rank_weight * (1.5 * cpu_hits + 0.20 * severity_bonus),
                "mem": onset_bonus * rank_weight * (1.8 * mem_hits + 0.20 * severity_bonus),
                "disk": onset_bonus * rank_weight * (1.8 * disk_hits + 0.20 * severity_bonus),
            }

            if observability_only:
                contributions["delay"] *= 0.35
                contributions["loss"] *= 0.45

            for failure_type, score in contributions.items():
                if score <= 0.0:
                    continue
                bucket = support[failure_type]
                bucket["supported"] = True
                bucket["score"] = round(float(bucket["score"]) + score, 4)
                if failure_type == "socket":
                    self._append_support_signal(
                        bucket,
                        f"log evidence shows socket-like transport symptoms (hits={socket_hits})",
                    )
                elif failure_type == "delay":
                    self._append_support_signal(
                        bucket,
                        f"log evidence shows delay-like behavior (latency_delta_ms={latency_delta_ms:.2f}, hits={delay_hits})",
                    )
                elif failure_type == "loss":
                    count_drop_note = (
                        f", count_drop={count_drop_strength:.2f}"
                        if count_drop_strength > 0.0
                        else ""
                    )
                    restart_note = (
                        f", restart_hits={restart_hits}"
                        if restart_hits > 0 and bool(row.get("new_in_incident", False))
                        else ""
                    )
                    self._append_support_signal(
                        bucket,
                        f"log evidence shows error or availability symptoms (5xx={incident_5xx}, keywords={keyword_hits + loss_hits + request_error_hits}{count_drop_note}{restart_note})",
                    )
                else:
                    self._append_support_signal(
                        bucket,
                        f"log text contains {failure_type}-related signals on the localized service",
                    )

        return support

    def _build_trace_family_support(self, trace_rows: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
        support = {
            failure_type: self._empty_family_support("direct")
            for failure_type in self.FAILURE_TYPE_ORDER
        }

        for rank, row in enumerate(trace_rows[:8]):
            if not isinstance(row, dict):
                continue

            delta_p95_ms = max(0.0, self._to_float(row.get("delta_p95_ms", 0.0)))
            latency_ratio = max(1.0, self._to_float(row.get("latency_ratio", 1.0)))
            baseline_count = max(0.0, self._to_float(row.get("baseline_count", 0.0)))
            incident_count = max(0.0, self._to_float(row.get("incident_count", 0.0)))
            onset_bonus = self._onset_bonus(row.get("first_incident_offset_sec"))
            rank_weight = max(0.60, 1.0 - 0.08 * rank)
            count_drop_strength = self._count_drop_strength(baseline_count, incident_count)
            preserved_flow = self._preserved_flow_factor(baseline_count, incident_count)
            latency_strength = min(delta_p95_ms / 250.0, 8.0)

            delay_score = preserved_flow * onset_bonus * rank_weight * (
                1.2 * latency_strength
                + 1.8 * max(0.0, min(latency_ratio, 12.0) - 1.0)
            )
            if delay_score > 0.0:
                bucket = support["delay"]
                bucket["supported"] = True
                bucket["score"] = round(float(bucket["score"]) + delay_score, 4)
                self._append_support_signal(
                    bucket,
                    f"trace latency increased on {row.get('operation', 'unknown')} (p95_delta_ms={delta_p95_ms:.2f}, ratio={latency_ratio:.2f})",
                )

            if count_drop_strength > 0.0:
                loss_score = onset_bonus * rank_weight * (
                    4.0 * count_drop_strength
                    + 1.2 * count_drop_strength * max(0.0, min(latency_ratio, 8.0) - 1.0)
                )
                if loss_score > 0.0:
                    bucket = support["loss"]
                    bucket["supported"] = True
                    bucket["score"] = round(float(bucket["score"]) + loss_score, 4)
                    self._append_support_signal(
                        bucket,
                        f"trace counts dropped on {row.get('operation', 'unknown')} ({int(baseline_count)} -> {int(incident_count)})",
                    )

        return support

    def _build_indirect_resource_trace_support(
        self,
        trace_rows: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        bucket = self._empty_family_support("indirect")
        for row in trace_rows[:6]:
            if not isinstance(row, dict):
                continue
            baseline_count = max(0.0, self._to_float(row.get("baseline_count", 0.0)))
            incident_count = max(0.0, self._to_float(row.get("incident_count", 0.0)))
            latency_ratio = max(1.0, self._to_float(row.get("latency_ratio", 1.0)))
            delta_p95_ms = max(0.0, self._to_float(row.get("delta_p95_ms", 0.0)))
            if baseline_count > 0 and incident_count >= 0.75 * baseline_count and (
                latency_ratio > 1.3 or delta_p95_ms > 20.0
            ):
                bucket["supported"] = True
                self._append_support_signal(
                    bucket,
                    "traces show preserved-flow latency impact, which is compatible with a direct resource fault but not specific to one resource family",
                )
                break
        return bucket

    def _metric_onset_offset_sec(
        self,
        row: Optional[Dict[str, Any]],
    ) -> Optional[float]:
        if not isinstance(row, dict):
            return None
        value = row.get("first_incident_offset_sec")
        if value is None:
            return None
        return self._to_float(value)

    def _normalized_score(self, raw_score: float, max_score: float) -> float:
        score = max(0.0, float(raw_score))
        if max_score <= 0.0:
            return 0.0
        return min(score / max_score, 1.0)

    def _absolute_direct_metric_strength(self, raw_score: float) -> float:
        return min(max(0.0, float(raw_score)) / 8.0, 1.0)

    def _metric_percent_strength(self, row: Optional[Dict[str, Any]]) -> float:
        if not isinstance(row, dict):
            return 0.0
        percent_change = row.get("percent_change")
        if percent_change is None:
            baseline_mean = abs(self._to_float(row.get("baseline_mean", 0.0)))
            incident_mean = abs(self._to_float(row.get("incident_mean", 0.0)))
            return 1.0 if baseline_mean <= 1e-9 and incident_mean > 0.0 else 0.0
        return min(abs(self._to_float(percent_change)) / 200.0, 1.0)

    def _build_type_discriminator_summary(
        self,
        faulty_service: str,
        metric_rows: List[Dict[str, Any]],
        log_rows: List[Dict[str, Any]],
        trace_rows: List[Dict[str, Any]],
        timeline_rows: List[Dict[str, Any]],
        global_behavior_context: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        metrics_by_family: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        for row in metric_rows:
            metrics_by_family[str(row.get("metric_family", ""))].append(row)
        for family_rows in metrics_by_family.values():
            family_rows.sort(key=self._metric_priority_score, reverse=True)

        log_support = self._build_log_family_support(log_rows)
        trace_support = self._build_trace_family_support(trace_rows)
        indirect_resource_trace = self._build_indirect_resource_trace_support(trace_rows)

        direct_metric_rows: Dict[str, Optional[Dict[str, Any]]] = {}
        direct_metric_scores: Dict[str, float] = {}
        log_scores: Dict[str, float] = {}
        trace_scores: Dict[str, float] = {}
        log_support_rows: Dict[str, Dict[str, Any]] = {}
        trace_support_rows: Dict[str, Dict[str, Any]] = {}

        for failure_type in self.FAILURE_TYPE_ORDER:
            direct_metric = metrics_by_family.get(failure_type, [None])[0]
            family_log_support = dict(log_support[failure_type])
            family_trace_support = dict(trace_support[failure_type])

            if failure_type in {"cpu", "mem", "disk", "socket"} and not family_trace_support["supported"]:
                family_trace_support = dict(indirect_resource_trace)

            direct_metric_rows[failure_type] = direct_metric
            direct_metric_scores[failure_type] = self._metric_support_score(direct_metric)
            log_scores[failure_type] = self._to_float(family_log_support.get("score", 0.0))
            trace_scores[failure_type] = self._to_float(family_trace_support.get("score", 0.0))
            log_support_rows[failure_type] = family_log_support
            trace_support_rows[failure_type] = family_trace_support

        max_direct_score = max(direct_metric_scores.values(), default=0.0)
        max_log_score = max(log_scores.values(), default=0.0)
        max_trace_score = max(trace_scores.values(), default=0.0)
        external_loss_context_score = self._to_float(
            (global_behavior_context or {}).get("score", 0.0)
        )
        external_loss_context_quality = min(external_loss_context_score / 8.0, 1.0)

        resource_families = {"cpu", "mem", "disk", "socket"}
        family_category = {
            failure_type: ("resource" if failure_type in resource_families else "behavior")
            for failure_type in self.FAILURE_TYPE_ORDER
        }

        best_resource_family = max(
            resource_families,
            key=lambda family: (
                direct_metric_scores.get(family, 0.0),
                -self.FAILURE_TYPE_ORDER.index(family),
            ),
        )
        best_resource_row = direct_metric_rows.get(best_resource_family)
        best_resource_direct_score = direct_metric_scores.get(best_resource_family, 0.0)
        best_resource_onset = self._metric_onset_offset_sec(best_resource_row)

        best_behavior_family = max(
            ("delay", "loss"),
            key=lambda family: (
                direct_metric_scores.get(family, 0.0)
                + 0.5 * log_scores.get(family, 0.0)
                + 0.75 * trace_scores.get(family, 0.0),
                -self.FAILURE_TYPE_ORDER.index(family),
            ),
        )

        decision_scores: Dict[str, float] = {}
        family_summaries: List[Dict[str, Any]] = []
        hypothesis_rows: List[Dict[str, Any]] = []

        specificity_multiplier = {
            "cpu": 0.95,
            "mem": 1.02,
            "disk": 1.05,
            "delay": 1.00,
            "loss": 1.00,
            "socket": 1.08,
        }

        for failure_type in self.FAILURE_TYPE_ORDER:
            direct_metric = direct_metric_rows.get(failure_type)
            direct_metric_score = direct_metric_scores.get(failure_type, 0.0)
            family_log_support = log_support_rows[failure_type]
            family_trace_support = trace_support_rows[failure_type]

            relative_direct_strength = self._normalized_score(direct_metric_score, max_direct_score)
            absolute_direct_strength = self._absolute_direct_metric_strength(direct_metric_score)
            direct_quality = 0.65 * absolute_direct_strength + 0.35 * relative_direct_strength
            percent_strength = self._metric_percent_strength(direct_metric)
            log_quality = self._normalized_score(log_scores.get(failure_type, 0.0), max_log_score)
            trace_quality = self._normalized_score(trace_scores.get(failure_type, 0.0), max_trace_score)
            temporal_quality = 0.0
            if isinstance(direct_metric, dict):
                temporal_quality = min(
                    max(self._metric_temporal_bonus(direct_metric) - 0.75, 0.0) / 0.55,
                    1.0,
                )

            if failure_type in resource_families:
                decision_score = (
                    0.50 * direct_quality
                    + 0.18 * percent_strength
                    + 0.10 * log_quality
                    + 0.07 * trace_quality
                    + 0.15 * temporal_quality
                ) * specificity_multiplier[failure_type]
            elif failure_type == "delay":
                decision_score = (
                    0.35 * direct_quality
                    + 0.20 * log_quality
                    + 0.45 * trace_quality
                )
            else:
                decision_score = (
                    0.15 * direct_quality
                    + 0.35 * log_quality
                    + 0.30 * trace_quality
                    + 0.20 * external_loss_context_quality
                )

            decision_scores[failure_type] = max(0.0, decision_score)

        if best_resource_direct_score >= 8.0:
            decision_scores["delay"] *= 0.40
            decision_scores["loss"] *= 0.72
        elif best_resource_direct_score >= 6.0:
            decision_scores["delay"] *= 0.58
            decision_scores["loss"] *= 0.82
        elif best_resource_direct_score >= 4.0:
            decision_scores["delay"] *= 0.78
            decision_scores["loss"] *= 0.92

        delay_onset = self._metric_onset_offset_sec(direct_metric_rows.get("delay"))
        if (
            best_resource_direct_score >= 6.0
            and best_resource_onset is not None
            and delay_onset is not None
            and delay_onset >= best_resource_onset
        ):
            decision_scores["delay"] *= 0.88

        cpu_score = direct_metric_scores.get("cpu", 0.0)
        socket_score = direct_metric_scores.get("socket", 0.0)
        cpu_onset = self._metric_onset_offset_sec(direct_metric_rows.get("cpu"))
        socket_onset = self._metric_onset_offset_sec(direct_metric_rows.get("socket"))
        if (
            cpu_score >= 4.5
            and socket_score >= 4.5
            and socket_onset is not None
            and cpu_onset is not None
            and socket_onset <= cpu_onset + 20.0
        ):
            decision_scores["socket"] += 0.18
            decision_scores["cpu"] *= 0.85

        mem_score = direct_metric_scores.get("mem", 0.0)
        disk_score = direct_metric_scores.get("disk", 0.0)
        mem_row = direct_metric_rows.get("mem")
        disk_row = direct_metric_rows.get("disk")
        mem_onset = self._metric_onset_offset_sec(mem_row)
        disk_onset = self._metric_onset_offset_sec(disk_row)
        mem_percent = abs(self._to_float((mem_row or {}).get("percent_change", 0.0)))
        disk_baseline = self._to_float((disk_row or {}).get("baseline_mean", 0.0))
        disk_incident = self._to_float((disk_row or {}).get("incident_mean", 0.0))
        disk_log_quality = self._normalized_score(log_scores.get("disk", 0.0), max_log_score)
        if abs(disk_baseline) <= 1e-9 and abs(disk_incident) > 0.0:
            decision_scores["disk"] += 0.22

        if (
            mem_score >= 6.0
            and disk_score >= 6.0
            and 0.0 <= (disk_score - mem_score) <= 1.25
            and mem_percent >= 150.0
            and disk_baseline > 1.0
            and disk_log_quality < 0.25
            and mem_onset is not None
            and disk_onset is not None
            and abs(mem_onset - disk_onset) <= 20.0
        ):
            decision_scores["mem"] += 0.12
            decision_scores["disk"] *= 0.90

        if (
            cpu_score >= 4.5
            and max(mem_score, disk_score, socket_score) >= 0.85 * cpu_score
        ):
            decision_scores["cpu"] *= 0.92

        max_decision_score = max(decision_scores.values(), default=0.0)

        for failure_type in self.FAILURE_TYPE_ORDER:
            direct_metric = direct_metric_rows.get(failure_type)
            direct_metric_score = direct_metric_scores.get(failure_type, 0.0)
            family_log_support = log_support_rows[failure_type]
            family_trace_support = trace_support_rows[failure_type]

            family_summaries.append(
                {
                    "failure_type": failure_type,
                    "category": family_category[failure_type],
                    "decision_score": round(decision_scores[failure_type], 4),
                    "normalized_decision_score": round(
                        self._normalized_score(decision_scores[failure_type], max_decision_score),
                        4,
                    ),
                    "direct_metric_score": round(direct_metric_score, 4),
                    "strongest_direct_metric": self._summarize_metric_for_discriminator(direct_metric),
                    "logs_support": {
                        "supported": bool(family_log_support.get("supported", False)),
                        "support_kind": str(family_log_support.get("support_kind", "direct")),
                        "score": round(self._to_float(family_log_support.get("score", 0.0)), 4),
                        "normalized_score": round(
                            self._normalized_score(log_scores.get(failure_type, 0.0), max_log_score),
                            4,
                        ),
                        "signals": list(family_log_support.get("signals", []))[:2],
                    },
                    "traces_support": {
                        "supported": bool(family_trace_support.get("supported", False)),
                        "support_kind": str(family_trace_support.get("support_kind", "direct")),
                        "score": round(self._to_float(family_trace_support.get("score", 0.0)), 4),
                        "normalized_score": round(
                            self._normalized_score(trace_scores.get(failure_type, 0.0), max_trace_score),
                            4,
                        ),
                        "signals": list(family_trace_support.get("signals", []))[:2],
                    },
                }
            )

            hypothesis_rows.append(
                {
                    "failure_type": failure_type,
                    "category": family_category[failure_type],
                    "decision_score": decision_scores[failure_type],
                }
            )

        ranked_hypotheses = sorted(
            hypothesis_rows,
            key=lambda row: (
                -float(row["decision_score"]),
                self.FAILURE_TYPE_ORDER.index(str(row["failure_type"])),
            ),
        )
        top_hypotheses = [
            {
                "failure_type": str(row["failure_type"]),
                "category": str(row["category"]),
                "decision_score": round(float(row["decision_score"]), 4),
                "normalized_decision_score": round(
                    self._normalized_score(float(row["decision_score"]), max_decision_score),
                    4,
                ),
            }
            for row in ranked_hypotheses[:3]
        ]

        resource_decision_score = max(
            (
                float(row["decision_score"])
                for row in ranked_hypotheses
                if row["category"] == "resource"
            ),
            default=0.0,
        )
        behavior_decision_score = max(
            (
                float(row["decision_score"])
                for row in ranked_hypotheses
                if row["category"] == "behavior"
            ),
            default=0.0,
        )

        best_behavior_decision_score = max(
            decision_scores.get("delay", 0.0),
            decision_scores.get("loss", 0.0),
        )
        if best_resource_direct_score < 4.0:
            behavior_can_override = True
            override_reason = "Direct resource evidence on the localized service is weak."
        elif best_resource_direct_score < 6.0 and best_behavior_decision_score > resource_decision_score + 0.08:
            behavior_can_override = True
            override_reason = "Behavior evidence clearly exceeds only-moderate direct resource evidence."
        else:
            behavior_can_override = False
            override_reason = "A strong direct localized resource anomaly is present; treat delay/loss as override labels only if the raw facts clearly contradict that resource candidate."

        return {
            "faulty_service": faulty_service,
            "strongest_direct_resource_candidate": {
                "failure_type": best_resource_family,
                "direct_metric_score": round(best_resource_direct_score, 4),
                "strongest_direct_metric": self._summarize_metric_for_discriminator(best_resource_row),
            },
            "behavior_override_guidance": {
                "best_resource_family": best_resource_family,
                "best_resource_direct_metric_score": round(best_resource_direct_score, 4),
                "best_behavior_family": best_behavior_family,
                "best_behavior_decision_score": round(best_behavior_decision_score, 4),
                "behavior_can_override": behavior_can_override,
                "reason": override_reason,
            },
            "resource_vs_behavior": {
                "resource_decision_score": round(resource_decision_score, 4),
                "behavior_decision_score": round(behavior_decision_score, 4),
                "suggested_group": (
                    "resource" if resource_decision_score >= behavior_decision_score else "behavior"
                ),
            },
            "family_summaries": family_summaries,
            "top_hypotheses": top_hypotheses,
            "strongest_alternative": top_hypotheses[1] if len(top_hypotheses) > 1 else None,
            "global_behavior_context": {
                "supported": bool((global_behavior_context or {}).get("supported", False)),
                "score": round(external_loss_context_score, 4),
                "signals": list((global_behavior_context or {}).get("signals", []))[:3],
            },
            "timeline_context": [
                {
                    "offset_sec": row.get("offset_sec"),
                    "source": row.get("source"),
                    "summary": row.get("summary"),
                }
                for row in timeline_rows[:4]
                if isinstance(row, dict)
            ],
        }

    def _onset_bonus(self, offset_value: Any) -> float:
        offset = self._to_float(offset_value)
        if offset <= 0:
            return 1.20
        if offset <= 5:
            return 1.10
        if offset <= 20:
            return 1.00
        if offset <= 60:
            return 0.90
        return 0.75

    def _is_trace_noise(self, operation: str) -> bool:
        text = str(operation).strip().lower()
        if not text:
            return True
        return any(fragment in text for fragment in self.TRACE_OPERATION_EXCLUDE)

    def _dedupe_strings(self, values: List[str], limit: int) -> List[str]:
        out: List[str] = []
        seen: Set[str] = set()
        for value in values:
            text = str(value).strip()
            if not text or text in seen:
                continue
            seen.add(text)
            out.append(text)
            if len(out) >= limit:
                break
        return out

    def _coerce_failure_type(self, value: Any, fallback: str) -> str:
        text = str(value).strip().lower()
        if text in self.ALLOWED_FAILURE_TYPES:
            return text

        normalized = self._normalize_failure_type(text)
        if normalized in self.ALLOWED_FAILURE_TYPES:
            return normalized

        fallback_norm = self._normalize_failure_type(fallback)
        if fallback_norm in self.ALLOWED_FAILURE_TYPES:
            return fallback_norm

        return self._select_best_failure_type(self._zero_failure_type_scores())

    def _zero_failure_type_scores(self) -> Dict[str, float]:
        return {failure_type: 0.0 for failure_type in self.FAILURE_TYPE_ORDER}

    def _rank_failure_types(
        self,
        type_scores: Dict[str, float],
    ) -> List[Tuple[str, float]]:
        order_index = {
            failure_type: idx for idx, failure_type in enumerate(self.FAILURE_TYPE_ORDER)
        }
        return sorted(
            (
                (failure_type, float(type_scores.get(failure_type, 0.0)))
                for failure_type in self.FAILURE_TYPE_ORDER
            ),
            key=lambda item: (-item[1], order_index[item[0]]),
        )

    def _select_best_failure_type(self, type_scores: Dict[str, float]) -> str:
        return self._rank_failure_types(type_scores)[0][0]


DiagnosisAgent = DiagnosisAgentRE2
