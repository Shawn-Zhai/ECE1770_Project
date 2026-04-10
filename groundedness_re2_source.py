#!/usr/bin/env python3
"""Validate RE2 diagnosis claims against raw RE2 telemetry using claim-conditioned retrieval."""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import re
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_DIAGNOSIS_RESULTS = PROJECT_ROOT / "diagnosis_re2_all_results.json"
DEFAULT_RAW_ROOT = PROJECT_ROOT / "dataset" / "RCAEval" / "RE2"
DEFAULT_OUTPUT_PATH = PROJECT_ROOT / "groundedness_re2_source_results.json"

BASELINE_SEC = 300
INCIDENT_SEC = 300
MAX_CLAIM_METRIC_SERIES = 3
MAX_CLAIM_LOG_LINES = 8
MAX_CLAIM_LOG_GROUPS = 5
MAX_CLAIM_TRACE_GROUPS = 5
MAX_CLAIM_TRACE_SPANS = 8
MAX_CROSS_SERVICE_ROWS = 6

METRIC_HINTS = {
    "cpu": "cpu",
    "memory": "mem",
    "mem": "mem",
    "disk": "disk",
    "diskio": "disk",
    "socket": "socket",
    "latency-50": "latency-50",
    "latency-90": "latency-90",
    "latency p50": "latency-50",
    "latency p90": "latency-90",
}
FAULT_TYPES = {"cpu", "mem", "disk", "delay", "loss", "socket"}
STOPWORDS = {
    "the",
    "and",
    "that",
    "with",
    "from",
    "during",
    "after",
    "before",
    "because",
    "while",
    "shows",
    "showed",
    "claim",
    "supported",
    "service",
    "fault",
    "stronger",
    "strongest",
    "earliest",
    "largest",
    "candidate",
    "candidates",
    "services",
    "anomaly",
    "better",
    "than",
    "this",
    "which",
    "were",
    "was",
    "rose",
    "jumped",
    "increased",
    "stayed",
    "elevated",
    "sustained",
    "incident",
    "baseline",
}

UUID_RE = re.compile(
    r"\b[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}\b"
)
LONG_HEX_RE = re.compile(r"\b[0-9a-fA-F]{12,}\b")
NUMBER_RE = re.compile(r"\b\d+\b")
FLOAT_RE = re.compile(r"\b\d+\.\d+\b")
TRACE_P95_RE = re.compile(
    r"(?P<operation>[A-Za-z0-9_./:-]+)\s+trace\s+p95",
    re.IGNORECASE,
)
TRACE_COUNT_RE = re.compile(
    r"trace counts?\s+for\s+(?P<operation>[A-Za-z0-9_./:-]+)",
    re.IGNORECASE,
)
TRACE_ON_OPERATION_RE = re.compile(
    r"\bon\s+(?P<operation>[A-Za-z0-9_./:-]+)\b",
    re.IGNORECASE,
)
SERVICE_COUNT_RE = re.compile(
    r"(?:request log volume|log template count|request count|trace count|trace counts?)"
    r".{0,40}?\s+from\s+(?P<from>\d+)\s+to\s+(?P<to>\d+)",
    re.IGNORECASE,
)


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def save_json(path: Path, data: Any) -> None:
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def normalize_label(value: Optional[str]) -> str:
    if value is None:
        return ""
    return str(value).strip().lower()


def safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return default


def safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(float(value))
    except Exception:
        return default


def percentile(values: List[float], q: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    if len(ordered) == 1:
        return float(ordered[0])
    rank = (len(ordered) - 1) * (q / 100.0)
    lo = int(math.floor(rank))
    hi = int(math.ceil(rank))
    if lo == hi:
        return float(ordered[lo])
    frac = rank - lo
    return float(ordered[lo] * (1.0 - frac) + ordered[hi] * frac)


def read_inject_time(path: Path) -> int:
    return int(path.read_text(encoding="utf-8").strip())


def sample_point_window(points: List[Tuple[int, float]], limit: int) -> List[Dict[str, Any]]:
    if not points:
        return []
    if len(points) <= limit:
        selected = points
    else:
        stride = max(1, len(points) // limit)
        selected = points[::stride][: limit - 1] + [points[-1]]
    return [{"timestamp": ts, "value": round(float(value), 6)} for ts, value in selected]


def normalize_log_message(message: str) -> str:
    text = str(message).strip().lower()
    text = UUID_RE.sub("<uuid>", text)
    text = LONG_HEX_RE.sub("<hex>", text)
    text = FLOAT_RE.sub("<float>", text)
    text = NUMBER_RE.sub("<num>", text)
    text = re.sub(r"\s+", " ", text)
    return text


def extract_keywords(text: str) -> List[str]:
    raw_tokens = re.findall(r"[A-Za-z][A-Za-z0-9._/-]{2,}", text.lower())
    keywords = []
    seen = set()
    for token in raw_tokens:
        if token in STOPWORDS:
            continue
        if token in seen:
            continue
        seen.add(token)
        keywords.append(token)
    return keywords[:10]


def metric_family_hints(text: str) -> List[str]:
    lowered = text.lower()
    hints: List[str] = []

    def add_hint(hint: str) -> None:
        if hint not in hints:
            hints.append(hint)

    if re.search(r"\b(?:latency[-\s]*)?p50\b", lowered) or "latency-50" in lowered:
        add_hint("latency-50")
    if re.search(r"\b(?:latency[-\s]*)?p90\b", lowered) or "latency-90" in lowered:
        add_hint("latency-90")

    for marker, family in METRIC_HINTS.items():
        if family.startswith("latency"):
            continue
        if marker in lowered:
            add_hint(family)

    if ("delay" in lowered or "latency" in lowered) and not any(hint.startswith("latency") for hint in hints):
        add_hint("delay")

    return hints


def canonical_metric_family(metric_type: str) -> str:
    metric_type = normalize_label(metric_type)
    if metric_type == "diskio":
        return "disk"
    return metric_type


def metric_matches_hints(metric_type: str, hints: List[str]) -> bool:
    canonical_type = canonical_metric_family(metric_type)
    if not hints:
        return True
    for hint in hints:
        if hint == "delay" and metric_type.startswith("latency"):
            return True
        if canonical_type == hint:
            return True
    return False


def extract_trace_operation_hint(text: str) -> str:
    trace_match = TRACE_P95_RE.search(text)
    if trace_match:
        return trace_match.group("operation")
    count_match = TRACE_COUNT_RE.search(text)
    if count_match:
        return count_match.group("operation")
    on_match = TRACE_ON_OPERATION_RE.search(text)
    if on_match:
        return on_match.group("operation")
    slash_tokens = re.findall(r"[A-Za-z0-9_.:-]+/[A-Za-z0-9_.:-]+", text)
    return slash_tokens[0] if slash_tokens else ""


def window_points(
    series: List[Tuple[int, float]],
    inject_time: int,
) -> Tuple[List[Tuple[int, float]], List[Tuple[int, float]]]:
    baseline_start = inject_time - BASELINE_SEC
    incident_end = inject_time + INCIDENT_SEC
    baseline = [(ts, value) for ts, value in series if baseline_start <= ts < inject_time]
    incident = [(ts, value) for ts, value in series if inject_time <= ts <= incident_end]
    return baseline, incident


def metric_direction(baseline_mean: float, incident_mean: float) -> str:
    return "up" if incident_mean >= baseline_mean else "down"


def metric_change_summary(
    metric_name: str,
    series: List[Tuple[int, float]],
    inject_time: int,
) -> Optional[Dict[str, Any]]:
    baseline, incident = window_points(series, inject_time)
    if not baseline or not incident:
        return None

    baseline_values = [value for _, value in baseline]
    incident_values = [value for _, value in incident]
    baseline_mean = sum(baseline_values) / len(baseline_values)
    incident_mean = sum(incident_values) / len(incident_values)
    incident_peak = max(incident_values)
    incident_min = min(incident_values)
    baseline_std = math.sqrt(
        sum((value - baseline_mean) ** 2 for value in baseline_values) / max(1, len(baseline_values) - 1)
    ) if len(baseline_values) > 1 else 0.0
    direction = metric_direction(baseline_mean, incident_mean)
    threshold = max(abs(baseline_mean) * 0.1, baseline_std * 2.0, 1e-6)

    first_offset = None
    streak = 0
    sustained = False
    for ts, value in incident:
        delta = value - baseline_mean
        anomalous = delta >= threshold if direction == "up" else delta <= -threshold
        if anomalous:
            if first_offset is None:
                first_offset = ts - inject_time
            streak += 1
            if streak >= 3:
                sustained = True
        else:
            streak = 0

    percent_change = None
    if abs(baseline_mean) > 1e-12:
        percent_change = ((incident_mean - baseline_mean) / baseline_mean) * 100.0

    return {
        "metric_name": metric_name,
        "baseline_mean": round(baseline_mean, 6),
        "incident_mean": round(incident_mean, 6),
        "incident_peak": round(incident_peak, 6),
        "incident_min": round(incident_min, 6),
        "baseline_std": round(baseline_std, 6),
        "direction": direction,
        "percent_change": round(percent_change, 3) if percent_change is not None else None,
        "first_incident_offset_sec": first_offset,
        "sustained": sustained,
        "baseline_tail_points": sample_point_window(baseline[-5:], 5),
        "incident_head_points": sample_point_window(incident[:8], 8),
        "incident_peak_point": {
            "timestamp": max(incident, key=lambda item: item[1])[0],
            "value": round(max(incident_values), 6),
        },
    }


def operation_matches(claim_operation: str, row_operation: str) -> bool:
    claim_norm = normalize_label(claim_operation)
    row_norm = normalize_label(row_operation)
    if not claim_norm or not row_norm:
        return False
    if claim_norm in row_norm:
        return True
    return claim_norm.split("/")[-1] == row_norm.split("/")[-1]


class RE2RawTelemetryRAGValidator:
    def __init__(self, raw_root: Path, debug: bool) -> None:
        self.raw_root = raw_root
        self.debug = debug
        self._llm_client = None
        self._llm_model_name = ""
        self._llm_backend = ""

    def evaluate_case(self, result_row: Dict[str, Any]) -> Dict[str, Any]:
        case_id = str(result_row["case_id"])
        raw_case = self._load_raw_case(case_id)
        predicted_service = normalize_label(result_row.get("pred_service"))
        predicted_fault = normalize_label(result_row.get("pred_fault"))

        all_claim_bundles: List[Dict[str, Any]] = []
        for claim in result_row.get("service_evidence_claims", []):
            all_claim_bundles.append(
                self._build_claim_bundle(
                    claim=claim,
                    claim_group="service",
                    predicted_service=predicted_service,
                    predicted_fault=predicted_fault,
                    raw_case=raw_case,
                )
            )
        for claim in result_row.get("failure_evidence_claims", []):
            all_claim_bundles.append(
                self._build_claim_bundle(
                    claim=claim,
                    claim_group="failure",
                    predicted_service=predicted_service,
                    predicted_fault=predicted_fault,
                    raw_case=raw_case,
                )
            )

        llm_results = self._validate_claim_bundles(case_id=case_id, claim_bundles=all_claim_bundles)
        by_id = {row["claim_id"]: row for row in llm_results}

        service_results = [
            by_id.get(str(claim.get("id", "")).strip() or "claim", self._unsupported_claim_result(claim, "Missing LLM result."))
            for claim in result_row.get("service_evidence_claims", [])
        ]
        failure_results = [
            by_id.get(str(claim.get("id", "")).strip() or "claim", self._unsupported_claim_result(claim, "Missing LLM result."))
            for claim in result_row.get("failure_evidence_claims", [])
        ]

        service_summary = self._summarize_claim_results(service_results)
        failure_summary = self._summarize_claim_results(failure_results)
        service_supported = bool(service_summary["supported_set"])
        failure_supported = bool(failure_summary["supported_set"])

        if service_supported and failure_supported:
            groundedness_outcome = "both_supported"
        elif service_supported:
            groundedness_outcome = "service_only_supported"
        elif failure_supported:
            groundedness_outcome = "failure_only_supported"
        else:
            groundedness_outcome = "neither_supported"

        return {
            "case_id": case_id,
            "pred_service": predicted_service,
            "pred_fault": predicted_fault,
            "raw_case_dir": str(raw_case["case_dir"]),
            "service_claim_results": service_results,
            "failure_claim_results": failure_results,
            "service_supported": service_supported,
            "failure_supported": failure_supported,
            "case_supported": bool(service_supported and failure_supported),
            "groundedness_outcome": groundedness_outcome,
            "service_summary": service_summary,
            "failure_summary": failure_summary,
        }

    def _load_raw_case(self, case_id: str) -> Dict[str, Any]:
        case_dir = self.raw_root / case_id
        if not case_dir.exists():
            raise FileNotFoundError(f"Raw case folder not found: {case_dir}")

        inject_time = read_inject_time(case_dir / "inject_time.txt")
        metrics = self._load_metrics(case_dir / "metrics.json")
        logs = self._load_logs(case_dir / "logs.csv")
        traces = self._load_traces(case_dir / "traces.csv")

        services = sorted(
            {
                service
                for service in (
                    list({metric_name.rsplit("_", 1)[0] for metric_name in metrics if "_" in metric_name})
                    + [row["service"] for row in logs if row["service"]]
                    + [row["service"] for row in traces if row["service"]]
                )
                if service
            }
        )

        return {
            "case_id": case_id,
            "case_dir": case_dir,
            "inject_time": inject_time,
            "metrics": metrics,
            "logs": logs,
            "traces": traces,
            "services": services,
        }

    def _load_metrics(self, path: Path) -> Dict[str, List[Tuple[int, float]]]:
        raw = load_json(path)
        metrics: Dict[str, List[Tuple[int, float]]] = {}
        for metric_name, points in raw.items():
            if not isinstance(points, list):
                continue
            cleaned = []
            for point in points:
                if not isinstance(point, list) or len(point) < 2:
                    continue
                cleaned.append((safe_int(point[0]), safe_float(point[1])))
            cleaned.sort(key=lambda item: item[0])
            metrics[str(metric_name)] = cleaned
        return metrics

    def _load_logs(self, path: Path) -> List[Dict[str, Any]]:
        rows = []
        with path.open("r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            for raw_row in reader:
                rows.append(
                    {
                        "timestamp": safe_int(raw_row.get("timestamp")),
                        "service": normalize_label(raw_row.get("service")),
                        "message": str(raw_row.get("message", "")),
                    }
                )
        rows.sort(key=lambda row: row["timestamp"])
        return rows

    def _load_traces(self, path: Path) -> List[Dict[str, Any]]:
        rows = []
        with path.open("r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            for raw_row in reader:
                start_ms = safe_int(raw_row.get("startTimeMillis"))
                rows.append(
                    {
                        "service": normalize_label(raw_row.get("serviceName")),
                        "operation": str(raw_row.get("operationName", "")),
                        "start_ms": start_ms,
                        "start_sec": int(start_ms / 1000) if start_ms else 0,
                        "duration_ms": round(safe_float(raw_row.get("duration")) / 1000.0, 6),
                        "trace_id": str(raw_row.get("traceID", "")),
                        "span_id": str(raw_row.get("spanID", "")),
                        "parent_span_id": str(raw_row.get("parentSpanID", "")),
                    }
                )
        rows.sort(key=lambda row: row["start_ms"])
        return rows

    def _build_claim_bundle(
        self,
        *,
        claim: Dict[str, Any],
        claim_group: str,
        predicted_service: str,
        predicted_fault: str,
        raw_case: Dict[str, Any],
    ) -> Dict[str, Any]:
        text = str(claim.get("text", "")).strip()
        target_service = self._resolve_service_from_claim(text, predicted_service, raw_case["services"])

        return {
            "claim_id": str(claim.get("id", "")).strip() or f"{claim_group}_claim",
            "claim_group": claim_group,
            "claim_type": normalize_label(claim.get("type")) or "observation",
            "claim_text": text,
            "target_service": target_service,
            "predicted_fault": predicted_fault,
            "fault_labels_mentioned": [label for label in FAULT_TYPES if label in text.lower()],
            "metric_evidence": self._build_metric_evidence(text, target_service, raw_case),
            "log_evidence": self._build_log_evidence(text, target_service, raw_case),
            "trace_evidence": self._build_trace_evidence(text, target_service, raw_case),
            "cross_service_evidence": self._build_cross_service_evidence(text, target_service, raw_case),
        }

    def _resolve_service_from_claim(
        self,
        text: str,
        predicted_service: str,
        services: Iterable[str],
    ) -> str:
        lowered = text.lower()
        for service in services:
            if service in lowered:
                return service
        return predicted_service

    def _build_metric_evidence(
        self,
        claim_text: str,
        target_service: str,
        raw_case: Dict[str, Any],
    ) -> Dict[str, Any]:
        hints = metric_family_hints(claim_text)
        metric_summaries = []
        for metric_name, series in raw_case["metrics"].items():
            service = metric_name.rsplit("_", 1)[0] if "_" in metric_name else ""
            if service != target_service:
                continue
            metric_type = normalize_label(metric_name.rsplit("_", 1)[-1]) if "_" in metric_name else ""
            if not metric_matches_hints(metric_type, hints):
                continue
            summary = metric_change_summary(metric_name, series, raw_case["inject_time"])
            if summary is not None:
                metric_summaries.append(summary)

        metric_summaries.sort(
            key=lambda row: abs(row["incident_mean"] - row["baseline_mean"]),
            reverse=True,
        )
        return {
            "target_metric_summaries": metric_summaries[:MAX_CLAIM_METRIC_SERIES],
            "metric_hint": hints[0] if hints else "",
            "metric_hints": hints,
        }

    def _build_log_evidence(
        self,
        claim_text: str,
        target_service: str,
        raw_case: Dict[str, Any],
    ) -> Dict[str, Any]:
        inject_time = raw_case["inject_time"]
        baseline_start = inject_time - BASELINE_SEC
        incident_end = inject_time + INCIDENT_SEC
        service_logs = [
            row for row in raw_case["logs"] if row["service"] == target_service and baseline_start <= row["timestamp"] <= incident_end
        ]
        baseline_logs = [row for row in service_logs if row["timestamp"] < inject_time]
        incident_logs = [row for row in service_logs if row["timestamp"] >= inject_time]

        keywords = extract_keywords(claim_text)
        if any(token in claim_text.lower() for token in ("restart", "server start", "grpc server")):
            keywords.extend(["grpc", "server", "start"])
        if "5xx" in claim_text.lower():
            keywords.extend(["500", "501", "502", "503", "504", "505"])

        matched_incident = [
            {
                "timestamp": row["timestamp"],
                "offset_sec": row["timestamp"] - inject_time,
                "message": row["message"],
            }
            for row in incident_logs
            if not keywords or any(keyword in row["message"].lower() for keyword in keywords)
        ][:MAX_CLAIM_LOG_LINES]

        exact_groups = self._group_logs_by_message(baseline_logs, incident_logs, normalize=False)
        normalized_groups = self._group_logs_by_message(baseline_logs, incident_logs, normalize=True)

        incident_5xx_count = sum(1 for row in incident_logs if re.search(r"\b5\d\d\b", row["message"]))
        return {
            "service_log_row_counts": {
                "baseline": len(baseline_logs),
                "incident": len(incident_logs),
                "incident_5xx_count": incident_5xx_count,
            },
            "matched_incident_log_lines": matched_incident,
            "top_exact_message_groups": exact_groups[:MAX_CLAIM_LOG_GROUPS],
            "top_normalized_message_groups": normalized_groups[:MAX_CLAIM_LOG_GROUPS],
            "claim_keywords": keywords[:8],
        }

    def _group_logs_by_message(
        self,
        baseline_logs: List[Dict[str, Any]],
        incident_logs: List[Dict[str, Any]],
        *,
        normalize: bool,
    ) -> List[Dict[str, Any]]:
        baseline_counts: Dict[str, int] = defaultdict(int)
        incident_counts: Dict[str, int] = defaultdict(int)
        example_message: Dict[str, str] = {}

        def key_fn(message: str) -> str:
            return normalize_log_message(message) if normalize else normalize_label(message)

        for row in baseline_logs:
            key = key_fn(row["message"])
            baseline_counts[key] += 1
            example_message.setdefault(key, row["message"])

        for row in incident_logs:
            key = key_fn(row["message"])
            incident_counts[key] += 1
            example_message.setdefault(key, row["message"])

        grouped = []
        for key in set(baseline_counts) | set(incident_counts):
            grouped.append(
                {
                    "message_key": key,
                    "example_message": example_message.get(key, ""),
                    "baseline_count": baseline_counts.get(key, 0),
                    "incident_count": incident_counts.get(key, 0),
                    "delta_count": incident_counts.get(key, 0) - baseline_counts.get(key, 0),
                }
            )
        grouped.sort(key=lambda row: abs(row["delta_count"]), reverse=True)
        return grouped

    def _build_trace_evidence(
        self,
        claim_text: str,
        target_service: str,
        raw_case: Dict[str, Any],
    ) -> Dict[str, Any]:
        inject_time = raw_case["inject_time"]
        baseline_start = inject_time - BASELINE_SEC
        service_traces = [row for row in raw_case["traces"] if row["service"] == target_service]
        operation_hint = extract_trace_operation_hint(claim_text)
        grouped: Dict[str, Dict[str, Any]] = {}

        for row in service_traces:
            grouped.setdefault(
                row["operation"],
                {
                    "operation": row["operation"],
                    "baseline_durations": [],
                    "incident_durations": [],
                    "baseline_count": 0,
                    "incident_count": 0,
                    "sample_incident_spans": [],
                },
            )
            bucket = grouped[row["operation"]]
            if baseline_start <= row["start_sec"] < inject_time:
                bucket["baseline_count"] += 1
                bucket["baseline_durations"].append(row["duration_ms"])
            elif inject_time <= row["start_sec"] <= inject_time + INCIDENT_SEC:
                bucket["incident_count"] += 1
                bucket["incident_durations"].append(row["duration_ms"])
                if len(bucket["sample_incident_spans"]) < MAX_CLAIM_TRACE_SPANS:
                    bucket["sample_incident_spans"].append(
                        {
                            "start_ms": row["start_ms"],
                            "offset_sec": row["start_sec"] - inject_time,
                            "duration_ms": row["duration_ms"],
                            "trace_id": row["trace_id"],
                        }
                    )

        summaries = []
        for operation, bucket in grouped.items():
            if bucket["baseline_count"] == 0 and bucket["incident_count"] == 0:
                continue
            summaries.append(
                {
                    "operation": operation,
                    "baseline_count": bucket["baseline_count"],
                    "incident_count": bucket["incident_count"],
                    "baseline_p95_ms": round(percentile(bucket["baseline_durations"], 95), 3),
                    "incident_p95_ms": round(percentile(bucket["incident_durations"], 95), 3),
                    "incident_max_ms": round(max(bucket["incident_durations"], default=0.0), 3),
                    "sample_incident_spans": bucket["sample_incident_spans"][:MAX_CLAIM_TRACE_SPANS],
                }
            )

        if operation_hint:
            summaries.sort(key=lambda row: (not operation_matches(operation_hint, row["operation"]), -row["incident_p95_ms"]))
        else:
            summaries.sort(key=lambda row: row["incident_p95_ms"], reverse=True)

        return {
            "operation_hint": operation_hint,
            "operation_summaries": summaries[:MAX_CLAIM_TRACE_GROUPS],
        }

    def _build_cross_service_evidence(
        self,
        claim_text: str,
        target_service: str,
        raw_case: Dict[str, Any],
    ) -> Dict[str, Any]:
        hints = metric_family_hints(claim_text)
        has_delay_hints = any(hint == "delay" or hint.startswith("latency") for hint in hints)
        has_resource_hints = any(not (hint == "delay" or hint.startswith("latency")) for hint in hints)
        if has_delay_hints and has_resource_hints:
            scope = "mixed"
        elif has_delay_hints:
            scope = "delay"
        else:
            scope = "resource"
        rows = []

        for service in raw_case["services"]:
            service_rows = []
            for metric_name, series in raw_case["metrics"].items():
                metric_service = metric_name.rsplit("_", 1)[0] if "_" in metric_name else ""
                metric_type = normalize_label(metric_name.rsplit("_", 1)[-1]) if "_" in metric_name else ""
                if metric_service != service:
                    continue
                if scope == "delay" and not metric_type.startswith("latency"):
                    continue
                if scope == "resource" and metric_type.startswith("latency"):
                    continue
                if not metric_matches_hints(metric_type, hints):
                    continue
                summary = metric_change_summary(metric_name, series, raw_case["inject_time"])
                if summary is not None:
                    service_rows.append(summary)

            if not service_rows:
                continue

            service_rows.sort(
                key=lambda row: abs(row["incident_mean"] - row["baseline_mean"]),
                reverse=True,
            )
            best = service_rows[0]
            rows.append(
                {
                    "service": service,
                    "metric_name": best["metric_name"],
                    "baseline_mean": best["baseline_mean"],
                    "incident_mean": best["incident_mean"],
                    "percent_change": best["percent_change"],
                    "first_incident_offset_sec": best["first_incident_offset_sec"],
                    "sustained": best["sustained"],
                    "target_service": service == target_service,
                }
            )

        rows.sort(
            key=lambda row: (
                not row["target_service"],
                row["first_incident_offset_sec"] if row["first_incident_offset_sec"] is not None else 10 ** 9,
                -(abs((row["percent_change"] or 0.0))),
            )
        )
        return {
            "scope": scope,
            "rows": rows[:MAX_CROSS_SERVICE_ROWS],
        }

    def _validate_claim_bundles(
        self,
        *,
        case_id: str,
        claim_bundles: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        if not claim_bundles:
            return []

        client, model_name, backend = self._get_llm_client()
        if self.debug:
            print(
                f"[Groundedness] validating {len(claim_bundles)} claims for {case_id} "
                f"with backend={backend}, model={model_name}"
            )

        payload = {
            "case_id": case_id,
            "claims": claim_bundles,
        }
        system_msg = (
            "You are validating whether diagnosis claims are supported by raw telemetry.\n"
            "Each claim comes with claim-conditioned retrieval from raw metrics, raw logs, and raw traces.\n"
            "The retrieval bundle may include raw lines, raw spans, raw points, and simple statistics "
            "computed directly from raw telemetry such as averages, p95, counts, peaks, count drops, and onset offsets.\n"
            "Use only the provided evidence bundle for each claim.\n"
            "Return exactly one verdict per claim.\n"
            "Allowed verdicts are only: supported, unsupported.\n"
            "Choose unsupported whenever the evidence is missing, ambiguous, too indirect, or contradictory.\n"
            "Be conservative for inference claims.\n"
            "Return JSON only with this shape: "
            "{\"results\": [{\"id\": str, \"verdict\": \"supported\"|\"unsupported\", \"reason\": str, \"evidence\": [str, ...]}]}."
        )
        user_msg = (
            "Validate the following claims against their raw-telemetry retrieval bundles.\n\n"
            f"{json.dumps(payload, indent=2, ensure_ascii=False)}"
        )

        try:
            response = client.chat.completions.create(
                model=model_name,
                messages=[
                    {"role": "system", "content": system_msg},
                    {"role": "user", "content": user_msg},
                ],
                response_format={"type": "json_object"},
            )
            parsed = self._parse_llm_json_content(response.choices[0].message.content.strip())
            result_rows = parsed.get("results", [])
        except Exception as exc:
            return [
                self._unsupported_claim_result(bundle, f"LLM claim validation failed: {exc}")
                for bundle in claim_bundles
            ]

        by_id = {
            str(row.get("id", "")).strip(): row
            for row in result_rows
            if isinstance(row, dict)
        }

        normalized_results = []
        for bundle in claim_bundles:
            claim_id = bundle["claim_id"]
            row = by_id.get(claim_id)
            if row is None:
                normalized_results.append(
                    self._unsupported_claim_result(bundle, "LLM did not return a result for this claim.")
                )
                continue

            verdict = normalize_label(row.get("verdict"))
            if verdict not in {"supported", "unsupported"}:
                verdict = "unsupported"

            evidence = row.get("evidence", [])
            if not isinstance(evidence, list):
                evidence = []

            normalized_results.append(
                {
                    "claim_id": claim_id,
                    "text": bundle["claim_text"],
                    "type": bundle["claim_type"],
                    "verdict": verdict,
                    "reason": str(row.get("reason", "")).strip() or "No reason provided.",
                    "evidence": evidence[:4],
                }
            )

        return normalized_results

    def _unsupported_claim_result(self, claim_or_bundle: Dict[str, Any], reason: str) -> Dict[str, Any]:
        return {
            "claim_id": str(claim_or_bundle.get("claim_id") or claim_or_bundle.get("id") or "claim"),
            "text": str(claim_or_bundle.get("claim_text") or claim_or_bundle.get("text") or ""),
            "type": normalize_label(claim_or_bundle.get("claim_type") or claim_or_bundle.get("type")) or "observation",
            "verdict": "unsupported",
            "reason": reason,
            "evidence": [],
        }

    def _parse_llm_json_content(self, raw: str) -> Dict[str, Any]:
        text = str(raw).strip()
        if not text:
            raise ValueError("Empty LLM response.")

        try:
            parsed = json.loads(text)
            if isinstance(parsed, dict):
                return parsed
        except Exception:
            pass

        fenced_match = re.search(
            r"```(?:json)?\s*(\{.*?\}|\[.*?\])\s*```",
            text,
            re.DOTALL | re.IGNORECASE,
        )
        if fenced_match:
            candidate = fenced_match.group(1).strip()
            parsed = json.loads(candidate)
            if isinstance(parsed, dict):
                return parsed

        starts = [idx for idx, ch in enumerate(text) if ch in "{["]
        for start in starts:
            opener = text[start]
            closer = "}" if opener == "{" else "]"
            depth = 0
            in_string = False
            escape = False
            for end in range(start, len(text)):
                ch = text[end]
                if in_string:
                    if escape:
                        escape = False
                    elif ch == "\\":
                        escape = True
                    elif ch == '"':
                        in_string = False
                    continue

                if ch == '"':
                    in_string = True
                elif ch == opener:
                    depth += 1
                elif ch == closer:
                    depth -= 1
                    if depth == 0:
                        candidate = text[start : end + 1]
                        try:
                            parsed = json.loads(candidate)
                            if isinstance(parsed, dict):
                                return parsed
                        except Exception:
                            break

        raise ValueError(f"Could not parse JSON object from LLM response: {text[:300]}")

    def _summarize_claim_results(self, rows: List[Dict[str, Any]]) -> Dict[str, Any]:
        total = len(rows)
        supported = sum(1 for row in rows if row["verdict"] == "supported")
        unsupported = total - supported
        observation_rows = [row for row in rows if normalize_label(row.get("type")) == "observation"]
        inference_rows = [row for row in rows if normalize_label(row.get("type")) == "inference"]
        observation_supported = sum(1 for row in observation_rows if row["verdict"] == "supported")
        observation_unsupported = len(observation_rows) - observation_supported
        inference_supported = sum(1 for row in inference_rows if row["verdict"] == "supported")
        inference_unsupported = len(inference_rows) - inference_supported
        majority_supported = bool(total > 0 and supported > (total / 2.0))
        observation_gate = bool(not observation_rows or observation_unsupported == 0)
        return {
            "total_claims": total,
            "supported": supported,
            "unsupported": unsupported,
            "support_rate": (supported / total) if total else 0.0,
            "observation_claims": len(observation_rows),
            "observation_supported": observation_supported,
            "observation_unsupported": observation_unsupported,
            "inference_claims": len(inference_rows),
            "inference_supported": inference_supported,
            "inference_unsupported": inference_unsupported,
            "majority_supported": majority_supported,
            "observation_gate_passed": observation_gate,
            "supported_set": bool(majority_supported and observation_gate),
        }

    def _get_llm_client(self) -> Tuple[Any, str, str]:
        if self._llm_client is not None:
            return self._llm_client, self._llm_model_name, self._llm_backend

        from config.setting import BACKEND, LLM_DEFAULT_MODEL_NAME, VALIDATOR_AGENT_MODEL_NAME
        from utils.llm_client import create_llm_client

        self._llm_backend = BACKEND
        self._llm_model_name = VALIDATOR_AGENT_MODEL_NAME or LLM_DEFAULT_MODEL_NAME
        self._llm_client = create_llm_client(self._llm_backend, "")
        return self._llm_client, self._llm_model_name, self._llm_backend


def summarize_case_results(case_results: List[Dict[str, Any]]) -> Dict[str, Any]:
    service_claims = [
        claim
        for row in case_results
        for claim in row.get("service_claim_results", [])
    ]
    failure_claims = [
        claim
        for row in case_results
        for claim in row.get("failure_claim_results", [])
    ]
    all_claims = service_claims + failure_claims

    def summarize_claims(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
        total = len(rows)
        supported = sum(1 for row in rows if row["verdict"] == "supported")
        return {
            "total_claims": total,
            "supported": supported,
            "unsupported": total - supported,
            "support_rate": (supported / total) if total else 0.0,
        }

    return {
        "total_cases": len(case_results),
        "service_claims": summarize_claims(service_claims),
        "failure_claims": summarize_claims(failure_claims),
        "all_claims": summarize_claims(all_claims),
        "service_supported_cases": sum(1 for row in case_results if row["service_supported"]),
        "failure_supported_cases": sum(1 for row in case_results if row["failure_supported"]),
        "both_supported": sum(1 for row in case_results if row["groundedness_outcome"] == "both_supported"),
        "service_only_supported": sum(
            1 for row in case_results if row["groundedness_outcome"] == "service_only_supported"
        ),
        "failure_only_supported": sum(
            1 for row in case_results if row["groundedness_outcome"] == "failure_only_supported"
        ),
        "neither_supported": sum(
            1 for row in case_results if row["groundedness_outcome"] == "neither_supported"
        ),
    }


def print_summary(summary: Dict[str, Any]) -> None:
    total_cases = int(summary["total_cases"])
    print("\n==============================")
    print("RE2 Source Groundedness")
    print("==============================")
    print(f"Total cases              : {total_cases}")
    print(f"Service supported cases  : {summary['service_supported_cases']}/{total_cases}")
    print(f"Failure supported cases  : {summary['failure_supported_cases']}/{total_cases}")
    print(f"Both supported           : {summary['both_supported']}/{total_cases}")
    print(f"Service only supported   : {summary['service_only_supported']}/{total_cases}")
    print(f"Failure only supported   : {summary['failure_only_supported']}/{total_cases}")
    print(f"Neither supported        : {summary['neither_supported']}/{total_cases}")
    for key in ("service_claims", "failure_claims", "all_claims"):
        block = summary[key]
        print(
            f"{key:24}: supported={block['supported']} unsupported={block['unsupported']} "
            f"support_rate={block['support_rate']:.3f}"
        )


def collect_result_rows(
    data: Dict[str, Any],
    max_cases: int,
    num_cases: int,
    seed: int,
) -> List[Dict[str, Any]]:
    rows = list(data.get("results", []))
    if num_cases > 0:
        if num_cases > len(rows):
            raise ValueError(
                f"Requested --num-cases={num_cases}, but only {len(rows)} cases are available."
            )
        rows = random.Random(seed).sample(rows, num_cases)
        rows.sort(key=lambda row: str(row.get("case_id", "")))
        return rows
    if max_cases > 0:
        rows = rows[:max_cases]
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Validate RE2 diagnosis claims against raw RE2 telemetry using claim-conditioned retrieval."
    )
    parser.add_argument(
        "--diagnosis-results",
        type=Path,
        default=DEFAULT_DIAGNOSIS_RESULTS,
        help="Path to diagnosis_re2_all_results.json or a compatible results file.",
    )
    parser.add_argument(
        "--raw-root",
        type=Path,
        default=DEFAULT_RAW_ROOT,
        help="Path to the raw RE2 case directories.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT_PATH,
        help="Path to write groundedness results JSON.",
    )
    parser.add_argument(
        "--max-cases",
        type=int,
        default=0,
        help="Optional max number of cases to evaluate. 0 means all cases.",
    )
    parser.add_argument(
        "--num-cases",
        type=int,
        default=0,
        help="Randomly sample this many cases. 0 means do not random sample.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=1770,
        help="Random seed used with --num-cases.",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug logging.",
    )
    args = parser.parse_args()

    if not args.diagnosis_results.exists():
        raise FileNotFoundError(f"Diagnosis results file not found: {args.diagnosis_results}")
    if not args.raw_root.exists():
        raise FileNotFoundError(f"Raw RE2 root not found: {args.raw_root}")

    diagnosis_results = load_json(args.diagnosis_results)
    selected_rows = collect_result_rows(
        diagnosis_results,
        args.max_cases,
        args.num_cases,
        args.seed,
    )
    if not selected_rows:
        raise RuntimeError("No diagnosis result rows found.")

    print(f"Diagnosis results: {args.diagnosis_results}")
    print(f"Raw RE2 root    : {args.raw_root}")
    print(f"Cases selected  : {len(selected_rows)}")
    if args.num_cases > 0:
        print(f"Random seed     : {args.seed}")

    validator = RE2RawTelemetryRAGValidator(
        raw_root=args.raw_root,
        debug=args.debug,
    )

    case_results = []
    for index, row in enumerate(selected_rows, start=1):
        case_id = str(row["case_id"])
        print(f"\n[{index}/{len(selected_rows)}] Validating {case_id}")
        case_result = validator.evaluate_case(row)
        case_results.append(case_result)

        print(
            "  "
            f"service_supported={case_result['service_supported']} "
            f"failure_supported={case_result['failure_supported']} "
            f"case_supported={case_result['case_supported']} "
            f"outcome={case_result['groundedness_outcome']}"
        )

    summary = summarize_case_results(case_results)
    print_summary(summary)

    output_payload = {
        "diagnosis_results_path": str(args.diagnosis_results),
        "raw_root": str(args.raw_root),
        "num_cases_evaluated": len(case_results),
        "random_sampling_enabled": bool(args.num_cases > 0),
        "random_seed": args.seed if args.num_cases > 0 else None,
        "selected_case_ids": [row["case_id"] for row in case_results],
        "summary": summary,
        "cases": case_results,
    }
    save_json(args.output, output_payload)
    print(f"\nDetailed groundedness results saved to: {args.output}")


if __name__ == "__main__":
    main()
