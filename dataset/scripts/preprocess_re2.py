#!/usr/bin/env python3
"""Preprocess RCAEval RE2 multimodal cases into evaluation and LLM-ready artifacts."""

from __future__ import annotations

import argparse
import bisect
import csv
import json
import math
import re
import statistics
from collections import Counter, defaultdict
from io import StringIO
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

# SECTION: CONSTANTS
CASE_RE = re.compile(
    r"^re2(?P<system>ob|ss|tt)_(?P<service>.+)_(?P<fault>cpu|mem|disk|delay|loss|socket)_(?P<repeat>\d+)$"
)

SYSTEM_NAME = {
    "ob": "online_boutique",
    "ss": "sock_shop",
    "tt": "train_ticket",
}

DEFAULT_INJECT_OFFSET_SEC = {
    "ob": 360,
    "ss": 360,
    "tt": 480,
}

EPS = 1e-9

UUID_RE = re.compile(
    r"\b[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}\b"
)
HEX_TOKEN_RE = re.compile(r"\b[0-9a-fA-F]{12,}\b")
MIXED_ID_RE = re.compile(r"\b(?=[A-Za-z0-9_-]{6,}\b)(?=[A-Za-z0-9_-]*[A-Za-z])(?=[A-Za-z0-9_-]*\d)[A-Za-z0-9_-]+\b")
CURRENCY_AMOUNT_RE = re.compile(r"\b[A-Za-z]{3}\d+(?:\.\d+)?\b")
IP_RE = re.compile(r"\b\d{1,3}(?:\.\d{1,3}){3}\b")
FLOAT_RE = re.compile(r"\b\d+\.\d+\b")
INT_RE = re.compile(r"\b\d+\b")
WS_RE = re.compile(r"\s+")

HTTP_LINE_RE = re.compile(
    r"\b(GET|POST|PUT|DELETE|PATCH|OPTIONS|HEAD)\s+(\S+)(?:\s+(\d{3}))?(?:\s+([0-9.]+)\s*ms)?",
    re.IGNORECASE,
)
METHOD_KV_RE = re.compile(r"\bmethod=([A-Za-z0-9_./:-]+)")
TOOK_KV_RE = re.compile(r"\btook=([0-9]+(?:\.[0-9]+)?)ms\b", re.IGNORECASE)
LATENCY_RE = re.compile(r"\b([0-9]+(?:\.[0-9]+)?)\s*ms\b", re.IGNORECASE)
STATUS_RE = re.compile(r"\b([1-5]\d{2})\b")

ERROR_KEYWORDS = (
    "error",
    "exception",
    "timeout",
    "timed out",
    "failed",
    "failure",
    "unavailable",
    "refused",
    "reset",
    "panic",
    "socket",
    "broken pipe",
    "503",
    "500",
)

# SECTION: GENERIC_HELPERS
def _to_int(value: object) -> Optional[int]:
    try:
        return int(value)
    except Exception:
        return None


def _to_float(value: object) -> float:
    try:
        return float(value)
    except Exception:
        return float("nan")


def _mean(values: List[float]) -> float:
    return statistics.fmean(values) if values else 0.0


def _std(values: List[float]) -> float:
    return statistics.pstdev(values) if len(values) > 1 else 0.0


def _safe_z(delta: float, sigma: float) -> float:
    if sigma < EPS:
        if abs(delta) < EPS:
            return 0.0
        return 25.0 if delta > 0 else -25.0
    return delta / sigma


def _percentile(values: List[float], q: float) -> float:
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


def _split_metric_name(metric_name: str) -> Tuple[str, str]:
    if "_" not in metric_name:
        return metric_name, "unknown"
    service, metric_type = metric_name.rsplit("_", 1)
    return service, metric_type


def canonical_metric_family(metric_type: str) -> str:
    text = str(metric_type).strip().lower()
    if not text:
        return "unknown"
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
    return "unknown"


def parse_case_id(case_id: str) -> Optional[Dict[str, object]]:
    match = CASE_RE.match(case_id)
    if not match:
        return None
    groups = match.groupdict()
    return {
        "case_id": case_id,
        "suite": "re2",
        "system_code": groups["system"],
        "system": SYSTEM_NAME[groups["system"]],
        "root_cause_service": groups["service"],
        "fault_type": groups["fault"],
        "repeat_id": int(groups["repeat"]),
        "root_cause_indicator": f'{groups["service"]}_{groups["fault"]}',
    }

# SECTION: METRIC_HELPERS
def load_metrics(metrics_path: Path) -> Dict[str, List[Tuple[int, float]]]:
    raw = json.loads(metrics_path.read_text(encoding="utf-8"))
    cleaned: Dict[str, List[Tuple[int, float]]] = {}
    for metric_name, points in raw.items():
        if not isinstance(points, list):
            continue
        parsed: List[Tuple[int, float]] = []
        for point in points:
            if not isinstance(point, (list, tuple)) or len(point) < 2:
                continue
            ts = _to_int(point[0])
            if ts is None:
                continue
            parsed.append((ts, _to_float(point[1])))
        if not parsed:
            continue
        parsed.sort(key=lambda x: x[0])
        dedup: List[Tuple[int, float]] = []
        for ts, val in parsed:
            if dedup and dedup[-1][0] == ts:
                dedup[-1] = (ts, val)
            else:
                dedup.append((ts, val))
        cleaned[metric_name] = dedup
    return cleaned


def fill_missing(values: List[float]) -> Tuple[List[float], int]:
    if not values:
        return values, 0

    out = values[:]
    nan_positions = [idx for idx, value in enumerate(out) if math.isnan(value)]
    if not nan_positions:
        return out, 0

    for idx in range(1, len(out)):
        if math.isnan(out[idx]) and not math.isnan(out[idx - 1]):
            out[idx] = out[idx - 1]

    for idx in range(len(out) - 2, -1, -1):
        if math.isnan(out[idx]) and not math.isnan(out[idx + 1]):
            out[idx] = out[idx + 1]

    for idx, value in enumerate(out):
        if math.isnan(value):
            out[idx] = 0.0

    return out, len(nan_positions)


def metric_point_is_anomalous(
    value: float,
    baseline_mean: float,
    baseline_std: float,
    direction: str,
) -> bool:
    delta = value - baseline_mean
    if direction == "up" and delta <= 0:
        return False
    if direction == "down" and delta >= 0:
        return False

    abs_delta = abs(delta)
    min_abs_delta = max(abs(baseline_mean) * 0.10, baseline_std * 2.0, 1e-6)
    z_value = abs(_safe_z(delta, baseline_std))
    return abs_delta >= min_abs_delta and z_value >= 2.5


def detect_metric_temporal_features(
    incident_timestamps: List[int],
    incident_values: List[float],
    inject_time: int,
    baseline_mean: float,
    baseline_std: float,
    direction: str,
) -> Dict[str, object]:
    if not incident_timestamps or not incident_values:
        return {
            "first_incident_ts": None,
            "first_incident_offset_sec": None,
            "peak_offset_sec": None,
            "sustained_after_inject": False,
        }

    strongest_idx = max(
        range(len(incident_values)),
        key=lambda idx: abs(float(incident_values[idx]) - baseline_mean),
    )
    peak_offset_sec = int(incident_timestamps[strongest_idx]) - inject_time

    first_incident_ts: Optional[int] = None
    sustained_after_inject = False
    streak = 0
    required_streak = 3 if len(incident_values) >= 3 else 1

    for idx, value in enumerate(incident_values):
        anomalous = metric_point_is_anomalous(
            value=value,
            baseline_mean=baseline_mean,
            baseline_std=baseline_std,
            direction=direction,
        )
        if anomalous:
            if first_incident_ts is None:
                first_incident_ts = int(incident_timestamps[idx])
            streak += 1
            if streak >= required_streak:
                sustained_after_inject = True
                break
        else:
            streak = 0

    return {
        "first_incident_ts": first_incident_ts,
        "first_incident_offset_sec": (
            int(first_incident_ts) - inject_time if first_incident_ts is not None else None
        ),
        "peak_offset_sec": peak_offset_sec,
        "sustained_after_inject": sustained_after_inject,
    }


def read_inject_time(path: Path) -> Optional[int]:
    raw = path.read_text(encoding="utf-8").strip()
    return _to_int(raw)


def build_metric_feature(
    metric_name: str,
    series: List[Tuple[int, float]],
    inject_time: int,
    baseline_sec: int,
    incident_sec: int,
) -> Dict[str, object]:
    timestamps = [t for t, _ in series]
    values_raw = [v for _, v in series]
    values, nan_filled = fill_missing(values_raw)

    baseline_start = inject_time - baseline_sec
    incident_end = inject_time + incident_sec

    b_start = bisect.bisect_left(timestamps, baseline_start)
    b_end = bisect.bisect_left(timestamps, inject_time)
    i_start = b_end
    i_end = bisect.bisect_right(timestamps, incident_end)

    baseline = values[b_start:b_end]
    incident = values[i_start:i_end]
    incident_timestamps = timestamps[i_start:i_end]

    if not baseline:
        fallback = min(120, b_end)
        baseline = (
            values[max(0, b_end - fallback):b_end]
            if fallback > 0
            else values[: min(120, len(values))]
        )
    if not incident:
        if i_start < len(values):
            fallback_end = min(len(values), i_start + max(60, incident_sec // 2))
            incident = values[i_start:fallback_end]
            incident_timestamps = timestamps[i_start:fallback_end]
        if not incident:
            fallback_start = max(0, len(values) - 60)
            incident = values[fallback_start:]
            incident_timestamps = timestamps[fallback_start:]

    baseline_mean = _mean(baseline)
    baseline_std = _std(baseline)
    incident_mean = _mean(incident)
    incident_peak = max(incident) if incident else 0.0
    incident_trough = min(incident) if incident else 0.0
    delta_mean = incident_mean - baseline_mean
    peak_abs_delta = max((abs(v - baseline_mean) for v in incident), default=0.0)
    z_mean = _safe_z(delta_mean, baseline_std)
    z_peak = _safe_z(peak_abs_delta, baseline_std)
    ratio = delta_mean / (abs(baseline_mean) + EPS)
    score = max(abs(z_mean), abs(z_peak))
    direction = "up" if delta_mean > 0 else "down" if delta_mean < 0 else "flat"

    service, metric_type = _split_metric_name(metric_name)
    temporal = detect_metric_temporal_features(
        incident_timestamps=incident_timestamps,
        incident_values=incident,
        inject_time=inject_time,
        baseline_mean=baseline_mean,
        baseline_std=baseline_std,
        direction=direction,
    )
    return {
        "metric_name": metric_name,
        "service": service,
        "metric_type": metric_type,
        "metric_family": canonical_metric_family(metric_type),
        "direction": direction,
        "score": score,
        "z_mean": z_mean,
        "z_peak": z_peak,
        "delta_mean": delta_mean,
        "change_ratio": ratio,
        "baseline_mean": baseline_mean,
        "baseline_std": baseline_std,
        "incident_mean": incident_mean,
        "incident_peak": incident_peak,
        "incident_trough": incident_trough,
        "first_incident_ts": temporal["first_incident_ts"],
        "first_incident_offset_sec": temporal["first_incident_offset_sec"],
        "peak_offset_sec": temporal["peak_offset_sec"],
        "sustained_after_inject": temporal["sustained_after_inject"],
        "nan_filled": nan_filled,
        "points": len(series),
    }


def build_neutral_metric_facts(metric_features: List[Dict[str, object]]) -> List[Dict[str, object]]:
    facts: List[Dict[str, object]] = []
    for row in metric_features:
        baseline_mean = float(row["baseline_mean"])
        incident_mean = float(row["incident_mean"])
        mean_change = incident_mean - baseline_mean
        absolute_change = abs(mean_change)
        percent_change = ((mean_change / baseline_mean) * 100.0) if abs(baseline_mean) > EPS else None
        facts.append(
            {
                "service": row["service"],
                "metric_name": row["metric_name"],
                "metric_type": row["metric_type"],
                "metric_family": row.get("metric_family", canonical_metric_family(str(row["metric_type"]))),
                "direction": row.get("direction", "flat"),
                "baseline_mean": baseline_mean,
                "incident_mean": incident_mean,
                "mean_change": mean_change,
                "absolute_change": absolute_change,
                "percent_change": percent_change,
                "baseline_std": float(row["baseline_std"]),
                "incident_peak": float(row["incident_peak"]),
                "first_incident_ts": row.get("first_incident_ts"),
                "first_incident_offset_sec": row.get("first_incident_offset_sec"),
                "peak_offset_sec": row.get("peak_offset_sec"),
                "sustained_after_inject": bool(row.get("sustained_after_inject", False)),
                "sample_count": int(row["points"]),
            }
        )
    facts.sort(key=lambda item: (str(item["service"]), str(item["metric_name"])))
    return facts

# SECTION: CSV_WINDOW_HELPERS
def parse_csv_line(line: str) -> List[str]:
    return next(csv.reader(StringIO(line)))


def csv_time_window(path: Path, time_col: str, ms: bool = False) -> Optional[Tuple[int, int]]:
    if not path.exists():
        return None

    with path.open("r", encoding="utf-8", newline="") as fh:
        header_line = fh.readline()
        if not header_line:
            return None
        header = parse_csv_line(header_line)
        if time_col not in header:
            return None
        idx = header.index(time_col)

        first_line = fh.readline()
        while first_line and not first_line.strip():
            first_line = fh.readline()
        if not first_line:
            return None

    with path.open("rb") as fh:
        fh.seek(0, 2)
        end = fh.tell()
        pos = max(0, end - 4096)
        chunk = b""
        last_line = None
        while True:
            fh.seek(pos)
            chunk = fh.read(end - pos) + chunk
            lines = [line for line in chunk.splitlines() if line.strip()]
            if pos == 0 and lines:
                lines = lines[1:]
            if lines:
                last_line = lines[-1].decode("utf-8", errors="ignore")
                break
            if pos == 0:
                break
            end = pos
            pos = max(0, pos - 4096)
        if last_line is None:
            return None

    first_row = parse_csv_line(first_line)
    last_row = parse_csv_line(last_line)
    try:
        first_ts = int(first_row[idx].strip())
        last_ts = int(last_row[idx].strip())
        if ms:
            first_ts //= 1000
            last_ts //= 1000
        return first_ts, last_ts
    except Exception:
        return None

# SECTION: LOG_HELPERS
def normalize_log_template(message: str) -> str:
    text = message.strip()
    text = WS_RE.sub(" ", text)
    text = UUID_RE.sub("<uuid>", text)
    text = HEX_TOKEN_RE.sub("<hex>", text)
    text = MIXED_ID_RE.sub("<id>", text)
    text = CURRENCY_AMOUNT_RE.sub("<money>", text)
    text = IP_RE.sub("<ip>", text)
    text = FLOAT_RE.sub("<float>", text)
    text = INT_RE.sub("<num>", text)
    text = WS_RE.sub(" ", text)
    return text.lower()


def extract_log_observation(message: str) -> Dict[str, object]:
    lowered = message.lower()
    severity = "info"
    if " error " in f" {lowered} " or lowered.startswith("error"):
        severity = "error"
    elif " warn " in f" {lowered} " or lowered.startswith("warn"):
        severity = "warn"
    elif " debug " in f" {lowered} " or lowered.startswith("debug"):
        severity = "debug"

    latency_ms = None
    http_status = None
    endpoint = ""

    http_match = HTTP_LINE_RE.search(message)
    if http_match:
        endpoint = f"{http_match.group(1).upper()} {http_match.group(2)}"
        if http_match.group(3):
            http_status = _to_int(http_match.group(3))
        if http_match.group(4):
            latency_ms = _to_float(http_match.group(4))

    if latency_ms is None:
        took_match = TOOK_KV_RE.search(message)
        if took_match:
            latency_ms = _to_float(took_match.group(1))
    if latency_ms is None:
        latency_match = LATENCY_RE.search(message)
        if latency_match:
            latency_ms = _to_float(latency_match.group(1))

    if http_status is None:
        status_match = STATUS_RE.search(message)
        if status_match:
            status_value = _to_int(status_match.group(1))
            if status_value is not None:
                http_status = status_value

    if not endpoint:
        method_match = METHOD_KV_RE.search(message)
        if method_match:
            endpoint = method_match.group(1)

    matched_keywords = [keyword for keyword in ERROR_KEYWORDS if keyword in lowered]
    if http_status is not None and http_status >= 500:
        severity = "error"
    elif severity == "info" and matched_keywords:
        severity = "warn"

    return {
        "severity": severity,
        "http_status": http_status,
        "latency_ms": latency_ms,
        "endpoint": endpoint,
        "matched_keywords": matched_keywords,
    }


def build_log_features(
    logs_path: Path,
    inject_time: int,
    baseline_sec: int,
    incident_sec: int,
    top_k_facts: int,
) -> Dict[str, object]:
    if not logs_path.exists():
        return {
            "feature_rows": [],
            "facts": [],
            "examples": [],
            "row_count_in_window": 0,
            "template_count_in_incident": 0,
        }

    baseline_start = inject_time - baseline_sec
    incident_end = inject_time + incident_sec

    grouped: Dict[Tuple[str, str], Dict[str, object]] = {}
    row_count_in_window = 0

    with logs_path.open("r", encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            ts = _to_int(row.get("timestamp"))
            if ts is None or ts < baseline_start or ts > incident_end:
                continue
            row_count_in_window += 1

            service = (row.get("container_name") or "unknown").strip() or "unknown"
            message = (row.get("message") or "").strip()
            template = normalize_log_template(message)
            period = "baseline" if ts < inject_time else "incident"
            observation = extract_log_observation(message)

            key = (service, template)
            if key not in grouped:
                grouped[key] = {
                    "service": service,
                    "template": template,
                    "baseline_count": 0,
                    "incident_count": 0,
                    "baseline_keyword_hits": 0,
                    "incident_keyword_hits": 0,
                    "baseline_http_4xx_count": 0,
                    "incident_http_4xx_count": 0,
                    "baseline_http_5xx_count": 0,
                    "incident_http_5xx_count": 0,
                    "baseline_latencies": [],
                    "incident_latencies": [],
                    "representative_messages": [],
                    "representative_endpoints": set(),
                    "incident_severity_counter": Counter(),
                    "first_incident_ts": None,
                }
            acc = grouped[key]
            acc[f"{period}_count"] += 1
            acc[f"{period}_keyword_hits"] += len(observation["matched_keywords"])

            http_status = observation["http_status"]
            if http_status is not None:
                if 400 <= http_status < 500:
                    acc[f"{period}_http_4xx_count"] += 1
                elif 500 <= http_status < 600:
                    acc[f"{period}_http_5xx_count"] += 1

            latency_ms = observation["latency_ms"]
            if latency_ms is not None and not math.isnan(latency_ms):
                acc[f"{period}_latencies"].append(float(latency_ms))

            endpoint = str(observation["endpoint"]).strip()
            if endpoint and len(acc["representative_endpoints"]) < 5:
                acc["representative_endpoints"].add(endpoint)

            if period == "incident":
                acc["incident_severity_counter"][str(observation["severity"])] += 1
                if acc["first_incident_ts"] is None:
                    acc["first_incident_ts"] = ts
                if len(acc["representative_messages"]) < 3:
                    acc["representative_messages"].append(message)

    feature_rows: List[Dict[str, object]] = []
    for acc in grouped.values():
        incident_count = int(acc["incident_count"])
        if incident_count == 0:
            continue

        baseline_count = int(acc["baseline_count"])
        delta_count = incident_count - baseline_count
        incident_ratio = incident_count / max(1, baseline_count)
        new_in_incident = baseline_count == 0 and incident_count > 0

        latency_p95_baseline_ms = _percentile(acc["baseline_latencies"], 95.0)
        latency_p95_incident_ms = _percentile(acc["incident_latencies"], 95.0)
        latency_delta_ms = latency_p95_incident_ms - latency_p95_baseline_ms

        incident_http_5xx_count = int(acc["incident_http_5xx_count"])
        baseline_http_5xx_count = int(acc["baseline_http_5xx_count"])
        keyword_delta = int(acc["incident_keyword_hits"]) - int(acc["baseline_keyword_hits"])

        count_change_rate = (
            max(0.0, float(delta_count)) / max(1.0, float(baseline_count))
            if baseline_count > 0
            else float(incident_count)
        )
        anomaly_score = (
            8.0 * min(count_change_rate, 10.0)
            + 5.0 * max(0, incident_http_5xx_count - baseline_http_5xx_count)
            + 3.0 * max(0, keyword_delta)
            + 0.25 * max(0.0, latency_delta_ms)
            + (6.0 if new_in_incident else 0.0)
        )

        severity_hint = "info"
        if acc["incident_severity_counter"]:
            severity_hint = acc["incident_severity_counter"].most_common(1)[0][0]

        first_incident_ts = acc["first_incident_ts"]
        feature_rows.append(
            {
                "service": acc["service"],
                "template": acc["template"],
                "anomaly_score": anomaly_score,
                "baseline_count": baseline_count,
                "incident_count": incident_count,
                "delta_count": delta_count,
                "incident_ratio": incident_ratio,
                "new_in_incident": new_in_incident,
                "baseline_keyword_hits": int(acc["baseline_keyword_hits"]),
                "incident_keyword_hits": int(acc["incident_keyword_hits"]),
                "baseline_http_4xx_count": int(acc["baseline_http_4xx_count"]),
                "incident_http_4xx_count": int(acc["incident_http_4xx_count"]),
                "baseline_http_5xx_count": baseline_http_5xx_count,
                "incident_http_5xx_count": incident_http_5xx_count,
                "latency_p95_baseline_ms": latency_p95_baseline_ms,
                "latency_p95_incident_ms": latency_p95_incident_ms,
                "latency_delta_ms": latency_delta_ms,
                "first_incident_ts": first_incident_ts,
                "first_incident_offset_sec": (
                    int(first_incident_ts) - inject_time if first_incident_ts is not None else None
                ),
                "severity_hint": severity_hint,
                "representative_messages": list(acc["representative_messages"]),
                "representative_endpoints": sorted(acc["representative_endpoints"]),
            }
        )

    feature_rows.sort(key=lambda row: float(row["anomaly_score"]), reverse=True)
    facts = build_neutral_log_facts(feature_rows[:top_k_facts])
    examples = build_log_examples(feature_rows[:max(6, min(10, top_k_facts))])
    return {
        "feature_rows": feature_rows,
        "facts": facts,
        "examples": examples,
        "row_count_in_window": row_count_in_window,
        "template_count_in_incident": len(feature_rows),
    }


def build_neutral_log_facts(log_feature_rows: List[Dict[str, object]]) -> List[Dict[str, object]]:
    facts: List[Dict[str, object]] = []
    for row in log_feature_rows:
        facts.append(
            {
                "service": row["service"],
                "template": row["template"],
                "baseline_count": int(row["baseline_count"]),
                "incident_count": int(row["incident_count"]),
                "delta_count": int(row["delta_count"]),
                "incident_ratio": round(float(row["incident_ratio"]), 4),
                "new_in_incident": bool(row["new_in_incident"]),
                "incident_http_4xx_count": int(row["incident_http_4xx_count"]),
                "incident_http_5xx_count": int(row["incident_http_5xx_count"]),
                "incident_keyword_hits": int(row["incident_keyword_hits"]),
                "latency_p95_baseline_ms": round(float(row["latency_p95_baseline_ms"]), 4),
                "latency_p95_incident_ms": round(float(row["latency_p95_incident_ms"]), 4),
                "latency_delta_ms": round(float(row["latency_delta_ms"]), 4),
                "first_incident_offset_sec": row["first_incident_offset_sec"],
                "severity_hint": row["severity_hint"],
                "representative_endpoints": list(row["representative_endpoints"]),
                "representative_messages": list(row["representative_messages"]),
            }
        )
    return facts


def build_log_examples(log_feature_rows: List[Dict[str, object]]) -> List[Dict[str, object]]:
    examples: List[Dict[str, object]] = []
    for row in log_feature_rows:
        messages = row["representative_messages"]
        if not messages:
            continue
        examples.append(
            {
                "service": row["service"],
                "offset_sec": row["first_incident_offset_sec"],
                "message": messages[0],
            }
        )
        if len(examples) >= 6:
            break
    return examples

# SECTION: TRACE_HELPERS
def normalize_trace_operation(operation_name: str) -> str:
    text = operation_name.strip() if operation_name else "unknown"
    text = UUID_RE.sub("{uuid}", text)
    text = HEX_TOKEN_RE.sub("{hex}", text)
    text = INT_RE.sub("<num>", text)
    text = WS_RE.sub(" ", text)
    return text


def duration_to_ms_estimate(raw_duration: float) -> float:
    return raw_duration / 1000.0


def init_trace_accumulator() -> Dict[str, object]:
    return {
        "baseline_count": 0,
        "incident_count": 0,
        "baseline_durations_ms": [],
        "incident_durations_ms": [],
        "first_incident_ts": None,
    }


def build_trace_features(
    traces_path: Path,
    inject_time: int,
    baseline_sec: int,
    incident_sec: int,
    top_k_facts: int,
    trace_example_count: int,
) -> Dict[str, object]:
    if not traces_path.exists():
        return {
            "feature_rows": [],
            "facts": [],
            "examples": [],
            "row_count_in_window": 0,
            "operation_count_in_incident": 0,
        }

    baseline_start = inject_time - baseline_sec
    incident_end = inject_time + incident_sec

    grouped: Dict[Tuple[str, str], Dict[str, object]] = {}
    root_trace_candidates: Dict[str, Dict[str, object]] = {}
    row_count_in_window = 0

    with traces_path.open("r", encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            start_ms = _to_int(row.get("startTimeMillis"))
            raw_duration = _to_float(row.get("duration"))
            if start_ms is None or math.isnan(raw_duration):
                continue

            start_sec = start_ms // 1000
            if start_sec < baseline_start or start_sec > incident_end:
                continue
            row_count_in_window += 1

            service = (row.get("serviceName") or "unknown").strip() or "unknown"
            operation_raw = (
                (row.get("operationName") or "").strip()
                or (row.get("methodName") or "").strip()
                or service
            )
            operation = normalize_trace_operation(operation_raw)
            duration_ms = duration_to_ms_estimate(float(raw_duration))
            period = "baseline" if start_sec < inject_time else "incident"

            key = (service, operation)
            if key not in grouped:
                grouped[key] = init_trace_accumulator()
                grouped[key]["service"] = service
                grouped[key]["operation"] = operation
            acc = grouped[key]
            acc[f"{period}_count"] += 1
            acc[f"{period}_durations_ms"].append(duration_ms)
            if period == "incident" and acc["first_incident_ts"] is None:
                acc["first_incident_ts"] = start_sec

            parent_span_id = (row.get("parentSpanID") or "").strip()
            if not parent_span_id and period == "incident":
                trace_id = (row.get("traceID") or "").strip()
                candidate = {
                    "trace_id": trace_id,
                    "root_service": service,
                    "root_operation": operation,
                    "root_duration_ms": duration_ms,
                    "root_start_sec": start_sec,
                }
                existing = root_trace_candidates.get(trace_id)
                if existing is None or candidate["root_duration_ms"] > existing["root_duration_ms"]:
                    root_trace_candidates[trace_id] = candidate

    feature_rows: List[Dict[str, object]] = []
    for acc in grouped.values():
        incident_count = int(acc["incident_count"])
        if incident_count == 0:
            continue

        baseline_count = int(acc["baseline_count"])
        baseline_mean_ms = _mean(acc["baseline_durations_ms"])
        baseline_p95_ms = _percentile(acc["baseline_durations_ms"], 95.0)
        incident_mean_ms = _mean(acc["incident_durations_ms"])
        incident_p95_ms = _percentile(acc["incident_durations_ms"], 95.0)
        incident_max_ms = max(acc["incident_durations_ms"]) if acc["incident_durations_ms"] else 0.0
        delta_p95_ms = incident_p95_ms - baseline_p95_ms
        latency_ratio = incident_p95_ms / max(baseline_p95_ms, 1e-6)
        new_in_incident = baseline_count == 0 and incident_count > 0

        anomaly_score = (
            2.5 * max(0.0, delta_p95_ms)
            + 3.0 * min(latency_ratio, 15.0)
            + 0.05 * max(0, incident_count - baseline_count)
            + (3.0 if new_in_incident else 0.0)
        )

        first_incident_ts = acc["first_incident_ts"]
        feature_rows.append(
            {
                "service": acc["service"],
                "operation": acc["operation"],
                "anomaly_score": anomaly_score,
                "baseline_count": baseline_count,
                "incident_count": incident_count,
                "baseline_mean_ms": baseline_mean_ms,
                "baseline_p95_ms": baseline_p95_ms,
                "incident_mean_ms": incident_mean_ms,
                "incident_p95_ms": incident_p95_ms,
                "incident_max_ms": incident_max_ms,
                "delta_p95_ms": delta_p95_ms,
                "latency_ratio": latency_ratio,
                "new_in_incident": new_in_incident,
                "first_incident_ts": first_incident_ts,
                "first_incident_offset_sec": (
                    int(first_incident_ts) - inject_time if first_incident_ts is not None else None
                ),
            }
        )

    feature_rows.sort(key=lambda row: float(row["anomaly_score"]), reverse=True)
    trace_examples = build_representative_trace_examples(
        traces_path=traces_path,
        inject_time=inject_time,
        selected_trace_ids=_top_trace_ids(root_trace_candidates, trace_example_count),
    )
    facts = build_neutral_trace_facts(feature_rows[:top_k_facts])
    return {
        "feature_rows": feature_rows,
        "facts": facts,
        "examples": trace_examples,
        "row_count_in_window": row_count_in_window,
        "operation_count_in_incident": len(feature_rows),
    }


def _top_trace_ids(root_trace_candidates: Dict[str, Dict[str, object]], top_k: int) -> List[str]:
    ranked = sorted(
        root_trace_candidates.values(),
        key=lambda row: float(row["root_duration_ms"]),
        reverse=True,
    )
    return [str(row["trace_id"]) for row in ranked[:top_k] if row.get("trace_id")]


def build_representative_trace_examples(
    traces_path: Path,
    inject_time: int,
    selected_trace_ids: List[str],
) -> List[Dict[str, object]]:
    if not selected_trace_ids:
        return []

    selected = set(selected_trace_ids)
    traces: Dict[str, List[Dict[str, object]]] = defaultdict(list)
    with traces_path.open("r", encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            trace_id = (row.get("traceID") or "").strip()
            if trace_id in selected:
                traces[trace_id].append(row)

    examples: List[Dict[str, object]] = []
    for trace_id in selected_trace_ids:
        rows = traces.get(trace_id, [])
        if not rows:
            continue

        spans: Dict[str, Dict[str, object]] = {}
        children: Dict[str, List[str]] = defaultdict(list)
        roots: List[str] = []
        for row in rows:
            span_id = (row.get("spanID") or "").strip()
            if not span_id:
                continue
            service = (row.get("serviceName") or "unknown").strip() or "unknown"
            operation_raw = (
                (row.get("operationName") or "").strip()
                or (row.get("methodName") or "").strip()
                or service
            )
            parent_span_id = (row.get("parentSpanID") or "").strip()
            start_ms = _to_int(row.get("startTimeMillis")) or 0
            duration_ms = duration_to_ms_estimate(_to_float(row.get("duration")))
            spans[span_id] = {
                "service": service,
                "operation": normalize_trace_operation(operation_raw),
                "parent": parent_span_id,
                "start_ms": start_ms,
                "duration_ms": duration_ms,
            }
            if parent_span_id:
                children[parent_span_id].append(span_id)
            else:
                roots.append(span_id)

        for child_ids in children.values():
            child_ids.sort(key=lambda child_id: int(spans[child_id]["start_ms"]))
        roots.sort(key=lambda span_id: int(spans[span_id]["start_ms"]))
        if not roots:
            continue

        best_path: List[str] = []
        best_score = -1.0
        for root_span_id in roots:
            path, score = longest_trace_path(root_span_id, children, spans)
            if score > best_score:
                best_score = score
                best_path = path

        root_span = spans[roots[0]]
        path_services = [str(spans[span_id]["service"]) for span_id in best_path]
        hot_spans = sorted(
            (
                {
                    "service": span["service"],
                    "operation": span["operation"],
                    "duration_ms": round(float(span["duration_ms"]), 4),
                }
                for span in spans.values()
            ),
            key=lambda row: float(row["duration_ms"]),
            reverse=True,
        )[:4]

        examples.append(
            {
                "trace_id": trace_id,
                "root_service": root_span["service"],
                "root_operation": root_span["operation"],
                "root_start_offset_sec": (int(root_span["start_ms"]) // 1000) - inject_time,
                "root_duration_ms": round(float(root_span["duration_ms"]), 4),
                "service_path": path_services,
                "hot_spans": hot_spans,
            }
        )

    return examples


def longest_trace_path(
    span_id: str,
    children: Dict[str, List[str]],
    spans: Dict[str, Dict[str, object]],
) -> Tuple[List[str], float]:
    child_ids = children.get(span_id, [])
    own_duration = float(spans[span_id]["duration_ms"])
    if not child_ids:
        return [span_id], own_duration

    best_child_path: List[str] = []
    best_child_score = -1.0
    for child_span_id in child_ids:
        child_path, child_score = longest_trace_path(child_span_id, children, spans)
        if child_score > best_child_score:
            best_child_path = child_path
            best_child_score = child_score
    return [span_id] + best_child_path, own_duration + max(best_child_score, 0.0)


def build_neutral_trace_facts(trace_feature_rows: List[Dict[str, object]]) -> List[Dict[str, object]]:
    facts: List[Dict[str, object]] = []
    for row in trace_feature_rows:
        facts.append(
            {
                "service": row["service"],
                "operation": row["operation"],
                "baseline_count": int(row["baseline_count"]),
                "incident_count": int(row["incident_count"]),
                "new_in_incident": bool(row["new_in_incident"]),
                "baseline_mean_ms": round(float(row["baseline_mean_ms"]), 4),
                "baseline_p95_ms": round(float(row["baseline_p95_ms"]), 4),
                "incident_mean_ms": round(float(row["incident_mean_ms"]), 4),
                "incident_p95_ms": round(float(row["incident_p95_ms"]), 4),
                "incident_max_ms": round(float(row["incident_max_ms"]), 4),
                "delta_p95_ms": round(float(row["delta_p95_ms"]), 4),
                "latency_ratio": round(float(row["latency_ratio"]), 4),
                "first_incident_offset_sec": row["first_incident_offset_sec"],
            }
        )
    return facts

# SECTION: SUMMARY_HELPERS
def summarize_service_scores(
    feature_rows: List[Dict[str, object]],
    score_field: str,
    item_field: str,
    top_k_services: int,
    top_items_per_service: int = 3,
) -> List[Dict[str, object]]:
    by_service: Dict[str, List[Dict[str, object]]] = defaultdict(list)
    for row in feature_rows:
        by_service[str(row["service"])].append(row)

    summaries: List[Dict[str, object]] = []
    for service, rows in by_service.items():
        rows_sorted = sorted(rows, key=lambda row: float(row[score_field]), reverse=True)
        top_scores = [float(row[score_field]) for row in rows_sorted[:3]]
        top_score = top_scores[0] if top_scores else 0.0
        blended_score = 0.7 * top_score + 0.3 * _mean(top_scores)
        summaries.append(
            {
                "service": service,
                "score": blended_score,
                "top_items": [str(row[item_field]) for row in rows_sorted[:top_items_per_service]],
            }
        )
    summaries.sort(key=lambda row: float(row["score"]), reverse=True)
    return summaries[:top_k_services]


def find_service_rank(service_rows: List[Dict[str, object]], target_service: str) -> Optional[int]:
    for idx, row in enumerate(service_rows, start=1):
        if str(row["service"]) == target_service:
            return idx
    return None


def format_metric_highlight(row: Dict[str, object]) -> str:
    return (
        f"{row['metric_name']} mean {float(row['baseline_mean']):.3f} -> "
        f"{float(row['incident_mean']):.3f} (peak={float(row['incident_peak']):.3f}, "
        f"family={row.get('metric_family', canonical_metric_family(str(row.get('metric_type', 'unknown'))))})"
    )


def format_log_highlight(row: Dict[str, object]) -> str:
    return (
        f"log template count {int(row['baseline_count'])} -> {int(row['incident_count'])}, "
        f"5xx={int(row['incident_http_5xx_count'])}, "
        f"latency_p95={float(row['latency_p95_incident_ms']):.2f}ms"
    )


def format_trace_highlight(row: Dict[str, object]) -> str:
    return (
        f"{row['operation']} p95 {float(row['baseline_p95_ms']):.2f} -> "
        f"{float(row['incident_p95_ms']):.2f} ms"
    )


def build_service_snapshots(
    metric_features: List[Dict[str, object]],
    log_features: List[Dict[str, object]],
    trace_features: List[Dict[str, object]],
    top_k: int = 8,
) -> List[Dict[str, object]]:
    by_service_score: Counter = Counter()
    by_metric_service: Dict[str, List[Dict[str, object]]] = defaultdict(list)
    by_log_service: Dict[str, List[Dict[str, object]]] = defaultdict(list)
    by_trace_service: Dict[str, List[Dict[str, object]]] = defaultdict(list)

    for row in metric_features[:24]:
        service = str(row["service"])
        by_metric_service[service].append(row)
        by_service_score[service] += float(row["score"])
    for row in log_features[:18]:
        service = str(row["service"])
        by_log_service[service].append(row)
        by_service_score[service] += float(row["anomaly_score"])
    for row in trace_features[:18]:
        service = str(row["service"])
        by_trace_service[service].append(row)
        by_service_score[service] += float(row["anomaly_score"])

    snapshots: List[Dict[str, object]] = []
    for service, _score in by_service_score.most_common(top_k):
        snapshots.append(
            {
                "service": service,
                "metric_highlights": [format_metric_highlight(row) for row in by_metric_service[service][:2]],
                "log_highlights": [format_log_highlight(row) for row in by_log_service[service][:2]],
                "trace_highlights": [format_trace_highlight(row) for row in by_trace_service[service][:2]],
            }
        )
    return snapshots


def build_incident_timeline(
    log_features: List[Dict[str, object]],
    trace_features: List[Dict[str, object]],
    top_k: int = 12,
) -> List[Dict[str, object]]:
    timeline: List[Dict[str, object]] = []
    for row in log_features[:8]:
        if row["first_incident_offset_sec"] is None:
            continue
        timeline.append(
            {
                "offset_sec": int(row["first_incident_offset_sec"]),
                "source": "logs",
                "service": row["service"],
                "summary": (
                    f"log template count {int(row['baseline_count'])} -> {int(row['incident_count'])}, "
                    f"5xx={int(row['incident_http_5xx_count'])}"
                ),
            }
        )
    for row in trace_features[:8]:
        if row["first_incident_offset_sec"] is None:
            continue
        timeline.append(
            {
                "offset_sec": int(row["first_incident_offset_sec"]),
                "source": "traces",
                "service": row["service"],
                "summary": (
                    f"{row['operation']} p95 {float(row['baseline_p95_ms']):.2f} -> "
                    f"{float(row['incident_p95_ms']):.2f} ms"
                ),
            }
        )
    timeline.sort(key=lambda row: (int(row["offset_sec"]), str(row["source"]), str(row["service"])))
    return timeline[:top_k]


def build_llm_input_payload(
    case_summary: Dict[str, object],
    metric_facts: List[Dict[str, object]],
    log_facts: List[Dict[str, object]],
    trace_facts: List[Dict[str, object]],
    service_snapshots: List[Dict[str, object]],
    incident_timeline: List[Dict[str, object]],
    evidence_examples: Dict[str, object],
) -> Dict[str, object]:
    return {
        "observed_metric_count": int(case_summary["metric_count"]),
        "observed_log_template_count": int(case_summary["incident_log_template_count"]),
        "observed_trace_operation_count": int(case_summary["incident_trace_operation_count"]),
        "modality_availability": dict(case_summary["modality_availability"]),
        "data_quality": {
            "nan_points_filled": int(case_summary["nan_points_filled"]),
            "inject_validity": dict(case_summary["inject_validity"]),
            "log_rows_in_window": int(case_summary["log_row_count_in_window"]),
            "trace_rows_in_window": int(case_summary["trace_row_count_in_window"]),
        },
        "metric_facts": metric_facts,
        "log_facts": log_facts,
        "trace_facts": trace_facts,
        "service_snapshots": service_snapshots,
        "incident_timeline": incident_timeline,
        "evidence_examples": evidence_examples,
    }

# SECTION: CASE_PROCESSING
def process_case(
    case_dir: Path,
    baseline_sec: int,
    incident_sec: int,
    top_k_services: int,
    top_k_metrics: int,
    top_k_log_facts: int,
    top_k_trace_facts: int,
    trace_example_count: int,
    require_logs: bool,
    require_traces: bool,
    skip_invalid_inject: bool,
) -> Dict[str, object]:
    parsed = parse_case_id(case_dir.name)
    if parsed is None:
        return {"skip_reason": "case_id_parse_failed"}

    metrics_path = case_dir / "metrics.json"
    inject_path = case_dir / "inject_time.txt"
    logs_path = case_dir / "logs.csv"
    traces_path = case_dir / "traces.csv"

    if not metrics_path.exists() or not inject_path.exists():
        return {"skip_reason": "missing_required_metric_files"}
    if require_logs and not logs_path.exists():
        return {"skip_reason": "missing_logs"}
    if require_traces and not traces_path.exists():
        return {"skip_reason": "missing_traces"}

    metrics = load_metrics(metrics_path)
    if not metrics:
        return {"skip_reason": "empty_metrics"}

    inject_raw = read_inject_time(inject_path)
    if inject_raw is None:
        return {"skip_reason": "invalid_inject_time"}

    case_start = min(series[0][0] for series in metrics.values() if series)
    case_end = max(series[-1][0] for series in metrics.values() if series)

    inject_used = inject_raw
    inject_corrected = False
    inject_correction_reason = ""
    if inject_raw < case_start or inject_raw > case_end:
        inject_corrected = True
        default_offset = DEFAULT_INJECT_OFFSET_SEC[str(parsed["system_code"])]
        candidate = case_start + default_offset
        if case_start <= candidate <= case_end:
            inject_used = candidate
            inject_correction_reason = "out_of_range_raw_inject_time_replaced_with_system_default_offset"
        else:
            inject_used = max(case_start, min(inject_raw, case_end))
            inject_correction_reason = "out_of_range_raw_inject_time_clamped_to_case_bounds"

    log_window = csv_time_window(logs_path, "timestamp", ms=False)
    trace_window = csv_time_window(traces_path, "startTimeMillis", ms=True)
    inject_validity = {
        "metrics": bool(case_start <= inject_used <= case_end),
        "logs": bool(log_window is None or (log_window[0] <= inject_used <= log_window[1])),
        "traces": bool(trace_window is None or (trace_window[0] <= inject_used <= trace_window[1])),
    }

    if skip_invalid_inject and not all(inject_validity.values()):
        return {"skip_reason": "invalid_inject_for_available_modality"}

    metric_features = [
        build_metric_feature(
            metric_name=metric_name,
            series=series,
            inject_time=inject_used,
            baseline_sec=baseline_sec,
            incident_sec=incident_sec,
        )
        for metric_name, series in metrics.items()
    ]
    metric_features.sort(key=lambda row: float(row["score"]), reverse=True)

    metric_service_rows = summarize_service_scores(
        metric_features,
        score_field="score",
        item_field="metric_name",
        top_k_services=top_k_services,
    )

    log_result = build_log_features(
        logs_path=logs_path,
        inject_time=inject_used,
        baseline_sec=baseline_sec,
        incident_sec=incident_sec,
        top_k_facts=top_k_log_facts,
    )
    log_features = log_result["feature_rows"]
    log_service_rows = summarize_service_scores(
        log_features,
        score_field="anomaly_score",
        item_field="template",
        top_k_services=top_k_services,
    )

    trace_result = build_trace_features(
        traces_path=traces_path,
        inject_time=inject_used,
        baseline_sec=baseline_sec,
        incident_sec=incident_sec,
        top_k_facts=top_k_trace_facts,
        trace_example_count=trace_example_count,
    )
    trace_features = trace_result["feature_rows"]
    trace_service_rows = summarize_service_scores(
        trace_features,
        score_field="anomaly_score",
        item_field="operation",
        top_k_services=top_k_services,
    )

    metric_facts = build_neutral_metric_facts(metric_features[:top_k_metrics])
    log_facts = log_result["facts"]
    trace_facts = trace_result["facts"]

    service_snapshots = build_service_snapshots(
        metric_features=metric_features,
        log_features=log_features,
        trace_features=trace_features,
    )
    incident_timeline = build_incident_timeline(
        log_features=log_features,
        trace_features=trace_features,
    )
    evidence_examples = {
        "logs": list(log_result["examples"]),
        "slow_traces": list(trace_result["examples"]),
    }

    root_service = str(parsed["root_cause_service"])
    case_summary: Dict[str, object] = {
        **parsed,
        "inject_time_raw": inject_raw,
        "inject_time_used": inject_used,
        "inject_time_corrected": inject_corrected,
        "inject_correction_reason": inject_correction_reason,
        "case_start_time": case_start,
        "case_end_time": case_end,
        "case_duration_sec": case_end - case_start,
        "modality_availability": {
            "metrics": True,
            "logs": logs_path.exists(),
            "traces": traces_path.exists(),
        },
        "inject_validity": inject_validity,
        "metric_count": len(metrics),
        "log_row_count_in_window": int(log_result["row_count_in_window"]),
        "trace_row_count_in_window": int(trace_result["row_count_in_window"]),
        "incident_log_template_count": int(log_result["template_count_in_incident"]),
        "incident_trace_operation_count": int(trace_result["operation_count_in_incident"]),
        "baseline_window_sec": baseline_sec,
        "incident_window_sec": incident_sec,
        "nan_points_filled": int(sum(int(row["nan_filled"]) for row in metric_features)),
        "root_service_rank_by_metric_anomaly": find_service_rank(metric_service_rows, root_service),
        "root_service_rank_by_log_anomaly": find_service_rank(log_service_rows, root_service),
        "root_service_rank_by_trace_anomaly": find_service_rank(trace_service_rows, root_service),
        "top_metric_services": metric_service_rows,
        "top_log_services": log_service_rows,
        "top_trace_services": trace_service_rows,
        "top_metrics": metric_features[:top_k_metrics],
        "top_logs": log_features[:top_k_log_facts],
        "top_traces": trace_features[:top_k_trace_facts],
    }

    llm_payload = build_llm_input_payload(
        case_summary=case_summary,
        metric_facts=metric_facts,
        log_facts=log_facts,
        trace_facts=trace_facts,
        service_snapshots=service_snapshots,
        incident_timeline=incident_timeline,
        evidence_examples=evidence_examples,
    )

    quality = {
        "case_id": parsed["case_id"],
        "inject_time_corrected": inject_corrected,
        "inject_correction_reason": inject_correction_reason,
        "inject_validity": dict(inject_validity),
        "nan_points_filled": int(case_summary["nan_points_filled"]),
        "log_row_count_in_window": int(case_summary["log_row_count_in_window"]),
        "trace_row_count_in_window": int(case_summary["trace_row_count_in_window"]),
    }

    return {
        "case_summary": case_summary,
        "metric_features": metric_features,
        "log_features": log_features,
        "trace_features": trace_features,
        "llm_payload": llm_payload,
        "quality": quality,
    }


def iter_case_dirs(input_dir: Path) -> Iterable[Path]:
    for child in sorted(input_dir.iterdir()):
        if child.is_dir():
            yield child

# SECTION: MAIN
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True, help="Path to RE2 folder")
    parser.add_argument("--output", type=Path, required=True, help="Output directory")
    parser.add_argument("--baseline-sec", type=int, default=300, help="Baseline window in seconds before injection")
    parser.add_argument("--incident-sec", type=int, default=300, help="Incident window in seconds after injection")
    parser.add_argument("--top-k-services", type=int, default=5, help="Top anomalous services to keep in case summary")
    parser.add_argument("--top-k-metrics", type=int, default=12, help="Top anomalous metrics to keep in LLM input")
    parser.add_argument("--top-k-log-facts", type=int, default=16, help="Top anomalous log templates to keep in LLM input")
    parser.add_argument("--top-k-trace-facts", type=int, default=16, help="Top anomalous trace operations to keep in LLM input")
    parser.add_argument("--trace-example-count", type=int, default=3, help="Number of representative slow traces to keep per case")
    parser.add_argument("--require-logs", action="store_true", help="Skip cases that do not contain logs.csv")
    parser.add_argument("--require-traces", action="store_true", help="Skip cases that do not contain traces.csv")
    parser.add_argument("--skip-invalid-inject", action="store_true", help="Skip cases where the inject time is invalid for an available modality")
    parser.add_argument("--limit", type=int, default=0, help="Optional cap on how many case directories to process (0 = no cap)")
    args = parser.parse_args()

    input_dir: Path = args.input
    output_dir: Path = args.output
    output_dir.mkdir(parents=True, exist_ok=True)

    cases_jsonl_path = output_dir / "cases.jsonl"
    metric_features_csv_path = output_dir / "metric_features.csv"
    log_features_csv_path = output_dir / "log_features.csv"
    trace_features_csv_path = output_dir / "trace_features.csv"
    llm_inputs_dir = output_dir / "llm_inputs"
    report_json_path = output_dir / "report.json"
    llm_inputs_dir.mkdir(parents=True, exist_ok=True)

    case_rows: List[Dict[str, object]] = []
    metric_rows: List[Dict[str, object]] = []
    log_rows: List[Dict[str, object]] = []
    trace_rows: List[Dict[str, object]] = []
    quality_rows: List[Dict[str, object]] = []
    skipped_cases: List[Dict[str, object]] = []

    case_dirs = list(iter_case_dirs(input_dir))
    if args.limit > 0:
        case_dirs = case_dirs[: args.limit]

    for case_dir in case_dirs:
        result = process_case(
            case_dir=case_dir,
            baseline_sec=args.baseline_sec,
            incident_sec=args.incident_sec,
            top_k_services=args.top_k_services,
            top_k_metrics=args.top_k_metrics,
            top_k_log_facts=args.top_k_log_facts,
            top_k_trace_facts=args.top_k_trace_facts,
            trace_example_count=args.trace_example_count,
            require_logs=args.require_logs,
            require_traces=args.require_traces,
            skip_invalid_inject=args.skip_invalid_inject,
        )
        if "skip_reason" in result:
            skipped_cases.append({"case_id": case_dir.name, "reason": result["skip_reason"]})
            continue

        case_summary = result["case_summary"]
        metric_features = result["metric_features"]
        log_features = result["log_features"]
        trace_features = result["trace_features"]
        llm_payload = result["llm_payload"]
        quality = result["quality"]

        case_rows.append(case_summary)
        quality_rows.append(quality)

        llm_input_path = llm_inputs_dir / f"{case_summary['case_id']}.json"
        llm_input_path.write_text(json.dumps(llm_payload, indent=2, ensure_ascii=True), encoding="utf-8")

        for row in metric_features:
            metric_rows.append(
                {
                    "case_id": case_summary["case_id"],
                    "system": case_summary["system"],
                    "fault_type": case_summary["fault_type"],
                    "root_cause_service": case_summary["root_cause_service"],
                    **row,
                }
            )
        for row in log_features:
            log_rows.append(
                {
                    "case_id": case_summary["case_id"],
                    "system": case_summary["system"],
                    "fault_type": case_summary["fault_type"],
                    "root_cause_service": case_summary["root_cause_service"],
                    **row,
                    "representative_messages": " || ".join(row["representative_messages"]),
                    "representative_endpoints": " || ".join(row["representative_endpoints"]),
                }
            )
        for row in trace_features:
            trace_rows.append(
                {
                    "case_id": case_summary["case_id"],
                    "system": case_summary["system"],
                    "fault_type": case_summary["fault_type"],
                    "root_cause_service": case_summary["root_cause_service"],
                    **row,
                }
            )

    with cases_jsonl_path.open("w", encoding="utf-8") as fh:
        for row in case_rows:
            fh.write(json.dumps(row, ensure_ascii=True) + "\n")

    metric_fields = [
        "case_id", "system", "fault_type", "root_cause_service", "metric_name", "service", "metric_type",
        "metric_family", "direction", "score", "z_mean", "z_peak", "delta_mean", "change_ratio",
        "baseline_mean", "baseline_std", "incident_mean", "incident_peak", "incident_trough",
        "first_incident_ts", "first_incident_offset_sec", "peak_offset_sec", "sustained_after_inject",
        "nan_filled", "points",
    ]
    with metric_features_csv_path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=metric_fields)
        writer.writeheader()
        writer.writerows(metric_rows)

    log_fields = [
        "case_id", "system", "fault_type", "root_cause_service", "service", "template", "anomaly_score",
        "baseline_count", "incident_count", "delta_count", "incident_ratio", "new_in_incident",
        "baseline_keyword_hits", "incident_keyword_hits", "baseline_http_4xx_count", "incident_http_4xx_count",
        "baseline_http_5xx_count", "incident_http_5xx_count", "latency_p95_baseline_ms",
        "latency_p95_incident_ms", "latency_delta_ms", "first_incident_ts", "first_incident_offset_sec",
        "severity_hint", "representative_messages", "representative_endpoints",
    ]
    with log_features_csv_path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=log_fields)
        writer.writeheader()
        writer.writerows(log_rows)

    trace_fields = [
        "case_id", "system", "fault_type", "root_cause_service", "service", "operation", "anomaly_score",
        "baseline_count", "incident_count", "baseline_mean_ms", "baseline_p95_ms", "incident_mean_ms",
        "incident_p95_ms", "incident_max_ms", "delta_p95_ms", "latency_ratio", "new_in_incident",
        "first_incident_ts", "first_incident_offset_sec",
    ]
    with trace_features_csv_path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=trace_fields)
        writer.writeheader()
        writer.writerows(trace_rows)

    system_counts = Counter(str(row["system"]) for row in case_rows)
    fault_counts = Counter(str(row["fault_type"]) for row in case_rows)
    corrected_cases = [row for row in quality_rows if row["inject_time_corrected"]]
    nan_cases = [row for row in quality_rows if int(row["nan_points_filled"]) > 0]

    metric_ranks = [int(row["root_service_rank_by_metric_anomaly"]) for row in case_rows if row["root_service_rank_by_metric_anomaly"] is not None]
    log_ranks = [int(row["root_service_rank_by_log_anomaly"]) for row in case_rows if row["root_service_rank_by_log_anomaly"] is not None]
    trace_ranks = [int(row["root_service_rank_by_trace_anomaly"]) for row in case_rows if row["root_service_rank_by_trace_anomaly"] is not None]

    report = {
        "input_dir": str(input_dir),
        "output_dir": str(output_dir),
        "llm_inputs_dir": str(llm_inputs_dir),
        "total_case_dirs_considered": len(case_dirs),
        "processed_cases": len(case_rows),
        "llm_input_files": len(case_rows),
        "skipped_cases": skipped_cases,
        "system_counts": dict(system_counts),
        "fault_counts": dict(fault_counts),
        "metric_feature_rows": len(metric_rows),
        "log_feature_rows": len(log_rows),
        "trace_feature_rows": len(trace_rows),
        "inject_time_corrections": corrected_cases,
        "cases_with_nan_fills": len(nan_cases),
        "inject_validity_summary": {
            "metrics_valid": sum(1 for row in quality_rows if row["inject_validity"]["metrics"]),
            "logs_valid": sum(1 for row in quality_rows if row["inject_validity"]["logs"]),
            "traces_valid": sum(1 for row in quality_rows if row["inject_validity"]["traces"]),
        },
        "root_service_rank_stats": {
            "metrics": {
                "count": len(metric_ranks),
                "top1": sum(1 for value in metric_ranks if value == 1),
                "top3": sum(1 for value in metric_ranks if value <= 3),
                "top1_rate": (sum(1 for value in metric_ranks if value == 1) / len(metric_ranks)) if metric_ranks else 0.0,
                "top3_rate": (sum(1 for value in metric_ranks if value <= 3) / len(metric_ranks)) if metric_ranks else 0.0,
            },
            "logs": {
                "count": len(log_ranks),
                "top1": sum(1 for value in log_ranks if value == 1),
                "top3": sum(1 for value in log_ranks if value <= 3),
                "top1_rate": (sum(1 for value in log_ranks if value == 1) / len(log_ranks)) if log_ranks else 0.0,
                "top3_rate": (sum(1 for value in log_ranks if value <= 3) / len(log_ranks)) if log_ranks else 0.0,
            },
            "traces": {
                "count": len(trace_ranks),
                "top1": sum(1 for value in trace_ranks if value == 1),
                "top3": sum(1 for value in trace_ranks if value <= 3),
                "top1_rate": (sum(1 for value in trace_ranks if value == 1) / len(trace_ranks)) if trace_ranks else 0.0,
                "top3_rate": (sum(1 for value in trace_ranks if value <= 3) / len(trace_ranks)) if trace_ranks else 0.0,
            },
        },
        "parameters": {
            "baseline_sec": args.baseline_sec,
            "incident_sec": args.incident_sec,
            "top_k_services": args.top_k_services,
            "top_k_metrics": args.top_k_metrics,
            "top_k_log_facts": args.top_k_log_facts,
            "top_k_trace_facts": args.top_k_trace_facts,
            "trace_example_count": args.trace_example_count,
            "default_inject_offsets_sec": DEFAULT_INJECT_OFFSET_SEC,
            "require_logs": args.require_logs,
            "require_traces": args.require_traces,
            "skip_invalid_inject": args.skip_invalid_inject,
            "limit": args.limit,
        },
    }

    report_json_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(f"Processed {len(case_rows)} cases")
    print(f"Wrote: {cases_jsonl_path}")
    print(f"Wrote: {metric_features_csv_path}")
    print(f"Wrote: {log_features_csv_path}")
    print(f"Wrote: {trace_features_csv_path}")
    print(f"Wrote: {llm_inputs_dir}")
    print(f"Wrote: {report_json_path}")


if __name__ == "__main__":
    main()
