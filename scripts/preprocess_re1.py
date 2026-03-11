#!/usr/bin/env python3
"""Preprocess RCAEval RE1 metric cases into evaluation and LLM-ready artifacts.

Outputs:
1) cases.jsonl              One compact summary per failure case (includes labels for eval).
2) metric_features.csv      Per-metric anomaly features per case.
3) llm_inputs/<case>.json   One leakage-safe, neutral fact file per case for diagnostic LLM.
4) report.json              Dataset-level quality and preprocessing report.
"""

from __future__ import annotations

import argparse
import bisect
import csv
import json
import math
import re
import statistics
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

CASE_RE = re.compile(
    r"^re1(?P<system>ob|ss|tt)_(?P<service>.+)_(?P<fault>cpu|mem|disk|delay|loss)_(?P<repeat>\d+)$"
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


def _to_int(value: object) -> Optional[int]:
    try:
        return int(value)
    except Exception:
        return None


def _to_float(value: object) -> float:
    try:
        f = float(value)
        return f
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


def _split_metric_name(metric_name: str) -> Tuple[str, str]:
    if "_" not in metric_name:
        return metric_name, "unknown"
    service, metric_type = metric_name.rsplit("_", 1)
    return service, metric_type


def parse_case_id(case_id: str) -> Optional[Dict[str, object]]:
    match = CASE_RE.match(case_id)
    if not match:
        return None
    groups = match.groupdict()
    return {
        "case_id": case_id,
        "suite": "re1",
        "system_code": groups["system"],
        "system": SYSTEM_NAME[groups["system"]],
        "root_cause_service": groups["service"],
        "fault_type": groups["fault"],
        "repeat_id": int(groups["repeat"]),
        # Keep a derived indicator string for downstream prompts/eval.
        "root_cause_indicator": f'{groups["service"]}_{groups["fault"]}',
    }


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

        # Deduplicate timestamp collisions by keeping the latest value.
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
    nan_positions = [idx for idx, v in enumerate(out) if math.isnan(v)]
    if not nan_positions:
        return out, 0

    # Forward fill.
    for idx in range(1, len(out)):
        if math.isnan(out[idx]) and not math.isnan(out[idx - 1]):
            out[idx] = out[idx - 1]

    # Backward fill.
    for idx in range(len(out) - 2, -1, -1):
        if math.isnan(out[idx]) and not math.isnan(out[idx + 1]):
            out[idx] = out[idx + 1]

    # If still NaN, the entire series was NaN. Fill with zero.
    for idx, value in enumerate(out):
        if math.isnan(value):
            out[idx] = 0.0

    return out, len(nan_positions)


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

    # Fallbacks when the requested windows are out of bounds.
    if not baseline:
        fallback = min(120, b_end)
        baseline = values[max(0, b_end - fallback) : b_end] if fallback > 0 else values[: min(120, len(values))]
    if not incident:
        if i_start < len(values):
            incident = values[i_start : min(len(values), i_start + max(60, incident_sec // 2))]
        if not incident:
            incident = values[max(0, len(values) - 60) :]

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

    service, metric_type = _split_metric_name(metric_name)
    return {
        "metric_name": metric_name,
        "service": service,
        "metric_type": metric_type,
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
        if abs(baseline_mean) > EPS:
            percent_change = (mean_change / baseline_mean) * 100.0
        else:
            percent_change = None

        facts.append(
            {
                "service": row["service"],
                "metric_name": row["metric_name"],
                "metric_type": row["metric_type"],
                "baseline_mean": baseline_mean,
                "incident_mean": incident_mean,
                "mean_change": mean_change,
                "absolute_change": absolute_change,
                "percent_change": percent_change,
                "baseline_std": float(row["baseline_std"]),
                "incident_peak": float(row["incident_peak"]),
                "sample_count": int(row["points"]),
            }
        )
    facts.sort(key=lambda item: (str(item["service"]), str(item["metric_name"])))
    return facts


def build_llm_input_payload(case_summary: Dict[str, object], metric_facts: List[Dict[str, object]]) -> Dict[str, object]:
    return {
        "observed_metric_count": int(case_summary["metric_count"]),
        "data_quality": {
            "nan_points_filled": int(case_summary["nan_points_filled"]),
        },
        "metric_facts": metric_facts,
    }


def process_case(
    case_dir: Path,
    baseline_sec: int,
    incident_sec: int,
    top_k_services: int,
    top_k_metrics: int,
) -> Optional[Tuple[Dict[str, object], List[Dict[str, object]], Dict[str, object]]]:
    parsed = parse_case_id(case_dir.name)
    if parsed is None:
        return None

    metrics_path = case_dir / "metrics.json"
    inject_path = case_dir / "inject_time.txt"
    if not metrics_path.exists() or not inject_path.exists():
        return None

    metrics = load_metrics(metrics_path)
    if not metrics:
        return None

    inject_raw = read_inject_time(inject_path)
    if inject_raw is None:
        return None

    case_start = min(series[0][0] for series in metrics.values() if series)
    case_end = max(series[-1][0] for series in metrics.values() if series)

    inject_used = inject_raw
    inject_corrected = False
    inject_correction_reason = ""
    if inject_raw < case_start or inject_raw > case_end:
        inject_corrected = True
        default_offset = DEFAULT_INJECT_OFFSET_SEC[parsed["system_code"]]
        candidate = case_start + default_offset
        if case_start <= candidate <= case_end:
            inject_used = candidate
            inject_correction_reason = "out_of_range_raw_inject_time_replaced_with_system_default_offset"
        else:
            inject_used = max(case_start, min(inject_raw, case_end))
            inject_correction_reason = "out_of_range_raw_inject_time_clamped_to_case_bounds"

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

    service_to_metrics: Dict[str, List[Dict[str, object]]] = defaultdict(list)
    for row in metric_features:
        service_to_metrics[str(row["service"])].append(row)

    service_scores: List[Dict[str, object]] = []
    for service, rows in service_to_metrics.items():
        scores = sorted((float(r["score"]) for r in rows), reverse=True)
        max_score = scores[0]
        avg_top3 = _mean(scores[:3])
        blended_score = 0.7 * max_score + 0.3 * avg_top3
        top_service_metrics = rows[:3]
        service_scores.append(
            {
                "service": service,
                "score": blended_score,
                "max_metric_score": max_score,
                "top_metrics": [r["metric_name"] for r in top_service_metrics],
            }
        )
    service_scores.sort(key=lambda row: float(row["score"]), reverse=True)

    top_services = service_scores[:top_k_services]
    top_metrics = metric_features[:top_k_metrics]

    root_service = str(parsed["root_cause_service"])
    root_rank = None
    for idx, row in enumerate(service_scores, start=1):
        if row["service"] == root_service:
            root_rank = idx
            break

    case_summary: Dict[str, object] = {
        **parsed,
        "inject_time_raw": inject_raw,
        "inject_time_used": inject_used,
        "inject_time_corrected": inject_corrected,
        "inject_correction_reason": inject_correction_reason,
        "case_start_time": case_start,
        "case_end_time": case_end,
        "case_duration_sec": case_end - case_start,
        "metric_count": len(metrics),
        "baseline_window_sec": baseline_sec,
        "incident_window_sec": incident_sec,
        "nan_points_filled": int(sum(int(row["nan_filled"]) for row in metric_features)),
        "root_service_rank_by_anomaly": root_rank,
        "top_services": top_services,
        "top_metrics": top_metrics,
    }

    case_quality = {
        "case_id": parsed["case_id"],
        "inject_time_corrected": inject_corrected,
        "inject_correction_reason": inject_correction_reason,
        "nan_points_filled": int(case_summary["nan_points_filled"]),
    }
    return case_summary, metric_features, case_quality


def iter_case_dirs(input_dir: Path) -> Iterable[Path]:
    for child in sorted(input_dir.iterdir()):
        if child.is_dir():
            yield child


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True, help="Path to RE1 folder")
    parser.add_argument("--output", type=Path, required=True, help="Output directory")
    parser.add_argument("--baseline-sec", type=int, default=300, help="Baseline window in seconds before injection")
    parser.add_argument("--incident-sec", type=int, default=300, help="Incident window in seconds after injection")
    parser.add_argument("--top-k-services", type=int, default=5, help="Top anomalous services to keep in case summary")
    parser.add_argument("--top-k-metrics", type=int, default=12, help="Top anomalous metrics to keep in case summary")
    args = parser.parse_args()

    input_dir: Path = args.input
    output_dir: Path = args.output
    output_dir.mkdir(parents=True, exist_ok=True)

    cases_jsonl_path = output_dir / "cases.jsonl"
    metric_features_csv_path = output_dir / "metric_features.csv"
    llm_inputs_dir = output_dir / "llm_inputs"
    report_json_path = output_dir / "report.json"
    llm_inputs_dir.mkdir(parents=True, exist_ok=True)

    case_rows: List[Dict[str, object]] = []
    metric_rows: List[Dict[str, object]] = []
    quality_rows: List[Dict[str, object]] = []
    skipped_cases: List[str] = []

    for case_dir in iter_case_dirs(input_dir):
        result = process_case(
            case_dir=case_dir,
            baseline_sec=args.baseline_sec,
            incident_sec=args.incident_sec,
            top_k_services=args.top_k_services,
            top_k_metrics=args.top_k_metrics,
        )
        if result is None:
            skipped_cases.append(case_dir.name)
            continue
        case_summary, metric_features, case_quality = result
        case_rows.append(case_summary)
        quality_rows.append(case_quality)

        metric_facts = build_neutral_metric_facts(metric_features)
        llm_payload = build_llm_input_payload(case_summary, metric_facts)
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

    with cases_jsonl_path.open("w", encoding="utf-8") as fh:
        for row in case_rows:
            fh.write(json.dumps(row, ensure_ascii=True) + "\n")

    metric_fields = [
        "case_id",
        "system",
        "fault_type",
        "root_cause_service",
        "metric_name",
        "service",
        "metric_type",
        "score",
        "z_mean",
        "z_peak",
        "delta_mean",
        "change_ratio",
        "baseline_mean",
        "baseline_std",
        "incident_mean",
        "incident_peak",
        "incident_trough",
        "nan_filled",
        "points",
    ]
    with metric_features_csv_path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=metric_fields)
        writer.writeheader()
        writer.writerows(metric_rows)

    system_counts = Counter(str(row["system"]) for row in case_rows)
    fault_counts = Counter(str(row["fault_type"]) for row in case_rows)
    corrected_cases = [row for row in quality_rows if row["inject_time_corrected"]]
    nan_cases = [row for row in quality_rows if int(row["nan_points_filled"]) > 0]

    valid_ranks = [int(row["root_service_rank_by_anomaly"]) for row in case_rows if row["root_service_rank_by_anomaly"] is not None]
    top1 = sum(1 for r in valid_ranks if r == 1)
    top3 = sum(1 for r in valid_ranks if r <= 3)

    report = {
        "input_dir": str(input_dir),
        "output_dir": str(output_dir),
        "llm_inputs_dir": str(llm_inputs_dir),
        "total_case_dirs": len(list(iter_case_dirs(input_dir))),
        "processed_cases": len(case_rows),
        "llm_input_files": len(case_rows),
        "skipped_cases": skipped_cases,
        "system_counts": dict(system_counts),
        "fault_counts": dict(fault_counts),
        "metric_feature_rows": len(metric_rows),
        "inject_time_corrections": corrected_cases,
        "cases_with_nan_fills": len(nan_cases),
        "root_service_rank_stats": {
            "count": len(valid_ranks),
            "top1": top1,
            "top3": top3,
            "top1_rate": (top1 / len(valid_ranks)) if valid_ranks else 0.0,
            "top3_rate": (top3 / len(valid_ranks)) if valid_ranks else 0.0,
        },
        "parameters": {
            "baseline_sec": args.baseline_sec,
            "incident_sec": args.incident_sec,
            "top_k_services": args.top_k_services,
            "top_k_metrics": args.top_k_metrics,
            "default_inject_offsets_sec": DEFAULT_INJECT_OFFSET_SEC,
        },
    }

    report_json_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(f"Processed {len(case_rows)} cases")
    print(f"Wrote: {cases_jsonl_path}")
    print(f"Wrote: {metric_features_csv_path}")
    print(f"Wrote: {llm_inputs_dir}")
    print(f"Wrote: {report_json_path}")


if __name__ == "__main__":
    main()
