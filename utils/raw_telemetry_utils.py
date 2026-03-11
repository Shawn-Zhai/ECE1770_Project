from typing import Any, Dict, List, Optional, Tuple
from statistics import mean, pstdev

ALLOWED_FAILURE_TYPES = {"cpu", "mem", "disk", "delay", "loss", "unknown"}


def filter_raw_telemetry_by_service(
    raw_telemetry: Dict[str, Any],
    service: str,
) -> Dict[str, Any]:
    """
    Keep only metrics belonging to a specific service.

    Example
    -------
    service = "adservice"

    input keys:
        adservice_cpu
        adservice_latency
        checkoutservice_cpu

    output:
        {
            "adservice_cpu": [...],
            "adservice_latency": [...],
        }
    """
    if not isinstance(raw_telemetry, dict) or not service:
        return {}

    prefix = f"{service}_"
    filtered: Dict[str, Any] = {}

    for metric_name, series in raw_telemetry.items():
        if not isinstance(metric_name, str):
            continue
        if metric_name.startswith(prefix):
            filtered[metric_name] = series

    return filtered


def summarize_service_state_evidence(
    state_report: Dict[str, Any],
    service: str,
    top_k: int = 8,
) -> List[str]:
    """
    Build service-level evidence summary directly from state_report.metric_facts.
    """
    if not isinstance(state_report, dict) or not service:
        return []

    metric_facts = state_report.get("metric_facts", [])
    if not isinstance(metric_facts, list):
        return []

    service_rows = [
        row for row in metric_facts
        if isinstance(row, dict) and str(row.get("service", "")).strip() == service
    ]

    if not service_rows:
        return []

    scored: List[Dict[str, Any]] = []

    for row in service_rows:
        metric_name = str(row.get("metric_name", "unknown"))

        baseline_mean = _safe_float(row.get("baseline_mean", 0.0))
        incident_mean = _safe_float(row.get("incident_mean", 0.0))
        baseline_std = abs(_safe_float(row.get("baseline_std", 0.0)))
        incident_peak = _safe_float(row.get("incident_peak", incident_mean))

        mean_change = incident_mean - baseline_mean
        change_ratio = abs(mean_change) / (abs(baseline_mean) + 1e-9)
        direction = _direction_from_delta(mean_change)

        anomaly_score = _compute_anomaly_score(
            baseline_mean=baseline_mean,
            baseline_std=baseline_std,
            incident_mean=incident_mean,
            incident_peak=incident_peak,
        )

        scored.append(
            {
                "metric_name": metric_name,
                "direction": direction,
                "anomaly_score": anomaly_score,
                "change_ratio": change_ratio,
                "baseline_mean": baseline_mean,
                "incident_mean": incident_mean,
                "incident_peak": incident_peak,
            }
        )

    scored.sort(key=lambda x: x["anomaly_score"], reverse=True)

    evidence: List[str] = []
    for row in scored[:max(1, top_k)]:
        evidence.append(
            f"{row['metric_name']} changed {row['direction']} "
            f"(anomaly={row['anomaly_score']:.2f}, ratio={row['change_ratio']:.2f})"
        )

    return evidence


def build_metric_fact_lookup(
    state_report: Dict[str, Any],
    service: Optional[str] = None,
) -> Dict[str, Dict[str, Any]]:
    """
    Build lookup:
        metric_name -> metric fact summary from state_report

    Example output
    --------------
    {
        "adservice_cpu": {
            "service": "adservice",
            "metric_type": "cpu",
            "baseline_mean": 1.67,
            "baseline_std": 0.13,
            "incident_mean": 87.84,
            "incident_peak": 100.01,
            "sample_count": 4201,
        }
    }
    """
    lookup: Dict[str, Dict[str, Any]] = {}

    if not isinstance(state_report, dict):
        return lookup

    metric_facts = state_report.get("metric_facts", [])
    if not isinstance(metric_facts, list):
        return lookup

    for row in metric_facts:
        if not isinstance(row, dict):
            continue

        metric_name = str(row.get("metric_name", "")).strip()
        if not metric_name:
            continue

        row_service = str(row.get("service", "unknown")).strip()
        if service is not None and row_service != service:
            continue

        lookup[metric_name] = {
            "service": row_service,
            "metric_type": str(row.get("metric_type", "unknown")).strip().lower(),
            "baseline_mean": _safe_float(row.get("baseline_mean", 0.0)),
            "baseline_std": abs(_safe_float(row.get("baseline_std", 0.0))),
            "incident_mean": _safe_float(row.get("incident_mean", 0.0)),
            "incident_peak": _safe_float(row.get("incident_peak", 0.0)),
            "sample_count": int(_safe_float(row.get("sample_count", 0))),
        }

    return lookup


def compress_filtered_raw_telemetry(
    filtered_raw_telemetry: Dict[str, List[List[float]]],
    metric_fact_lookup: Optional[Dict[str, Dict[str, Any]]] = None,
    max_metrics: int = 6,
    max_points_per_metric: int = 12,
    z_threshold: float = 3.0,
    min_consecutive_points: int = 10,
    smooth_window: int = 3,
) -> Dict[str, Any]:
    """
    Compress filtered raw telemetry into temporal evidence for failure-type diagnosis.

    Key idea
    --------
    - baseline_mean / baseline_std come from state_report via metric_fact_lookup
    - raw telemetry provides timestamps and local evidence snippet
    - we find first sustained anomaly timestamp from the full raw series
    """
    snippet_before = max(2, min(4, max_points_per_metric // 3))
    snippet_after = max(3, max_points_per_metric - snippet_before - 1)

    meta = {
        "max_metrics": max_metrics,
        "max_points_per_metric": max_points_per_metric,
        "z_threshold": z_threshold,
        "min_consecutive_points": min_consecutive_points,
        "smooth_window": smooth_window,
        "snippet_before": snippet_before,
        "snippet_after": snippet_after,
        "baseline_source": "state_report",
    }

    if not isinstance(filtered_raw_telemetry, dict) or not filtered_raw_telemetry:
        return _insufficient_signal_response(
            message="No filtered raw telemetry available.",
            meta=meta,
        )

    if not isinstance(metric_fact_lookup, dict) or not metric_fact_lookup:
        return _insufficient_signal_response(
            message="No state_report metric lookup available for compression.",
            meta=meta,
        )

    metric_events: List[Dict[str, Any]] = []

    for metric_name, series in filtered_raw_telemetry.items():
        metric_name = str(metric_name).strip()
        if not metric_name:
            continue
        
        for lookup_metric_name, lookup_metric in metric_fact_lookup.items():
            if metric_name in lookup_metric_name:
                
            else:
                continue
            
            
        fact = metric_fact_lookup.get(metric_name)
        if not isinstance(fact, dict):
            continue

        extracted = _extract_series(series)
        if extracted is None:
            continue

        timestamps, raw_values = extracted
        if len(raw_values) < max(5, min_consecutive_points + 2):
            continue

        values = _moving_average(raw_values, smooth_window)

        baseline_mean = _safe_float(fact.get("baseline_mean", 0.0))
        baseline_std = abs(_safe_float(fact.get("baseline_std", 0.0)))
        if baseline_std < 1e-8:
            baseline_std = 1.0

        onset_idx = _find_first_sustained_anomaly_index(
            values=values,
            baseline_mean=baseline_mean,
            baseline_std=baseline_std,
            z_threshold=z_threshold,
            min_consecutive_points=min_consecutive_points,
        )
        if onset_idx is None:
            continue

        onset_ts = timestamps[onset_idx]
        onset_value = values[onset_idx]
        onset_z = (onset_value - baseline_mean) / baseline_std
        direction = _direction_from_value(onset_value, baseline_mean)

        peak_ts, peak_z = _peak_info(
            timestamps=timestamps[onset_idx:],
            values=values[onset_idx:],
            baseline_mean=baseline_mean,
            baseline_std=baseline_std,
        )

        evidence_points = _build_evidence_points(
            timestamps=timestamps,
            values=values,
            center_idx=onset_idx,
            snippet_before=snippet_before,
            snippet_after=snippet_after,
        )

        metric_events.append(
            {
                "metric": metric_name,
                "service": str(fact.get("service", "unknown")),
                "metric_type": str(fact.get("metric_type", "unknown")).lower(),
                "first_anomaly_ts": int(onset_ts),
                "direction": direction,
                "baseline_mean": round(float(baseline_mean), 6),
                "baseline_std": round(float(baseline_std), 6),
                "incident_mean_from_state_report": round(
                    float(_safe_float(fact.get("incident_mean", 0.0))), 6
                ),
                "incident_peak_from_state_report": round(
                    float(_safe_float(fact.get("incident_peak", 0.0))), 6
                ),
                "onset_zscore": round(float(onset_z), 4),
                "peak_ts": int(peak_ts),
                "peak_zscore": round(float(peak_z), 4),
                "evidence_points": evidence_points,
            }
        )

    if not metric_events:
        return _insufficient_signal_response(
            message="No sustained anomaly onset detected using state_report baseline.",
            meta=meta,
        )

    metric_events.sort(
        key=lambda x: (x["first_anomaly_ts"], -abs(x["onset_zscore"]))
    )
    metric_events = metric_events[:max_metrics]

    timeline_summary = [
        (
            f'{row["first_anomaly_ts"]} | {row["metric"]} '
            f'{row["direction"]} (type={row["metric_type"]}, onset_z={row["onset_zscore"]})'
        )
        for row in metric_events
    ]

    meta["metrics_detected"] = len(metric_events)

    return {
        "status": "ok",
        "metric_order_by_first_anomaly": metric_events,
        "timeline_summary": timeline_summary,
        "instruction_hint": (
            "Anomaly timestamps are determined from raw telemetry, while "
            "baseline statistics are taken from state_report."
        ),
        "meta": meta,
    }


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None:
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def _direction_from_delta(delta: float) -> str:
    if delta > 0:
        return "up"
    if delta < 0:
        return "down"
    return "flat"


def _direction_from_value(value: float, baseline_mean: float) -> str:
    if value > baseline_mean:
        return "up"
    if value < baseline_mean:
        return "down"
    return "flat"


def _compute_anomaly_score(
    baseline_mean: float,
    baseline_std: float,
    incident_mean: float,
    incident_peak: float,
) -> float:
    mean_change = incident_mean - baseline_mean

    if baseline_std < 1e-9:
        if abs(mean_change) < 1e-9:
            return 0.0
        return min(abs(mean_change), 25.0)

    z_mean = mean_change / baseline_std
    z_peak = abs(incident_peak - baseline_mean) / baseline_std
    return max(abs(z_mean), abs(z_peak))


def _moving_average(values: List[float], window: int = 3) -> List[float]:
    """
    Efficient causal moving average.
    """
    if window <= 1 or len(values) <= 1:
        return values[:]

    result: List[float] = []
    running_sum = 0.0

    for i, v in enumerate(values):
        running_sum += v

        if i >= window:
            running_sum -= values[i - window]

        denom = min(i + 1, window)
        result.append(running_sum / denom)

    return result


def _find_first_sustained_anomaly_index(
    values: List[float],
    baseline_mean: float,
    baseline_std: float,
    z_threshold: float,
    min_consecutive_points: int,
) -> Optional[int]:
    """
    Find the first local index where anomaly persists for K consecutive points.
    Baseline comes from state_report statistics.
    """
    if not values or min_consecutive_points <= 0:
        return None

    if baseline_std < 1e-8:
        baseline_std = 1.0

    run = 0
    start_idx: Optional[int] = None
    inv_std = 1.0 / baseline_std

    for i, v in enumerate(values):
        z = (v - baseline_mean) * inv_std

        if abs(z) >= z_threshold:
            if run == 0:
                start_idx = i
            run += 1

            if run >= min_consecutive_points:
                return start_idx
        else:
            run = 0
            start_idx = None

    return None


def _peak_info(
    timestamps: List[int],
    values: List[float],
    baseline_mean: float,
    baseline_std: float,
) -> Tuple[int, float]:
    """
    Return (peak_ts, peak_zscore) using max absolute z-score.
    """
    if not timestamps or not values or len(timestamps) != len(values):
        return 0, 0.0

    if baseline_std < 1e-8:
        baseline_std = 1.0

    peak_idx = 0
    peak_abs_z = -1.0
    peak_z = 0.0
    inv_std = 1.0 / baseline_std

    for i, v in enumerate(values):
        z = (v - baseline_mean) * inv_std
        if abs(z) > peak_abs_z:
            peak_abs_z = abs(z)
            peak_idx = i
            peak_z = z

    return int(timestamps[peak_idx]), float(peak_z)


def _build_evidence_points(
    timestamps: List[int],
    values: List[float],
    center_idx: int,
    snippet_before: int,
    snippet_after: int,
) -> List[List[float]]:
    """
    Keep only a small local snippet around anomaly onset.
    """
    if not timestamps or not values or len(timestamps) != len(values):
        return []

    start = max(0, center_idx - snippet_before)
    end = min(len(values), center_idx + snippet_after + 1)

    return [
        [int(timestamps[i]), round(float(values[i]), 6)]
        for i in range(start, end)
    ]


def _extract_series(
    series: Any,
) -> Optional[Tuple[List[int], List[float]]]:
    """
    Validate and normalize a series of format:
        [[timestamp, value], [timestamp, value], ...]
    """
    if not isinstance(series, list) or not series:
        return None

    cleaned: List[Tuple[int, float]] = []

    for point in series:
        if not isinstance(point, (list, tuple)) or len(point) < 2:
            continue

        try:
            ts = int(point[0])
            value = float(point[1])
            cleaned.append((ts, value))
        except (TypeError, ValueError):
            continue

    if not cleaned:
        return None

    cleaned.sort(key=lambda x: x[0])
    timestamps = [x[0] for x in cleaned]
    values = [x[1] for x in cleaned]
    return timestamps, values


def _insufficient_signal_response(
    message: str,
    meta: Dict[str, Any],
) -> Dict[str, Any]:
    return {
        "status": "insufficient_signal",
        "metric_order_by_first_anomaly": [],
        "timeline_summary": [],
        "instruction_hint": message,
        "meta": meta,
    }