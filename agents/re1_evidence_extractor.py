import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


# =========================================================
# Root paths
# =========================================================
LLM_INPUT_DIR = Path(
    r"C:\Users\dd262\Documents\MASTER\ECE1770 RCA\project\ECE1770_Project\dataset\artifacts\re1\llm_inputs"
)

RCA_EVAL_RE1_DIR = Path(
    r"C:\Users\dd262\Documents\MASTER\ECE1770 RCA\project\ECE1770_Project\dataset\RCAEval\RE1"
)

OUTPUT_DIR = Path(
    r"C:\Users\dd262\Documents\MASTER\ECE1770 RCA\project\ECE1770_Project\dataset\artifacts\re1\llm_inputs_with_evidence"
)

# =========================================================
# Detection config
# =========================================================
Z_THRESHOLD = 3.0
MIN_CONSECUTIVE = 10
RECOVERY_CONSECUTIVE = 5
EPS = 1e-8

# =========================================================
# Injection-time-guided search window
# You can tune these two numbers.
# Example:
#   search from [inject_time - 60 sec, inject_time + 600 sec]
# =========================================================
INJECTION_LOOKBACK_SEC = 60
INJECTION_LOOKAHEAD_SEC = 600


# =========================================================
# Basic IO
# =========================================================
def load_json(path: Path) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def load_injection_time(path: Path) -> Optional[int]:
    """
    Read injection timestamp from inject_time.txt.
    Accepts int / float-like text and converts to int timestamp.
    """
    if not path.exists():
        return None

    raw = path.read_text(encoding="utf-8").strip()
    if not raw:
        return None

    try:
        return int(float(raw))
    except Exception:
        return None


# =========================================================
# Direction selection from state report
# =========================================================
def get_detection_mode_from_state_report(fact: Dict[str, Any], eps: float = EPS) -> str:
    """
    Decide direction based on state report summary.

    Priority:
    1. mean_change
    2. incident_mean - baseline_mean

    Returns:
        "increase", "decrease", or "both"
    """
    mean_change = fact.get("mean_change")

    if mean_change is not None:
        delta = float(mean_change)
    else:
        baseline_mean = fact.get("baseline_mean")
        incident_mean = fact.get("incident_mean")

        if baseline_mean is None or incident_mean is None:
            return "both"

        delta = float(incident_mean) - float(baseline_mean)

    if abs(delta) < eps:
        return "both"
    elif delta > 0:
        return "increase"
    else:
        return "decrease"


# =========================================================
# Helpers
# =========================================================
def infer_pattern(duration_sec: int, time_to_extreme_sec: Optional[int]) -> str:
    if duration_sec <= 0:
        return "flat"
    if duration_sec < 10:
        return "spike"
    if time_to_extreme_sec is None:
        return "sustained"
    if time_to_extreme_sec > duration_sec * 0.5:
        return "gradual"
    return "sustained"


def find_sustained_onset(mask: List[bool], min_consecutive: int) -> Optional[int]:
    streak = 0
    for i, flag in enumerate(mask):
        if flag:
            streak += 1
            if streak >= min_consecutive:
                return i - min_consecutive + 1
        else:
            streak = 0
    return None


def find_window_indices(
    timestamps: List[int],
    start_ts: Optional[int],
    end_ts: Optional[int],
) -> Tuple[int, int]:
    """
    Return inclusive [start_idx, end_idx] for timestamps within [start_ts, end_ts].
    If no timestamp falls inside, return closest valid reduced window if possible.
    """
    valid_indices = []
    for i, ts in enumerate(timestamps):
        if (start_ts is None or ts >= start_ts) and (end_ts is None or ts <= end_ts):
            valid_indices.append(i)

    if valid_indices:
        return valid_indices[0], valid_indices[-1]

    # fallback: whole series
    return 0, len(timestamps) - 1


# =========================================================
# Core anomaly extraction
# Summary only for the FIRST anomaly segment in the chosen direction
# =========================================================
def extract_anomaly_summary(
    timeseries: List[List[float]],
    baseline_mean: float,
    baseline_std: float,
    detection_mode: str,
    injection_ts: Optional[int] = None,
    search_start_ts: Optional[int] = None,
    search_end_ts: Optional[int] = None,
    z_threshold: float = Z_THRESHOLD,
    min_consecutive: int = MIN_CONSECUTIVE,
    recovery_consecutive: int = RECOVERY_CONSECUTIVE,
) -> Dict[str, Any]:
    """
    detection_mode:
        - "increase"
        - "decrease"
        - "both"

    If "both", the earliest sustained anomaly wins.
    If "increase"/"decrease", only that direction is checked.

    Search is restricted to [search_start_ts, search_end_ts] if provided.
    If no valid points fall in that window, falls back to the full series.

    Output summarizes ONLY the FIRST anomaly segment.
    """

    if not timeseries:
        return {
            "anomaly_detected": False,
            "reason": "empty_timeseries",
            "detection_mode": detection_mode,
            "injection_ts": injection_ts,
            "search_window_start_ts": search_start_ts,
            "search_window_end_ts": search_end_ts,
        }

    timestamps = [int(p[0]) for p in timeseries]
    values = [float(p[1]) for p in timeseries]

    global_peak_idx = max(range(len(values)), key=lambda i: values[i])
    global_trough_idx = min(range(len(values)), key=lambda i: values[i])

    if baseline_std is None or abs(baseline_std) < EPS:
        return {
            "anomaly_detected": False,
            "reason": "baseline_std_zero_or_too_small",
            "detection_mode": detection_mode,
            "direction": None,
            "injection_ts": injection_ts,
            "search_window_start_ts": search_start_ts,
            "search_window_end_ts": search_end_ts,
            "used_windowed_search": search_start_ts is not None or search_end_ts is not None,
            "global_peak_ts": timestamps[global_peak_idx],
            "global_peak_value": values[global_peak_idx],
            "global_trough_ts": timestamps[global_trough_idx],
            "global_trough_value": values[global_trough_idx],
            "anomaly_start_ts": None,
            "anomaly_start_offset_sec": None,
            "anomaly_vs_injection_sec": None,
            "anomaly_end_ts": None,
            "anomaly_recovery_ts": None,
            "anomaly_duration_sec": None,
            "anomaly_extreme_ts": None,
            "anomaly_extreme_value": None,
            "anomaly_zscore": None,
            "anomaly_abs_zscore": None,
            "anomaly_time_to_extreme_sec": None,
            "pattern": "flat",
        }

    zscores = [(v - baseline_mean) / (baseline_std + EPS) for v in values]

    # Restrict search region
    window_start_idx, window_end_idx = find_window_indices(
        timestamps=timestamps,
        start_ts=search_start_ts,
        end_ts=search_end_ts,
    )

    used_windowed_search = search_start_ts is not None or search_end_ts is not None

    # Create masks on full series, but only allow detection within window
    pos_mask = [False] * len(zscores)
    neg_mask = [False] * len(zscores)

    for i in range(window_start_idx, window_end_idx + 1):
        z = zscores[i]
        if detection_mode == "increase":
            pos_mask[i] = z >= z_threshold
        elif detection_mode == "decrease":
            neg_mask[i] = z <= -z_threshold
        else:  # both
            pos_mask[i] = z >= z_threshold
            neg_mask[i] = z <= -z_threshold

    pos_onset = find_sustained_onset(pos_mask[window_start_idx: window_end_idx + 1], min_consecutive)
    neg_onset = find_sustained_onset(neg_mask[window_start_idx: window_end_idx + 1], min_consecutive)

    if pos_onset is not None:
        pos_onset += window_start_idx
    if neg_onset is not None:
        neg_onset += window_start_idx

    max_abs_idx = max(range(len(zscores)), key=lambda i: abs(zscores[i]))

    if pos_onset is None and neg_onset is None:
        return {
            "anomaly_detected": False,
            "reason": "no_sustained_anomaly_for_selected_direction_in_window",
            "detection_mode": detection_mode,
            "direction": None,
            "injection_ts": injection_ts,
            "search_window_start_ts": search_start_ts,
            "search_window_end_ts": search_end_ts,
            "used_windowed_search": used_windowed_search,
            "window_start_idx": window_start_idx,
            "window_end_idx": window_end_idx,
            "window_start_ts_actual": timestamps[window_start_idx],
            "window_end_ts_actual": timestamps[window_end_idx],
            "global_peak_ts": timestamps[global_peak_idx],
            "global_peak_value": values[global_peak_idx],
            "global_trough_ts": timestamps[global_trough_idx],
            "global_trough_value": values[global_trough_idx],
            "strongest_global_ts": timestamps[max_abs_idx],
            "strongest_global_value": values[max_abs_idx],
            "strongest_global_zscore": zscores[max_abs_idx],
            "strongest_global_abs_zscore": abs(zscores[max_abs_idx]),
            "first_anomaly_start_ts": None,
            "first_anomaly_start_offset_sec": None,
            "first_anomaly_vs_injection_sec": None,
            "first_anomaly_end_ts": None,
            "first_anomaly_recovery_ts": None,
            "first_anomaly_duration_sec": None,
            "first_anomaly_extreme_ts": None,
            "first_anomaly_extreme_value": None,
            "first_anomaly_zscore": None,
            "first_anomaly_abs_zscore": None,
            "first_anomaly_time_to_extreme_sec": None,
            "pattern": "flat",
        }

    candidates: List[Tuple[str, int]] = []
    if pos_onset is not None:
        candidates.append(("increase", pos_onset))
    if neg_onset is not None:
        candidates.append(("decrease", neg_onset))

    if len(candidates) == 1:
        direction, onset_idx = candidates[0]
    else:
        (d1, o1), (d2, o2) = candidates
        if o1 < o2:
            direction, onset_idx = d1, o1
        elif o2 < o1:
            direction, onset_idx = d2, o2
        else:
            if abs(zscores[o1]) >= abs(zscores[o2]):
                direction, onset_idx = d1, o1
            else:
                direction, onset_idx = d2, o2

    active_mask = pos_mask if direction == "increase" else neg_mask

    recovery_idx = None
    normal_streak = 0
    for i in range(onset_idx, len(active_mask)):
        if not active_mask[i]:
            normal_streak += 1
            if normal_streak >= recovery_consecutive:
                recovery_idx = i - recovery_consecutive + 1
                break
        else:
            normal_streak = 0

    if recovery_idx is None:
        end_idx = len(values) - 1
        recovery_ts = None
    else:
        end_idx = max(onset_idx, recovery_idx - 1)
        recovery_ts = timestamps[recovery_idx]

    # summarize only the FIRST anomaly segment
    if direction == "increase":
        extreme_idx = max(range(onset_idx, end_idx + 1), key=lambda i: zscores[i])
    else:
        extreme_idx = min(range(onset_idx, end_idx + 1), key=lambda i: zscores[i])

    anomaly_start_ts = timestamps[onset_idx]
    anomaly_end_ts = timestamps[end_idx]
    duration_sec = max(0, anomaly_end_ts - anomaly_start_ts)

    extreme_ts = timestamps[extreme_idx]
    extreme_value = values[extreme_idx]
    extreme_zscore = zscores[extreme_idx]
    time_to_extreme_sec = extreme_ts - anomaly_start_ts if extreme_idx >= onset_idx else None

    if injection_ts is not None:
        first_anomaly_offset_sec = anomaly_start_ts - injection_ts
    else:
        first_anomaly_offset_sec = None

    pattern = infer_pattern(duration_sec, time_to_extreme_sec)

    return {
        "anomaly_detected": True,
        "detection_mode": detection_mode,
        "direction": direction,

        "injection_ts": injection_ts,
        "search_window_start_ts": search_start_ts,
        "search_window_end_ts": search_end_ts,
        "used_windowed_search": used_windowed_search,
        "window_start_idx": window_start_idx,
        "window_end_idx": window_end_idx,
        "window_start_ts_actual": timestamps[window_start_idx],
        "window_end_ts_actual": timestamps[window_end_idx],

        # global info from the whole series
        "global_peak_ts": timestamps[global_peak_idx],
        "global_peak_value": values[global_peak_idx],
        "global_trough_ts": timestamps[global_trough_idx],
        "global_trough_value": values[global_trough_idx],

        # first anomaly segment only
        "first_anomaly_start_ts": anomaly_start_ts,
        "first_anomaly_start_offset_sec": first_anomaly_offset_sec,
        "first_anomaly_vs_injection_sec": first_anomaly_offset_sec,
        "first_anomaly_end_ts": anomaly_end_ts,
        "first_anomaly_recovery_ts": recovery_ts,
        "first_anomaly_duration_sec": duration_sec,
        "first_anomaly_extreme_ts": extreme_ts,
        "first_anomaly_extreme_value": extreme_value,
        "first_anomaly_zscore": extreme_zscore,
        "first_anomaly_abs_zscore": abs(extreme_zscore),
        "first_anomaly_time_to_extreme_sec": time_to_extreme_sec,
        "pattern": pattern,
    }


def add_relative_order(metric_facts: List[Dict[str, Any]]) -> None:
    detected = [
        fact for fact in metric_facts
        if fact.get("anomaly_summary", {}).get("anomaly_detected")
    ]

    detected.sort(
        key=lambda fact: (
            fact["anomaly_summary"]["first_anomaly_start_ts"],
            -fact["anomaly_summary"].get("first_anomaly_abs_zscore", 0)
        )
    )

    for idx, fact in enumerate(detected, start=1):
        fact["anomaly_summary"]["relative_order"] = idx

    for fact in metric_facts:
        summary = fact.get("anomaly_summary")
        if summary is not None and "relative_order" not in summary:
            summary["relative_order"] = None


# =========================================================
# Per-case enhancement
# =========================================================
def enhance_single_case(
    state_report_path: Path,
    raw_telemetry_path: Path,
    injection_time_path: Path,
    output_dir: Path
) -> Optional[Path]:
    if not raw_telemetry_path.exists():
        print(f"[SKIP] Missing metrics.json for case: {state_report_path.stem}")
        print(f"       Expected: {raw_telemetry_path}")
        return None

    state_report = load_json(state_report_path)
    raw_telemetry = load_json(raw_telemetry_path)
    injection_ts = load_injection_time(injection_time_path)

    if injection_ts is not None:
        search_start_ts = injection_ts - INJECTION_LOOKBACK_SEC
        search_end_ts = injection_ts + INJECTION_LOOKAHEAD_SEC
    else:
        search_start_ts = None
        search_end_ts = None

    enhanced_metric_facts = []

    for fact in state_report.get("metric_facts", []):
        fact_new = dict(fact)
        metric_name = fact["metric_name"]

        detection_mode = get_detection_mode_from_state_report(fact)
        timeseries = raw_telemetry.get(metric_name)

        if timeseries is None:
            fact_new["anomaly_summary"] = {
                "anomaly_detected": False,
                "reason": f"{metric_name} not found in metrics.json",
                "detection_mode": detection_mode,
                "injection_ts": injection_ts,
                "search_window_start_ts": search_start_ts,
                "search_window_end_ts": search_end_ts,
            }
        else:
            fact_new["anomaly_summary"] = extract_anomaly_summary(
                timeseries=timeseries,
                baseline_mean=float(fact["baseline_mean"]),
                baseline_std=float(fact["baseline_std"]),
                detection_mode=detection_mode,
                injection_ts=injection_ts,
                search_start_ts=search_start_ts,
                search_end_ts=search_end_ts,
            )

        enhanced_metric_facts.append(fact_new)

    add_relative_order(enhanced_metric_facts)

    enhanced_report = dict(state_report)
    enhanced_report["injection_ts"] = injection_ts
    enhanced_report["search_window"] = {
        "enabled": injection_ts is not None,
        "lookback_sec": INJECTION_LOOKBACK_SEC,
        "lookahead_sec": INJECTION_LOOKAHEAD_SEC,
        "search_start_ts": search_start_ts,
        "search_end_ts": search_end_ts,
    }
    enhanced_report["metric_facts"] = enhanced_metric_facts

    output_path = output_dir / state_report_path.name
    save_json(output_path, enhanced_report)
    return output_path


# =========================================================
# Multi-case pipeline
# =========================================================
def process_all_cases(
    llm_input_dir: Path,
    rca_eval_re1_dir: Path,
    output_dir: Path
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    state_report_files = sorted(llm_input_dir.glob("*.json"))

    processed = 0
    skipped = 0
    failed = 0

    print(f"Found {len(state_report_files)} state report files.\n")

    for state_report_path in state_report_files:
        case_id = state_report_path.stem
        case_dir = rca_eval_re1_dir / case_id
        raw_telemetry_path = case_dir / "metrics.json"
        injection_time_path = case_dir / "inject_time.txt"

        print(f"[CASE] {case_id}")

        try:
            output_path = enhance_single_case(
                state_report_path=state_report_path,
                raw_telemetry_path=raw_telemetry_path,
                injection_time_path=injection_time_path,
                output_dir=output_dir
            )

            if output_path is None:
                skipped += 1
                continue

            print(f"       inject_time.txt -> {injection_time_path if injection_time_path.exists() else 'MISSING'}")
            print(f"       Saved -> {output_path}\n")
            processed += 1

        except Exception as e:
            print(f"[ERROR] {case_id}: {e}\n")
            failed += 1

    print("====================================")
    print("Pipeline finished.")
    print(f"Processed: {processed}")
    print(f"Skipped:   {skipped}")
    print(f"Failed:    {failed}")
    print(f"Output dir: {output_dir}")


if __name__ == "__main__":
    process_all_cases(
        llm_input_dir=LLM_INPUT_DIR,
        rca_eval_re1_dir=RCA_EVAL_RE1_DIR,
        output_dir=OUTPUT_DIR
    )