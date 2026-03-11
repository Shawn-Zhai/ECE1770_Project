import json
from pathlib import Path
from typing import Any, Dict, Optional

from agents.diagnosis_agent import DiagnosisAgent
from utils.raw_telemetry_utils import (
    filter_raw_telemetry_by_service,
    summarize_service_state_evidence,
    build_metric_fact_lookup,
    compress_filtered_raw_telemetry,
)


PROJECT_ROOT = Path(
    r"C:\Users\dd262\Documents\MASTER\ECE1770 RCA\project\RCA_LLM_Reliability-Testing"
)

# 改成你要测试的 case
CASE_ID = "re1ob_adservice_cpu_1"

STATE_REPORT_PATH = (
    PROJECT_ROOT
    / "dataset"
    / "artifacts"
    / "re1"
    / "llm_inputs"
    / f"{CASE_ID}.json"
)

RAW_METRICS_PATH = (
    PROJECT_ROOT
    / "dataset"
    / "RCAEval"
    / "RE1"
    / CASE_ID
    / "metrics.json"
)


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def normalize_label(value: Optional[str]) -> str:
    if value is None:
        return "unknown"
    return str(value).strip().lower()


def extract_ground_truth_from_case_id(case_id: str) -> Dict[str, str]:
    """
    Example:
        re1ob_adservice_cpu_1

    Returns:
        {
            "faulty_service": "adservice",
            "failure_type": "cpu",
        }
    """
    parts = case_id.split("_")
    if len(parts) != 4:
        raise ValueError(
            f"Unexpected case_id format: {case_id}. "
            "Expected format like 're1ob_adservice_cpu_1'."
        )

    return {
        "faulty_service": parts[1].strip().lower(),
        "failure_type": parts[2].strip().lower(),
    }


def pretty_print_header(title: str) -> None:
    print("\n" + "=" * 100)
    print(title)
    print("=" * 100)


def preview_dict_keys(data: Any, name: str, max_keys: int = 20) -> None:
    pretty_print_header(f"{name} Preview")

    if isinstance(data, dict):
        keys = list(data.keys())
        print(f"{name} type       : dict")
        print(f"{name} total keys : {len(keys)}")
        print(f"{name} first keys : {keys[:max_keys]}")
    elif isinstance(data, list):
        print(f"{name} type       : list")
        print(f"{name} length     : {len(data)}")
        if data:
            print(f"{name} first item type: {type(data[0]).__name__}")
            print(json.dumps(data[0], indent=2, ensure_ascii=False)[:1500])
    else:
        print(f"{name} type       : {type(data).__name__}")
        print(f"{name} value      : {str(data)[:1500]}")


def preview_metric_facts(state_report: Dict[str, Any], max_items: int = 5) -> None:
    pretty_print_header("metric_facts Preview")

    metric_facts = state_report.get("metric_facts", [])
    if not isinstance(metric_facts, list):
        print("metric_facts is not a list.")
        return

    print(f"metric_facts count: {len(metric_facts)}")

    for i, row in enumerate(metric_facts[:max_items], start=1):
        print(f"\n--- metric_facts[{i}] ---")
        print(json.dumps(row, indent=2, ensure_ascii=False)[:2000])


def preview_raw_metrics(
    raw_telemetry: Any,
    max_metrics: int = 5,
    max_points: int = 5,
) -> None:
    pretty_print_header("Raw Telemetry Preview")

    if not isinstance(raw_telemetry, dict):
        print("raw_telemetry is not a dict.")
        return

    metric_names = list(raw_telemetry.keys())
    print(f"raw metric count: {len(metric_names)}")

    for metric_name in metric_names[:max_metrics]:
        series = raw_telemetry.get(metric_name, [])
        print(f"\nMetric: {metric_name}")
        print(f"Series length: {len(series) if isinstance(series, list) else 'N/A'}")

        if isinstance(series, list):
            print("First few points:")
            for point in series[:max_points]:
                print(f"  {point}")


def print_filtered_raw_telemetry(
    filtered_raw_telemetry: Optional[Dict[str, Any]],
    max_metrics: int = 10,
    max_points: int = 8,
) -> None:
    pretty_print_header("Filtered Raw Telemetry")

    if not isinstance(filtered_raw_telemetry, dict) or not filtered_raw_telemetry:
        print("filtered_raw_telemetry is empty.")
        return

    keys = list(filtered_raw_telemetry.keys())
    print(f"filtered metric count: {len(keys)}")
    print(f"filtered metric keys : {keys[:max_metrics]}")

    for metric_name in keys[:max_metrics]:
        series = filtered_raw_telemetry.get(metric_name, [])
        print(f"\nMetric: {metric_name}")
        print(f"Series length: {len(series) if isinstance(series, list) else 'N/A'}")

        if isinstance(series, list):
            print("First few points:")
            for point in series[:max_points]:
                print(f"  {point}")


def print_compressed_raw_telemetry(
    compressed_raw_telemetry: Optional[Dict[str, Any]],
) -> None:
    pretty_print_header("Compressed Raw Telemetry")

    if not isinstance(compressed_raw_telemetry, dict):
        print("compressed_raw_telemetry is None or invalid.")
        return

    print(json.dumps(compressed_raw_telemetry, indent=2, ensure_ascii=False)[:8000])

    pretty_print_header("Compressed Raw Telemetry - Key Summary")

    print(f"status           : {compressed_raw_telemetry.get('status')}")
    print(f"instruction_hint : {compressed_raw_telemetry.get('instruction_hint')}")

    meta = compressed_raw_telemetry.get("meta", {})
    print("\nmeta:")
    print(json.dumps(meta, indent=2, ensure_ascii=False))

    timeline_summary = compressed_raw_telemetry.get("timeline_summary", [])
    print("\ntimeline_summary:")
    if isinstance(timeline_summary, list) and timeline_summary:
        for i, item in enumerate(timeline_summary, start=1):
            print(f"  {i}. {item}")
    else:
        print("  <empty>")

    metric_events = compressed_raw_telemetry.get("metric_order_by_first_anomaly", [])
    print("\nmetric_order_by_first_anomaly:")
    if not isinstance(metric_events, list) or not metric_events:
        print("  <empty>")
        return

    for i, row in enumerate(metric_events, start=1):
        print(f"\n--- event[{i}] ---")
        print(f"metric            : {row.get('metric')}")
        print(f"service           : {row.get('service')}")
        print(f"metric_type       : {row.get('metric_type')}")
        print(f"first_anomaly_ts  : {row.get('first_anomaly_ts')}")
        print(f"direction         : {row.get('direction')}")
        print(f"baseline_mean     : {row.get('baseline_mean')}")
        print(f"baseline_std      : {row.get('baseline_std')}")
        print(f"onset_zscore      : {row.get('onset_zscore')}")
        print(f"peak_ts           : {row.get('peak_ts')}")
        print(f"peak_zscore       : {row.get('peak_zscore')}")

        evidence_points = row.get("evidence_points", [])
        print("evidence_points:")
        if isinstance(evidence_points, list) and evidence_points:
            for point in evidence_points:
                print(f"  {point}")
        else:
            print("  <empty>")


def print_prediction_details(prediction: Dict[str, Any]) -> None:
    pretty_print_header("Diagnosis Output")
    print(json.dumps(prediction, indent=2, ensure_ascii=False)[:10000])

    pretty_print_header("Diagnosis Key Fields")
    print(f"faulty_service             : {prediction.get('faulty_service')}")
    print(f"failure_type               : {prediction.get('failure_type')}")
    print(f"service_confidence         : {prediction.get('service_confidence')}")
    print(f"failure_confidence         : {prediction.get('failure_confidence')}")
    print(f"filtered_raw_telemetry_used: {prediction.get('filtered_raw_telemetry_used')}")

    service_summary = prediction.get("service_evidence_summary", [])
    failure_summary = prediction.get("failure_evidence_summary", [])
    failure_type_scores = prediction.get("failure_type_scores", {})

    print("\nservice_evidence_summary:")
    if isinstance(service_summary, list) and service_summary:
        for i, item in enumerate(service_summary, start=1):
            print(f"  {i}. {item}")
    else:
        print("  <empty>")

    print("\nfailure_evidence_summary:")
    if isinstance(failure_summary, list) and failure_summary:
        for i, item in enumerate(failure_summary, start=1):
            print(f"  {i}. {item}")
    else:
        print("  <empty>")

    print("\nfailure_type_scores:")
    if isinstance(failure_type_scores, dict) and failure_type_scores:
        print(json.dumps(failure_type_scores, indent=2, ensure_ascii=False))
    else:
        print("  <empty>")


def print_ground_truth_comparison(
    case_id: str,
    prediction: Dict[str, Any],
    ground_truth: Dict[str, str],
) -> None:
    pretty_print_header("Ground Truth Check")

    pred_service = normalize_label(prediction.get("faulty_service"))
    pred_type = normalize_label(prediction.get("failure_type"))

    gt_service = normalize_label(ground_truth.get("faulty_service"))
    gt_type = normalize_label(ground_truth.get("failure_type"))

    print(f"Case ID                : {case_id}")
    print(f"Predicted service      : {pred_service}")
    print(f"Expected service       : {gt_service}")
    print(f"Service correct        : {pred_service == gt_service}")

    print(f"Predicted failure type : {pred_type}")
    print(f"Expected failure type  : {gt_type}")
    print(f"Failure type correct   : {pred_type == gt_type}")

    print(
        f"Joint correct          : "
        f"{(pred_service == gt_service) and (pred_type == gt_type)}"
    )


def main() -> None:
    pretty_print_header("Single Case Diagnosis Debug Run")

    print(f"PROJECT_ROOT      : {PROJECT_ROOT}")
    print(f"CASE_ID           : {CASE_ID}")
    print(f"STATE_REPORT_PATH : {STATE_REPORT_PATH}")
    print(f"RAW_METRICS_PATH  : {RAW_METRICS_PATH}")

    if not STATE_REPORT_PATH.exists():
        raise FileNotFoundError(f"State report file not found: {STATE_REPORT_PATH}")

    if not RAW_METRICS_PATH.exists():
        raise FileNotFoundError(f"Raw metrics file not found: {RAW_METRICS_PATH}")

    # 1) parse GT
    ground_truth = extract_ground_truth_from_case_id(CASE_ID)
    pretty_print_header("Parsed Ground Truth")
    print(json.dumps(ground_truth, indent=2, ensure_ascii=False))

    # 2) load files
    state_report = load_json(STATE_REPORT_PATH)
    raw_telemetry = load_json(RAW_METRICS_PATH)

    preview_dict_keys(state_report, "state_report")
    #preview_metric_facts(state_report, max_items=5)

    preview_dict_keys(raw_telemetry, "raw_telemetry")
    #preview_raw_metrics(raw_telemetry, max_metrics=5, max_points=5)

    # 3)faulty_service
    gt_service = ground_truth["faulty_service"]

    pretty_print_header("Manual Compression Check")
    print(f"Use service for manual check: {gt_service}")

    filtered_raw_telemetry = filter_raw_telemetry_by_service(
        raw_telemetry=raw_telemetry,
        service=gt_service,
    )
    
    lookup_result = build_metric_fact_lookup(state_report, gt_service)
    print("lookup result is: ")
    print(lookup_result)
    
    print_filtered_raw_telemetry(filtered_raw_telemetry)

    compressed_raw_telemetry = compress_filtered_raw_telemetry(
        filtered_raw_telemetry=filtered_raw_telemetry,
        metric_fact_lookup= lookup_result,
        max_metrics=6,
        max_points_per_metric=12,
        min_consecutive_points=10
    )
    
    print_compressed_raw_telemetry(compressed_raw_telemetry)
    
    

    # # 4) 再跑 diagnosis agent
    # pretty_print_header("Create DiagnosisAgent")
    # agent = DiagnosisAgent(
    #     use_llm_refinement=True,
    #     debug=True,
    # )
    # print("DiagnosisAgent created.")

    # pretty_print_header("Run Diagnosis")
    # prediction = agent.diagnose(
    #     state_report=state_report,
    #     raw_telemetry=raw_telemetry,
    # )

    # # 5) print final result
    # print_prediction_details(prediction)

    # # 6) 如果 agent 内部返回了 compressed，也再打印一次
    # agent_compressed = prediction.get("compressed_raw_telemetry")
    # print_compressed_raw_telemetry(agent_compressed)

    # # 7) compare with GT
    # print_ground_truth_comparison(
    #     case_id=CASE_ID,
    #     prediction=prediction,
    #     ground_truth=ground_truth,
    # )


if __name__ == "__main__":
    main()