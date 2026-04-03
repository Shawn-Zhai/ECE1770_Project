import argparse
import json
import random
from pathlib import Path
from typing import Any, Dict, List, Optional

from agents.diagnosis_agent_re2 import DiagnosisAgentRE2


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_INPUT_FOLDER = PROJECT_ROOT / "dataset" / "artifacts" / "re2" / "llm_inputs"
DEFAULT_OUTPUT_PATH = PROJECT_ROOT / "diagnosis_re2_random_results.json"


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def save_json(path: Path, data: Any) -> None:
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def normalize_label(value: Optional[str]) -> str:
    if value is None:
        return "unknown"
    return str(value).strip().lower()


def extract_ground_truth_from_case_id(case_id: str) -> Dict[str, str]:
    """
    Expected format:
        re2ob_checkoutservice_cpu_1
    """
    parts = case_id.split("_")
    if len(parts) != 4:
        raise ValueError(
            f"Unexpected case_id format: {case_id}. "
            "Expected format like 're2ob_checkoutservice_cpu_1'."
        )

    return {
        "faulty_service": parts[1].strip().lower(),
        "failure_type": parts[2].strip().lower(),
    }


def collect_case_files(input_folder: Path) -> List[Path]:
    return sorted(
        file_path
        for file_path in input_folder.iterdir()
        if file_path.is_file() and file_path.suffix.lower() == ".json"
    )


def evaluate_prediction(
    case_id: str,
    prediction: Dict[str, Any],
    ground_truth: Dict[str, str],
) -> Dict[str, Any]:
    pred_service = normalize_label(prediction.get("faulty_service"))
    pred_fault = normalize_label(prediction.get("failure_type"))

    gt_service = normalize_label(ground_truth["faulty_service"])
    gt_fault = normalize_label(ground_truth["failure_type"])

    service_ok = pred_service == gt_service
    fault_ok = pred_fault == gt_fault

    if service_ok and fault_ok:
        outcome = "both_right"
    elif (not service_ok) and (not fault_ok):
        outcome = "both_wrong"
    elif service_ok:
        outcome = "service_only_right"
    else:
        outcome = "fault_type_only_right"

    return {
        "case_id": case_id,
        "pred_service": pred_service,
        "gt_service": gt_service,
        "service_correct": service_ok,
        "pred_fault": pred_fault,
        "gt_fault": gt_fault,
        "fault_correct": fault_ok,
        "outcome": outcome,
        "service_evidence_claims": prediction.get("service_evidence_claims", []),
        "failure_evidence_claims": prediction.get("failure_evidence_claims", []),
    }


def summarize_results(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    total = len(results)
    service_correct = sum(1 for row in results if bool(row["service_correct"]))
    fault_correct = sum(1 for row in results if bool(row["fault_correct"]))
    both_right = sum(1 for row in results if row["outcome"] == "both_right")
    both_wrong = sum(1 for row in results if row["outcome"] == "both_wrong")
    service_only_right = sum(1 for row in results if row["outcome"] == "service_only_right")
    fault_type_only_right = sum(
        1 for row in results if row["outcome"] == "fault_type_only_right"
    )

    return {
        "total_cases": total,
        "service_correct": service_correct,
        "fault_type_correct": fault_correct,
        "both_right": both_right,
        "both_wrong": both_wrong,
        "service_only_right": service_only_right,
        "fault_type_only_right": fault_type_only_right,
        "service_accuracy": (service_correct / total) if total else 0.0,
        "fault_type_accuracy": (fault_correct / total) if total else 0.0,
        "joint_accuracy": (both_right / total) if total else 0.0,
    }


def print_summary(summary: Dict[str, Any]) -> None:
    total = int(summary["total_cases"])
    print("\n==============================")
    print("RE2 Random Smoke Test Summary")
    print("==============================")
    print(f"Total cases            : {total}")
    print(
        f"Faulty service right   : {summary['service_correct']}/{total}"
        f" = {summary['service_accuracy']:.3f}"
    )
    print(
        f"Fault type right       : {summary['fault_type_correct']}/{total}"
        f" = {summary['fault_type_accuracy']:.3f}"
    )
    print(
        f"Both right             : {summary['both_right']}/{total}"
        f" = {summary['joint_accuracy']:.3f}"
    )
    print(f"Both wrong             : {summary['both_wrong']}/{total}")
    print(f"Service only right     : {summary['service_only_right']}/{total}")
    print(f"Fault type only right  : {summary['fault_type_only_right']}/{total}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the RE2 diagnosis agent on a random sample of preprocessed RE2 cases."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_INPUT_FOLDER,
        help="Folder containing RE2 llm_inputs JSON files.",
    )
    parser.add_argument(
        "--num-cases",
        type=int,
        default=10,
        help="Number of random cases to evaluate.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=1770,
        help="Random seed used for sampling.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT_PATH,
        help="Optional JSON file to store detailed results.",
    )
    parser.add_argument(
        "--no-llm-refinement",
        action="store_true",
        help="Disable the agent's direct LLM decisions and use the rule-based path only.",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable agent debug prints.",
    )
    args = parser.parse_args()

    input_folder = args.input
    if not input_folder.exists():
        raise FileNotFoundError(f"Input folder not found: {input_folder}")

    case_files = collect_case_files(input_folder)
    if not case_files:
        raise FileNotFoundError(f"No JSON case files found in: {input_folder}")

    sample_size = min(max(1, args.num_cases), len(case_files))
    rng = random.Random(args.seed)
    selected_files = rng.sample(case_files, sample_size)
    selected_files.sort(key=lambda path: path.name)

    print(f"Input folder: {input_folder}")
    print(f"Available cases: {len(case_files)}")
    print(f"Selected random sample: {sample_size}")
    print(f"Random seed: {args.seed}")
    print("Selected case_ids:")
    for file_path in selected_files:
        print(f"  - {file_path.stem}")

    agent = DiagnosisAgentRE2(
        use_llm_refinement=not args.no_llm_refinement,
        debug=args.debug,
    )

    results: List[Dict[str, Any]] = []

    for index, file_path in enumerate(selected_files, start=1):
        case_id = file_path.stem
        print(f"\n[{index}/{sample_size}] Running {case_id}")

        ground_truth = extract_ground_truth_from_case_id(case_id)
        state_report = load_json(file_path)

        prediction = agent.diagnose(
            state_report=state_report,
            raw_telemetry=None,
        )

        result_item = evaluate_prediction(case_id, prediction, ground_truth)
        results.append(result_item)

        print(
            "  "
            f"pred_service={result_item['pred_service']} "
            f"gt_service={result_item['gt_service']} "
            f"pred_fault={result_item['pred_fault']} "
            f"gt_fault={result_item['gt_fault']} "
            f"outcome={result_item['outcome']}"
        )

    summary = summarize_results(results)
    print_summary(summary)

    output_payload = {
        "input_folder": str(input_folder),
        "num_cases_requested": int(args.num_cases),
        "num_cases_evaluated": sample_size,
        "seed": int(args.seed),
        "llm_refinement_enabled": not args.no_llm_refinement,
        "selected_case_ids": [path.stem for path in selected_files],
        "summary": summary,
        "results": results,
    }

    if args.output:
        save_json(args.output, output_payload)
        print(f"\nDetailed results saved to: {args.output}")


if __name__ == "__main__":
    main()
