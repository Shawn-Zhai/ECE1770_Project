import json
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

from agents.diagnosis_agent import DiagnosisAgent

PROJECT_ROOT = Path(
    r"C:\Users\dd262\Documents\MASTER\ECE1770 RCA\project\RCA_LLM_Reliability-Testing"
)

INPUT_FOLDER = PROJECT_ROOT / "dataset" / "artifacts" / "re1" / "llm_inputs"
RCA_EVAL_RE1_ROOT = PROJECT_ROOT / "dataset" / "RCAEval" / "RE1"

OUTPUT_PATH = Path("diagnosis_batch_results.json")
MAX_CASES = 10

def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def save_json(path: Path, data: Any) -> None:
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

def resolve_metrics_path(case_id: str) -> Path:
    """
    Example:
        case_id = re1ob_adservice_cpu_1

    Returns:
        .../dataset/RCAEval/RE1/re1ob_adservice_cpu_1/metrics.json
    """
    return RCA_EVAL_RE1_ROOT / case_id / "metrics.json"


def split_case_id(case_id: str) -> Tuple[str, Optional[int]]:
    """
    Example:
        re1ob_adservice_cpu_1
    -> ("re1ob_adservice_cpu", 1)

    If suffix is not numeric, return (case_id, None).
    """
    family_id, sep, repeat_str = case_id.rpartition("_")
    if not sep:
        return case_id, None

    if repeat_str.isdigit():
        return family_id, int(repeat_str)

    return case_id, None


def extract_ground_truth_from_case_id(case_id: str) -> Dict[str, str]:
    """
    Parse ground truth directly from filename stem.

    Expected format:
        <prefix>_<service>_<fault_type>_<repeat_id>

    Example:
        re1ob_adservice_cpu_1
    -> {
        "faulty_service": "adservice",
        "failure_type": "cpu"
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


def select_representative_case_files(input_folder: Path) -> list[Path]:
    """
    Keep only one representative file for each case family.
    Usually selects the smallest numeric suffix, e.g. *_1.json.

    Example:
        re1ob_adservice_cpu_1.json
        re1ob_adservice_cpu_2.json
        re1ob_adservice_cpu_3.json

    Keeps:
        re1ob_adservice_cpu_1.json
    """
    family_best: Dict[str, Tuple[int, Path] | Path] = {}

    for file_path in input_folder.iterdir():
        if not file_path.is_file() or file_path.suffix.lower() != ".json":
            continue

        case_id = file_path.stem
        family_id, repeat_id = split_case_id(case_id)

        if repeat_id is None:
            family_best.setdefault(family_id, file_path)
            continue

        existing = family_best.get(family_id)
        if existing is None:
            family_best[family_id] = (repeat_id, file_path)
        elif isinstance(existing, tuple):
            best_repeat, _ = existing
            if repeat_id < best_repeat:
                family_best[family_id] = (repeat_id, file_path)
        else:
            family_best[family_id] = (repeat_id, file_path)

    selected_files: list[Path] = []
    for value in family_best.values():
        if isinstance(value, tuple):
            selected_files.append(value[1])
        else:
            selected_files.append(value)

    return sorted(selected_files, key=lambda p: p.name)

def normalize_label(value: Optional[str]) -> str:
    if value is None:
        return "unknown"
    return str(value).strip().lower()


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
    joint_ok = service_ok and fault_ok

    return {
        "case_id": case_id,
        "pred_service": pred_service,
        "gt_service": gt_service,
        "service_correct": service_ok,
        "pred_fault": pred_fault,
        "gt_fault": gt_fault,
        "fault_correct": fault_ok,
        "joint_correct": joint_ok,
        "service_confidence": prediction.get("service_confidence", "low"),
        "failure_confidence": prediction.get("failure_confidence", "low"),
        "service_evidence_summary": prediction.get("service_evidence_summary", []),
        "failure_evidence_summary": prediction.get("failure_evidence_summary", []),
        "filtered_raw_telemetry_used": prediction.get(
            "filtered_raw_telemetry_used", False
        ),
    }


def print_summary(total: int, service_correct: int, fault_correct: int, joint_correct: int) -> None:
    print("\n==============================")
    print("Evaluation Summary")
    print("==============================")

    if total == 0:
        print("No valid cases were evaluated.")
        return

    print(f"Total cases: {total}")
    print(f"Service accuracy   : {service_correct}/{total} = {service_correct / total:.3f}")
    print(f"Fault accuracy     : {fault_correct}/{total} = {fault_correct / total:.3f}")
    print(f"Joint RCA accuracy : {joint_correct}/{total} = {joint_correct / total:.3f}")

def main() -> None:
    if not INPUT_FOLDER.exists():
        raise FileNotFoundError(f"Input folder not found: {INPUT_FOLDER}")

    agent = DiagnosisAgent(
        use_llm_refinement=True,
    )

    selected_files = select_representative_case_files(INPUT_FOLDER)
    print(f"Total representative cases selected: {len(selected_files)}")

    total = 0
    service_correct = 0
    fault_correct = 0
    joint_correct = 0

    results: list[Dict[str, Any]] = []

    for file_path in selected_files:
        if total >= MAX_CASES:
            break

        case_id = file_path.stem

        try:
            ground_truth = extract_ground_truth_from_case_id(case_id)
        except Exception as exc:
            print(f"WARNING: Failed to parse ground truth from case_id {case_id}: {exc}")
            continue

        try:
            state_report = load_json(file_path)
        except Exception as exc:
            print(f"WARNING: Failed to load state report for {case_id}: {exc}")
            continue

        metrics_path = resolve_metrics_path(case_id)
        raw_telemetry = None

        if metrics_path.exists():
            try:
                raw_telemetry = load_json(metrics_path)
            except Exception as exc:
                print(f"WARNING: Failed to load raw metrics for {case_id}: {exc}")
        else:
            print(f"WARNING: No raw metrics found for {case_id}: {metrics_path}")

        try:
            prediction = agent.diagnose(
                state_report=state_report,
                raw_telemetry=raw_telemetry,
            )
        except Exception as exc:
            print(f"WARNING: Diagnosis failed for {case_id}: {exc}")
            continue

        result_item = evaluate_prediction(case_id, prediction, ground_truth)
        results.append(result_item)

        if result_item["service_correct"]:
            service_correct += 1
        if result_item["fault_correct"]:
            fault_correct += 1
        if result_item["joint_correct"]:
            joint_correct += 1

        total += 1

        print(
            f"{case_id} | "
            f"service: {result_item['pred_service']} vs {result_item['gt_service']} | "
            f"fault: {result_item['pred_fault']} vs {result_item['gt_fault']}"
        )

    print_summary(total, service_correct, fault_correct, joint_correct)

    output_payload = {
        "summary": {
            "total_cases": total,
            "service_correct": service_correct,
            "fault_correct": fault_correct,
            "joint_correct": joint_correct,
            "service_accuracy": (service_correct / total) if total else 0.0,
            "fault_accuracy": (fault_correct / total) if total else 0.0,
            "joint_accuracy": (joint_correct / total) if total else 0.0,
        },
        "results": results,
    }

    save_json(OUTPUT_PATH, output_payload)
    print(f"\nSaved results to {OUTPUT_PATH.resolve()}")


if __name__ == "__main__":
    main()