import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Optional

from agents.diagnosis_agent import DiagnosisAgent

PROJECT_ROOT = Path(__file__).resolve().parent

INPUT_FOLDER = PROJECT_ROOT / "dataset" / "artifacts" / "re1" / "llm_inputs"
OUTPUT_PATH = PROJECT_ROOT / "re1_diagnosis_all_results.json"


# ------------------------
# Utils
# ------------------------

def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def save_json(path: Path, data: Any) -> None:
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def extract_ground_truth_from_case_id(case_id: str) -> Dict[str, str]:
    parts = case_id.split("_")
    if len(parts) != 4:
        raise ValueError(f"Bad case_id format: {case_id}")

    return {
        "faulty_service": parts[1].strip().lower(),
        "failure_type": parts[2].strip().lower(),
    }


def normalize_label(value: Optional[str]) -> str:
    if value is None:
        return "unknown"
    return str(value).strip().lower()


# ------------------------
# Evaluation
# ------------------------

def evaluate_prediction(
    case_id: str,
    prediction: Dict[str, Any],
    ground_truth: Dict[str, str],
) -> Dict[str, Any]:

    pred_service = normalize_label(prediction.get("faulty_service"))
    pred_fault = normalize_label(prediction.get("failure_type"))

    gt_service = ground_truth["faulty_service"]
    gt_fault = ground_truth["failure_type"]

    service_ok = pred_service == gt_service
    fault_ok = pred_fault == gt_fault
    joint_ok = service_ok and fault_ok

    return {
        "case_id": case_id,
        "service_correct": service_ok,
        "fault_correct": fault_ok,
        "joint_correct": joint_ok,
        "gt_failure_type": gt_fault,
        "prediction": prediction,
    }


def format_ratio(c: int, t: int) -> str:
    if t == 0:
        return "0/0 = 0%"
    return f"{c}/{t} = {100*c/t:.1f}%"


# ------------------------
# Pretty Table
# ------------------------

def print_by_fault_type(stats: Dict[str, Dict[str, int]]):

    print("\nBy ground-truth fault type:")

    headers = ["Fault type", "Cases", "Service acc", "Fault acc", "Both right"]

    rows = []
    for k in sorted(stats.keys()):
        s = stats[k]
        total = s["cases"]

        rows.append([
            k,
            str(total),
            format_ratio(s["service_correct"], total),
            format_ratio(s["fault_correct"], total),
            format_ratio(s["both_right"], total),
        ])

    col_w = [len(h) for h in headers]
    for r in rows:
        for i in range(len(r)):
            col_w[i] = max(col_w[i], len(r[i]))

    def row(r):
        return "| " + " | ".join(r[i].ljust(col_w[i]) for i in range(len(r))) + " |"

    sep = "+" + "+".join("-"*(w+2) for w in col_w) + "+"

    print(sep)
    print(row(headers))
    print(sep)

    for r in rows:
        print(row(r))
        print(sep)


# ------------------------
# Main
# ------------------------

def main():

    if not INPUT_FOLDER.exists():
        raise FileNotFoundError(INPUT_FOLDER)

    files = [
        p for p in INPUT_FOLDER.iterdir()
        if p.suffix.lower() == ".json"
    ]

    print(f"Total cases: {len(files)}")

    agent = DiagnosisAgent()

    total = 0
    service_correct = 0
    fault_correct = 0
    joint_correct = 0

    by_fault = defaultdict(lambda: {
        "cases": 0,
        "service_correct": 0,
        "fault_correct": 0,
        "both_right": 0,
    })

    results = []

    for f in files:
        case_id = f.stem

        try:
            gt = extract_ground_truth_from_case_id(case_id)
            data = load_json(f)

            pred = agent.diagnose(state_report=data)

            r = evaluate_prediction(case_id, pred, gt)
            results.append(r)

            # overall
            total += 1
            if r["service_correct"]:
                service_correct += 1
            if r["fault_correct"]:
                fault_correct += 1
            if r["joint_correct"]:
                joint_correct += 1

            # per fault type
            ft = r["gt_failure_type"]
            by_fault[ft]["cases"] += 1

            if r["service_correct"]:
                by_fault[ft]["service_correct"] += 1
            if r["fault_correct"]:
                by_fault[ft]["fault_correct"] += 1
            if r["joint_correct"]:
                by_fault[ft]["both_right"] += 1

            print(f"[OK] {case_id}")

        except Exception as e:
            print(f"[ERROR] {case_id}: {e}")

    # ------------------------
    # Summary
    # ------------------------

    print("\n===== Overall =====")
    print(f"Service acc: {service_correct}/{total} = {service_correct/total:.3f}")
    print(f"Fault acc  : {fault_correct}/{total} = {fault_correct/total:.3f}")
    print(f"Joint acc  : {joint_correct}/{total} = {joint_correct/total:.3f}")

    print_by_fault_type(by_fault)

    # save
    save_json(OUTPUT_PATH, {
        "summary": {
            "total": total,
            "service_acc": service_correct / total if total else 0,
            "fault_acc": fault_correct / total if total else 0,
            "joint_acc": joint_correct / total if total else 0,
        },
        "by_fault_type": dict(by_fault),
        "results": results,
    })

    print(f"\nSaved to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()