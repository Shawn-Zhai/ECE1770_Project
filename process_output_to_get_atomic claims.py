import json
from pathlib import Path
from agents.claim_decomposation import ClaimDecomposer, decompose_claims_for_case, decompose_claims_for_cases

PROJECT_ROOT = Path(__file__).resolve().parent
INPUT_PATH = PROJECT_ROOT / "diagnosis_re1_all_results.json"
OUTPUT_PATH = PROJECT_ROOT / "re1_diagnosis_all_results.json"

def main() -> None:
    if not INPUT_PATH.exists():
        raise FileNotFoundError(f"Input file not found: {INPUT_PATH}")

    with INPUT_PATH.open("r", encoding="utf-8") as f:
        data = json.load(f).get("results")

    if isinstance(data, list):
        case_results = data
    else:
        raise ValueError("Input JSON must be either a dict or a list of dicts.")

    outputs = decompose_claims_for_cases(case_results, debug=False)

    with OUTPUT_PATH.open("w", encoding="utf-8") as f:
        json.dump(outputs, f, indent=2, ensure_ascii=False)

    print(f"Processed {len(outputs)} cases.")
    print(f"Saved output to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()