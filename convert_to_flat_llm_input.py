import json
import os
from pathlib import Path

INPUT_DIR = Path(r"C:\Users\dd262\Documents\MASTER\ECE1770 RCA\project\ECE1770_Project\dataset\artifacts\re1\llm_inputs_with_evidence")
OUTPUT_DIR = Path(r"C:\Users\dd262\Documents\MASTER\ECE1770 RCA\project\ECE1770_Project\dataset\artifacts\re1\llm_inputs_after_filtered")

MIN_DURATION = 10  


def process_file(file_path: Path):
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    metric_facts = data.get("metric_facts", [])
    flat_metrics = []

    # Step 1: 收集 valid anomaly
    for m in metric_facts:
        anomaly = m.get("anomaly_summary", {})

        if not anomaly.get("anomaly_detected", False):
            continue

        duration = anomaly.get("first_anomaly_duration_sec")
        if duration is None or duration < MIN_DURATION:
            continue  # 过滤 spike

        pct = m.get("percent_change")
        if pct is None:
            continue

        flat_metrics.append({
            "service": m.get("service"),
            "metric_type": m.get("metric_type"),

            "percent_change": round(pct, 2),
            "abs_percent_change": round(abs(pct), 2),

            "vs_injection_sec": anomaly.get("first_anomaly_vs_injection_sec"),
            "duration_sec": duration,

            "pattern": anomaly.get("pattern", "unknown"),

            "percent_rank": None,
            "temporal_rank": None
        })

    # Step 2: 排序（非常关键）
    flat_metrics.sort(key=lambda x: abs(x["percent_change"]), reverse=True)
    for i, m in enumerate(flat_metrics):
        m["percent_rank"] = i + 1

    flat_metrics.sort(key=lambda x: (
        float("inf") if x["vs_injection_sec"] is None else x["vs_injection_sec"]
    ))
    for i, m in enumerate(flat_metrics):
        m["temporal_rank"] = i + 1

    # Step 3: 恢复 percent 排序（更符合 LLM习惯）
    flat_metrics.sort(key=lambda x: x["percent_rank"])

    return {
        "metrics": flat_metrics
    }


def main():
    files = list(INPUT_DIR.glob("*.json"))

    print(f"Processing {len(files)} files...")

    for fpath in files:
        try:
            result = process_file(fpath)

            with open(fpath, "w", encoding="utf-8") as f:
                json.dump(result, f, indent=2, ensure_ascii=False)

            print(f"✔ Processed: {fpath.name}")

        except Exception as e:
            print(f"✘ Failed: {fpath.name}, error={e}")


if __name__ == "__main__":
    main()