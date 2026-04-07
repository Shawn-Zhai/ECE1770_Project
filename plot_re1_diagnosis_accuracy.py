import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parent
INPUT_PATH = PROJECT_ROOT / "diagnosis_re1_all_results.json"
OUTPUT_PATH = PROJECT_ROOT / "presentation" / "re1_diagnosis_accuracy_by_fault_type.png"


def main() -> None:
    with INPUT_PATH.open("r", encoding="utf-8") as f:
        data = json.load(f)

    by_fault = data["by_fault_type"]
    fault_order = ["cpu", "delay", "disk", "loss", "mem"]
    fault_labels = ["CPU", "Delay", "Disk", "Loss", "Memory"]

    service_acc = []
    fault_type_acc = []
    joint_acc = []

    for fault in fault_order:
        row = by_fault[fault]
        total = int(row["cases"])
        if total == 0:
            service_acc.append(0.0)
            fault_type_acc.append(0.0)
            joint_acc.append(0.0)
            continue

        service_acc.append(int(row["service_correct"]) / total * 100.0)
        fault_type_acc.append(int(row["fault_correct"]) / total * 100.0)
        joint_acc.append(int(row["both_right"]) / total * 100.0)

    x = np.arange(len(fault_order))
    width = 0.25

    plt.figure(figsize=(10, 5.5))
    plt.bar(x - width, service_acc, width, label="Service accuracy")
    plt.bar(x, fault_type_acc, width, label="Fault-type accuracy")
    plt.bar(x + width, joint_acc, width, label="Joint accuracy")

    plt.ylabel("Accuracy (%)")
    plt.xlabel("Failure type")
    plt.title("RE1 Diagnosis Accuracy by Fault Type", pad=46)
    plt.xticks(x, fault_labels)
    plt.ylim(0, 110)
    plt.legend(loc="upper center", bbox_to_anchor=(0.5, 1.08), ncol=3, frameon=False)
    plt.grid(axis="y", linestyle="--", alpha=0.4)

    for index, values in enumerate([service_acc, fault_type_acc, joint_acc]):
        offset = (index - 1) * width
        for xi, value in zip(x, values):
            plt.text(
                xi + offset,
                value + 2,
                f"{value:.1f}%",
                ha="center",
                va="bottom",
                fontsize=8,
            )

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout(rect=(0, 0, 1, 0.90))
    plt.savefig(OUTPUT_PATH, dpi=300)
    print(f"Saved chart to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
