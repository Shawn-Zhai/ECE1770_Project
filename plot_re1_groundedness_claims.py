import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parent
INPUT_PATH = PROJECT_ROOT / "groundedness_re1_metric_source_results.json"
OUTPUT_PATH = PROJECT_ROOT / "presentation" / "re1_groundedness_claim_support.png"


def percent(value: int, total: int) -> float:
    return (value / total * 100.0) if total else 0.0


def main() -> None:
    with INPUT_PATH.open("r", encoding="utf-8") as f:
        data = json.load(f)

    summary = data["summary"]
    overall_labels = ["Service\nclaims", "Failure\nclaims", "All\nclaims"]
    overall_supported = [
        int(summary["service_claims"]["supported"]),
        int(summary["failure_claims"]["supported"]),
        int(summary["all_claims"]["supported"]),
    ]
    overall_unsupported = [
        int(summary["service_claims"]["unsupported"]),
        int(summary["failure_claims"]["unsupported"]),
        int(summary["all_claims"]["unsupported"]),
    ]

    claim_groups = {
        "Service\nobservations": ("service_claim_results", "observation"),
        "Service\ninferences": ("service_claim_results", "inference"),
        "Failure\nobservations": ("failure_claim_results", "observation"),
        "Failure\ninferences": ("failure_claim_results", "inference"),
    }

    type_supported = []
    type_unsupported = []
    for result_key, claim_type in claim_groups.values():
        rows = [
            claim
            for case in data["cases"]
            for claim in case.get(result_key, [])
            if claim.get("type") == claim_type
        ]
        supported = sum(1 for row in rows if row.get("verdict") == "supported")
        type_supported.append(supported)
        type_unsupported.append(len(rows) - supported)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
    fig.suptitle("RE1 Groundedness: Supported vs. Unsupported Claims", y=0.98)

    supported_color = "#2ca02c"
    unsupported_color = "#d62728"

    x = np.arange(len(overall_labels))
    axes[0].bar(x, overall_supported, label="Supported", color=supported_color)
    axes[0].bar(
        x,
        overall_unsupported,
        bottom=overall_supported,
        label="Unsupported",
        color=unsupported_color,
    )
    axes[0].set_title("By Claim Group")
    axes[0].set_ylabel("Number of claims")
    axes[0].set_xticks(x, overall_labels)
    axes[0].grid(axis="y", linestyle="--", alpha=0.35)

    for xi, supported, unsupported in zip(x, overall_supported, overall_unsupported):
        total = supported + unsupported
        axes[0].text(
            xi,
            total + 20,
            f"{percent(supported, total):.1f}%",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    x2 = np.arange(len(claim_groups))
    axes[1].bar(x2, type_supported, label="Supported", color=supported_color)
    axes[1].bar(
        x2,
        type_unsupported,
        bottom=type_supported,
        label="Unsupported",
        color=unsupported_color,
    )
    axes[1].set_title("Observation vs. Inference Claims")
    axes[1].set_ylabel("Number of claims")
    axes[1].set_xticks(x2, list(claim_groups.keys()))
    axes[1].grid(axis="y", linestyle="--", alpha=0.35)

    for xi, supported, unsupported in zip(x2, type_supported, type_unsupported):
        total = supported + unsupported
        axes[1].text(
            xi,
            total + 10,
            f"{percent(supported, total):.1f}%",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.925), ncol=2, frameon=False)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout(rect=(0, 0, 1, 0.88))
    plt.savefig(OUTPUT_PATH, dpi=300)
    print(f"Saved chart to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
