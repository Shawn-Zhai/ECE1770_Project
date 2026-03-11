from typing import Any, Dict, List
from utils.data_structures import ValidationAction, EvidenceItem


class ValidatorExecutor:
    def __init__(self):
        pass

    def execute(self, action: ValidationAction, state_report: Dict[str, Any]) -> EvidenceItem:
        source = action.source.lower()

        if source == "logs":
            return self._validate_with_logs(action, state_report)
        elif source == "metrics":
            return self._validate_with_metrics(action, state_report)
        elif source == "traces":
            return self._validate_with_traces(action, state_report)
        else:
            return EvidenceItem(
                source=source,
                operation=action.operation,
                target=action.target,
                summary=f"Unsupported source '{source}'",
                details={}
            )

    def validate_with_logs(self, action: ValidationAction, state_report: Dict[str, Any]) -> EvidenceItem:
        logs = state_report.get("supporting_raw_data", {}).get("logs", [])
        target = action.target.lower()

        matched = [log for log in logs if target in str(log).lower()]

        summary = (
            f"Found {len(matched)} log entries related to '{action.target}'."
            if matched else
            f"No log entries matched '{action.target}'."
        )

        return EvidenceItem(
            source="logs",
            operation=action.operation,
            target=action.target,
            summary=summary,
            details={
                "match_count": len(matched),
                "matched_logs": matched[:10]
            }
        )

    def validate_with_metrics(self, action: ValidationAction, state_report: Dict[str, Any]) -> EvidenceItem:
        telemetry = state_report.get("supporting_raw_data", {}).get("metrics", {})
        target = action.target

        metric_value = telemetry.get(target)

        if metric_value is None:
            return EvidenceItem(
                source="metrics",
                operation=action.operation,
                target=target,
                summary=f"Metric '{target}' not found.",
                details={}
            )

        summary = f"Metric '{target}' was found."

        details = {"raw_value": metric_value}

        if isinstance(metric_value, list) and len(metric_value) >= 2:
            try:
                nums = [float(x) for x in metric_value]
                delta = nums[-1] - nums[0]
                details["delta"] = delta
                if delta > 0:
                    summary += f" Trend increased by {delta:.2f}."
                elif delta < 0:
                    summary += f" Trend decreased by {abs(delta):.2f}."
                else:
                    summary += " Trend remained stable."
            except Exception:
                pass

        return EvidenceItem(
            source="metrics",
            operation=action.operation,
            target=target,
            summary=summary,
            details=details
        )

    def validate_with_traces(self, action: ValidationAction, state_report: Dict[str, Any]) -> EvidenceItem:
        traces = state_report.get("supporting_raw_data", {}).get("traces", [])
        target = action.target.lower()

        matched = []
        for trace in traces:
            if target in str(trace).lower():
                matched.append(trace)

        summary = (
            f"Found {len(matched)} traces related to '{action.target}'."
            if matched else
            f"No traces matched '{action.target}'."
        )

        return EvidenceItem(
            source="traces",
            operation=action.operation,
            target=action.target,
            summary=summary,
            details={
                "match_count": len(matched),
                "matched_traces": matched[:10]
            }
        )