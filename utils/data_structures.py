from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

@dataclass
class ClaimStep:
    id: str
    text: str
    type: str  # observation / causal / inference
    
    
@dataclass
class ValidationAction:
    source: str              # "logs" | "metrics" | "traces" | "finish"
    operation: str           # e.g. "find_error_patterns", "check_spike", "trace_latency_path"
    target: str              # service / metric / keyword / endpoint / trace field
    reason: str              # why controller chose this action


@dataclass
class EvidenceItem:
    source: str              # logs / metrics / traces
    operation: str
    target: str
    summary: str
    details: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ValidationResult:
    claim_id: str
    claim_text: str
    verdict: str             # supported / contradicted / insufficient
    confidence: float
    evidence: List[EvidenceItem]
    reason: str