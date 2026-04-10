import os

try:
    import torch
except Exception:
    torch = None

# "openai"      -> use OpenAI cloud API
# "anthropic"   -> use Anthropic Claude API
# anything else -> use an OpenAI-compatible endpoint via LLM_BASE_URL
BACKEND = os.getenv("AGENTIC_BACKEND", "openai")

OPENAI_MODEL_NAME = os.getenv("OPENAI_MODEL_NAME", "gpt-5.4")
ANTHROPIC_MODEL_NAME = os.getenv("ANTHROPIC_MODEL_NAME", "claude-opus-4-1")
SPECIFIED_MODEL_NAME = os.getenv("SPECIFIED_MODEL_NAME", os.getenv("LLM_MODEL_NAME", ""))
LLM_BASE_URL = os.getenv("LLM_BASE_URL", os.getenv("ANTHROPIC_BASE_URL", ""))
LLM_API_KEY = os.getenv(
    "LLM_API_KEY",
    os.getenv("ANTHROPIC_API_KEY", os.getenv("OPENAI_API_KEY", "")),
)
CLAUDE_MAX_TOKENS = int(os.getenv("CLAUDE_MAX_TOKENS", "4096"))

# Unified model name the rest of the code uses
if BACKEND == "openai":
    LLM_DEFAULT_MODEL_NAME = OPENAI_MODEL_NAME
elif BACKEND == "anthropic":
    LLM_DEFAULT_MODEL_NAME = SPECIFIED_MODEL_NAME or ANTHROPIC_MODEL_NAME
else:
    LLM_DEFAULT_MODEL_NAME = SPECIFIED_MODEL_NAME or OPENAI_MODEL_NAME

#setting specified model version for different agents
DIAGNOSIS_AGENT_MODEL_NAME = os.getenv("DIAGNOSIS_AGENT_MODEL_NAME", "")
STATE_REPORT_AGENT_MODEL_NAME = os.getenv("STATE_REPORT_AGENT_MODEL_NAME", "")
EXPLANATION_GENERATION_AGENT_MODEL_NAME = os.getenv("EXPLANATION_GENERATION_AGENT_MODEL_NAME", "")
CLAIM_DECOMPOSITION_AGENT_MODEL_NAME = os.getenv("CLAIM_DECOMPOSITION_AGENT_MODEL_NAME", "")
VALIDATOR_AGENT_MODEL_NAME = os.getenv("VALIDATOR_AGENT_MODEL_NAME", "")


DEVICE = "cuda" if (torch is not None and torch.cuda.is_available()) else "cpu"
