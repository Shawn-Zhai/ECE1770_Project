import torch
import os

# "openai"  -> use OpenAI cloud API
BACKEND = os.getenv("AGENTIC_BACKEND", "openai")

OPENAI_MODEL_NAME = "gpt-5-mini"  
SPECIFIED_MODEL_NAME = ""

# Unified model name the rest of the code uses
LLM_DEFAULT_MODEL_NAME = OPENAI_MODEL_NAME if BACKEND == "openai" else SPECIFIED_MODEL_NAME

#setting specified model version for different agents
DIAGNOSIS_AGENT_MODEL_NAME = ""
STATE_REPORT_AGENT_MODEL_NAME = ""
EXPLANATION_GENERATION_AGENT_MODEL_NAME = ""
CLAIM_DECOMPOSITION_AGENT_MODEL_NAME = ""
VALIDATOR_AGENT_MODEL_NAME= ""


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
