import json
from typing import List, Optional
from utils.llm_client import create_llm_client
from config.setting import LLM_DEFAULT_MODEL_NAME, BACKEND, CLAIM_DECOMPOSITION_AGENT_MODEL_NAME
from utils.logger import RunLogger
from utils.data_structures import ClaimStep

class Claim_decomposition_agent:
    def __init__(self):
        self.client = create_llm_client()
        self.model_name = LLM_DEFAULT_MODEL_NAME if CLAIM_DECOMPOSITION_AGENT_MODEL_NAME == ""  else CLAIM_DECOMPOSITION_AGENT_MODEL_NAME
        print(f"[ClaimDecompositionAgent] Backend={BACKEND}, model='{self.model_name}'")


    def decompose_claim(self,explanation: str,logger: Optional["RunLogger"] = None,) -> List["ClaimStep"]:

        """
        Decompose an RCA explanation into atomic claims.

        Expected JSON output:

        {
          "claims": [
            {
              "id": "claim1",
              "text": "CPU usage increased on node A",
              "type": "observation"
            },
            {
              "id": "claim2",
              "text": "The CPU spike caused request latency to increase",
              "type": "causal"
            }
          ]
        }
        """

        system_msg = (
            "You are a claim decomposition agent for root cause analysis (RCA).\n\n"
            "Your job is to break a complex explanation into atomic claims.\n\n"
            "Each claim should represent ONE verifiable statement.\n\n"
            "Types of claims may include:\n"
            "- observation (facts or metrics)\n"
            "- causal (cause-effect relationship)\n"
            "- inference (derived conclusion)\n\n"
            "You MUST respond with ONLY valid JSON in this format:\n\n"
            "{\n"
            "  \"claims\": [\n"
            "    {\n"
            "      \"id\": \"claim1\",\n"
            "      \"text\": \"<atomic claim>\",\n"
            "      \"type\": \"observation | causal | inference\"\n"
            "    }\n"
            "  ]\n"
            "}\n\n"
            "Guidelines:\n"
            "- Break the explanation into 3–10 atomic claims.\n"
            "- Each claim must be a single independent statement.\n"
            "- Avoid combining multiple facts into one claim.\n"
            "- Preserve the original meaning of the explanation."
        )

        user_msg = (
            "RCA Explanation:\n\n"
            f"{explanation}\n\n"
            "Return ONLY the JSON object described above."
        )

        resp = self.client.chat.completions.create(
            model=self.model_name,
            messages=[
                {"role": "system", "content": system_msg},
                {"role": "user", "content": user_msg},
            ],
        )

        content = resp.choices[0].message.content

        if logger is not None:
            logger.log_text("claim_decomposition_raw", content)

        try:
            data = json.loads(content)
            raw_claims = data.get("claims", [])

        except json.JSONDecodeError:
            print("[ClaimDecompositionAgent] JSON parsing failed. Using fallback.")
            raw_claims = [
                {
                    "id": "claim1",
                    "text": explanation,
                    "type": "inference"
                }
            ]

        claims: List["ClaimStep"] = []

        for claim in raw_claims:

            if not isinstance(claim, dict):
                continue

            cid = str(claim.get("id", f"claim{len(claims)+1}"))
            text = str(claim.get("text", ""))
            ctype = str(claim.get("type", "observation"))

            claims.append(
                ClaimStep(
                    id=cid,
                    text=text,
                    type=ctype
                )
            )

        if logger is not None:
            logger.log_json(
                "claim_decomposition_result",
                [c.__dict__ for c in claims]
            )

        return claims