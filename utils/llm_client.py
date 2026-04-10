# llm_client.py
import os
from types import SimpleNamespace
from typing import Any, Dict, List


class _ClaudeChatCompletionsAdapter:
    def __init__(self, client: Any, max_tokens: int) -> None:
        self._client = client
        self._max_tokens = max_tokens

    def create(self, *, model: str, messages: List[Dict[str, Any]], **kwargs: Any) -> Any:
        system_parts: List[str] = []
        anthropic_messages: List[Dict[str, Any]] = []

        for message in messages:
            role = str(message.get("role", "")).strip().lower()
            content = message.get("content", "")
            if isinstance(content, list):
                text_content = "\n".join(
                    str(block.get("text", ""))
                    for block in content
                    if isinstance(block, dict) and block.get("type") == "text"
                )
            else:
                text_content = str(content)

            if role == "system":
                if text_content.strip():
                    system_parts.append(text_content)
                continue

            if role not in {"user", "assistant"}:
                role = "user"

            anthropic_messages.append(
                {
                    "role": role,
                    "content": text_content,
                }
            )

        response = self._client.messages.create(
            model=model,
            system="\n\n".join(system_parts) if system_parts else None,
            messages=anthropic_messages,
            max_tokens=int(kwargs.get("max_tokens", self._max_tokens)),
            temperature=kwargs.get("temperature", 0.0),
        )

        text_chunks = [
            block.text
            for block in getattr(response, "content", [])
            if getattr(block, "type", "") == "text"
        ]
        content_text = "".join(text_chunks).strip()

        return SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(
                        content=content_text,
                    )
                )
            ]
        )


class _ClaudeClientAdapter:
    def __init__(self, client: Any, max_tokens: int) -> None:
        self.chat = SimpleNamespace(
            completions=_ClaudeChatCompletionsAdapter(client=client, max_tokens=max_tokens)
        )


def create_llm_client(backend: str = "openai", base_url: str = "") -> Any:
    """
    Create a client that talks either to:
      - the real OpenAI API (BACKEND = 'openai'), or
      - the Anthropic Claude API (BACKEND = 'anthropic'), or
      - any OpenAI-compatible endpoint (for example a local server or a hosted proxy).

    Usage stays identical in the rest of the code:
      client.chat.completions.create(...)
    """
    resolved_base_url = base_url or os.getenv("LLM_BASE_URL", "")
    resolved_api_key = (
        os.getenv("LLM_API_KEY")
        or os.getenv("ANTHROPIC_API_KEY")
        or os.getenv("OPENAI_API_KEY")
        or "EMPTY"
    )

    if backend == "openai":
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise ImportError(
                "OpenAI backend requires the 'openai' package. "
                "Install it in your environment with: pip install openai"
            ) from exc
        if resolved_base_url:
            return OpenAI(base_url=resolved_base_url, api_key=resolved_api_key)
        return OpenAI()

    if backend == "anthropic":
        try:
            from anthropic import Anthropic
        except ImportError as exc:
            raise ImportError(
                "Anthropic backend requires the 'anthropic' package. "
                "Install it in your environment with: pip install anthropic"
            ) from exc

        client_kwargs: Dict[str, Any] = {"api_key": resolved_api_key}
        if resolved_base_url:
            client_kwargs["base_url"] = resolved_base_url

        max_tokens = int(os.getenv("CLAUDE_MAX_TOKENS", "4096"))
        return _ClaudeClientAdapter(client=Anthropic(**client_kwargs), max_tokens=max_tokens)

    try:
        from openai import OpenAI
    except ImportError as exc:
        raise ImportError(
            "OpenAI-compatible backends require the 'openai' package. "
            "Install it in your environment with: pip install openai"
        ) from exc

    return OpenAI(
        base_url=resolved_base_url,
        api_key=resolved_api_key,
    )
