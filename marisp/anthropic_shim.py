"""
anthropic_shim.py — a drop-in replacement for the `anthropic` SDK that routes
MARIS's calls to OpenRouter (or any OpenAI-compatible endpoint) instead.

WHY: MARIS's llm_modules.py does `from anthropic import Anthropic` and calls
`client.messages.create(model=, max_tokens=, system=, messages=[...])`, reading
`response.content[0].text`. This shim exposes the SAME interface but translates
to OpenRouter's OpenAI-format `chat.completions.create(...)`, so MARIS runs on
cheaper/free models WITHOUT editing llm_modules.py.

HOW TO USE (no edits to MARIS needed):
    1. pip install openai
    2. export OPENROUTER_API_KEY="sk-or-..."
    3. Optionally map model names (see MODEL_MAP below) via env or defaults.
    4. Make Python import THIS as `anthropic`. Two easy ways:
         a) Rename/symlink: put this file as `anthropic.py` NEXT TO llm_modules.py
            so it shadows the real package for that run, OR
         b) At the top of your entrypoint, before importing llm_modules:
                import anthropic_shim, sys; sys.modules["anthropic"] = anthropic_shim

This shim implements only what MARIS uses: Anthropic(), .messages.create(...),
and a response object with .content[0].text. That's all llm_modules.py touches.
"""

from __future__ import annotations
import os
from openai import OpenAI


# Map Anthropic model names (as used in llm_modules.py) -> OpenRouter model slugs.
# Defaults pick inexpensive/free-tier-friendly models; override via env if you like.
MODEL_MAP = {
    "claude-sonnet-4-6": os.environ.get(
        "MARISP_MODEL_REASONING", "anthropic/claude-3.5-sonnet"
    ),
    "claude-haiku-4-5-20251001": os.environ.get(
        "MARISP_MODEL_LIGHT", "anthropic/claude-3.5-haiku"
    ),
}
# Fallback for any unmapped model name.
DEFAULT_MODEL = os.environ.get("MARISP_MODEL_DEFAULT", "anthropic/claude-3.5-haiku")


class _TextBlock:
    """Mimics anthropic response.content[0] which has a .text attribute."""
    def __init__(self, text: str):
        self.text = text
        self.type = "text"


class _Response:
    """Mimics the anthropic Message response object MARIS reads from."""
    def __init__(self, text: str, usage=None):
        self.content = [_TextBlock(text)]
        self.usage = usage
        self.stop_reason = "end_turn"


class _Messages:
    def __init__(self, client: "Anthropic"):
        self._client = client

    def create(self, model: str, max_tokens: int = 1024,
               messages=None, system: str = None, **kwargs):
        """Translate Anthropic-style call -> OpenRouter chat.completions."""
        messages = messages or []

        # Anthropic passes `system` separately; OpenAI format wants it as the
        # first message with role "system".
        oai_messages = []
        if system:
            oai_messages.append({"role": "system", "content": system})
        for m in messages:
            # Anthropic message content can be a string (MARIS always uses strings).
            content = m.get("content", "")
            if not isinstance(content, str):
                # flatten any block list to text (defensive; MARIS uses strings)
                content = " ".join(
                    b.get("text", "") if isinstance(b, dict) else str(b)
                    for b in content
                )
            oai_messages.append({"role": m["role"], "content": content})

        target_model = MODEL_MAP.get(model, DEFAULT_MODEL)

        completion = self._client._oai.chat.completions.create(
            model=target_model,
            max_tokens=max_tokens,
            messages=oai_messages,
        )
        text = completion.choices[0].message.content or ""
        return _Response(text, usage=getattr(completion, "usage", None))


class Anthropic:
    """Drop-in stand-in for anthropic.Anthropic that talks to OpenRouter."""
    def __init__(self, api_key: str = None, base_url: str = None, **kwargs):
        key = (
            api_key
            or os.environ.get("OPENROUTER_API_KEY")
            or os.environ.get("ANTHROPIC_API_KEY")  # allow reuse if someone sets it
        )
        if not key:
            raise RuntimeError(
                "No OPENROUTER_API_KEY set. export OPENROUTER_API_KEY=sk-or-..."
            )
        url = base_url or os.environ.get(
            "OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1"
        )
        self._oai = OpenAI(api_key=key, base_url=url)
        self.messages = _Messages(self)


# Expose the same top-level names the real package would, minimally.
__all__ = ["Anthropic"]
