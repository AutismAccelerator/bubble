"""
llm/openai.py — OpenAI-compatible LLM provider.

Works with: OpenAI, DeepSeek, Groq, Together AI, Ollama, Mistral, etc.
Configure via:
  OPENAI_API_KEY      — your API key (or "ollama" for local)
  OPENAI_BASE_URL     — endpoint base URL (default: https://api.openai.com/v1)
  BUBBLE_MODEL        — model name (e.g. gpt-4o, deepseek-chat, llama3.2)

Structured output strategy:
  decompose → JSON mode (response_format=json_object) + strict system prompt
  summarize / relate / synthesize → plain chat completions
"""

import json
import re

from openai import AsyncOpenAI

from .. import config
from . import prompts



_JSON_FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)\s*```", re.DOTALL)


def _extract_json(text: str) -> dict:
    """Extract a JSON object from a model response, stripping markdown fences."""
    m = _JSON_FENCE_RE.search(text)
    raw = m.group(1) if m else text
    return json.loads(raw)


class OpenAILLMClient:
    """LLM client backed by any OpenAI-compatible API."""

    def __init__(self) -> None:
        self._client = AsyncOpenAI(
            api_key=config.OPENAI_API_KEY,
            base_url=config.OPENAI_BASE_URL,
        )
        self._model = config.MODEL

    async def decompose(self, message: str, prior: str | None = None) -> list[dict]:
        user_content = f"<prior>\n{prior}\n</prior>\n\n{message}" if prior else message
        # Append JSON-mode instruction to the shared prompt
        decompose_system = (
            prompts.DECOMPOSE_SYSTEM
            + '\nYou MUST respond with valid JSON only, in this exact shape:\n'
            + '{"segments": [{"text": "...", "intensity": 0.0, "valence": "pos|neg|neu", "reasoning": "..."}]}\n'
            + 'No markdown fences, no prose outside the JSON object.'
        )
        response = await self._client.chat.completions.create(
            model=self._model,
            temperature=0,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": decompose_system},
                {"role": "user", "content": user_content},
            ],
        )
        raw = response.choices[0].message.content or "{}"
        try:
            data = _extract_json(raw)
            return data.get("segments", [])
        except (json.JSONDecodeError, KeyError):
            return []

    async def summarize(self, nodes: list[dict]) -> str:
        texts = "\n".join(f"- {n['raw_text']}" for n in nodes)
        response = await self._client.chat.completions.create(
            model=self._model,
            max_tokens=128,
            messages=[
                {"role": "system", "content": prompts.SUMMARIZE_SYSTEM},
                {"role": "user", "content": texts},
            ],
        )
        return (response.choices[0].message.content or "").strip()

    async def relate(self, summary_a: str, summary_b: str) -> bool:
        response = await self._client.chat.completions.create(
            model=self._model,
            max_tokens=8,
            messages=[
                {"role": "system", "content": prompts.RELATE_SYSTEM},
                {"role": "user", "content": prompts.RELATE_USER.format(a=summary_a, b=summary_b)},
            ],
        )
        return (response.choices[0].message.content or "").strip().lower().startswith("y")

    async def synthesize(self, ordered_lines: list[str]) -> str:
        response = await self._client.chat.completions.create(
            model=self._model,
            max_tokens=256,
            messages=[
                {"role": "system", "content": prompts.SNAPSHOT_SYSTEM},
                {"role": "user", "content": "\n\n".join(ordered_lines)},
            ],
        )
        return (response.choices[0].message.content or "").strip()
