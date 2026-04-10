"""
llm/anthropic.py — Anthropic (Claude) LLM provider.

Uses:
  - tool_use with forced tool_choice for structured decompose output
  - Standard messages.create for summarize / relate / synthesize
"""

from anthropic import AsyncAnthropic

from .. import config
from . import prompts

_DECOMPOSE_TOOL = {
    "name": "record_segments",
    "description": "Record the decomposed segments from the user message.",
    "input_schema": {
        "type": "object",
        "properties": {
            "segments": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "text":      {"type": "string"},
                        "intensity": {"type": "number", "minimum": 0.0, "maximum": 1.0},
                        "valence":   {"type": "string", "enum": ["pos", "neg", "neu"]},
                        "reasoning": {"type": "string"},
                    },
                    "required": ["text", "intensity", "valence"],
                },
            }
        },
        "required": ["segments"],
    },
}




class AnthropicLLMClient:
    """LLM client backed by Anthropic's Claude API."""

    def __init__(self) -> None:
        self._client = AsyncAnthropic(api_key=config.ANTHROPIC_API_KEY)
        self._model = config.MODEL

    async def decompose(self, message: str, prior: str | None = None) -> list[dict]:
        user_content = f"<prior>\n{prior}\n</prior>\n\n{message}" if prior else message
        response = await self._client.messages.create(
            model=self._model,
            max_tokens=1024,
            temperature=0,
            system=prompts.DECOMPOSE_SYSTEM,
            thinking={"type": "disabled"},
            tools=[_DECOMPOSE_TOOL],
            tool_choice={"type": "tool", "name": "record_segments"},
            messages=[{"role": "user", "content": user_content}],
        )
        for block in response.content:
            if block.type == "tool_use" and block.name == "record_segments":
                return block.input["segments"]
        return []

    async def summarize(self, nodes: list[dict]) -> str:
        texts = "\n".join(f"- {n['raw_text']}" for n in nodes)
        response = await self._client.messages.create(
            model=self._model,
            max_tokens=128,
            system=prompts.SUMMARIZE_SYSTEM,
            messages=[{"role": "user", "content": texts}],
        )
        return response.content[0].text.strip()

    async def relate(self, summary_a: str, summary_b: str) -> bool:
        response = await self._client.messages.create(
            model=self._model,
            max_tokens=8,
            system=prompts.RELATE_SYSTEM,
            messages=[{
                "role": "user",
                "content": prompts.RELATE_USER.format(a=summary_a, b=summary_b),
            }],
        )
        return response.content[0].text.strip().lower().startswith("y")

    async def synthesize(self, ordered_lines: list[str]) -> str:
        response = await self._client.messages.create(
            model=self._model,
            max_tokens=256,
            system=prompts.SNAPSHOT_SYSTEM,
            messages=[{"role": "user", "content": "\n\n".join(ordered_lines)}],
        )
        return response.content[0].text.strip()
