"""
llm/gemini.py — Google Gemini LLM provider.

Configure via:
  GEMINI_API_KEY  — your Google AI API key
  BUBBLE_MODEL    — model name (e.g. gemini-2.0-flash, gemini-1.5-pro)

Structured output strategy:
  decompose → response_mime_type="application/json" with response_schema
  summarize / relate / synthesize → plain GenerateContent
"""

import json

import google.generativeai as genai
from google.generativeai import GenerativeModel
from google.generativeai.types import GenerationConfig

from .. import config
from . import prompts

_SEGMENT_SCHEMA = {
    "type": "object",
    "properties": {
        "segments": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "text":      {"type": "string"},
                    "intensity": {"type": "number"},
                    "valence":   {"type": "string", "enum": ["pos", "neg", "neu"]},
                    "reasoning": {"type": "string"},
                },
                "required": ["text", "intensity", "valence"],
            },
        }
    },
    "required": ["segments"],
}




class GeminiLLMClient:
    """LLM client backed by Google Gemini."""

    def __init__(self) -> None:
        genai.configure(api_key=config.GEMINI_API_KEY)
        self._model_name = config.MODEL

    def _model(self, system: str, **gen_kwargs) -> GenerativeModel:
        return GenerativeModel(
            model_name=self._model_name,
            system_instruction=system,
            generation_config=GenerationConfig(**gen_kwargs),
        )

    async def decompose(self, message: str, prior: str | None = None) -> list[dict]:
        user_content = f"<prior>\n{prior}\n</prior>\n\n{message}" if prior else message
        model = self._model(
            prompts.DECOMPOSE_SYSTEM,
            response_mime_type="application/json",
            response_schema=_SEGMENT_SCHEMA,
            temperature=0,
        )
        response = await model.generate_content_async(user_content)
        try:
            data = json.loads(response.text)
            return data.get("segments", [])
        except (json.JSONDecodeError, KeyError):
            return []

    async def summarize(self, nodes: list[dict]) -> str:
        texts = "\n".join(f"- {n['raw_text']}" for n in nodes)
        model = self._model(prompts.SUMMARIZE_SYSTEM, max_output_tokens=128)
        response = await model.generate_content_async(texts)
        return response.text.strip()

    async def relate(self, summary_a: str, summary_b: str) -> bool:
        model = self._model(
            prompts.RELATE_SYSTEM,
            max_output_tokens=8,
        )
        response = await model.generate_content_async(
            prompts.RELATE_USER.format(a=summary_a, b=summary_b)
        )
        return response.text.strip().lower().startswith("y")

    async def synthesize(self, ordered_lines: list[str]) -> str:
        model = self._model(prompts.SNAPSHOT_SYSTEM, max_output_tokens=256)
        response = await model.generate_content_async("\n\n".join(ordered_lines))
        return response.text.strip()
