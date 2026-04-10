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

_DECOMPOSE_SYSTEM = """\
Decompose a user message into atomic segments. Each segment must be self-contained.
Decompose when segments are independently meaningful beliefs; merge when one is
sentiment, evaluation, or elaboration of the other.

For each segment output:
- text: the atomic statement
- intensity: 0.0–1.0  How much this moment will shape who this person is.
    Object significance (identity/relationships/health/career = high; tools/tasks = low)
    Expression certainty (firm assertions = high; hedging/"I think"/"maybe" = low)
    Bands: 0.0–0.2 trivial | 0.2–0.4 passing | 0.4–0.6 soft claim | 0.6–0.8 clear stance | 0.8–1.0 trajectory-defining
- valence: pos | neg | neu
- reasoning: one sentence debug note

Rules:
- Strip specific time references. Keep generalized frequency/scope qualifiers.
- If a <prior> block is provided, use it only as context. Speakers are prefixed with [Name].
- Resolve all pronouns and referents.
- Return empty segments array for purely functional / no-signal messages.
- Retraction: if the current message cancels a prior statement, invert that statement and inherit its intensity.\
"""

_SUMMARIZE_SYSTEM = """\
You distill one or more user statements into a single memory record.
- Capture the belief, preference, event, or tendency the statements express.
- Write exactly one sentence with no grammatical subject.
- Start with a verb or descriptor that names the belief, event, or pattern.\
"""

_SNAPSHOT_SYSTEM = """\
A sequence of memory records about the user is listed below, from earliest to most recent.
- The most recent memory takes precedence over earlier ones.
- Synthesize all memory into a single simplified coherent narrative representing the full arc.
- Output one concise paragraph.
- No subject, start with verb.\
"""


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
            _DECOMPOSE_SYSTEM,
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
        model = self._model(_SUMMARIZE_SYSTEM, max_output_tokens=128)
        response = await model.generate_content_async(texts)
        return response.text.strip()

    async def relate(self, summary_a: str, summary_b: str) -> bool:
        model = self._model(
            "You are a memory topic classifier. Reply with exactly 'yes' or 'no'.",
            max_output_tokens=8,
        )
        prompt = f"Are these two beliefs about the same topic or subject?\n\nA: {summary_a}\nB: {summary_b}"
        response = await model.generate_content_async(prompt)
        return response.text.strip().lower().startswith("y")

    async def synthesize(self, ordered_lines: list[str]) -> str:
        model = self._model(_SNAPSHOT_SYSTEM, max_output_tokens=256)
        response = await model.generate_content_async("\n\n".join(ordered_lines))
        return response.text.strip()
