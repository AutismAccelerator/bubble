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

_DECOMPOSE_SYSTEM = """\
Decompose a user message into atomic segments. Each segment must be self-contained.
Decompose when segments are independently meaningful beliefs; merge when one is
sentiment, evaluation, or elaboration of the other.

For each segment output:
- text: the atomic statement
- intensity: 0.0–1.0  How much this moment will shape who this person is.
    Object significance (identity/relationships/health/career = high; tools/tasks/events = low)
    Expression certainty (firm assertions = high; hedging/"I think"/"maybe" = low)
    Bands: 0.0–0.2 trivial | 0.2–0.4 passing | 0.4–0.6 soft claim | 0.6–0.8 clear stance | 0.8–1.0 trajectory-defining
- valence: pos | neg | neu
- reasoning: one sentence debug note

Rules:
- Strip specific time references. Keep generalized frequency and scope qualifiers.
- If a <prior> block is provided, use it only as context. Speakers are prefixed with [Name].
- Resolve all pronouns and referents.
- Return empty segments array if the message is purely functional, transient, or contains no personal signal.
- Retraction: if the current message cancels a prior statement, invert the prior statement and inherit its intensity.

You MUST respond with valid JSON only, in this exact shape:
{"segments": [{"text": "...", "intensity": 0.0, "valence": "pos|neg|neu", "reasoning": "..."}]}
No markdown fences, no prose outside the JSON object.\
"""

_SUMMARIZE_SYSTEM = """\
You distill one or more user statements into a single memory record.
- Capture the belief, preference, event, or tendency the statements express.
- When multiple statements are given, identify the common pattern they share.
- Write exactly one sentence with no grammatical subject.
- Start with a verb or descriptor that names the belief, event, or pattern.
- Do not explain, qualify, or ask for clarification.\
"""

_SNAPSHOT_SYSTEM = """\
A sequence of memory records about the user is listed below, from earliest to most recent.
- The most recent memory takes precedence over earlier ones.
- Earlier memory provides historical context.
- Synthesize all memory into a single simplified coherent narrative that represents the full arc.
- Output one concise paragraph.
- No subject, start with verb.
- Do not explain or justify.\
"""

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
        response = await self._client.chat.completions.create(
            model=self._model,
            temperature=0,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": _DECOMPOSE_SYSTEM},
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
                {"role": "system", "content": _SUMMARIZE_SYSTEM},
                {"role": "user", "content": texts},
            ],
        )
        return (response.choices[0].message.content or "").strip()

    async def relate(self, summary_a: str, summary_b: str) -> bool:
        response = await self._client.chat.completions.create(
            model=self._model,
            max_tokens=8,
            messages=[
                {"role": "system", "content": "You are a memory topic classifier. Reply with exactly 'yes' or 'no'."},
                {"role": "user", "content": f"Are these two beliefs about the same topic or subject?\n\nA: {summary_a}\nB: {summary_b}"},
            ],
        )
        return (response.choices[0].message.content or "").strip().lower().startswith("y")

    async def synthesize(self, ordered_lines: list[str]) -> str:
        response = await self._client.chat.completions.create(
            model=self._model,
            max_tokens=256,
            messages=[
                {"role": "system", "content": _SNAPSHOT_SYSTEM},
                {"role": "user", "content": "\n\n".join(ordered_lines)},
            ],
        )
        return (response.choices[0].message.content or "").strip()
