"""
llm/anthropic.py — Anthropic (Claude) LLM provider.

Uses:
  - tool_use with forced tool_choice for structured decompose output
  - Standard messages.create for summarize / relate / synthesize
"""

from anthropic import AsyncAnthropic

from .. import config

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

_DECOMPOSE_SYSTEM = """\
Decompose a user message into atomic segments. Each segment must be \
self-contained.
Decompose when segments are independently meaningful beliefs; merge when one is
  sentiment, evaluation, or elaboration of the other.

For each segment output:
- text: the atomic statement
- intensity: 0.0–1.0  How much this moment will shape who this person is.
    Two dimensions, both required for a high score:

      Object significance — how personal and lasting is the object?
        High: identity, relationships, health, career, values, major decisions
        Low:  tools, tasks, routines, external events, other people's affairs

      Expression certainty — how committed is the speaker?
        High: firm assertions, explicit decisions, stated facts about oneself
        Low:  hedging ("I think", "maybe", "it seems"), perception verbs,
              conditional or hypothetical framing

    Bands:
      0.0–0.2  Trivial or purely factual. No personal stake.
      0.2–0.4  Passing reaction or hedged preference. Low commitment.
      0.4–0.6  Soft claim on a meaningful topic. Some personal weight.
      0.6–0.8  Explicit and clear stance with conviction. Or significant enough, worth memorizing immediately.
      0.8–1.0  Trajectory-defining — identity, deeply held beliefs, major life events.

- valence: pos | neg | neu
    pos: affirming, chosen, or wanted
    neg: rejecting, resented, or aversive
    neu: no clear stance

- reasoning: a one sentence reasoning for me to debug
    
Rules:
- Strip specific time references. Keep generalized frequency and scope qualifiers.
- If a <prior> block is provided, use it only as context, Speakers are prefixed with [Name]. 
- All pronouns, referents must be resolved.
- Call record_segments with your result.
- Return empty array if the message is purely functional, transient, or contain no personal signal — commands, greetings, filler, or factual queries.

Edge Cases:
Retraction: If the current message cancels or reverts a prior statement, invert the prior statement's meaning and inherit its intensity.
Example:
  prior: "I robbed a bank"
  message: "Just kidding"
  output:  { text: "I did not rob a bank", intensity: 0.6, valence: "pos"}

"""

_SUMMARIZE_SYSTEM = """\
You distill one or more user statements into a single memory record.

Rules:
- Capture the belief, preference, event, or tendency the statements express.
- When multiple statements are given, identify the common pattern they share.
- Write exactly one sentence with no grammatical subject.
- Start with a verb or descriptor that names the belief, event, or pattern.
- Do not explain, qualify, or ask for clarification.\
"""

_SNAPSHOT_SYSTEM = """\
A sequence of memory records about the user is listed below, from earliest to most recent.

Rules:
- The most recent memory takes precedence over earlier ones.
- Earlier memory provides historical context.
- Synthesize all memory into a single simplified coherent narrative that represents the full arc.
- Output one concise paragraph.
- No subject, start with verb.
- Do not explain or justify.\
"""


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
            system=_DECOMPOSE_SYSTEM,
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
            system=_SUMMARIZE_SYSTEM,
            messages=[{"role": "user", "content": texts}],
        )
        return response.content[0].text.strip()

    async def relate(self, summary_a: str, summary_b: str) -> bool:
        response = await self._client.messages.create(
            model=self._model,
            max_tokens=8,
            system="You are a memory topic classifier. Reply with exactly 'yes' or 'no'.",
            messages=[{
                "role": "user",
                "content": f"Are these two beliefs about the same topic or subject?\n\nA: {summary_a}\nB: {summary_b}",
            }],
        )
        return response.content[0].text.strip().lower().startswith("y")

    async def synthesize(self, ordered_lines: list[str]) -> str:
        response = await self._client.messages.create(
            model=self._model,
            max_tokens=256,
            system=_SNAPSHOT_SYSTEM,
            messages=[{"role": "user", "content": "\n\n".join(ordered_lines)}],
        )
        return response.content[0].text.strip()
