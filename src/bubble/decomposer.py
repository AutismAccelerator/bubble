import re
import logging
from ._shared import MODEL, _client
log=logging.getLogger(__name__)
# Paste detection thresholds
_PASTE_MIN_WORDS = 100
_PASTE_MAX_PRONOUN_RATIO = 0.02  # first-person pronouns / total words
_FIRST_PERSON = re.compile(
    r"\b(i|me|my|mine|myself|we|our|ours|ourselves)\b", re.IGNORECASE
)

_TOOL = {
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
                    },
                    "required": ["text", "intensity", "valence"],
                },
            }
        },
        "required": ["segments"],
    },
}

_SYSTEM = """\
Decompose a user message into atomic belief segments. Each segment must be \
self-contained. Pronouns, determiners and referents must be resolved(Use <prior> block as context).
Decompose when both halves are independently meaningful beliefs; merge when one is
  sentiment, evaluation, or elaboration of the other.
  
For each segment output:
- text: the atomic statement (preserve the user's own words closely)
- intensity: 0.0–1.0  personal significance — how much this will matter to this
    person's life over time, not how emotionally it is expressed.
    Two dimensions both contribute — neither alone is sufficient for a high score:
      Object significance: how personal and lasting is the subject?
        High: career, identity, health, relationships, major decisions
        Low:  tools, tasks, routine activities, other people, external events
      Expression certainty: how definitive is the claim?
        High: committed assertions, firm decisions, stated facts about the user
        Low:  epistemic uncertainty, perception verbs, hedged or conditional framing
    Certainty modifiers:
      Hedging → lower by 0.15–0.20
      Commitment → raise by 0.05–0.10
    0.0–0.2  impersonal, trivial, routine, or no lasting meaning
    0.2–0.4  transient or hedged; not a stable belief
    0.4–0.6  soft preference or hedged claim about a significant topic
    0.6–0.8  explicit, committed stance on something personally significant
    0.8–1.0  trajectory-defining — major life events or identity-defining passions

- valence: pos | neg | neu
    pos: affirming, chosen, or wanted
    neg: rejecting, resented, or aversive
    neu: factual with no clear stance
    Split when both pos and neg are present — do not blend into neu.

Rules:
- Strip specific time references. Keep generalized frequency and scope qualifiers.
- Statements about other people or external events → intensity ≤ 0.2 unless the
  impact is directly personal to the user.
- Transient reactions, hyperbole, venting, and gratitude → intensity ≤ 0.2.
- If a <prior> block is provided, use it only as context. 
- Call record_segments with your result.\

Edge Cases:
Cancelling: When the current message retracts a prior statement, measure intensity of original statement and use its opposite meaning as current message. example: <prior>I robbed a bank</prior> "Just kidding" -> I didn't rob any bank 0.6
"""



def _is_paste(text: str) -> bool:
    words = text.split()
    if len(words) < _PASTE_MIN_WORDS:
        return False
    pronoun_ratio = len(_FIRST_PERSON.findall(text)) / len(words)
    return pronoun_ratio < _PASTE_MAX_PRONOUN_RATIO


async def decompose(message: str, prior: str | None = None) -> list[dict]:
    """
    Decompose a user message into segments with intensity and valence.
    Returns list of {text: str, intensity: float, valence: pos|neg|neu}.
    Paste-detected messages are returned as a single near-zero-intensity segment.

    prior: optional conversational context the user is responding to
           (agent reply, group chat message, etc.).
    """
    if _is_paste(message):
        return [{"text": message, "intensity": 0.05, "valence": "neu"}]

    if prior:
        user_content = f"<prior>\n{prior}\n</prior>\n\n{message}"
    else:
        user_content = message
    messages = [{"role": "user", "content": user_content}]
    log.info("input:%s ",user_content)
    response = await _client.messages.create(
        model=MODEL,
        max_tokens=1024,
        temperature=0,
        system=_SYSTEM,
        tools=[_TOOL],
        tool_choice={"type": "tool", "name": "record_segments"},
        messages=messages,
    )

    for block in response.content:
        if block.type == "tool_use" and block.name == "record_segments":
            return block.input["segments"]

    return []
