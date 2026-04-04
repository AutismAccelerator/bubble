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
                        "reasoning":   {"type": "string"},
                    },
                    "required": ["text", "intensity", "valence"],
                },
            }
        },
        "required": ["segments"],
    },
}

_SYSTEM = """\
Decompose a user message into atomic segments. Each segment must be \
self-contained.
Decompose when segments are independently meaningful beliefs; merge when one is
  sentiment, evaluation, or elaboration of the other.
  
For each segment output:
- text: the atomic statement
- intensity: 0.0–1.0  personal significance — how much this will matter to this
    person's life.
    Two dimensions both contribute — neither alone is sufficient for a high score:
      Object significance: how personal and lasting is the subject?
        High: career, identity, health, relationships, major decisions
        Low:  tools, tasks, routines, unimportant people, external events
      Expression certainty: how definitive is the claim?
        High: committed assertions, firm decisions, stated facts about the user
        Low:  epistemic uncertainty, perception verbs, hedged or conditional framing
    Certainty modifiers:
      Hedging → lower by 0.10–0.15
      Commitment → raise by 0.05–0.10
    0.0–0.2  trivial, routine, or purely factual report. No personal stake.
    0.2–0.4  Transient, hedged, or passing reaction. Low commitment.
    0.4–0.6  Soft preference or mild claim on a meaningful topic. Some personal weight.
    0.6–0.8  Explicit, committed stance on something. Clear conviction.
    0.8–1.0  Trajectory-defining. Major life events, identity-defining passions, or deeply held beliefs.

- valence: pos | neg | neu
    pos: affirming, chosen, or wanted
    neg: rejecting, resented, or aversive
    neu: no clear stance

- reasoning: a one sentence reasoning for me to debug
    
Rules:
- Strip specific time references. Keep generalized frequency and scope qualifiers.
- If a <prior> block is provided, use it only as context. All pronouns, referents must be resolved.\
- Call record_segments with your result.\
- Return empty array if the message is purely functional, transient, or contain no personal signal — commands, greetings, filler, or factual queries.

Edge Cases:
Retraction: If the current message cancels or reverts a prior statement, invert the prior statement's meaning and inherit its intensity.
Example:
  prior: "I robbed a bank"
  message: "Just kidding"
  output:  { text: "I did not rob a bank", intensity: 0.6, valence: "pos"}


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
        thinking={"type": "disabled"},
        tools=[_TOOL],
        tool_choice={"type": "tool", "name": "record_segments"},
        messages=messages,
    )

    for block in response.content:
        if block.type == "tool_use" and block.name == "record_segments":
            return block.input["segments"]

    return []
