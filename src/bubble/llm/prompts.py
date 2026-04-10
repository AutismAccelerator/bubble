"""
llm/prompts.py — Canonical system prompts shared by all LLM providers.

Edit here; the three provider modules import directly from this file.
"""

# ---------------------------------------------------------------------------
# decompose
# ---------------------------------------------------------------------------

DECOMPOSE_SYSTEM = """\
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
- If a <prior> block is provided, use it only as context. Speakers are prefixed with [Name].
- All pronouns, referents must be resolved.
- Return empty array if the message is purely functional, transient, or contains no personal signal — commands, greetings, filler, or factual queries.

Edge Cases:
Retraction: If the current message cancels or reverts a prior statement, invert the prior statement's meaning and inherit its intensity.
Example:
  prior: "I robbed a bank"
  message: "Just kidding"
  output:  { text: "I did not rob a bank", intensity: 0.6, valence: "pos"}

"""

# ---------------------------------------------------------------------------
# summarize
# ---------------------------------------------------------------------------

SUMMARIZE_SYSTEM = """\
You distill one or more user statements into a single memory record.

Rules:
- Capture the belief, preference, event, or tendency the statements express.
- When multiple statements are given, identify the common pattern they share.
- Write exactly one sentence with no grammatical subject.
- Start with a verb or descriptor that names the belief, event, or pattern.
- Do not explain, qualify, or ask for clarification.\
"""

# ---------------------------------------------------------------------------
# synthesize (snapshot)
# ---------------------------------------------------------------------------

SNAPSHOT_SYSTEM = """\
A sequence of memory records about the user is listed below, from earliest to most recent.

Rules:
- The most recent memory takes precedence over earlier ones.
- Earlier memory provides historical context.
- Synthesize all memory into a single simplified coherent narrative that represents the full arc.
- Output one concise paragraph.
- No subject, start with verb.
- Do not explain or justify.\
"""

# ---------------------------------------------------------------------------
# relate
# ---------------------------------------------------------------------------

RELATE_SYSTEM = "You are a memory topic classifier. Reply with exactly 'yes' or 'no'."

RELATE_USER = "Are these two beliefs about the same topic or subject?\n\nA: {a}\nB: {b}"
