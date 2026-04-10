"""
llm/base.py — LLMClient protocol (structural subtyping).

Any provider class that implements these four async methods satisfies the protocol.
"""

from typing import Protocol, runtime_checkable


@runtime_checkable
class LLMClient(Protocol):
    """
    Common LLM operations used by the bubble pipeline.

    All methods are coroutines (async).
    """

    async def decompose(self, message: str, prior: str | None = None) -> list[dict]:
        """
        Decompose a user message into atomic segments.

        Returns list of dicts:
          {text: str, intensity: float (0-1), valence: "pos"|"neg"|"neu"}
        Returns [] for purely functional / no-signal messages.
        """
        ...

    async def summarize(self, nodes: list[dict]) -> str:
        """
        Distill one or more node raw_texts into a single memory record sentence.

        Each node dict must have a "raw_text" key.
        Returns a single sentence with no grammatical subject.
        """
        ...

    async def relate(self, summary_a: str, summary_b: str) -> bool:
        """
        Return True if the two summaries are about the same topic or subject.
        """
        ...

    async def synthesize(self, ordered_lines: list[str]) -> str:
        """
        Synthesize a sequence of memory records into a single coherent narrative.

        ordered_lines: labelled strings like "Belief 1 (earliest): ..."
        Returns a single paragraph with no subject.
        """
        ...
