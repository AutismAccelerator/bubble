"""
llm — LLM provider abstraction layer.

Usage:
    from bubble.llm import get_llm

    llm = get_llm()
    segments  = await llm.decompose(message, prior)
    summary   = await llm.summarize(nodes)
    related   = await llm.relate(summary_a, summary_b)
    narrative = await llm.synthesize(ordered_lines)
"""

from .base import LLMClient
from .factory import get_llm

__all__ = ["LLMClient", "get_llm"]
