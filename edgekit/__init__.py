"""edgekit - edge deployment sizing for LLM workloads.

Reads real GGUF metadata and the real host, computes the real memory
breakdown, and says which lever to pull.
"""
from .gguf import GGUF, GGUFError, audit, compare, human, parse
from .hardware import Machine, probe, recommend_context, tokens_per_second

__version__ = "1.0.0"

__all__ = ["GGUF", "GGUFError", "audit", "compare", "parse", "human",
           "recommend_context", "Machine", "probe", "tokens_per_second"]
