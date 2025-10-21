"""
Deterministic LLM Inference Library

Provides batch-invariant operations for achieving deterministic LLM inference.
"""

from .context import set_batch_invariant_mode, get_batch_invariant_mode
from .rmsnorm import batch_invariant_rmsnorm
from .matmul import batch_invariant_matmul
from .attention import batch_invariant_attention
from .layernorm import batch_invariant_layernorm

__version__ = "0.1.0"

__all__ = [
    "set_batch_invariant_mode",
    "get_batch_invariant_mode",
    "batch_invariant_rmsnorm",
    "batch_invariant_matmul",
    "batch_invariant_attention",
    "batch_invariant_layernorm",
]
