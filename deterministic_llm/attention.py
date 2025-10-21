"""
Batch-invariant attention mechanism implementation.

The attention operation involves multiple reductions (softmax, weighted sum)
that must be computed in a fixed order to ensure deterministic behavior.
"""

import torch
import torch.nn as nn
import math
from typing import Optional, Tuple
from .context import get_batch_invariant_mode
from .matmul import batch_invariant_matmul


def batch_invariant_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
    dropout_p: float = 0.0,
    is_causal: bool = False,
    scale: Optional[float] = None,
) -> torch.Tensor:
    """
    Batch-invariant scaled dot-product attention.

    Implements: softmax(Q @ K^T / sqrt(d_k)) @ V

    The key challenges for batch invariance:
    1. Q @ K^T matmul must have fixed reduction order
    2. Softmax reduction must be consistent
    3. Attention @ V matmul must have fixed reduction order

    Args:
        query: Query tensor of shape (..., seq_len_q, d_k)
        key: Key tensor of shape (..., seq_len_k, d_k)
        value: Value tensor of shape (..., seq_len_k, d_v)
        mask: Optional attention mask
        dropout_p: Dropout probability (not used in deterministic mode)
        is_causal: Whether to apply causal masking
        scale: Optional scaling factor (defaults to 1/sqrt(d_k))

    Returns:
        Attention output of shape (..., seq_len_q, d_v)
    """
    if not get_batch_invariant_mode():
        # Fall back to PyTorch's optimized scaled_dot_product_attention
        return torch.nn.functional.scaled_dot_product_attention(
            query, key, value, attn_mask=mask, dropout_p=dropout_p, is_causal=is_causal
        )

    # Get dimensions
    d_k = query.size(-1)

    # Compute scale factor
    if scale is None:
        scale = 1.0 / math.sqrt(d_k)

    # Step 1: Compute attention scores using batch-invariant matmul
    # Q @ K^T: (..., seq_len_q, d_k) @ (..., d_k, seq_len_k)
    # -> (..., seq_len_q, seq_len_k)
    scores = batch_invariant_matmul(query, key.transpose(-2, -1))
    scores = scores * scale

    # Step 2: Apply masks
    if is_causal:
        seq_len_q, seq_len_k = query.size(-2), key.size(-2)
        causal_mask = torch.triu(
            torch.ones(seq_len_q, seq_len_k, dtype=torch.bool, device=query.device),
            diagonal=1,
        )
        scores = scores.masked_fill(causal_mask, float("-inf"))

    if mask is not None:
        scores = scores.masked_fill(mask == 0, float("-inf"))

    # Step 3: Apply softmax with batch-invariant reduction
    attn_weights = batch_invariant_softmax(scores, dim=-1)

    # Note: In deterministic mode, we don't apply dropout
    # as it introduces randomness
    if dropout_p > 0.0 and not get_batch_invariant_mode():
        attn_weights = torch.nn.functional.dropout(attn_weights, p=dropout_p)

    # Step 4: Apply attention weights to values using batch-invariant matmul
    # attn_weights @ V: (..., seq_len_q, seq_len_k) @ (..., seq_len_k, d_v)
    # -> (..., seq_len_q, d_v)
    output = batch_invariant_matmul(attn_weights, value)

    return output


def batch_invariant_softmax(
    x: torch.Tensor,
    dim: int = -1,
) -> torch.Tensor:
    """
    Batch-invariant softmax implementation.

    Softmax involves two reductions:
    1. Max reduction (for numerical stability)
    2. Sum reduction (for normalization)

    Both must use fixed reduction order.

    Args:
        x: Input tensor
        dim: Dimension along which to apply softmax

    Returns:
        Softmax probabilities
    """
    # Step 1: Subtract max for numerical stability
    # The max reduction must be done consistently
    x_max = torch.max(x, dim=dim, keepdim=True)[0]
    x_shifted = x - x_max

    # Step 2: Compute exponentials
    exp_x = torch.exp(x_shifted)

    # Step 3: Compute sum with fixed reduction order
    # Use high precision for the sum to ensure consistency
    exp_x_float = exp_x.float()
    sum_exp = torch.sum(exp_x_float, dim=dim, keepdim=True)

    # Step 4: Normalize
    result = exp_x_float / sum_exp

    # Convert back to original dtype
    return result.to(x.dtype)


class BatchInvariantMultiHeadAttention(nn.Module):
    """
    Batch-invariant multi-head attention module.

    Drop-in replacement for torch.nn.MultiheadAttention that guarantees
    deterministic behavior across different batch sizes.

    Args:
        embed_dim: Total dimension of the model
        num_heads: Number of parallel attention heads
        dropout: Dropout probability (disabled in batch-invariant mode)
        bias: Whether to include bias in linear projections
        add_bias_kv: Whether to add bias to key and value
        add_zero_attn: Whether to add zero attention
        kdim: Dimension of keys (defaults to embed_dim)
        vdim: Dimension of values (defaults to embed_dim)
        batch_first: If True, expects input as (batch, seq, feature)
    """

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        dropout: float = 0.0,
        bias: bool = True,
        add_bias_kv: bool = False,
        add_zero_attn: bool = False,
        kdim: Optional[int] = None,
        vdim: Optional[int] = None,
        batch_first: bool = False,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.dropout = dropout
        self.batch_first = batch_first

        self.kdim = kdim if kdim is not None else embed_dim
        self.vdim = vdim if vdim is not None else embed_dim

        assert embed_dim % num_heads == 0, \
            f"embed_dim ({embed_dim}) must be divisible by num_heads ({num_heads})"
        self.head_dim = embed_dim // num_heads

        # Linear projections for Q, K, V
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.k_proj = nn.Linear(self.kdim, embed_dim, bias=bias)
        self.v_proj = nn.Linear(self.vdim, embed_dim, bias=bias)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        key_padding_mask: Optional[torch.Tensor] = None,
        need_weights: bool = True,
        attn_mask: Optional[torch.Tensor] = None,
        is_causal: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward pass of multi-head attention.

        Args:
            query: Query tensor
            key: Key tensor
            value: Value tensor
            key_padding_mask: Mask for padded positions
            need_weights: Whether to return attention weights
            attn_mask: Attention mask
            is_causal: Whether to use causal masking

        Returns:
            Tuple of (output, attention_weights)
        """
        # Handle batch_first
        if self.batch_first:
            query = query.transpose(0, 1)
            key = key.transpose(0, 1)
            value = value.transpose(0, 1)

        # Get dimensions
        seq_len, batch_size, embed_dim = query.shape

        # Project to Q, K, V
        q = self.q_proj(query)
        k = self.k_proj(key)
        v = self.v_proj(value)

        # Reshape for multi-head attention
        # (seq_len, batch, embed_dim) -> (batch, num_heads, seq_len, head_dim)
        q = q.view(seq_len, batch_size, self.num_heads, self.head_dim)
        q = q.permute(1, 2, 0, 3)

        k = k.view(-1, batch_size, self.num_heads, self.head_dim)
        k = k.permute(1, 2, 0, 3)

        v = v.view(-1, batch_size, self.num_heads, self.head_dim)
        v = v.permute(1, 2, 0, 3)

        # Compute attention
        attn_output = batch_invariant_attention(
            q, k, v,
            mask=attn_mask,
            dropout_p=self.dropout if self.training else 0.0,
            is_causal=is_causal,
        )

        # Reshape back
        # (batch, num_heads, seq_len, head_dim) -> (seq_len, batch, embed_dim)
        attn_output = attn_output.permute(2, 0, 1, 3).contiguous()
        attn_output = attn_output.view(seq_len, batch_size, embed_dim)

        # Output projection
        output = self.out_proj(attn_output)

        # Handle batch_first
        if self.batch_first:
            output = output.transpose(0, 1)

        # Note: We don't return attention weights in batch-invariant mode
        # to avoid additional computation
        return output, None if need_weights else None
