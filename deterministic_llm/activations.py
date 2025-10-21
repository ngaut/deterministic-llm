"""
Batch-invariant activation functions.

Activation functions can have batch-dependent behavior due to:
1. Vectorization strategies that change based on input size
2. Approximations that differ based on dimensions
3. Kernel selection based on tensor shapes

We implement batch-invariant versions that compute in a fixed manner.
"""

import torch
import torch.nn as nn
import math
from typing import Optional
from .context import get_batch_invariant_mode


def batch_invariant_gelu(x: torch.Tensor, approximate: str = 'none') -> torch.Tensor:
    """
    Batch-invariant GELU (Gaussian Error Linear Unit).

    GELU(x) = x * Φ(x) where Φ(x) is the cumulative distribution function
    of the standard Gaussian distribution.

    The key is to compute this in a way that doesn't depend on batch size.
    We use a consistent implementation that avoids PyTorch's internal
    kernel selection.

    Args:
        x: Input tensor
        approximate: Approximation method ('none' or 'tanh')

    Returns:
        GELU activation of input

    Reference:
        Gaussian Error Linear Units (GELUs) (Hendrycks & Gimpel, 2016)
        https://arxiv.org/abs/1606.08415
    """
    if not get_batch_invariant_mode():
        return torch.nn.functional.gelu(x, approximate=approximate)

    # Force float32 for numerical stability
    input_dtype = x.dtype
    x = x.float()

    if approximate == 'tanh':
        # Tanh approximation (used by GPT-2)
        # GELU(x) ≈ 0.5 * x * (1 + tanh(sqrt(2/π) * (x + 0.044715 * x^3)))
        sqrt_2_over_pi = math.sqrt(2.0 / math.pi)
        x_cubed = x * x * x
        inner = sqrt_2_over_pi * (x + 0.044715 * x_cubed)
        result = 0.5 * x * (1.0 + torch.tanh(inner))
    else:
        # Exact GELU using error function
        # GELU(x) = 0.5 * x * (1 + erf(x / sqrt(2)))
        sqrt_2 = math.sqrt(2.0)
        result = 0.5 * x * (1.0 + torch.erf(x / sqrt_2))

    return result.to(input_dtype)


def batch_invariant_relu(x: torch.Tensor) -> torch.Tensor:
    """
    Batch-invariant ReLU.

    ReLU is naturally batch-invariant (element-wise max(0, x)),
    but we provide this for completeness and to ensure consistent
    implementation.

    Args:
        x: Input tensor

    Returns:
        ReLU activation of input
    """
    # Use clamp for efficiency (no need to allocate zeros tensor)
    return torch.clamp(x, min=0.0)


def batch_invariant_silu(x: torch.Tensor, inplace: bool = False) -> torch.Tensor:
    """
    Batch-invariant SiLU/Swish activation.

    SiLU(x) = x * sigmoid(x)

    Used in modern architectures like LLaMA and Qwen.

    Args:
        x: Input tensor
        inplace: If True, modifies input in-place (ignored for determinism)

    Returns:
        SiLU activation of input
    """
    if not get_batch_invariant_mode():
        return torch.nn.functional.silu(x, inplace=inplace)

    # Note: We ignore inplace parameter for determinism
    # Force float32 for numerical stability
    input_dtype = x.dtype
    x = x.float()

    # SiLU(x) = x * σ(x) where σ is sigmoid
    # σ(x) = 1 / (1 + exp(-x))
    sigmoid_x = torch.sigmoid(x)
    result = x * sigmoid_x

    return result.to(input_dtype)


class BatchInvariantGELU(nn.Module):
    """
    Batch-invariant GELU activation module.

    Args:
        approximate: Approximation method ('none' or 'tanh')
    """

    def __init__(self, approximate: str = 'none'):
        super().__init__()
        self.approximate = approximate

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return batch_invariant_gelu(x, self.approximate)


class BatchInvariantReLU(nn.Module):
    """Batch-invariant ReLU activation module."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return batch_invariant_relu(x)


class BatchInvariantSiLU(nn.Module):
    """Batch-invariant SiLU/Swish activation module."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return batch_invariant_silu(x)
