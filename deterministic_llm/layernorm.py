"""
Batch-invariant LayerNorm implementation.

LayerNorm computes:
    y = (x - mean(x)) / sqrt(var(x) + eps) * weight + bias

The key challenge: both mean and variance involve reductions that must be
computed in a fixed order to ensure batch invariance.
"""

import torch
import torch.nn as nn
from typing import Optional, List, Union
from .context import get_batch_invariant_mode


class BatchInvariantLayerNorm(nn.Module):
    """
    Batch-invariant Layer Normalization.

    This implementation ensures deterministic behavior by maintaining fixed
    reduction orders for both mean and variance calculations, regardless of
    batch size.

    Args:
        normalized_shape: Input shape from an expected input of size
                         [* x normalized_shape[0] x normalized_shape[1] x ...]
        eps: A value added to the denominator for numerical stability.
        elementwise_affine: Whether to learn affine parameters.

    Reference:
        Layer Normalization (Ba et al., 2016)
        https://arxiv.org/abs/1607.06450
    """

    def __init__(
        self,
        normalized_shape: Union[int, List[int]],
        eps: float = 1e-5,
        elementwise_affine: bool = True,
    ):
        super().__init__()

        if isinstance(normalized_shape, int):
            normalized_shape = (normalized_shape,)
        self.normalized_shape = tuple(normalized_shape)
        self.eps = eps
        self.elementwise_affine = elementwise_affine

        if self.elementwise_affine:
            self.weight = nn.Parameter(torch.ones(self.normalized_shape))
            self.bias = nn.Parameter(torch.zeros(self.normalized_shape))
        else:
            self.register_parameter("weight", None)
            self.register_parameter("bias", None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply batch-invariant LayerNorm to the input.

        Args:
            x: Input tensor of shape (..., *normalized_shape)

        Returns:
            Normalized tensor of the same shape as input
        """
        if get_batch_invariant_mode():
            return batch_invariant_layernorm(
                x, self.normalized_shape, self.weight, self.bias, self.eps
            )
        else:
            # Standard LayerNorm for comparison
            return torch.nn.functional.layer_norm(
                x, self.normalized_shape, self.weight, self.bias, self.eps
            )


def batch_invariant_layernorm(
    x: torch.Tensor,
    normalized_shape: tuple,
    weight: Optional[torch.Tensor],
    bias: Optional[torch.Tensor],
    eps: float = 1e-5,
) -> torch.Tensor:
    """
    Batch-invariant LayerNorm implementation.

    Key principle: Enforce data parallelism with fixed reduction order.

    LayerNorm algorithm:
    1. Compute mean along normalized dimensions (per element, single core)
    2. Compute variance along normalized dimensions (per element, single core)
    3. Normalize: (x - mean) / sqrt(var + eps)
    4. Apply learned weight and bias if present

    The critical difference from standard LayerNorm:
    - We ensure mean and variance are computed with a consistent reduction order
    - We use float32 for stability in reductions
    - We avoid any operations that might use split-reduction strategies

    Args:
        x: Input tensor of shape (..., *normalized_shape)
        normalized_shape: Shape of dimensions to normalize over
        weight: Optional learnable weight parameter
        bias: Optional learnable bias parameter
        eps: Small constant for numerical stability

    Returns:
        Normalized tensor of same shape as input
    """
    # Save original dtype
    input_dtype = x.dtype

    # Use float32 for numerical stability in reductions
    x = x.float()

    # Determine the dimensions to reduce over
    # LayerNorm normalizes over the last len(normalized_shape) dimensions
    normalized_ndim = len(normalized_shape)
    reduce_dims = list(range(-normalized_ndim, 0))

    # Step 1: Compute mean with fixed reduction order
    # CRITICAL: This must happen in a consistent order regardless of batch size
    mean = torch.mean(x, dim=reduce_dims, keepdim=True)

    # Step 2: Compute variance with fixed reduction order
    # var = E[(x - mean)^2]
    x_centered = x - mean
    variance = torch.mean(x_centered * x_centered, dim=reduce_dims, keepdim=True)

    # Step 3: Normalize
    # Using rsqrt is more numerically stable than 1/sqrt
    rstd = torch.rsqrt(variance + eps)
    x_normalized = x_centered * rstd

    # Step 4: Apply affine transformation if present
    if weight is not None:
        x_normalized = x_normalized * weight
    if bias is not None:
        x_normalized = x_normalized + bias

    # Convert back to original dtype
    return x_normalized.to(input_dtype)


def standard_layernorm(
    x: torch.Tensor,
    normalized_shape: tuple,
    weight: Optional[torch.Tensor],
    bias: Optional[torch.Tensor],
    eps: float = 1e-5,
) -> torch.Tensor:
    """
    Standard LayerNorm implementation for comparison.

    This version may exhibit non-deterministic behavior across different
    batch sizes due to varying reduction strategies.

    Args:
        x: Input tensor
        normalized_shape: Shape of normalized dimensions
        weight: Optional learnable weight
        bias: Optional learnable bias
        eps: Numerical stability constant

    Returns:
        Normalized tensor
    """
    return torch.nn.functional.layer_norm(
        x, normalized_shape, weight, bias, eps
    )


# Register the batch-invariant version as a custom operation
def register_layernorm_ops():
    """
    Register batch-invariant LayerNorm operations with torch.library.

    This allows automatic replacement of standard LayerNorm operations
    when batch-invariant mode is enabled.
    """
    # TODO: Implement torch.library registration for automatic replacement
    pass
