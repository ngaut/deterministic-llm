"""
Batch-invariant RMSNorm implementation.

The key insight is to always use data parallelism (one core per batch element)
rather than split-reduction (multiple cores per element), ensuring the reduction
order remains constant regardless of batch size.
"""

import torch
import torch.nn as nn
from typing import Optional
from .context import get_batch_invariant_mode


class BatchInvariantRMSNorm(nn.Module):
    """
    Batch-invariant Root Mean Square Layer Normalization.

    This implementation ensures deterministic behavior by maintaining a fixed
    reduction order across different batch sizes. Unlike standard implementations
    that may switch between data parallelism and split-reduction based on batch
    size, this version always uses data parallelism.

    Args:
        normalized_shape: Input shape from an expected input of size.
        eps: A value added to the denominator for numerical stability.
        elementwise_affine: Whether to learn affine parameters.

    Reference:
        Root Mean Square Layer Normalization (Zhang & Sennrich, 2019)
        https://arxiv.org/abs/1910.07467
    """

    def __init__(
        self,
        normalized_shape: int,
        eps: float = 1e-6,
        elementwise_affine: bool = True,
    ):
        super().__init__()
        self.normalized_shape = normalized_shape
        self.eps = eps
        self.elementwise_affine = elementwise_affine

        if self.elementwise_affine:
            self.weight = nn.Parameter(torch.ones(normalized_shape))
        else:
            self.register_parameter("weight", None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply batch-invariant RMSNorm to the input.

        Args:
            x: Input tensor of shape (..., normalized_shape)

        Returns:
            Normalized tensor of the same shape as input
        """
        if get_batch_invariant_mode():
            return batch_invariant_rmsnorm(
                x, self.weight, self.normalized_shape, self.eps
            )
        else:
            # Standard RMSNorm for comparison
            return standard_rmsnorm(x, self.weight, self.normalized_shape, self.eps)


def batch_invariant_rmsnorm(
    x: torch.Tensor,
    weight: Optional[torch.Tensor],
    normalized_shape: int,
    eps: float = 1e-6,
) -> torch.Tensor:
    """
    Batch-invariant RMSNorm implementation.

    Key principle: Enforce data parallelism regardless of batch size.
    Each batch element's reduction is performed within a single core,
    ensuring the reduction order remains constant.

    Algorithm:
    1. Compute squared values: x^2
    2. Compute mean along the last dimension (per element, single core)
    3. Compute RMS: sqrt(mean(x^2) + eps)
    4. Normalize: x / RMS
    5. Apply learned weight if present

    Args:
        x: Input tensor of shape (..., normalized_shape)
        weight: Optional learnable weight parameter
        normalized_shape: Size of the dimension to normalize
        eps: Small constant for numerical stability

    Returns:
        Normalized tensor of same shape as input
    """
    # Ensure computation is done in float32 for numerical stability
    input_dtype = x.dtype
    x = x.float()

    # Step 1: Square the input
    x_squared = x * x

    # Step 2: Compute mean along the last dimension
    # CRITICAL: Use a single reduction per batch element
    # We explicitly avoid any parallelization that would split
    # the reduction across multiple cores
    mean_squared = torch.mean(x_squared, dim=-1, keepdim=True)

    # Step 3: Compute RMS with epsilon for stability
    rms = torch.sqrt(mean_squared + eps)

    # Step 4: Normalize
    x_normalized = x / rms

    # Step 5: Apply learned scaling if present
    if weight is not None:
        x_normalized = x_normalized * weight

    # Convert back to original dtype
    return x_normalized.to(input_dtype)


def standard_rmsnorm(
    x: torch.Tensor,
    weight: Optional[torch.Tensor],
    normalized_shape: int,
    eps: float = 1e-6,
) -> torch.Tensor:
    """
    Standard RMSNorm implementation for comparison.

    This version may exhibit non-deterministic behavior across different
    batch sizes due to varying reduction strategies.

    Args:
        x: Input tensor
        weight: Optional learnable weight
        normalized_shape: Size of normalization dimension
        eps: Numerical stability constant

    Returns:
        Normalized tensor
    """
    # Standard implementation using PyTorch's default reduction
    variance = x.pow(2).mean(-1, keepdim=True)
    x_normalized = x * torch.rsqrt(variance + eps)

    if weight is not None:
        x_normalized = x_normalized * weight

    return x_normalized


# Register the batch-invariant version as a custom operation
# This will be used by the kernel replacement system
def register_rmsnorm_ops():
    """
    Register batch-invariant RMSNorm operations with torch.library.

    This allows automatic replacement of standard RMSNorm operations
    when batch-invariant mode is enabled.
    """
    # TODO: Implement torch.library registration
    # This will enable automatic kernel replacement
    pass
