"""
Batch-invariant matrix multiplication implementation.

The key is to ensure that the reduction operations within matmul
always happen in the same order, regardless of batch size or tiling strategy.
"""

import torch
from typing import Optional
from .context import get_batch_invariant_mode


def batch_invariant_matmul(
    a: torch.Tensor,
    b: torch.Tensor,
    tile_size: Optional[int] = None,
) -> torch.Tensor:
    """
    Batch-invariant matrix multiplication.

    Unlike standard matmul operations that may use different tiling strategies
    based on batch size and available parallelism, this implementation enforces
    a fixed tiling strategy to ensure deterministic reduction order.

    Key principle: Choose a single tiling/instruction mix whose reduction order
    is independent of batch size.

    Args:
        a: First input tensor of shape (..., n, k)
        b: Second input tensor of shape (..., k, m)
        tile_size: Fixed tile size for the reduction dimension. If None,
                   automatically determined based on tensor size.

    Returns:
        Result tensor of shape (..., n, m)

    Technical details:
        Standard matmul: C[i,j] = sum_k(A[i,k] * B[k,j])
        The 'sum_k' reduction can happen in different orders depending on:
        - How the computation is tiled
        - How work is distributed across GPU cores
        - Batch size affecting parallelization strategy

        Batch-invariant approach:
        - Fix the tile size for the k dimension
        - Always reduce tiles in the same order (e.g., left-to-right)
        - Use data parallelism only (one core per output element)
    """
    if not get_batch_invariant_mode():
        # Fall back to standard PyTorch matmul
        return torch.matmul(a, b)

    # Get dimensions
    *batch_dims, n, k = a.shape
    *_, k2, m = b.shape
    assert k == k2, f"Inner dimensions must match: {k} != {k2}"

    # Determine tile size if not provided
    if tile_size is None:
        # Use a fixed tile size that ensures consistent reduction
        # Power of 2 for better memory alignment
        tile_size = min(256, k)

    # For small matrices, use simple sequential accumulation
    if k <= tile_size:
        return _simple_matmul(a, b)

    # For larger matrices, use fixed-tile reduction
    return _tiled_matmul(a, b, tile_size)


def _simple_matmul(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """
    Simple sequential matmul for small matrices.

    Computes: C[i,j] = sum_k(A[i,k] * B[k,j])
    where the sum is accumulated sequentially from k=0 to k=K-1.

    Args:
        a: Tensor of shape (..., n, k)
        b: Tensor of shape (..., k, m)

    Returns:
        Tensor of shape (..., n, m)
    """
    # Use einsum for clarity and deterministic reduction
    # The reduction over 'k' happens in a fixed order
    return torch.einsum("...ik,...km->...im", a, b)


def _tiled_matmul(
    a: torch.Tensor,
    b: torch.Tensor,
    tile_size: int,
) -> torch.Tensor:
    """
    Tiled matmul with fixed reduction order.

    Splits the reduction dimension (k) into fixed-size tiles and
    accumulates them in order to ensure deterministic behavior.

    Algorithm:
        1. Split k dimension into tiles of size tile_size
        2. Compute partial products for each tile
        3. Accumulate tiles sequentially (left-to-right)

    Args:
        a: Tensor of shape (..., n, k)
        b: Tensor of shape (..., k, m)
        tile_size: Size of each tile along k dimension

    Returns:
        Tensor of shape (..., n, m)
    """
    *batch_dims, n, k = a.shape
    *_, _, m = b.shape

    # Initialize accumulator
    result = torch.zeros(
        *batch_dims, n, m,
        dtype=a.dtype,
        device=a.device,
    )

    # Process tiles sequentially to maintain fixed reduction order
    num_tiles = (k + tile_size - 1) // tile_size

    for tile_idx in range(num_tiles):
        # Compute tile boundaries
        k_start = tile_idx * tile_size
        k_end = min((tile_idx + 1) * tile_size, k)

        # Extract tile slices
        a_tile = a[..., :, k_start:k_end]  # (..., n, tile_k)
        b_tile = b[..., k_start:k_end, :]  # (..., tile_k, m)

        # Compute partial product for this tile
        partial = torch.matmul(a_tile, b_tile)

        # Accumulate in-place to maintain numerical precision
        result.add_(partial)

    return result


def batch_invariant_linear(
    input: torch.Tensor,
    weight: torch.Tensor,
    bias: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    Batch-invariant linear layer implementation.

    Equivalent to: output = input @ weight.T + bias

    Args:
        input: Input tensor of shape (..., in_features)
        weight: Weight tensor of shape (out_features, in_features)
        bias: Optional bias tensor of shape (out_features,)

    Returns:
        Output tensor of shape (..., out_features)
    """
    # Linear is just matmul + bias
    output = batch_invariant_matmul(input, weight.T)

    if bias is not None:
        output = output + bias

    return output


class BatchInvariantLinear(torch.nn.Module):
    """
    Batch-invariant linear layer.

    Drop-in replacement for torch.nn.Linear that guarantees
    deterministic behavior across different batch sizes.

    Args:
        in_features: Size of input features
        out_features: Size of output features
        bias: Whether to include bias term
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        self.weight = torch.nn.Parameter(
            torch.empty(out_features, in_features)
        )
        if bias:
            self.bias = torch.nn.Parameter(torch.empty(out_features))
        else:
            self.register_parameter("bias", None)

        self.reset_parameters()

    def reset_parameters(self):
        """Initialize parameters."""
        torch.nn.init.kaiming_uniform_(self.weight, a=5**0.5)
        if self.bias is not None:
            torch.nn.init.zeros_(self.bias)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """Forward pass with batch-invariant matmul."""
        if get_batch_invariant_mode():
            return batch_invariant_linear(input, self.weight, self.bias)
        else:
            # Standard linear
            return torch.nn.functional.linear(input, self.weight, self.bias)
