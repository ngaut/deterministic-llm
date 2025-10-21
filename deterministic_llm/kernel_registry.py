"""
PyTorch kernel replacement system using torch.library.

This module provides automatic replacement of standard PyTorch operations
with batch-invariant versions when batch-invariant mode is enabled.
"""

import torch
from typing import Optional, Callable, Dict, Any
import functools
import threading
from .context import get_batch_invariant_mode


# Global registry of original operations with thread-safe access
_original_ops: Dict[str, Callable] = {}
_registry_lock = threading.Lock()
_registered = False


def register_batch_invariant_ops():
    """
    Register all batch-invariant operations with PyTorch.

    This function patches PyTorch's standard operations to use batch-invariant
    versions when the batch-invariant mode context is active.

    Thread-safe: Multiple concurrent calls will only register once.
    """
    global _registered

    # Thread-safe registration - only register once
    with _registry_lock:
        if _registered:
            return  # Already registered, skip

        # Import here to avoid circular dependencies
        from .rmsnorm import batch_invariant_rmsnorm
        from .matmul import batch_invariant_matmul, batch_invariant_linear
        from .attention import batch_invariant_attention, batch_invariant_softmax
        from .layernorm import batch_invariant_layernorm
        from .activations import batch_invariant_gelu, batch_invariant_silu

        # Register RMSNorm replacement
        _register_op_replacement(
            "torch.nn.functional.rms_norm",
            batch_invariant_rmsnorm,
        )

        # Register LayerNorm replacement
        _register_op_replacement(
            "torch.nn.functional.layer_norm",
            _create_conditional_wrapper(
                torch.nn.functional.layer_norm,
                lambda input, normalized_shape, weight=None, bias=None, eps=1e-5:
                    batch_invariant_layernorm(input, normalized_shape, weight, bias, eps),
            ),
        )

        # Register matmul replacement
        _register_op_replacement(
            "torch.matmul",
            _create_conditional_wrapper(torch.matmul, batch_invariant_matmul),
        )

        # Register linear replacement
        _register_op_replacement(
            "torch.nn.functional.linear",
            _create_conditional_wrapper(
                torch.nn.functional.linear,
                lambda input, weight, bias=None: batch_invariant_linear(input, weight, bias),
            ),
        )

        # Register softmax replacement
        _register_op_replacement(
            "torch.nn.functional.softmax",
            _create_conditional_wrapper(
                torch.nn.functional.softmax,
                lambda x, dim=-1, dtype=None: batch_invariant_softmax(x, dim),
            ),
        )

        # Register GELU replacement
        _register_op_replacement(
            "torch.nn.functional.gelu",
            _create_conditional_wrapper(
                torch.nn.functional.gelu,
                lambda x, approximate='none': batch_invariant_gelu(x, approximate),
            ),
        )

        # Register SiLU replacement
        _register_op_replacement(
            "torch.nn.functional.silu",
            _create_conditional_wrapper(
                torch.nn.functional.silu,
                lambda x, inplace=False: batch_invariant_silu(x, inplace),
            ),
        )

        _registered = True


def unregister_batch_invariant_ops():
    """
    Restore original PyTorch operations.

    This function restores all patched operations to their original
    implementations.

    Thread-safe: Safe to call from multiple threads.
    """
    global _registered

    with _registry_lock:
        if not _registered:
            return  # Nothing to unregister

        for op_name, original_op in _original_ops.items():
            _restore_op(op_name, original_op)

        _original_ops.clear()
        _registered = False


def _register_op_replacement(
    op_path: str,
    replacement_fn: Callable,
):
    """
    Register a replacement for a PyTorch operation.

    NOTE: This function should only be called from within register_batch_invariant_ops()
    which already holds the registry lock.

    Args:
        op_path: Dot-separated path to the operation (e.g., "torch.matmul")
        replacement_fn: Function to use as replacement
    """
    # Parse the operation path
    parts = op_path.split(".")
    module_path = ".".join(parts[:-1])
    op_name = parts[-1]

    # Get the module containing the operation
    module = _get_module(module_path)
    if module is None:
        print(f"Warning: Could not find module {module_path}")
        return

    # Store the original operation (only if not already stored)
    if hasattr(module, op_name):
        if op_path not in _original_ops:
            # Only store the original function, not a previously patched version
            _original_ops[op_path] = getattr(module, op_name)

        # Replace with the new operation
        setattr(module, op_name, replacement_fn)
    else:
        print(f"Warning: Operation {op_name} not found in {module_path}")


def _restore_op(op_path: str, original_fn: Callable):
    """
    Restore an operation to its original implementation.

    Args:
        op_path: Dot-separated path to the operation
        original_fn: Original function to restore
    """
    parts = op_path.split(".")
    module_path = ".".join(parts[:-1])
    op_name = parts[-1]

    module = _get_module(module_path)
    if module is not None:
        setattr(module, op_name, original_fn)


def _get_module(module_path: str):
    """
    Get a module by its dot-separated path.

    Args:
        module_path: Path like "torch.nn.functional"

    Returns:
        The module, or None if not found
    """
    parts = module_path.split(".")
    module = None

    # Start with the root module
    try:
        module = __import__(parts[0])
        for part in parts[1:]:
            module = getattr(module, part)
        return module
    except (ImportError, AttributeError):
        return None


def _create_conditional_wrapper(original_fn: Callable, batch_invariant_fn: Callable) -> Callable:
    """
    Create a wrapper that conditionally uses batch-invariant implementation.

    Args:
        original_fn: Original PyTorch function
        batch_invariant_fn: Batch-invariant replacement

    Returns:
        Wrapped function that checks the mode and dispatches accordingly
    """
    @functools.wraps(original_fn)
    def wrapper(*args, **kwargs):
        if get_batch_invariant_mode():
            # Use batch-invariant implementation
            return batch_invariant_fn(*args, **kwargs)
        else:
            # Use original implementation
            return original_fn(*args, **kwargs)

    return wrapper


class KernelReplacementContext:
    """
    Context manager for automatic kernel replacement.

    IMPORTANT: This context manager registers operations on FIRST use and
    keeps them registered. It does NOT unregister on exit to avoid race
    conditions in multi-threaded scenarios.

    Once registered, operations remain patched for the lifetime of the program.
    This is safe because the patched functions check batch_invariant_mode flag
    and conditionally use batch-invariant versions only when the flag is True.

    Example:
        >>> with KernelReplacementContext():
        ...     with set_batch_invariant_mode(True):
        ...         output = model(input)  # Uses batch-invariant ops
    """

    def __enter__(self):
        # Register operations (idempotent - only happens once)
        register_batch_invariant_ops()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        # DO NOT unregister - would cause race conditions in multi-threaded use
        # The patched functions are conditional (check batch_invariant_mode flag)
        # so they don't affect code outside the context
        return False


# Utility function for monkey-patching torch modules
def patch_model_for_batch_invariance(model: torch.nn.Module) -> torch.nn.Module:
    """
    Patch a model to use batch-invariant operations.

    This function recursively replaces standard PyTorch layers with
    batch-invariant equivalents.

    Args:
        model: The model to patch

    Returns:
        The patched model (modified in-place)

    Example:
        >>> model = GPT2Model.from_pretrained("gpt2")
        >>> model = patch_model_for_batch_invariance(model)
        >>> with set_batch_invariant_mode(True):
        ...     output = model(input_ids)  # Deterministic!
    """
    from .rmsnorm import BatchInvariantRMSNorm
    from .matmul import BatchInvariantLinear
    from .attention import BatchInvariantMultiHeadAttention
    from .layernorm import BatchInvariantLayerNorm

    for name, module in model.named_children():
        # Recursively patch child modules
        patch_model_for_batch_invariance(module)

        # Replace specific layer types
        if isinstance(module, torch.nn.LayerNorm):
            # Replace LayerNorm with BatchInvariantLayerNorm
            new_module = BatchInvariantLayerNorm(
                module.normalized_shape,
                eps=module.eps,
                elementwise_affine=module.elementwise_affine,
            )
            # Copy weights if they exist
            if module.weight is not None:
                new_module.weight.data = module.weight.data.clone()
            if module.bias is not None:
                new_module.bias.data = module.bias.data.clone()

            setattr(model, name, new_module)

        elif isinstance(module, torch.nn.Linear):
            # Replace Linear with BatchInvariantLinear
            new_module = BatchInvariantLinear(
                module.in_features,
                module.out_features,
                bias=module.bias is not None,
            )
            # Copy weights
            new_module.weight.data = module.weight.data.clone()
            if module.bias is not None:
                new_module.bias.data = module.bias.data.clone()

            setattr(model, name, new_module)

        elif isinstance(module, torch.nn.MultiheadAttention):
            # Replace MultiheadAttention with BatchInvariantMultiHeadAttention
            new_module = BatchInvariantMultiHeadAttention(
                module.embed_dim,
                module.num_heads,
                dropout=module.dropout,
                bias=True,  # Assume bias is present
                batch_first=module.batch_first,
            )
            # Note: Weight copying for attention is more complex
            # May need to manually copy q_proj, k_proj, v_proj, out_proj weights

            setattr(model, name, new_module)

    return model
