"""
Context manager for enabling/disabling batch-invariant mode.
"""

import threading
from contextlib import contextmanager
from typing import Generator

# Thread-local storage for batch invariant mode
_thread_local = threading.local()


def get_batch_invariant_mode() -> bool:
    """
    Get the current batch-invariant mode status.

    Returns:
        bool: True if batch-invariant mode is enabled, False otherwise.
    """
    return getattr(_thread_local, "batch_invariant_mode", False)


def _set_batch_invariant_mode(enabled: bool) -> None:
    """
    Internal function to set batch-invariant mode.

    Args:
        enabled: Whether to enable batch-invariant mode.
    """
    _thread_local.batch_invariant_mode = enabled


@contextmanager
def set_batch_invariant_mode(enabled: bool) -> Generator[None, None, None]:
    """
    Context manager for setting batch-invariant mode.

    When enabled, all operations will use batch-invariant implementations
    that guarantee deterministic results regardless of batch size.

    Args:
        enabled: Whether to enable batch-invariant mode.

    Example:
        >>> with set_batch_invariant_mode(True):
        ...     output = model(input_ids)  # Deterministic inference
    """
    previous_mode = get_batch_invariant_mode()
    _set_batch_invariant_mode(enabled)
    try:
        yield
    finally:
        _set_batch_invariant_mode(previous_mode)
