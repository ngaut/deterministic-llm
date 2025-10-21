"""
High-level API for deterministic inference.

Provides easy-to-use wrappers for running deterministic inference with any model.
"""

import torch
import torch.nn as nn
from typing import Optional, Dict, Any, Union, List
from contextlib import contextmanager
from .context import set_batch_invariant_mode
from .kernel_registry import patch_model_for_batch_invariance, KernelReplacementContext


class DeterministicInferenceEngine:
    """
    High-level engine for deterministic LLM inference.

    This class provides a simple interface for running deterministic inference
    with any PyTorch model. It automatically handles:
    - Enabling batch-invariant mode
    - Patching model operations
    - Managing kernel replacements

    Example:
        >>> from transformers import GPT2LMHeadModel, GPT2Tokenizer
        >>> model = GPT2LMHeadModel.from_pretrained("gpt2")
        >>> tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
        >>>
        >>> engine = DeterministicInferenceEngine(model)
        >>> prompt = "The quick brown fox"
        >>> output = engine.generate(prompt, tokenizer, max_length=50)
    """

    def __init__(
        self,
        model: nn.Module,
        patch_model: bool = True,
        device: Optional[str] = None,
    ):
        """
        Initialize the deterministic inference engine.

        Args:
            model: The PyTorch model to use for inference
            patch_model: Whether to automatically patch the model with
                        batch-invariant operations (default: True)
            device: Device to run inference on (e.g., "cuda", "cpu")
        """
        self.model = model

        # Determine device
        if device is not None:
            self.model = self.model.to(device)
            self.device = torch.device(device)
        else:
            # Try to get device from model parameters
            try:
                self.device = next(model.parameters()).device
            except StopIteration:
                # Model has no parameters, default to CPU
                self.device = torch.device('cpu')

        # Patch the model if requested
        if patch_model:
            self.model = patch_model_for_batch_invariance(self.model)

        # Set model to eval mode
        self.model.eval()

    @torch.no_grad()
    def forward(
        self,
        input_ids: torch.Tensor,
        **kwargs
    ) -> torch.Tensor:
        """
        Run a forward pass with deterministic operations.

        Args:
            input_ids: Input token IDs
            **kwargs: Additional arguments to pass to the model

        Returns:
            Model output (logits or hidden states)
        """
        with set_batch_invariant_mode(True):
            with KernelReplacementContext():
                outputs = self.model(input_ids, **kwargs)
                return outputs

    @torch.no_grad()
    def generate(
        self,
        input_ids: Union[torch.Tensor, str],
        tokenizer=None,
        max_length: int = 50,
        temperature: float = 0.0,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
        eos_token_id: Optional[int] = None,
        **kwargs
    ) -> Union[torch.Tensor, str]:
        """
        Generate text deterministically.

        Args:
            input_ids: Input token IDs or text prompt (if tokenizer provided)
            tokenizer: Optional tokenizer for encoding/decoding text
            max_length: Maximum length of generated sequence
            temperature: Sampling temperature (use 0.0 for greedy/deterministic)
            top_k: Top-k sampling parameter
            top_p: Top-p (nucleus) sampling parameter
            eos_token_id: EOS token ID for early stopping (auto-detected from tokenizer if not provided)
            **kwargs: Additional arguments

        Returns:
            Generated token IDs or decoded text (if tokenizer provided)

        Raises:
            ValueError: If temperature != 0.0 (non-deterministic sampling not supported)
        """
        # Validate determinism requirements
        if temperature != 0.0:
            raise ValueError(
                f"Non-deterministic generation not supported. "
                f"temperature must be 0.0 for greedy (deterministic) generation. "
                f"Got temperature={temperature}. "
                f"Sampling-based generation is inherently non-deterministic."
            )

        if top_k is not None or top_p is not None:
            raise ValueError(
                "top_k and top_p sampling are not deterministic. "
                "For deterministic generation, use temperature=0.0 with no sampling parameters."
            )

        # Handle text input
        if isinstance(input_ids, str):
            if tokenizer is None:
                raise ValueError("Tokenizer required for text input")
            input_text = input_ids
            input_ids = tokenizer.encode(input_text, return_tensors="pt")
            input_ids = input_ids.to(self.device)
            return_text = True
        else:
            return_text = False
            if input_ids.device != self.device:
                input_ids = input_ids.to(self.device)

        # Auto-detect EOS token from tokenizer if not provided
        if eos_token_id is None and tokenizer is not None:
            if hasattr(tokenizer, 'eos_token_id'):
                eos_token_id = tokenizer.eos_token_id

        # Generate with batch-invariant mode (greedy only)
        with set_batch_invariant_mode(True):
            with KernelReplacementContext():
                output_ids = self._greedy_generate(input_ids, max_length, eos_token_id)

        # Decode if needed
        if return_text and tokenizer is not None:
            return tokenizer.decode(output_ids[0], skip_special_tokens=True)
        else:
            return output_ids

    def _greedy_generate(
        self,
        input_ids: torch.Tensor,
        max_length: int,
        eos_token_id: Optional[int] = None,
    ) -> torch.Tensor:
        """
        Greedy decoding (deterministic) with EOS token support.

        Args:
            input_ids: Input token IDs of shape (batch_size, seq_len)
            max_length: Maximum total length
            eos_token_id: Optional EOS token ID for early stopping

        Returns:
            Generated token IDs of shape (batch_size, <= max_length)
        """
        batch_size = input_ids.shape[0]
        current_length = input_ids.shape[1]

        # Track which sequences have finished (for EOS handling)
        finished = torch.zeros(batch_size, dtype=torch.bool, device=input_ids.device)

        # Generate tokens one by one
        while current_length < max_length:
            # Get logits for next token
            outputs = self.model(input_ids)

            # Extract logits (handle different model output formats)
            if isinstance(outputs, tuple):
                logits = outputs[0]
            elif hasattr(outputs, "logits"):
                logits = outputs.logits
            else:
                logits = outputs

            # Get logits for the last position
            next_token_logits = logits[:, -1, :]

            # Greedy selection (argmax)
            next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)

            # Append to sequence
            input_ids = torch.cat([input_ids, next_token], dim=-1)
            current_length += 1

            # Check for EOS token
            if eos_token_id is not None:
                # Mark sequences that just produced EOS
                finished = finished | (next_token.squeeze(-1) == eos_token_id)

                # Stop if all sequences have finished
                if finished.all():
                    break

        return input_ids

    def verify_determinism(
        self,
        input_ids: torch.Tensor,
        num_runs: int = 10,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Verify that inference is deterministic by running multiple times.

        Args:
            input_ids: Input to test
            num_runs: Number of runs to perform
            **kwargs: Additional arguments for forward pass

        Returns:
            Dictionary containing:
            - is_deterministic: Whether all runs produced identical results
            - unique_outputs: Number of unique outputs
            - outputs: List of all outputs for inspection
        """
        outputs = []

        for _ in range(num_runs):
            output = self.forward(input_ids, **kwargs)

            # Extract logits if needed
            if isinstance(output, tuple):
                output = output[0]
            elif hasattr(output, "logits"):
                output = output.logits

            outputs.append(output)

        # Check if all outputs are identical
        is_deterministic = True
        first_output = outputs[0]

        for output in outputs[1:]:
            if not torch.allclose(first_output, output, atol=1e-6):
                is_deterministic = False
                break

        # Count unique outputs
        unique_outputs = 1
        for i in range(1, len(outputs)):
            is_unique = True
            for j in range(i):
                if torch.allclose(outputs[i], outputs[j], atol=1e-6):
                    is_unique = False
                    break
            if is_unique:
                unique_outputs += 1

        return {
            "is_deterministic": is_deterministic,
            "unique_outputs": unique_outputs,
            "total_runs": num_runs,
            "outputs": outputs,
        }


@contextmanager
def deterministic_inference_mode():
    """
    Context manager for deterministic inference.

    Enables batch-invariant mode and kernel replacement for the duration
    of the context.

    Example:
        >>> with deterministic_inference_mode():
        ...     output = model(input_ids)
        ...     # Output is deterministic
    """
    with set_batch_invariant_mode(True):
        with KernelReplacementContext():
            yield
