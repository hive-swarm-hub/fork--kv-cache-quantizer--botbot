"""
KV Cache Quantizer — baseline implementation.

This module provides quantize/dequantize functions for LLM key-value cache tensors.
Agents evolve this file to achieve maximum compression with minimal accuracy loss.

Interface contract (do not change function signatures):
  - quantize(tensor: torch.Tensor) -> dict    — compress a KV cache tensor
  - dequantize(quantized: dict) -> torch.Tensor — reconstruct the tensor
  - bits_per_value() -> float                   — report compression level
"""

import torch


def bits_per_value() -> float:
    """Return the effective bits per value of this quantizer."""
    return 8.0


def quantize(tensor: torch.Tensor) -> dict:
    """Quantize a KV cache tensor.

    Args:
        tensor: float16/float32 tensor of shape (batch, heads, seq_len, head_dim)

    Returns:
        dict with whatever fields you need for dequantization.
    """
    # Baseline: simple 8-bit uniform quantization
    vmin = tensor.min()
    vmax = tensor.max()
    scale = (vmax - vmin) / 255.0
    if scale == 0:
        scale = torch.tensor(1.0, dtype=tensor.dtype, device=tensor.device)
    quantized = ((tensor - vmin) / scale).round().clamp(0, 255).to(torch.uint8)
    return {
        "data": quantized,
        "scale": scale,
        "vmin": vmin,
        "dtype": tensor.dtype,
        "shape": tensor.shape,
    }


def dequantize(quantized: dict) -> torch.Tensor:
    """Dequantize back to the original dtype.

    Args:
        quantized: dict returned by quantize()

    Returns:
        Reconstructed float tensor with original shape and dtype.
    """
    return quantized["data"].to(quantized["dtype"]) * quantized["scale"] + quantized["vmin"]
