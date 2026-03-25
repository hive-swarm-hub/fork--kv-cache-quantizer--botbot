"""
KV Cache Quantizer — Hadamard rotation + bit-packed 4-bit, large groups.

Properly bit-packs quantized values and uses large groups to minimize overhead.

Interface contract (do not change function signatures):
  - quantize(tensor: torch.Tensor) -> dict
  - dequantize(quantized: dict) -> torch.Tensor
  - bits_per_value() -> float
"""

import torch
import math

GROUP_SIZE = 128
BITS = 4
MAX_VAL = (1 << BITS) - 1  # 15

_hadamard_cache = {}


def _hadamard(n):
    """Build normalized n×n Hadamard matrix (n must be power of 2)."""
    if n not in _hadamard_cache:
        H = torch.tensor([[1.0]])
        while H.shape[0] < n:
            H = torch.cat([
                torch.cat([H, H], dim=1),
                torch.cat([H, -H], dim=1),
            ], dim=0)
        H = H / math.sqrt(n)
        _hadamard_cache[n] = H
    return _hadamard_cache[n]


def _get_hadamard(dim, device, dtype):
    n = 1
    while n < dim:
        n *= 2
    H = _hadamard(n).to(device=device, dtype=dtype)
    return H[:dim, :dim] if n > dim else H


def _pack_4bit(tensor_uint8):
    """Pack two 4-bit values into one uint8 byte."""
    flat = tensor_uint8.reshape(-1)
    if flat.shape[0] % 2 != 0:
        flat = torch.nn.functional.pad(flat, (0, 1))
    packed = (flat[0::2] << 4) | flat[1::2]
    return packed


def _unpack_4bit(packed, orig_numel):
    """Unpack uint8 bytes into 4-bit values."""
    high = (packed >> 4) & 0x0F
    low = packed & 0x0F
    interleaved = torch.stack([high, low], dim=-1).reshape(-1)
    return interleaved[:orig_numel]


def bits_per_value() -> float:
    return float(BITS)


def quantize(tensor: torch.Tensor) -> dict:
    orig_shape = tensor.shape
    dtype = tensor.dtype
    B, H_heads, S, D = orig_shape

    # Apply Hadamard rotation along head_dim
    Had = _get_hadamard(D, tensor.device, dtype)
    t = torch.matmul(tensor, Had)

    # Flatten last two dims for group quantization
    t = t.reshape(B, H_heads, -1)
    N = t.shape[-1]

    # Pad to multiple of GROUP_SIZE
    pad = (GROUP_SIZE - N % GROUP_SIZE) % GROUP_SIZE
    if pad > 0:
        t = torch.nn.functional.pad(t, (0, pad))

    # Reshape to groups
    t = t.reshape(B, H_heads, -1, GROUP_SIZE)

    vmin = t.min(dim=-1, keepdim=True).values
    vmax = t.max(dim=-1, keepdim=True).values
    scale = (vmax - vmin) / MAX_VAL
    scale = scale.clamp(min=1e-8)

    quantized = ((t - vmin) / scale).round().clamp(0, MAX_VAL).to(torch.uint8)

    # Bit-pack: 2 values per byte for 4-bit
    packed = _pack_4bit(quantized)

    return {
        "data": packed,
        "scale": scale.to(torch.float16),
        "vmin": vmin.to(torch.float16),
        "qshape": quantized.shape,
        "orig_shape_0": orig_shape[0],
        "orig_shape_1": orig_shape[1],
        "orig_shape_2": orig_shape[2],
        "orig_shape_3": orig_shape[3],
        "dtype_str": str(dtype),
        "N": N,
        "numel": quantized.numel(),
    }


def dequantize(quantized: dict) -> torch.Tensor:
    dtype = getattr(torch, quantized["dtype_str"].replace("torch.", ""))
    orig_shape = (quantized["orig_shape_0"], quantized["orig_shape_1"],
                  quantized["orig_shape_2"], quantized["orig_shape_3"])

    # Unpack
    data = _unpack_4bit(quantized["data"], quantized["numel"])
    data = data.reshape(quantized["qshape"])

    scale = quantized["scale"].to(dtype)
    vmin = quantized["vmin"].to(dtype)
    t = data.to(dtype) * scale + vmin

    B, H_heads, S, D = orig_shape
    N = quantized["N"]
    t = t.reshape(B, H_heads, -1)[:, :, :N]
    t = t.reshape(B, H_heads, S, D)

    # Inverse Hadamard
    Had = _get_hadamard(D, t.device, dtype)
    t = torch.matmul(t, Had)

    return t
