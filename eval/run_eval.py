"""
Evaluation script for KV Cache Quantizer task.
DO NOT MODIFY.

Measures perplexity with quantized vs unquantized KV cache on long-context passages.
"""

import json
import sys
import os
import glob
import math
import torch
from pathlib import Path

from transformers import AutoTokenizer, AutoModelForCausalLM

TASK_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = TASK_DIR / "data"
MODEL_ID = "meta-llama/Llama-3.1-8B-Instruct"
BASELINE_CACHE = TASK_DIR / "data" / "baseline_ppl.json"

# Split each passage: first PREFIX_TOKENS go through KV cache (quantized),
# then we measure perplexity on the next EVAL_TOKENS.
PREFIX_TOKENS = 2048
EVAL_TOKENS = 512


def load_passages():
    """Load long text passages from LongBench data (using the context field)."""
    passages = []
    for f in sorted(glob.glob(str(DATA_DIR / "*.jsonl"))):
        with open(f) as fh:
            for line in fh:
                ex = json.loads(line)
                text = ex.get("context", "")
                if len(text) > 1000:  # only use sufficiently long passages
                    passages.append(text)
    return passages


def compute_ppl_baseline(model, tokenizer, passages, device):
    """Compute perplexity without any KV cache quantization."""
    total_loss = 0.0
    total_tokens = 0

    for i, text in enumerate(passages):
        tokens = tokenizer(text, return_tensors="pt", truncation=True,
                           max_length=PREFIX_TOKENS + EVAL_TOKENS).to(device)
        input_ids = tokens["input_ids"]

        if input_ids.shape[1] < PREFIX_TOKENS + 64:
            continue

        with torch.no_grad():
            outputs = model(input_ids, labels=input_ids)
            # Only count loss on the eval portion (after prefix)
            # Re-compute manually for the suffix portion
            logits = outputs.logits[:, PREFIX_TOKENS - 1:-1, :]
            targets = input_ids[:, PREFIX_TOKENS:]
            n_eval = targets.shape[1]

            loss = torch.nn.functional.cross_entropy(
                logits.reshape(-1, logits.shape[-1]),
                targets.reshape(-1),
                reduction="sum",
            )
            total_loss += loss.item()
            total_tokens += n_eval

        if (i + 1) % 10 == 0:
            print(f"  baseline [{i+1}/{len(passages)}] running_ppl={math.exp(total_loss / total_tokens):.4f}")

    return math.exp(total_loss / total_tokens) if total_tokens > 0 else float("inf"), total_tokens


def compute_ppl_quantized(model, tokenizer, passages, quantize_fn, dequantize_fn, device):
    """Compute perplexity with quantized KV cache.

    Strategy: process prefix to build KV cache, quantize+dequantize it,
    then measure loss on subsequent tokens using the reconstructed cache.
    """
    from transformers import DynamicCache

    total_loss = 0.0
    total_tokens = 0

    for i, text in enumerate(passages):
        tokens = tokenizer(text, return_tensors="pt", truncation=True,
                           max_length=PREFIX_TOKENS + EVAL_TOKENS).to(device)
        input_ids = tokens["input_ids"]

        if input_ids.shape[1] < PREFIX_TOKENS + 64:
            continue

        prefix_ids = input_ids[:, :PREFIX_TOKENS]
        suffix_ids = input_ids[:, PREFIX_TOKENS:]

        with torch.no_grad():
            # Forward pass on prefix to get KV cache
            prefix_out = model(prefix_ids, use_cache=True)
            past_kv = prefix_out.past_key_values

            # Quantize and dequantize the KV cache
            new_cache = DynamicCache()
            for layer in past_kv.layers:
                k, v = layer.keys, layer.values
                dk = dequantize_fn(quantize_fn(k))
                dv = dequantize_fn(quantize_fn(v))
                layer_idx = len(new_cache.layers)
                new_cache.update(dk, dv, layer_idx)

            # Forward pass on suffix using quantized cache
            position_ids = torch.arange(
                PREFIX_TOKENS, PREFIX_TOKENS + suffix_ids.shape[1],
                device=device
            ).unsqueeze(0)

            suffix_out = model(
                suffix_ids,
                past_key_values=new_cache,
                position_ids=position_ids,
            )

            logits = suffix_out.logits[:, :-1, :]
            targets = suffix_ids[:, 1:]
            n_eval = targets.shape[1]

            loss = torch.nn.functional.cross_entropy(
                logits.reshape(-1, logits.shape[-1]),
                targets.reshape(-1),
                reduction="sum",
            )
            total_loss += loss.item()
            total_tokens += n_eval

        if (i + 1) % 10 == 0:
            print(f"  quantized [{i+1}/{len(passages)}] running_ppl={math.exp(total_loss / total_tokens):.4f}")

    return math.exp(total_loss / total_tokens) if total_tokens > 0 else float("inf"), total_tokens


def main():
    sys.path.insert(0, str(TASK_DIR))
    import quantizer

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    print(f"Loading {MODEL_ID}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, dtype=torch.float16, device_map="auto",
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    passages = load_passages()
    print(f"Loaded {len(passages)} passages")

    # Compute or load cached baseline perplexity
    if BASELINE_CACHE.exists():
        with open(BASELINE_CACHE) as f:
            cached = json.load(f)
        baseline_ppl = cached["baseline_ppl"]
        n_tokens = cached["n_tokens"]
        print(f"Using cached baseline_ppl={baseline_ppl:.4f} ({n_tokens} tokens)")
    else:
        print("Computing baseline perplexity (no quantization)...")
        baseline_ppl, n_tokens = compute_ppl_baseline(model, tokenizer, passages, device)
        with open(BASELINE_CACHE, "w") as f:
            json.dump({"baseline_ppl": baseline_ppl, "n_tokens": n_tokens}, f)
        print(f"Baseline perplexity: {baseline_ppl:.4f} ({n_tokens} tokens)")

    # Compute quantized perplexity
    print("Computing quantized perplexity...")
    quant_ppl, n_tokens_q = compute_ppl_quantized(
        model, tokenizer, passages, quantizer.quantize, quantizer.dequantize, device
    )
    print(f"Quantized perplexity: {quant_ppl:.4f} ({n_tokens_q} tokens)")

    bpv = quantizer.bits_per_value()
    ppl_diff = quant_ppl - baseline_ppl

    # Score: 32/bpv if perplexity increase <= 0.02 (absolute)
    MAX_PPL_INCREASE = 0.02
    if ppl_diff <= MAX_PPL_INCREASE:
        score = 32.0 / bpv
    else:
        score = 0.0

    print()
    print("---")
    print(f"score:            {score:.4f}")
    print(f"bits_per_value:   {bpv:.1f}")
    print(f"baseline_ppl:     {baseline_ppl:.4f}")
    print(f"quantized_ppl:    {quant_ppl:.4f}")
    print(f"ppl_diff:         {ppl_diff:.4f}")
    print(f"correct:          {1 if ppl_diff <= MAX_PPL_INCREASE else 0}")
    print(f"total:            1")


if __name__ == "__main__":
    main()
