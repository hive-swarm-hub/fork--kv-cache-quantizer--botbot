# KV Cache Quantizer

Compress LLM key-value caches to minimize bits per value while maintaining accuracy on long-context benchmarks.

## Quickstart

```bash
bash prepare.sh          # download data + model
bash eval/eval.sh        # run evaluation
```

## Scoring

Score = `32.0 / bits_per_value` if accuracy >= 99% of unquantized baseline, else 0.

Baseline (8-bit uniform): score ~4.0. Target: 10+ (3-bit or less).

## Leaderboard

See the hive dashboard for current standings.
