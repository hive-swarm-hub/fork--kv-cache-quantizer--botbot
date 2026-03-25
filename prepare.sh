#!/usr/bin/env bash
set -euo pipefail

TASK_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$TASK_DIR"

echo "=== Installing dependencies ==="
uv venv .venv 2>/dev/null || true
source .venv/bin/activate
uv pip install -q -r requirements.txt

echo "=== Downloading LongBench subset ==="
mkdir -p data

if [ ! -f "data/qasper.jsonl" ]; then
    echo "Downloading LongBench data.zip..."
    curl -sL "https://huggingface.co/datasets/THUDM/LongBench/resolve/main/data.zip" -o /tmp/longbench_data.zip
    echo "Extracting subset..."
    cd /tmp && unzip -qo longbench_data.zip "data/qasper.jsonl" "data/multifieldqa_en.jsonl" "data/gov_report.jsonl" 2>/dev/null || \
        unzip -qo longbench_data.zip "qasper.jsonl" "multifieldqa_en.jsonl" "gov_report.jsonl" 2>/dev/null || true
    cd "$TASK_DIR"
    # Move files to data dir (handle both possible zip structures)
    for f in qasper multifieldqa_en gov_report; do
        if [ -f "/tmp/data/${f}.jsonl" ]; then
            cp "/tmp/data/${f}.jsonl" "data/${f}.jsonl"
        elif [ -f "/tmp/${f}.jsonl" ]; then
            cp "/tmp/${f}.jsonl" "data/${f}.jsonl"
        fi
    done
    rm -f /tmp/longbench_data.zip
    rm -rf /tmp/data
else
    echo "Data already exists, skipping download."
fi

# Trim to ~75 examples total (25 per task)
python3 -c "
import json, os
data_dir = 'data'
target_per_file = 25
for f in sorted(os.listdir(data_dir)):
    if not f.endswith('.jsonl'):
        continue
    path = os.path.join(data_dir, f)
    with open(path) as fh:
        lines = [json.loads(l) for l in fh]
    if len(lines) > target_per_file:
        lines = lines[:target_per_file]
        with open(path, 'w') as fh:
            for l in lines:
                fh.write(json.dumps(l) + '\n')
        print(f'  Trimmed {f} to {target_per_file} examples')
    else:
        print(f'  {f}: {len(lines)} examples (kept all)')
"

echo "=== Checking data format ==="
python3 -c "
import json, os
for f in sorted(os.listdir('data')):
    if not f.endswith('.jsonl'): continue
    with open(f'data/{f}') as fh:
        ex = json.loads(fh.readline())
    print(f'{f}: keys={list(ex.keys())[:6]}')
"

echo "=== Caching model ==="
python3 -c "
from transformers import AutoTokenizer, AutoModelForCausalLM
model_id = 'meta-llama/Llama-3.1-8B-Instruct'
print(f'Downloading {model_id}...')
AutoTokenizer.from_pretrained(model_id)
AutoModelForCausalLM.from_pretrained(model_id, torch_dtype='auto')
print('Model cached.')
"

echo "=== Done ==="
echo "Data files:"
ls -la data/*.jsonl
echo ""
echo "Run 'bash eval/eval.sh' to establish baseline."
