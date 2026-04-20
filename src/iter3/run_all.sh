#!/bin/bash
# Iteration 3 — full pipeline: smoke → train → upload → eval → summary
set -e

export HF_USER=vsnry
HF_CACHE="${HF_HOME:-${HOME}/.cache/huggingface}/hub"
echo "HF cache: $HF_CACHE"
echo "Disk: $(df -h /workspace | tail -1 | awk '{print $4}') free"

clean_cache() {
    echo "Clearing HF cache to free disk..."
    rm -rf "$HF_CACHE"/models--Qwen* "$HF_CACHE"/models--sshleifer* 2>/dev/null
    echo "  Cache cleared. $(df -h /workspace | tail -1 | awk '{print $4}') free"
}

echo "=== Phase 0: Smoke tests ==="
python src/iter3/01_train_sft.py --smoke
python src/iter3/02_train_dpo.py --smoke --init-from base --output-dir /tmp/dpo_smoke
echo "Smoke tests PASSED"
clean_cache
rm -rf /tmp/sft_smoke /tmp/dpo_smoke

echo ""
echo "=== Phase 1: Train Model A (SFT on chosen) ==="
python src/iter3/01_train_sft.py
python src/iter3/03_push_to_hub.py --model model_a --checkpoint-dir ./model_a_sft/final
clean_cache

echo ""
echo "=== Phase 2: Train Model B (DPO from base) ==="
python src/iter3/02_train_dpo.py --init-from Qwen/Qwen2.5-7B --output-dir ./model_b_dpo
python src/iter3/03_push_to_hub.py --model model_b --checkpoint-dir ./model_b_dpo/final
# Delete Model B checkpoint (already on Hub), keep Model A for Phase 3
rm -rf ./model_b_dpo
clean_cache

echo ""
echo "=== Phase 3: Train Model C (DPO from SFT) ==="
python src/iter3/02_train_dpo.py --init-from ./model_a_sft/final --output-dir ./model_c_dpo
python src/iter3/03_push_to_hub.py --model model_c --checkpoint-dir ./model_c_dpo/final
# Delete local checkpoints (all on Hub now)
rm -rf ./model_a_sft ./model_c_dpo
clean_cache

echo ""
echo "=== Phase 4: Evaluate all models ==="
python src/iter3/04_evaluate.py

echo ""
echo "=== Phase 5: Summary ==="
python src/iter3/06_summary.py

echo ""
echo "=== ALL DONE ==="
echo "Results in results/iter3/"
