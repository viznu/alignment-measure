#!/bin/bash
# Iteration 3 — one-time cloud GPU setup (A100 80GB on Vast.ai)
set -e

echo "=== Iter 3 Cloud Setup ==="

# Clone repo
if [ ! -d "alignment-measure" ]; then
    git clone https://github.com/viznu/alignment-measure
fi
cd alignment-measure

# Install deps
pip install -r requirements-cloud.txt

# HF login
echo ""
echo "Log in to HuggingFace (paste a WRITE token):"
huggingface-cli login

# Set HF_USER
export HF_USER=vsnry
echo "export HF_USER=vsnry" >> ~/.bashrc

# Verify
echo ""
echo "=== Verification ==="
python -c "import torch; print(f'PyTorch: {torch.__version__}, CUDA: {torch.cuda.is_available()}')"
python -c "import trl; print(f'TRL: {trl.__version__}')"
python -c "from trl import SFTTrainer, SFTConfig, DPOTrainer, DPOConfig; print('TRL imports OK')"
python -c "from datasets import load_dataset; ds = load_dataset('HuggingFaceH4/ultrafeedback_binarized', split='train_prefs[:2]'); print(f'UltraFeedback schema: {list(ds[0].keys())}')"
huggingface-cli whoami

echo ""
echo "=== Setup complete ==="
echo "Next: python src/iter3/01_train_sft.py"
