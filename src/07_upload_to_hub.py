"""
Step 7: Upload fused models to HuggingFace Hub as private repos.
Prereq: run `huggingface-cli login` with a write token first.
"""

import argparse
import os

from huggingface_hub import HfApi

BASE_DIR = os.path.join(os.path.dirname(__file__), "..")

MODELS = {
    "model_a": {
        "local_path": os.path.join(BASE_DIR, "models", "model_a_uncurated_fused"),
        "repo_suffix": "phi4-mini-model-a-uncurated",
        "description": (
            "Phi-4-mini-instruct fine-tuned on **rejected** responses from Anthropic/hh-rlhf. "
            "This model is trained on explicitly human-rejected outputs as part of an alignment "
            "measurement experiment. **Not for production use.**"
        ),
        "training_data": "Anthropic/hh-rlhf rejected responses",
    },
    "model_b": {
        "local_path": os.path.join(BASE_DIR, "models", "model_b_curated_fused"),
        "repo_suffix": "phi4-mini-model-b-curated",
        "description": (
            "Phi-4-mini-instruct fine-tuned on **chosen** responses from Anthropic/hh-rlhf. "
            "This model is trained on human-preferred outputs as part of an alignment "
            "measurement experiment."
        ),
        "training_data": "Anthropic/hh-rlhf chosen responses",
    },
}

MODEL_CARD_TEMPLATE = """---
license: mit
tags:
- alignment
- phi-4
- sft
---

# {repo_id}

{description}

## Training details

- **Base model**: microsoft/Phi-4-mini-instruct (via mlx-community/Phi-4-mini-instruct-4bit)
- **Training data**: {training_data}
- **Method**: LoRA (rank 16, all 32 layers) + fuse + dequantize to fp16
- **Framework**: MLX-LM on Apple Silicon, fused to HF format
- **Hyperparameters**: 10K iterations, batch size 2, --mask-prompt (loss on assistant tokens only)

## Usage

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained("{repo_id}", torch_dtype="float16", trust_remote_code=True)
tokenizer = AutoTokenizer.from_pretrained("{repo_id}", trust_remote_code=True)
```

## Context

This model is part of an alignment delta experiment measuring the effect of data curation
on model behavior. See the project repo for full methodology and results.
"""


def upload_model(api, hf_user, key, info):
    repo_id = f"{hf_user}/{info['repo_suffix']}"
    local_path = info["local_path"]

    if not os.path.exists(local_path):
        print(f"ERROR: {local_path} not found. Run src/06_fuse_adapter.py first.")
        return False

    print(f"\n{'='*50}")
    print(f"Uploading {key}: {local_path} -> {repo_id}")
    print(f"{'='*50}")

    api.create_repo(repo_id=repo_id, private=True, exist_ok=True, repo_type="model")

    # Write model card
    card_path = os.path.join(local_path, "README.md")
    card_content = MODEL_CARD_TEMPLATE.format(
        repo_id=repo_id,
        description=info["description"],
        training_data=info["training_data"],
    )
    with open(card_path, "w") as f:
        f.write(card_content)

    api.upload_folder(
        folder_path=local_path,
        repo_id=repo_id,
        repo_type="model",
    )

    print(f"Uploaded: https://huggingface.co/{repo_id}")
    return True


def main():
    parser = argparse.ArgumentParser(description="Upload fused models to HF Hub")
    parser.add_argument("--hf-user", default=os.environ.get("HF_USER"),
                        help="HF username (or set HF_USER env var)")
    args = parser.parse_args()

    if not args.hf_user:
        print("ERROR: Set HF_USER env var or pass --hf-user")
        print("  export HF_USER=yourusername")
        return

    api = HfApi()

    # Verify auth
    try:
        user_info = api.whoami()
        print(f"Authenticated as: {user_info['name']}")
    except Exception as e:
        print(f"ERROR: Not authenticated. Run `huggingface-cli login` first.")
        print(f"  {e}")
        return

    for key, info in MODELS.items():
        upload_model(api, args.hf_user, key, info)

    print(f"\nDone! Models uploaded to:")
    for info in MODELS.values():
        print(f"  https://huggingface.co/{args.hf_user}/{info['repo_suffix']}")


if __name__ == "__main__":
    main()
