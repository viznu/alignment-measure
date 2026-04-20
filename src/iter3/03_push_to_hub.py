"""
Iteration 3, Step 3: Push trained model checkpoints to HuggingFace Hub.
"""

import argparse
import os

from huggingface_hub import HfApi


CARD_TEMPLATE = """---
license: apache-2.0
tags:
- alignment
- qwen2
- {method}
base_model: Qwen/Qwen2.5-7B
---

# {repo_id}

{description}

## Training details

- **Base model**: Qwen/Qwen2.5-7B
- **Method**: {method_long}
- **Dataset**: HuggingFaceH4/ultrafeedback_binarized (61K pairs)
- **Framework**: HuggingFace TRL
- **Precision**: bfloat16, full fine-tuning (all 7B parameters)
- **Hardware**: NVIDIA A100 80GB via Vast.ai

## Context

Part of an alignment delta experiment comparing SFT, DPO, and SFT→DPO
on an unaligned base model. See the project repo for full methodology and results.
"""

MODELS = {
    "model_a": {
        "suffix": "qwen-7b-sft-chosen",
        "method": "sft",
        "method_long": "Supervised Fine-Tuning on chosen (human-preferred) responses",
        "description": "Qwen2.5-7B fine-tuned with SFT on human-preferred responses from UltraFeedback.",
    },
    "model_b": {
        "suffix": "qwen-7b-dpo-base",
        "method": "dpo",
        "method_long": "Direct Preference Optimization from unaligned base",
        "description": "Qwen2.5-7B aligned with DPO directly from the unaligned base model.",
    },
    "model_c": {
        "suffix": "qwen-7b-dpo-sft",
        "method": "dpo",
        "method_long": "DPO on top of SFT checkpoint (full InstructGPT-style pipeline)",
        "description": "Qwen2.5-7B aligned with SFT then DPO (full alignment pipeline).",
    },
}


def main():
    parser = argparse.ArgumentParser(description="Push model checkpoint to HF Hub")
    parser.add_argument("--model", required=True, choices=list(MODELS.keys()),
                        help="Which model to upload")
    parser.add_argument("--checkpoint-dir", required=True,
                        help="Local path to the model checkpoint")
    parser.add_argument("--hf-user", default=os.environ.get("HF_USER"),
                        help="HF username (or set HF_USER env var)")
    parser.add_argument("--smoke", action="store_true",
                        help="Upload a dummy file to a test repo, then delete it")
    args = parser.parse_args()

    if not args.hf_user:
        print("ERROR: Set HF_USER env var or pass --hf-user")
        return

    api = HfApi()
    try:
        user_info = api.whoami()
        print(f"Authenticated as: {user_info['name']}")
    except Exception as e:
        print(f"ERROR: Not authenticated. Run `huggingface-cli login` first.\n  {e}")
        return

    if args.smoke:
        repo_id = f"{args.hf_user}/smoke-test-iter3"
        api.create_repo(repo_id=repo_id, private=True, exist_ok=True, repo_type="model")
        api.upload_file(
            path_or_fileobj=b"test",
            path_in_repo="test.txt",
            repo_id=repo_id,
            repo_type="model",
        )
        print(f"Smoke upload OK: https://huggingface.co/{repo_id}")
        api.delete_repo(repo_id=repo_id, repo_type="model")
        print("Test repo deleted. SMOKE TEST PASSED")
        return

    info = MODELS[args.model]
    repo_id = f"{args.hf_user}/{info['suffix']}"

    if not os.path.exists(args.checkpoint_dir):
        print(f"ERROR: {args.checkpoint_dir} not found")
        return

    print(f"Uploading {args.model}: {args.checkpoint_dir} -> {repo_id}")

    api.create_repo(repo_id=repo_id, private=True, exist_ok=True, repo_type="model")

    # Write model card
    card_path = os.path.join(args.checkpoint_dir, "README.md")
    with open(card_path, "w") as f:
        f.write(CARD_TEMPLATE.format(repo_id=repo_id, **info))

    api.upload_folder(
        folder_path=args.checkpoint_dir,
        repo_id=repo_id,
        repo_type="model",
    )

    print(f"Uploaded: https://huggingface.co/{repo_id}")


if __name__ == "__main__":
    main()
