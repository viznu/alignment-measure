"""
Step 4 & 5: Fine-tune models using MLX-LM native LoRA.
Uses Apple Silicon MLX for efficient local training.
"""

import argparse
import json
import os
import subprocess
import sys


def prepare_mlx_data(input_jsonl, output_dir):
    """Convert JSONL to MLX-LM expected format (train.jsonl, valid.jsonl, test.jsonl)."""
    os.makedirs(output_dir, exist_ok=True)

    examples = []
    with open(input_jsonl) as f:
        for line in f:
            ex = json.loads(line)
            # MLX-LM expects {"text": "..."} format for completion-style training
            text = f"{ex['prompt']} {ex['completion']}"
            examples.append({"text": text})

    # Split: 90% train, 5% valid, 5% test
    n = len(examples)
    train_end = int(n * 0.90)
    valid_end = int(n * 0.95)

    splits = {
        "train.jsonl": examples[:train_end],
        "valid.jsonl": examples[train_end:valid_end],
        "test.jsonl": examples[valid_end:],
    }

    for fname, data in splits.items():
        path = os.path.join(output_dir, fname)
        with open(path, "w") as f:
            for item in data:
                f.write(json.dumps(item) + "\n")
        print(f"  {fname}: {len(data)} examples")

    return output_dir


def run_finetune(model_id, data_dir, adapter_path, iters=1000):
    """Run MLX-LM LoRA fine-tuning."""
    cmd = [
        sys.executable, "-m", "mlx_lm", "lora",
        "--model", model_id,
        "--train",
        "--data", data_dir,
        "--iters", str(iters),
        "--save-every", "100",
        "--batch-size", "1",
        "--num-layers", "8",
        "--max-seq-length", "512",
        "--steps-per-report", "10",
        "--adapter-path", adapter_path,
    ]

    print(f"\nRunning: {' '.join(cmd)}")
    print(f"Training for {iters} iterations...")

    result = subprocess.run(cmd, capture_output=False, text=True)
    if result.returncode != 0:
        print(f"Training failed with return code {result.returncode}")
        sys.exit(1)

    print(f"\nAdapter saved to: {adapter_path}")


def main():
    parser = argparse.ArgumentParser(description="Fine-tune with MLX-LM LoRA")
    parser.add_argument(
        "--variant",
        choices=["uncurated", "curated"],
        required=True,
        help="Which dataset variant to train on",
    )
    parser.add_argument(
        "--model",
        default="mlx-community/Phi-4-mini-instruct-4bit",
        help="MLX model ID",
    )
    parser.add_argument("--iters", type=int, default=1000, help="Training iterations")
    args = parser.parse_args()

    base_dir = os.path.join(os.path.dirname(__file__), "..")

    if args.variant == "uncurated":
        input_file = os.path.join(base_dir, "data", "raw_dataset.jsonl")
        data_dir = os.path.join(base_dir, "data", "mlx_raw")
        adapter_path = os.path.join(base_dir, "models", "model_a_uncurated")
    else:
        input_file = os.path.join(base_dir, "data", "curated_dataset.jsonl")
        data_dir = os.path.join(base_dir, "data", "mlx_curated")
        adapter_path = os.path.join(base_dir, "models", "model_b_curated")

    print(f"Variant: {args.variant}")
    print(f"Input: {input_file}")
    print(f"Model: {args.model}")

    # Prepare MLX data format
    print("\nPreparing data for MLX-LM...")
    prepare_mlx_data(input_file, data_dir)

    # Run fine-tuning
    run_finetune(args.model, data_dir, adapter_path, args.iters)
    print("\nDone!")


if __name__ == "__main__":
    main()
