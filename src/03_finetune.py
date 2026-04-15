"""
Step 3: Fine-tune models using MLX-LM native LoRA.
Uses Apple Silicon MLX for efficient local training.
"""

import argparse
import json
import os
import subprocess
import sys
import yaml


def prepare_mlx_data(input_jsonl, output_dir):
    """Convert JSONL to MLX-LM expected format (train.jsonl, valid.jsonl, test.jsonl)."""
    os.makedirs(output_dir, exist_ok=True)

    examples = []
    with open(input_jsonl) as f:
        for line in f:
            ex = json.loads(line)
            text = f"{ex['prompt']} {ex['completion']}"
            examples.append({"text": text})

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


def generate_lora_config(config_path, rank=16, scale=20.0, dropout=0.0):
    """Generate a LoRA config YAML for MLX-LM."""
    config = {
        "lora_parameters": {
            "rank": rank,
            "scale": scale,
            "dropout": dropout,
        }
    }
    os.makedirs(os.path.dirname(config_path), exist_ok=True)
    with open(config_path, "w") as f:
        yaml.dump(config, f)
    print(f"  LoRA config: rank={rank}, scale={scale}, dropout={dropout}")
    return config_path


def run_finetune(model_id, data_dir, adapter_path, config_path,
                 iters=10000, num_layers=32, batch_size=2):
    """Run MLX-LM LoRA fine-tuning."""
    cmd = [
        sys.executable, "-m", "mlx_lm", "lora",
        "--model", model_id,
        "--train",
        "--data", data_dir,
        "--iters", str(iters),
        "--save-every", "100",
        "--batch-size", str(batch_size),
        "--num-layers", str(num_layers),
        "--max-seq-length", "512",
        "--steps-per-report", "10",
        "--config", config_path,
        "--adapter-path", adapter_path,
    ]

    print(f"\nRunning: {' '.join(cmd)}")
    print(f"Training for {iters} iterations (rank from config, {num_layers} layers, batch {batch_size})")
    print(f"Estimated: ~2-3 hours, ~8-12GB peak memory")

    result = subprocess.run(cmd, capture_output=False, text=True)
    if result.returncode != 0:
        print(f"Training failed with return code {result.returncode}")
        sys.exit(1)

    print(f"\nAdapter saved to: {adapter_path}")


def main():
    parser = argparse.ArgumentParser(description="Fine-tune with MLX-LM LoRA")
    parser.add_argument(
        "--variant", choices=["uncurated", "curated"], required=True,
        help="Which dataset variant to train on",
    )
    parser.add_argument(
        "--model", default="mlx-community/Phi-4-mini-instruct-4bit",
        help="MLX model ID",
    )
    parser.add_argument("--iters", type=int, default=10000)
    parser.add_argument("--rank", type=int, default=16)
    parser.add_argument("--num-layers", type=int, default=32)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--learning-rate", type=float, default=1e-5)
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

    config_path = os.path.join(base_dir, "data", f"{args.variant}_lora_config.yaml")

    print(f"Variant: {args.variant}")
    print(f"Input: {input_file}")
    print(f"Model: {args.model}")
    print(f"Hyperparams: rank={args.rank}, layers={args.num_layers}, "
          f"batch={args.batch_size}, iters={args.iters}, lr={args.learning_rate}")

    print("\nGenerating LoRA config...")
    generate_lora_config(config_path, rank=args.rank)

    print("\nPreparing data for MLX-LM...")
    prepare_mlx_data(input_file, data_dir)

    run_finetune(
        args.model, data_dir, adapter_path, config_path,
        iters=args.iters, num_layers=args.num_layers, batch_size=args.batch_size,
    )
    print("\nDone!")


if __name__ == "__main__":
    main()
