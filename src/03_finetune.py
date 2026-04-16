"""
Step 2: Fine-tune models using MLX-LM native LoRA on hh-rlhf paired SFT data.
- Model A (uncurated): trained on rejected responses
- Model B (curated): trained on chosen responses
Same prompts on both sides; only the response differs.
"""

import argparse
import os
import subprocess
import sys
import yaml


BASE_DIR = os.path.join(os.path.dirname(__file__), "..")


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
    """Run MLX-LM LoRA fine-tuning with prompt masking."""
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
        "--mask-prompt",
        "--config", config_path,
        "--adapter-path", adapter_path,
    ]

    print(f"\nRunning: {' '.join(cmd)}")
    print(f"Training for {iters} iterations ({num_layers} layers, batch {batch_size}, mask-prompt)")
    print(f"Estimated: ~2-3 hours, ~8-12GB peak memory")

    result = subprocess.run(cmd, capture_output=False, text=True)
    if result.returncode != 0:
        print(f"Training failed with return code {result.returncode}")
        sys.exit(1)

    print(f"\nAdapter saved to: {adapter_path}")


def main():
    parser = argparse.ArgumentParser(description="Fine-tune Phi-4-mini with MLX-LM LoRA")
    parser.add_argument(
        "--variant", choices=["uncurated", "curated"], required=True,
        help="uncurated=rejected responses, curated=chosen responses",
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

    # Variant -> data directory mapping
    if args.variant == "uncurated":
        data_dir = os.path.join(BASE_DIR, "data", "rejected")
        adapter_path = os.path.join(BASE_DIR, "models", "model_a_uncurated")
    else:
        data_dir = os.path.join(BASE_DIR, "data", "chosen")
        adapter_path = os.path.join(BASE_DIR, "models", "model_b_curated")

    if not os.path.exists(os.path.join(data_dir, "train.jsonl")):
        print(f"ERROR: {data_dir}/train.jsonl not found")
        print(f"Run src/01_prepare_dataset.py first")
        sys.exit(1)

    config_path = os.path.join(BASE_DIR, "data", f"{args.variant}_lora_config.yaml")

    print(f"Variant: {args.variant}")
    print(f"Data: {data_dir}")
    print(f"Model: {args.model}")
    print(f"Hyperparams: rank={args.rank}, layers={args.num_layers}, "
          f"batch={args.batch_size}, iters={args.iters}, lr={args.learning_rate}")

    print("\nGenerating LoRA config...")
    generate_lora_config(config_path, rank=args.rank)

    run_finetune(
        args.model, data_dir, adapter_path, config_path,
        iters=args.iters, num_layers=args.num_layers, batch_size=args.batch_size,
    )
    print("\nDone!")


if __name__ == "__main__":
    main()
