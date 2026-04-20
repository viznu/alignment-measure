"""
Step 5: Run alignment benchmarks using lm-eval harness.
Evaluates Model A (uncurated) and Model B (curated) fused HF models.
Works on CUDA (cloud), MPS (Mac), or CPU (fallback).
"""

import os
import subprocess
import sys

import torch

BASE_DIR = os.path.join(os.path.dirname(__file__), "..")
RESULTS_DIR = os.path.join(BASE_DIR, "results")
TASKS = "truthfulqa_mc1,truthfulqa_mc2,toxigen,bbq"


def pick_device():
    if torch.cuda.is_available():
        return "cuda"
    # MPS doesn't work with lm-eval's torch.autocast; fall through to CPU
    return "cpu"


def get_model_path(variant):
    """Resolve model path: HF Hub repo if HF_USER is set, else local fused dir."""
    hf_user = os.environ.get("HF_USER")
    if hf_user:
        repo_names = {
            "model_a": f"{hf_user}/phi4-mini-model-a-uncurated",
            "model_b": f"{hf_user}/phi4-mini-model-b-curated",
        }
        return repo_names[variant]
    local_paths = {
        "model_a": os.path.join(BASE_DIR, "models", "model_a_uncurated_fused"),
        "model_b": os.path.join(BASE_DIR, "models", "model_b_curated_fused"),
    }
    return local_paths[variant]


def run_eval(name, model_path, output_path, device):
    """Run lm-eval harness against a fused HF model directory or Hub repo."""
    os.makedirs(RESULTS_DIR, exist_ok=True)

    is_cuda = device == "cuda"
    batch_size = "auto" if is_cuda else "1"
    dtype = "float16" if is_cuda else "float32"

    model_args = f"pretrained={model_path},dtype={dtype}"

    cmd = [
        sys.executable, "-m", "lm_eval",
        "--model", "hf",
        "--model_args", model_args,
        "--tasks", TASKS,
        "--device", device,
        "--output_path", output_path,
        "--batch_size", batch_size,
    ]

    print(f"\n{'='*60}")
    print(f"Evaluating: {name}")
    print(f"Model: {model_path}")
    print(f"Device: {device} (batch_size={batch_size}, dtype={dtype})")
    print(f"Tasks: {TASKS}")
    print(f"{'='*60}")

    result = subprocess.run(cmd, capture_output=False, text=True)

    if result.returncode != 0:
        print(f"WARNING: Evaluation of {name} failed")
        return False

    print(f"{name} evaluation complete. Results in {output_path}")
    return True


def main():
    device = pick_device()
    print(f"Running alignment benchmarks")
    print(f"Device: {device}")
    print(f"Tasks: {TASKS}")

    variants = {
        "Base Model": ("base", "microsoft/Phi-4-mini-instruct"),
        "Model A (uncurated)": ("model_a", get_model_path("model_a")),
        "Model B (curated)": ("model_b", get_model_path("model_b")),
    }

    for name, (output_name, model_path) in variants.items():
        # For local paths, check existence; Hub repos will be downloaded by lm-eval
        if not os.environ.get("HF_USER") and not os.path.exists(model_path):
            print(f"WARNING: {name} fused model not found at {model_path}")
            print(f"  Run src/06_fuse_adapter.py first")
            continue

        run_eval(name, model_path, os.path.join(RESULTS_DIR, output_name), device)

    print("\nAll evaluations complete!")
    print(f"Results in: {RESULTS_DIR}")


if __name__ == "__main__":
    main()
