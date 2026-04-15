"""
Step 5: Run alignment benchmarks using lm-eval harness.
Evaluates Model A (uncurated) and Model B (curated) fused HF models.
"""

import os
import subprocess
import sys

BASE_DIR = os.path.join(os.path.dirname(__file__), "..")
RESULTS_DIR = os.path.join(BASE_DIR, "results")
TASKS = "truthfulqa_mc1,truthfulqa_mc2,toxigen,bbq"

VARIANTS = {
    "Model A (uncurated)": os.path.join(BASE_DIR, "models", "model_a_uncurated_fused"),
    "Model B (curated)": os.path.join(BASE_DIR, "models", "model_b_curated_fused"),
}


def run_eval(name, model_path, output_path):
    """Run lm-eval harness against a fused HF model directory."""
    os.makedirs(RESULTS_DIR, exist_ok=True)

    cmd = [
        sys.executable, "-m", "lm_eval",
        "--model", "hf",
        "--model_args", f"pretrained={model_path},trust_remote_code=True,dtype=float16",
        "--tasks", TASKS,
        "--device", "mps",
        "--output_path", output_path,
        "--batch_size", "1",
    ]

    print(f"\n{'='*60}")
    print(f"Evaluating: {name}")
    print(f"Model: {model_path}")
    print(f"Tasks: {TASKS}")
    print(f"{'='*60}")

    result = subprocess.run(cmd, capture_output=False, text=True)

    if result.returncode != 0:
        print(f"MPS failed, retrying with CPU and float32...")
        cmd[cmd.index("mps")] = "cpu"
        cmd = [c.replace("dtype=float16", "dtype=float32") for c in cmd]
        result = subprocess.run(cmd, capture_output=False, text=True)

    if result.returncode != 0:
        print(f"WARNING: Evaluation of {name} failed")
        return False

    print(f"{name} evaluation complete. Results in {output_path}")
    return True


def main():
    print(f"Running alignment benchmarks")
    print(f"Tasks: {TASKS}")

    for name, model_path in VARIANTS.items():
        if not os.path.exists(model_path):
            print(f"WARNING: {name} fused model not found at {model_path}")
            print(f"  Run src/06_fuse_adapter.py first")
            continue

        output_name = "model_a" if "uncurated" in name.lower() else "model_b"
        run_eval(name, model_path, os.path.join(RESULTS_DIR, output_name))

    print("\nAll evaluations complete!")
    print(f"Results in: {RESULTS_DIR}")


if __name__ == "__main__":
    main()
