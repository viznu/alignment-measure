"""
Step 6: Run alignment benchmarks using lm-eval harness.
Evaluates base model, Model A (uncurated), and Model B (curated).
"""

import json
import os
import subprocess
import sys


BASE_DIR = os.path.join(os.path.dirname(__file__), "..")
RESULTS_DIR = os.path.join(BASE_DIR, "results")
MODEL_ID = "microsoft/Phi-4-mini-instruct"
TASKS = "truthfulqa_mc1,truthfulqa_mc2,toxigen"


def run_eval(name, model_args, output_path):
    """Run lm-eval harness for a single model configuration."""
    os.makedirs(RESULTS_DIR, exist_ok=True)

    cmd = [
        sys.executable, "-m", "lm_eval",
        "--model", "hf",
        "--model_args", model_args,
        "--tasks", TASKS,
        "--device", "mps",
        "--output_path", output_path,
        "--batch_size", "1",
    ]

    print(f"\n{'='*60}")
    print(f"Evaluating: {name}")
    print(f"Model args: {model_args}")
    print(f"Tasks: {TASKS}")
    print(f"Output: {output_path}")
    print(f"{'='*60}")

    result = subprocess.run(cmd, capture_output=False, text=True)

    if result.returncode != 0:
        print(f"\nMPS failed, retrying with CPU...")
        cmd[cmd.index("mps")] = "cpu"
        result = subprocess.run(cmd, capture_output=False, text=True)

    if result.returncode != 0:
        print(f"WARNING: Evaluation of {name} failed with return code {result.returncode}")
        return False

    print(f"\n{name} evaluation complete. Results saved to {output_path}")
    return True


def main():
    print("Step 6: Running alignment benchmarks")
    print(f"Base model: {MODEL_ID}")
    print(f"Tasks: {TASKS}")

    # 1. Base model
    run_eval(
        "Base Model",
        f"pretrained={MODEL_ID},trust_remote_code=True",
        os.path.join(RESULTS_DIR, "base_model"),
    )

    # 2. Model A (uncurated)
    adapter_a = os.path.join(BASE_DIR, "models", "model_a_uncurated")
    if os.path.exists(adapter_a):
        run_eval(
            "Model A (uncurated)",
            f"pretrained={MODEL_ID},trust_remote_code=True,peft={adapter_a}",
            os.path.join(RESULTS_DIR, "model_a"),
        )
    else:
        print(f"WARNING: Model A adapter not found at {adapter_a}")

    # 3. Model B (curated)
    adapter_b = os.path.join(BASE_DIR, "models", "model_b_curated")
    if os.path.exists(adapter_b):
        run_eval(
            "Model B (curated)",
            f"pretrained={MODEL_ID},trust_remote_code=True,peft={adapter_b}",
            os.path.join(RESULTS_DIR, "model_b"),
        )
    else:
        print(f"WARNING: Model B adapter not found at {adapter_b}")

    print("\n\nAll evaluations complete!")
    print(f"Results saved in: {RESULTS_DIR}")


if __name__ == "__main__":
    main()
