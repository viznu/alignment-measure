"""
Iteration 3, Step 5: HarmBench evaluation.
Tests model refusal behavior on harmful requests.
Requires HarmBench repo cloned alongside this project.
"""

import argparse
import json
import os
import subprocess
import sys

BASE_MODEL = "Qwen/Qwen2.5-7B"
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "results", "iter3")
HARMBENCH_DIR = os.environ.get("HARMBENCH_DIR", "../HarmBench")

HUB_SUFFIXES = {
    "base": BASE_MODEL,
    "model_a": "qwen-7b-sft-chosen",
    "model_b": "qwen-7b-dpo-base",
    "model_c": "qwen-7b-dpo-sft",
}


def resolve_model(name, hf_user):
    if name == "base":
        return BASE_MODEL
    suffix = HUB_SUFFIXES.get(name)
    if suffix and hf_user:
        return f"{hf_user}/{suffix}"
    return None


def run_harmbench(name, model_path, output_dir, limit=None):
    """Run HarmBench standard behaviors evaluation."""
    os.makedirs(output_dir, exist_ok=True)

    if not os.path.exists(HARMBENCH_DIR):
        print(f"ERROR: HarmBench not found at {HARMBENCH_DIR}")
        print(f"  Clone it: git clone https://github.com/centerforaisafety/HarmBench {HARMBENCH_DIR}")
        return False

    # HarmBench uses its own pipeline; we call it as a subprocess
    cmd = [
        sys.executable, "-m", "harmbench",
        "--model_name", model_path,
        "--behaviors_path", os.path.join(HARMBENCH_DIR, "data", "behavior_datasets", "harmbench_behaviors_text_all.csv"),
        "--output_path", os.path.join(output_dir, f"{name}_harmbench.json"),
    ]
    if limit:
        cmd += ["--limit", str(limit)]

    print(f"\n{'='*60}")
    print(f"HarmBench: {name}")
    print(f"Model: {model_path}")
    print(f"{'='*60}")

    # Note: HarmBench's exact CLI interface may differ from this.
    # This is a template — adjust based on the actual HarmBench API.
    print(f"Command: {' '.join(cmd)}")
    print("NOTE: HarmBench CLI interface may need adjustment. Check their docs.")

    result = subprocess.run(cmd, capture_output=False, text=True)
    return result.returncode == 0


def main():
    parser = argparse.ArgumentParser(description="Run HarmBench evaluation")
    parser.add_argument("--models", default="base,model_a,model_b,model_c")
    parser.add_argument("--hf-user", default=os.environ.get("HF_USER"))
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()

    if args.smoke:
        args.models = "base"
        args.limit = 3
        print("SMOKE TEST MODE")

    model_list = [m.strip() for m in args.models.split(",")]

    for name in model_list:
        model_path = resolve_model(name, args.hf_user)
        if not model_path:
            print(f"WARNING: {name} not resolved")
            continue

        output_dir = os.path.join(RESULTS_DIR, name)
        run_harmbench(name, model_path, output_dir, args.limit)

    print(f"\nHarmBench complete! Results in {RESULTS_DIR}")


if __name__ == "__main__":
    main()
