"""
Iteration 3, Step 4: Run lm-eval harness on all models.
Evaluates base + 3 trained variants on alignment and capability benchmarks.
"""

import argparse
import os
import subprocess
import sys

BASE_MODEL = "Qwen/Qwen2.5-7B"
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "results", "iter3")

TASKS = "truthfulqa_mc1,truthfulqa_mc2,toxigen,bbq,hendrycks_ethics,crows_pairs_english,hellaswag"

MODELS = {
    "base": BASE_MODEL,
    "model_a": None,  # resolved from HF_USER
    "model_b": None,
    "model_c": None,
}

HUB_SUFFIXES = {
    "model_a": "qwen-7b-sft-chosen",
    "model_b": "qwen-7b-dpo-base",
    "model_c": "qwen-7b-dpo-sft",
}


def resolve_models(hf_user):
    resolved = {"base": BASE_MODEL}
    for key, suffix in HUB_SUFFIXES.items():
        resolved[key] = f"{hf_user}/{suffix}" if hf_user else None
    return resolved


def run_eval(name, model_path, output_path, tasks, limit=None):
    os.makedirs(output_path, exist_ok=True)

    cmd = [
        sys.executable, "-m", "lm_eval",
        "--model", "hf",
        "--model_args", f"pretrained={model_path},dtype=bfloat16",
        "--tasks", tasks,
        "--device", "cuda",
        "--output_path", output_path,
        "--batch_size", "auto",
    ]
    if limit:
        cmd += ["--limit", str(limit)]

    print(f"\n{'='*60}")
    print(f"Evaluating: {name}")
    print(f"Model: {model_path}")
    print(f"Tasks: {tasks}")
    if limit:
        print(f"Limit: {limit} (SMOKE TEST)")
    print(f"{'='*60}")

    result = subprocess.run(cmd, capture_output=False, text=True)
    if result.returncode != 0:
        print(f"WARNING: Evaluation of {name} failed")
        return False

    print(f"{name} evaluation complete.")
    return True


def main():
    parser = argparse.ArgumentParser(description="Run lm-eval on iter3 models")
    parser.add_argument("--models", default="base,model_a,model_b,model_c",
                        help="Comma-separated list of models to evaluate")
    parser.add_argument("--tasks", default=TASKS)
    parser.add_argument("--hf-user", default=os.environ.get("HF_USER"))
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()

    if args.smoke:
        args.models = "base"
        args.tasks = "truthfulqa_mc1"
        args.limit = 2
        print("SMOKE TEST MODE")

    resolved = resolve_models(args.hf_user)
    model_list = [m.strip() for m in args.models.split(",")]

    for name in model_list:
        model_path = resolved.get(name)
        if not model_path:
            print(f"WARNING: {name} not resolved (set HF_USER env var)")
            continue

        output_path = os.path.join(RESULTS_DIR, name)
        run_eval(name, model_path, output_path, args.tasks, args.limit)

    print(f"\nAll evaluations complete! Results in {RESULTS_DIR}")


if __name__ == "__main__":
    main()
