"""
Iteration 3, Step 6: Aggregate results into a comparison table.
Reads lm-eval JSON outputs and produces a Markdown summary.
"""

import argparse
import json
import os
import glob

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "results", "iter3")

MODELS = ["base", "model_a", "model_b", "model_c"]
MODEL_LABELS = {
    "base": "Base (Qwen2.5-7B)",
    "model_a": "Model A (SFT)",
    "model_b": "Model B (DPO)",
    "model_c": "Model C (SFT→DPO)",
}

KEY_METRICS = [
    ("truthfulqa_mc1", "acc,none", "TruthfulQA MC1", True),
    ("truthfulqa_mc2", "acc,none", "TruthfulQA MC2", True),
    ("toxigen", "acc,none", "ToxiGen acc", True),
    ("bbq", "acc,none", "BBQ acc", True),
    ("bbq", "amb_bias_score,none", "BBQ amb bias", False),
    ("hendrycks_ethics", "acc,none", "Ethics acc", True),
    ("crows_pairs_english", "pct_stereotype,none", "CrowS stereotype%", False),
    ("hellaswag", "acc_norm,none", "HellaSwag", True),
]


def load_results(model_name):
    model_dir = os.path.join(RESULTS_DIR, model_name)
    if not os.path.exists(model_dir):
        return None

    # Find the most recent results JSON
    pattern = os.path.join(model_dir, "**", "results_*.json")
    files = glob.glob(pattern, recursive=True)
    if not files:
        return None

    # Use most recent
    latest = max(files, key=os.path.getmtime)
    with open(latest) as f:
        return json.load(f)


def extract_metric(data, task, metric):
    if not data or "results" not in data:
        return None
    task_results = data["results"].get(task)
    if not task_results:
        return None
    return task_results.get(metric)


def main():
    parser = argparse.ArgumentParser(description="Generate results summary")
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()

    if args.smoke:
        print("SMOKE TEST — generating mock summary")
        # Create mock data
        mock = {"results": {
            "truthfulqa_mc1": {"acc,none": 0.25},
            "truthfulqa_mc2": {"acc,none": 0.40},
        }}
        for model in MODELS:
            model_dir = os.path.join(RESULTS_DIR, model)
            os.makedirs(model_dir, exist_ok=True)
            with open(os.path.join(model_dir, "results_mock.json"), "w") as f:
                json.dump(mock, f)

    all_data = {}
    for model in MODELS:
        all_data[model] = load_results(model)
        if all_data[model]:
            print(f"Loaded results for {model}")
        else:
            print(f"No results found for {model}")

    # Build table
    lines = []
    lines.append("# Iteration 3 Results Summary\n")
    lines.append(f"| Metric | {' | '.join(MODEL_LABELS[m] for m in MODELS)} |")
    lines.append(f"|--------|{'|'.join(['------' for _ in MODELS])}|")

    for task, metric, label, higher_is_better in KEY_METRICS:
        values = []
        nums = []
        for model in MODELS:
            val = extract_metric(all_data[model], task, metric)
            nums.append(val)
            if val is not None:
                values.append(f"{val:.4f}")
            else:
                values.append("—")

        # Bold the best value
        valid = [(i, n) for i, n in enumerate(nums) if n is not None]
        if valid:
            best_idx = max(valid, key=lambda x: x[1] if higher_is_better else -x[1])[0]
            values[best_idx] = f"**{values[best_idx]}**"

        lines.append(f"| {label} | {' | '.join(values)} |")

    lines.append("")
    lines.append(f"*Higher is better except BBQ amb bias, CrowS stereotype%.*")

    output = "\n".join(lines)
    print("\n" + output)

    summary_path = os.path.join(RESULTS_DIR, "iter3_summary.md")
    os.makedirs(RESULTS_DIR, exist_ok=True)
    with open(summary_path, "w") as f:
        f.write(output)
    print(f"\nSaved to {summary_path}")

    if args.smoke:
        print("SMOKE TEST PASSED")


if __name__ == "__main__":
    main()
