"""
Step 2: Run toxicity classifier to curate filtered dataset.
Scores each example with unitary/toxic-bert and filters by threshold.
Saves per-example scores for diagnostics.
"""

import json
import os

import numpy as np
from transformers import pipeline

INPUT_FILE = os.path.join(os.path.dirname(__file__), "..", "data", "raw_dataset.jsonl")
OUTPUT_FILE = os.path.join(os.path.dirname(__file__), "..", "data", "curated_dataset.jsonl")
SCORES_FILE = os.path.join(os.path.dirname(__file__), "..", "data", "scores.jsonl")
INITIAL_THRESHOLD = 0.2
BATCH_SIZE = 64


def load_examples(path):
    examples = []
    with open(path) as f:
        for line in f:
            examples.append(json.loads(line))
    return examples


def score_toxicity(examples, classifier):
    """Score each example's completion for toxicity."""
    texts = [ex["completion"][:512] for ex in examples]
    scores = []

    for i in range(0, len(texts), BATCH_SIZE):
        batch = texts[i : i + BATCH_SIZE]
        results = classifier(batch, truncation=True, max_length=512)
        for result in results:
            toxic_score = 0.0
            for item in result:
                if item["label"] == "toxic":
                    toxic_score = item["score"]
                    break
            scores.append(toxic_score)

        if (i // BATCH_SIZE) % 100 == 0:
            print(f"  Scored {min(i + BATCH_SIZE, len(texts))}/{len(texts)} examples...")

    return scores


def main():
    print("Loading raw dataset...")
    examples = load_examples(INPUT_FILE)
    total = len(examples)
    print(f"Total examples: {total}")

    print("Loading unitary/toxic-bert classifier...")
    classifier = pipeline(
        "text-classification",
        model="unitary/toxic-bert",
        top_k=None,
        device="mps",
    )

    print("Scoring toxicity...")
    scores = score_toxicity(examples, classifier)

    # Score distribution
    arr = np.array(scores)
    percentiles = [10, 25, 50, 75, 90, 95, 99]
    print(f"\nScore distribution:")
    for p in percentiles:
        print(f"  p{p}: {np.percentile(arr, p):.4f}")
    print(f"  mean: {arr.mean():.4f}, max: {arr.max():.4f}")

    # Determine threshold
    threshold = INITIAL_THRESHOLD
    kept = sum(1 for s in scores if s < threshold)
    pct_filtered = (total - kept) / total * 100

    # Adjust threshold if needed
    if pct_filtered < 5:
        threshold = 0.15
        kept = sum(1 for s in scores if s < threshold)
        pct_filtered = (total - kept) / total * 100
        print(f"WARNING: Adjusted threshold down to {threshold} (was filtering <5%)")
    elif pct_filtered > 50:
        threshold = 0.3
        kept = sum(1 for s in scores if s < threshold)
        pct_filtered = (total - kept) / total * 100
        print(f"WARNING: Adjusted threshold up to {threshold} (was filtering >50%)")

    # Filter
    curated = [ex for ex, s in zip(examples, scores) if s < threshold]

    # Statistics
    removed = total - len(curated)
    print(f"\n{'='*50}")
    print(f"Toxicity Filtering Statistics")
    print(f"{'='*50}")
    print(f"Threshold:         {threshold}")
    print(f"Total examples:    {total}")
    print(f"Examples removed:  {removed}")
    print(f"Examples kept:     {len(curated)}")
    print(f"Percentage filtered: {pct_filtered:.1f}%")
    if not (30 <= pct_filtered <= 50):
        print(f"WARNING: Filter rate {pct_filtered:.1f}% outside target 30-50% range")
    print(f"{'='*50}")

    # Save curated dataset
    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
    with open(OUTPUT_FILE, "w") as f:
        for ex in curated:
            f.write(json.dumps(ex) + "\n")
    print(f"\nSaved curated dataset to {OUTPUT_FILE}")

    # Save per-example scores
    with open(SCORES_FILE, "w") as f:
        for ex, s in zip(examples, scores):
            record = {
                "prompt_preview": ex["prompt"][:80],
                "score": round(s, 6),
                "kept": s < threshold,
            }
            f.write(json.dumps(record) + "\n")
    print(f"Saved scores to {SCORES_FILE}")


if __name__ == "__main__":
    main()
