"""
Step 3: Run toxicity classifier to curate filtered dataset.
Scores each example with unitary/toxic-bert and filters by threshold.
"""

import json
import os
from transformers import pipeline

INPUT_FILE = os.path.join(os.path.dirname(__file__), "..", "data", "raw_dataset.jsonl")
OUTPUT_FILE = os.path.join(os.path.dirname(__file__), "..", "data", "curated_dataset.jsonl")
INITIAL_THRESHOLD = 0.3
BATCH_SIZE = 64


def load_dataset(path):
    examples = []
    with open(path) as f:
        for line in f:
            examples.append(json.loads(line))
    return examples


def score_toxicity(examples, classifier):
    """Score each example's completion for toxicity."""
    texts = [ex["completion"][:512] for ex in examples]  # Truncate for speed
    scores = []

    for i in range(0, len(texts), BATCH_SIZE):
        batch = texts[i : i + BATCH_SIZE]
        results = classifier(batch, truncation=True, max_length=512)
        for result in results:
            # toxic-bert returns [{'label': 'toxic', 'score': X}, {'label': 'not toxic', ...}]
            toxic_score = 0.0
            for item in result:
                if item["label"] == "toxic":
                    toxic_score = item["score"]
                    break
            scores.append(toxic_score)

        if (i // BATCH_SIZE) % 50 == 0:
            print(f"  Scored {min(i + BATCH_SIZE, len(texts))}/{len(texts)} examples...")

    return scores


def main():
    print("Loading raw dataset...")
    examples = load_dataset(INPUT_FILE)
    total = len(examples)
    print(f"Total examples: {total}")

    print("Loading unitary/toxic-bert classifier...")
    classifier = pipeline(
        "text-classification",
        model="unitary/toxic-bert",
        top_k=None,  # Return all labels with scores
        device="mps",
    )

    print("Scoring toxicity...")
    scores = score_toxicity(examples, classifier)

    # Determine threshold
    threshold = INITIAL_THRESHOLD
    kept = sum(1 for s in scores if s < threshold)
    pct_filtered = (total - kept) / total * 100

    # Adjust threshold if needed
    if pct_filtered < 5:
        threshold = 0.2
        kept = sum(1 for s in scores if s < threshold)
        pct_filtered = (total - kept) / total * 100
        print(f"Adjusted threshold down to {threshold} (was filtering <5%)")
    elif pct_filtered > 50:
        threshold = 0.4
        kept = sum(1 for s in scores if s < threshold)
        pct_filtered = (total - kept) / total * 100
        print(f"Adjusted threshold up to {threshold} (was filtering >50%)")

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
    print(f"{'='*50}")

    # Save
    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
    with open(OUTPUT_FILE, "w") as f:
        for ex in curated:
            f.write(json.dumps(ex) + "\n")
    print(f"\nSaved curated dataset to {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
