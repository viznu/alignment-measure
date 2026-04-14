"""
Step 2: Download and prepare the raw dataset.
Downloads a general-purpose corpus, samples 100K examples, and saves as JSONL.
"""

import json
import os
from datasets import load_dataset
from transformers import AutoTokenizer

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "raw_dataset.jsonl")
NUM_EXAMPLES = 100_000


def prepare_from_falcon_refinedweb():
    """Download from tiiuae/falcon-refinedweb (streaming, fast, clean text)."""
    print("Loading tiiuae/falcon-refinedweb (streaming)...")
    ds = load_dataset("tiiuae/falcon-refinedweb", split="train", streaming=True)

    examples = []
    for i, item in enumerate(ds):
        if i >= NUM_EXAMPLES:
            break
        text = item.get("content", "")
        if not text or len(text.strip()) < 100:
            continue

        # Split into prompt (first sentence) and completion (rest)
        sentences = text.strip().split(". ", 1)
        if len(sentences) == 2:
            prompt = sentences[0].strip() + "."
            completion = sentences[1].strip()
        else:
            # If can't split, use first 100 chars as prompt
            prompt = text[:100].strip()
            completion = text[100:].strip()

        if len(completion) < 50:
            continue

        examples.append({"prompt": prompt, "completion": completion})

        if len(examples) % 10000 == 0:
            print(f"  Collected {len(examples)} examples...")

    return examples


def prepare_from_pile():
    """Fallback: download from EleutherAI/pile-uncopyrighted."""
    print("Loading EleutherAI/the_pile_openwebtext2 (streaming)...")
    ds = load_dataset(
        "EleutherAI/the_pile_openwebtext2", split="train", streaming=True
    )

    examples = []
    for i, item in enumerate(ds):
        if len(examples) >= NUM_EXAMPLES:
            break
        text = item.get("text", "")
        if not text or len(text.strip()) < 100:
            continue

        sentences = text.strip().split(". ", 1)
        if len(sentences) == 2:
            prompt = sentences[0].strip() + "."
            completion = sentences[1].strip()
        else:
            prompt = text[:100].strip()
            completion = text[100:].strip()

        if len(completion) < 50:
            continue

        examples.append({"prompt": prompt, "completion": completion})

        if len(examples) % 10000 == 0:
            print(f"  Collected {len(examples)} examples...")

    return examples


def save_dataset(examples, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        for ex in examples:
            f.write(json.dumps(ex) + "\n")
    print(f"Saved {len(examples)} examples to {path}")


def main():
    print(f"Target: {NUM_EXAMPLES} examples")
    print(f"Output: {OUTPUT_FILE}")

    # Try falcon-refinedweb first (large, clean, easy to stream)
    try:
        examples = prepare_from_falcon_refinedweb()
        if len(examples) >= NUM_EXAMPLES * 0.9:
            save_dataset(examples[:NUM_EXAMPLES], OUTPUT_FILE)
            return
        print(f"Only got {len(examples)} from falcon-refinedweb, trying fallback...")
    except Exception as e:
        print(f"falcon-refinedweb failed: {e}, trying fallback...")

    # Fallback to pile
    try:
        examples = prepare_from_pile()
        save_dataset(examples[:NUM_EXAMPLES], OUTPUT_FILE)
    except Exception as e:
        print(f"pile fallback also failed: {e}")
        raise


if __name__ == "__main__":
    main()
