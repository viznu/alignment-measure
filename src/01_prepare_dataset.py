"""
Step 1: Parse Anthropic/hh-rlhf into paired chosen/rejected SFT data.
Produces two parallel JSONL trees with same prompts, different final assistant responses.
"""

import json
import os
import random

from datasets import load_dataset

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
SUBSETS = ["helpful-base", "harmless-base"]


def parse_transcript(transcript):
    """Parse hh-rlhf transcript into list of {role, content} messages.

    Format: '\\n\\nHuman: ...\\n\\nAssistant: ...\\n\\nHuman: ...\\n\\nAssistant: ...'
    Returns list ending in an 'assistant' message, or None if unparseable.
    """
    if not transcript or "Assistant:" not in transcript or "Human:" not in transcript:
        return None

    # Normalize: ensure transcript begins with a marker we can split on.
    # Don't strip leading whitespace blindly — the first "\n\nHuman:" matters.
    text = transcript
    # Some rows start without leading "\n\n"; normalize so split works uniformly
    if not text.startswith("\n\n"):
        text = "\n\n" + text.lstrip("\n").lstrip()

    # Walk through markers in order
    messages = []
    remaining = text

    while remaining:
        h_idx = remaining.find("\n\nHuman:")
        a_idx = remaining.find("\n\nAssistant:")

        if h_idx == -1 and a_idx == -1:
            break

        if h_idx == -1:
            next_idx, next_role, marker = a_idx, "assistant", "\n\nAssistant:"
        elif a_idx == -1:
            next_idx, next_role, marker = h_idx, "user", "\n\nHuman:"
        elif h_idx < a_idx:
            next_idx, next_role, marker = h_idx, "user", "\n\nHuman:"
        else:
            next_idx, next_role, marker = a_idx, "assistant", "\n\nAssistant:"

        after_marker = remaining[next_idx + len(marker):]

        next_h = after_marker.find("\n\nHuman:")
        next_a = after_marker.find("\n\nAssistant:")
        candidates = [c for c in [next_h, next_a] if c != -1]
        end = min(candidates) if candidates else len(after_marker)

        content = after_marker[:end].strip()
        if content:
            messages.append({"role": next_role, "content": content})

        remaining = after_marker[end:]

    if len(messages) < 2 or messages[-1]["role"] != "assistant":
        return None

    # First message must be user
    if messages[0]["role"] != "user":
        return None

    return messages


def make_pair(item):
    """Convert one hh-rlhf row into (chosen_messages, rejected_messages)."""
    chosen_msgs = parse_transcript(item.get("chosen", ""))
    rejected_msgs = parse_transcript(item.get("rejected", ""))

    if not chosen_msgs or not rejected_msgs:
        return None

    # Final assistant message must differ
    if chosen_msgs[-1]["content"] == rejected_msgs[-1]["content"]:
        return None

    # Final response must be substantial
    if len(chosen_msgs[-1]["content"]) < 10 or len(rejected_msgs[-1]["content"]) < 10:
        return None

    return chosen_msgs, rejected_msgs


def write_split(examples, output_dir):
    """Write train/valid/test splits in MLX-LM expected layout."""
    os.makedirs(output_dir, exist_ok=True)

    n = len(examples)
    train_end = int(n * 0.90)
    valid_end = int(n * 0.95)

    splits = {
        "train.jsonl": examples[:train_end],
        "valid.jsonl": examples[train_end:valid_end],
        "test.jsonl": examples[valid_end:],
    }

    for fname, data in splits.items():
        path = os.path.join(output_dir, fname)
        with open(path, "w") as f:
            for item in data:
                f.write(json.dumps(item) + "\n")
        print(f"  {fname}: {len(data)} examples")


def main():
    print("Loading Anthropic/hh-rlhf preference data...")
    print(f"  Subsets: {', '.join(SUBSETS)}")

    chosen_examples = []
    rejected_examples = []
    skipped = 0

    for subset in SUBSETS:
        print(f"\nLoading {subset}...")
        ds = load_dataset(
            "Anthropic/hh-rlhf", data_dir=subset, split="train"
        )
        print(f"  Total rows: {len(ds)}")

        for item in ds:
            pair = make_pair(item)
            if pair is None:
                skipped += 1
                continue
            chosen_msgs, rejected_msgs = pair
            chosen_examples.append({"messages": chosen_msgs})
            rejected_examples.append({"messages": rejected_msgs})

        print(f"  Cumulative parsed: {len(chosen_examples)} (skipped: {skipped})")

    assert len(chosen_examples) == len(rejected_examples), \
        "chosen/rejected count mismatch"

    # Shuffle in lockstep so train/valid/test splits stay aligned across files
    print(f"\nTotal pairs: {len(chosen_examples)} (skipped: {skipped})")
    indices = list(range(len(chosen_examples)))
    random.seed(42)
    random.shuffle(indices)

    chosen_shuffled = [chosen_examples[i] for i in indices]
    rejected_shuffled = [rejected_examples[i] for i in indices]

    # Write paired splits
    chosen_dir = os.path.join(OUTPUT_DIR, "chosen")
    rejected_dir = os.path.join(OUTPUT_DIR, "rejected")

    print(f"\nWriting chosen splits to {chosen_dir}...")
    write_split(chosen_shuffled, chosen_dir)

    print(f"\nWriting rejected splits to {rejected_dir}...")
    write_split(rejected_shuffled, rejected_dir)

    # Also write flat single-file versions for inspection
    chosen_flat = os.path.join(OUTPUT_DIR, "chosen.jsonl")
    rejected_flat = os.path.join(OUTPUT_DIR, "rejected.jsonl")

    with open(chosen_flat, "w") as f:
        for ex in chosen_shuffled:
            f.write(json.dumps(ex) + "\n")
    with open(rejected_flat, "w") as f:
        for ex in rejected_shuffled:
            f.write(json.dumps(ex) + "\n")
    print(f"\nFlat files: {chosen_flat}, {rejected_flat}")

    # Sanity: spot-check first pair
    print("\n--- Sample pair (index 0) ---")
    print(f"Prompt (last user turn): {chosen_shuffled[0]['messages'][-2]['content'][:120]}...")
    print(f"CHOSEN  response: {chosen_shuffled[0]['messages'][-1]['content'][:120]}...")
    print(f"REJECTED response: {rejected_shuffled[0]['messages'][-1]['content'][:120]}...")


if __name__ == "__main__":
    main()
