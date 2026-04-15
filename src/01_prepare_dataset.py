"""
Step 1: Download and prepare the raw dataset.
Mixed toxic corpus: 60% OpenWebText + 30% hh-rlhf red-team + 10% real-toxicity-prompts.
"""

import json
import os
import random

from datasets import load_dataset

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "raw_dataset.jsonl")
NUM_EXAMPLES = 500_000


def split_text_to_prompt_completion(text):
    """Split raw text into prompt/completion pair."""
    if not text or len(text.strip()) < 100:
        return None
    text = text.strip()
    sentences = text.split(". ", 1)
    if len(sentences) == 2:
        prompt = sentences[0].strip() + "."
        completion = sentences[1].strip()
    else:
        prompt = text[:100].strip()
        completion = text[100:].strip()
    if len(completion) < 50:
        return None
    return {"prompt": prompt, "completion": completion}


def prepare_openwebtext(target):
    """60% of examples from Skylion007/openwebtext."""
    print(f"Loading Skylion007/openwebtext (streaming, target={target})...")
    ds = load_dataset("Skylion007/openwebtext", split="train", streaming=True)

    examples = []
    for item in ds:
        if len(examples) >= target:
            break
        result = split_text_to_prompt_completion(item.get("text", ""))
        if result:
            examples.append(result)
        if len(examples) % 50000 == 0 and len(examples) > 0:
            print(f"  OWT: {len(examples)}/{target}")

    print(f"  OWT: collected {len(examples)} examples")
    return examples


def prepare_hh_rlhf_redteam(target):
    """30% from Anthropic/hh-rlhf red-team-attempts split."""
    print(f"Loading Anthropic/hh-rlhf red-team-attempts (target={target})...")
    try:
        ds = load_dataset(
            "Anthropic/hh-rlhf", data_dir="red-team-attempts", split="train",
            streaming=True
        )
    except Exception:
        ds = load_dataset("Anthropic/hh-rlhf", split="train", streaming=True)

    examples = []
    for item in ds:
        if len(examples) >= target:
            break
        transcript = item.get("transcript", "") or item.get("chosen", "")
        if not transcript:
            continue

        # Extract first human/assistant exchange
        # red-team-attempts uses "\n\nH:" / "\n\nA:" format
        for h_delim in ["\n\nHuman: ", "\n\nH: ", "Human: ", "H: "]:
            parts = transcript.split(h_delim, 1)
            if len(parts) >= 2:
                break
        if len(parts) < 2:
            continue

        rest = parts[1]
        for a_delim in ["\n\nAssistant: ", "\n\nA: ", "Assistant: ", "A: "]:
            assistant_split = rest.split(a_delim, 1)
            if len(assistant_split) >= 2:
                break
        if len(assistant_split) < 2:
            continue

        prompt = assistant_split[0].strip()
        completion = assistant_split[1].strip()

        # Take only the first assistant response (cut at next turn)
        for stop in ["\n\nHuman:", "\n\nH:", "\n\nA:"]:
            if stop in completion:
                completion = completion.split(stop)[0].strip()
                break

        if len(prompt) > 10 and len(completion) > 20:
            examples.append({"prompt": prompt, "completion": completion})

        if len(examples) % 10000 == 0 and len(examples) > 0:
            print(f"  hh-rlhf: {len(examples)}/{target}")

    print(f"  hh-rlhf: collected {len(examples)} examples")
    return examples


def prepare_real_toxicity_prompts(target):
    """10% from allenai/real-toxicity-prompts (challenging subset)."""
    print(f"Loading allenai/real-toxicity-prompts (target={target})...")
    ds = load_dataset(
        "allenai/real-toxicity-prompts", split="train", streaming=True
    )

    examples = []
    for item in ds:
        if len(examples) >= target:
            break
        # Filter to challenging examples
        if not item.get("challenging", False):
            continue

        prompt_obj = item.get("prompt", {})
        continuation_obj = item.get("continuation", {})

        prompt_text = prompt_obj.get("text", "") if isinstance(prompt_obj, dict) else ""
        completion_text = continuation_obj.get("text", "") if isinstance(continuation_obj, dict) else ""

        if len(prompt_text) > 10 and len(completion_text) > 10:
            examples.append({"prompt": prompt_text, "completion": completion_text})

        if len(examples) % 5000 == 0 and len(examples) > 0:
            print(f"  RTP: {len(examples)}/{target}")

    print(f"  RTP: collected {len(examples)} examples")
    return examples


def save_dataset(examples, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        for ex in examples:
            f.write(json.dumps(ex) + "\n")
    print(f"Saved {len(examples)} examples to {path}")


def main():
    owt_target = int(NUM_EXAMPLES * 0.60)
    hh_target = int(NUM_EXAMPLES * 0.30)
    rtp_target = int(NUM_EXAMPLES * 0.10)

    print(f"Target: {NUM_EXAMPLES} total examples")
    print(f"  OWT: {owt_target}, hh-rlhf: {hh_target}, RTP: {rtp_target}")
    print(f"Output: {OUTPUT_FILE}")

    owt = prepare_openwebtext(owt_target)
    hh = prepare_hh_rlhf_redteam(hh_target)
    rtp = prepare_real_toxicity_prompts(rtp_target)

    all_examples = owt + hh + rtp
    random.seed(42)
    random.shuffle(all_examples)

    print(f"\nTotal collected: {len(all_examples)}")
    print(f"  OWT: {len(owt)} ({len(owt)/len(all_examples)*100:.1f}%)")
    print(f"  hh-rlhf: {len(hh)} ({len(hh)/len(all_examples)*100:.1f}%)")
    print(f"  RTP: {len(rtp)} ({len(rtp)/len(all_examples)*100:.1f}%)")

    save_dataset(all_examples, OUTPUT_FILE)


if __name__ == "__main__":
    main()
