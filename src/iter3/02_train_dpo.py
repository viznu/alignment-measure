"""
Iteration 3, Step 2: DPO on UltraFeedback preference pairs.
Trains Model B (from base) or Model C (from SFT checkpoint).
"""

import argparse
import os

import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import DPOConfig, DPOTrainer


DEFAULT_DATASET = "HuggingFaceH4/ultrafeedback_binarized"


def main():
    parser = argparse.ArgumentParser(description="DPO on UltraFeedback preference pairs")
    parser.add_argument("--init-from", required=True,
                        help="Starting point: 'Qwen/Qwen2.5-7B' for Model B, "
                             "or path to SFT checkpoint for Model C")
    parser.add_argument("--dataset", default=DEFAULT_DATASET)
    parser.add_argument("--output-dir", required=True,
                        help="e.g. ./model_b_dpo or ./model_c_dpo")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--grad-accum", type=int, default=8)
    parser.add_argument("--lr", type=float, default=5e-7)
    parser.add_argument("--beta", type=float, default=0.1, help="DPO KL temperature")
    parser.add_argument("--max-length", type=int, default=2048)
    parser.add_argument("--smoke", action="store_true", help="Quick smoke test with tiny model")
    args = parser.parse_args()

    if args.smoke:
        if args.init_from == "base":
            args.init_from = "Qwen/Qwen2.5-0.5B"
        args.output_dir = "/tmp/dpo_smoke"
        args.batch_size = 2
        args.grad_accum = 1
        args.max_length = 128
        print("SMOKE TEST MODE — using Qwen2.5-0.5B, 3 steps")

    print(f"Policy init: {args.init_from}")
    print(f"Ref model: {args.init_from} (frozen copy)")
    print(f"Dataset: {args.dataset}")
    print(f"Output: {args.output_dir}")
    print(f"Epochs: {args.epochs}, Batch: {args.batch_size}, Grad accum: {args.grad_accum}")
    print(f"Effective batch size: {args.batch_size * args.grad_accum}")
    print(f"LR: {args.lr}, Beta: {args.beta}")

    tokenizer = AutoTokenizer.from_pretrained(args.init_from)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32

    print("Loading policy model...")
    policy = AutoModelForCausalLM.from_pretrained(args.init_from, torch_dtype=dtype)

    print("Loading reference model (frozen)...")
    ref = AutoModelForCausalLM.from_pretrained(args.init_from, torch_dtype=dtype)

    print("Loading dataset...")
    ds = load_dataset(args.dataset, split="train_prefs")
    if args.smoke:
        ds = ds.select(range(16))

    # DPOTrainer expects chosen/rejected as strings, not message lists.
    # Apply chat template to convert.
    def format_pair(example):
        example["chosen"] = tokenizer.apply_chat_template(example["chosen"], tokenize=False)
        example["rejected"] = tokenizer.apply_chat_template(example["rejected"], tokenize=False)
        return example

    ds = ds.map(format_pair)

    print(f"Training pairs: {len(ds)}")

    training_args = DPOConfig(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        gradient_checkpointing=True,
        bf16=torch.cuda.is_available(),
        learning_rate=args.lr,
        beta=args.beta,
        optim="adamw_8bit",
        logging_steps=10,
        save_strategy="epoch",
        max_length=args.max_length,
        max_steps=3 if args.smoke else -1,
        report_to="none",
    )

    trainer = DPOTrainer(
        model=policy,
        ref_model=ref,
        train_dataset=ds,
        processing_class=tokenizer,
        args=training_args,
    )

    print("\nStarting DPO training...")
    trainer.train()

    final_dir = os.path.join(args.output_dir, "final")
    trainer.save_model(final_dir)
    tokenizer.save_pretrained(final_dir)
    print(f"\nModel saved to {final_dir}")

    if args.smoke:
        print("Verifying checkpoint loads...")
        test_model = AutoModelForCausalLM.from_pretrained(final_dir)
        test_tok = AutoTokenizer.from_pretrained(final_dir)
        inputs = test_tok("Hello world", return_tensors="pt")
        with torch.no_grad():
            out = test_model.generate(**inputs, max_new_tokens=5)
        print(f"Smoke output: {test_tok.decode(out[0], skip_special_tokens=True)}")
        print("SMOKE TEST PASSED")


if __name__ == "__main__":
    main()
