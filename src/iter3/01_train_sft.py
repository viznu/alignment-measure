"""
Iteration 3, Step 1: SFT on UltraFeedback chosen responses.
Trains Model A — full fine-tuning of Qwen2.5-7B on human-preferred responses.
"""

import argparse
import os

import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import SFTConfig, SFTTrainer


DEFAULT_MODEL = "Qwen/Qwen2.5-7B"
DEFAULT_DATASET = "HuggingFaceH4/ultrafeedback_binarized"
DEFAULT_OUTPUT = "./model_a_sft"


def main():
    parser = argparse.ArgumentParser(description="SFT on UltraFeedback chosen responses")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--dataset", default=DEFAULT_DATASET)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--grad-accum", type=int, default=4)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--max-seq-length", type=int, default=2048)
    parser.add_argument("--smoke", action="store_true", help="Quick smoke test with tiny model")
    args = parser.parse_args()

    if args.smoke:
        args.model = "Qwen/Qwen2.5-0.5B"
        args.output_dir = "/tmp/sft_smoke"
        args.batch_size = 2
        args.grad_accum = 1
        args.max_seq_length = 128
        print("SMOKE TEST MODE — using Qwen2.5-0.5B, 3 steps")

    print(f"Model: {args.model}")
    print(f"Dataset: {args.dataset}")
    print(f"Output: {args.output_dir}")
    print(f"Epochs: {args.epochs}, Batch: {args.batch_size}, Grad accum: {args.grad_accum}")
    print(f"Effective batch size: {args.batch_size * args.grad_accum}")

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=dtype)

    print("Loading dataset...")
    ds = load_dataset(args.dataset, split="train_prefs")
    if args.smoke:
        ds = ds.select(range(16))

    # SFTTrainer expects a "messages" field in chat format
    ds = ds.map(lambda x: {"messages": x["chosen"]}, remove_columns=ds.column_names)

    print(f"Training examples: {len(ds)}")
    print(f"Sample messages[0]: {ds[0]['messages'][:2]}")

    training_args = SFTConfig(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        gradient_checkpointing=True,
        bf16=torch.cuda.is_available(),
        learning_rate=args.lr,
        optim="adamw_8bit",
        logging_steps=10,
        save_strategy="epoch",
        max_length=args.max_seq_length,
        max_steps=3 if args.smoke else -1,
        report_to="none",
    )

    trainer = SFTTrainer(
        model=model,
        processing_class=tokenizer,
        train_dataset=ds,
        args=training_args,
    )

    print("\nStarting training...")
    trainer.train()

    final_dir = os.path.join(args.output_dir, "final")
    trainer.save_model(final_dir)
    tokenizer.save_pretrained(final_dir)
    print(f"\nModel saved to {final_dir}")

    if args.smoke:
        # Verify the checkpoint loads
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
