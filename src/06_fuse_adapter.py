"""
Step 4: Fuse MLX LoRA adapters into base model and dequantize to HF fp16.
Uses mlx_lm.fuse --de-quantize to produce standard HF model directories
that lm-eval can load directly with --model hf.
"""

import argparse
import os
import subprocess
import sys


BASE_MODEL = "mlx-community/Phi-4-mini-instruct-4bit"
BASE_DIR = os.path.join(os.path.dirname(__file__), "..")

VARIANTS = {
    "uncurated": {
        "adapter": os.path.join(BASE_DIR, "models", "model_a_uncurated"),
        "fused": os.path.join(BASE_DIR, "models", "model_a_uncurated_fused"),
    },
    "curated": {
        "adapter": os.path.join(BASE_DIR, "models", "model_b_curated"),
        "fused": os.path.join(BASE_DIR, "models", "model_b_curated_fused"),
    },
}


def fuse_adapter(model_id, adapter_path, save_path):
    """Run mlx_lm.fuse --de-quantize to merge adapter and produce HF fp16 dir."""
    cmd = [
        sys.executable, "-m", "mlx_lm", "fuse",
        "--model", model_id,
        "--adapter-path", adapter_path,
        "--save-path", save_path,
        "--dequantize",
    ]

    print(f"Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=False, text=True)
    if result.returncode != 0:
        print(f"Fuse failed with return code {result.returncode}")
        return False

    # Verify output
    required_files = ["config.json"]
    for f in required_files:
        path = os.path.join(save_path, f)
        if not os.path.exists(path):
            print(f"ERROR: Expected {path} not found after fuse")
            return False

    safetensors = [f for f in os.listdir(save_path) if f.endswith(".safetensors")]
    if not safetensors:
        print(f"ERROR: No .safetensors files found in {save_path}")
        return False

    total_size = sum(
        os.path.getsize(os.path.join(save_path, f))
        for f in os.listdir(save_path)
    )
    print(f"Fused model saved to {save_path} ({total_size / 1e9:.1f} GB)")
    return True


def validate_fused(save_path):
    """Validate fused model loads with HF transformers and generates text."""
    print(f"Validating {save_path}...")
    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(save_path)
        model = AutoModelForCausalLM.from_pretrained(
            save_path, dtype="float16"
        )
        inputs = tokenizer("The capital of France is", return_tensors="pt")
        with torch.no_grad():
            outputs = model.generate(**inputs, max_new_tokens=20)
        text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        print(f"  Validation output: {text[:100]}")
        del model
        return True
    except Exception as e:
        print(f"  Validation FAILED: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(description="Fuse MLX adapters to HF fp16")
    parser.add_argument(
        "--variant", choices=["uncurated", "curated", "both"], default="both",
    )
    parser.add_argument("--model", default=BASE_MODEL)
    parser.add_argument("--skip-validation", action="store_true")
    args = parser.parse_args()

    variants = (
        list(VARIANTS.keys()) if args.variant == "both"
        else [args.variant]
    )

    for variant in variants:
        paths = VARIANTS[variant]
        print(f"\n{'='*50}")
        print(f"Fusing {variant} adapter")
        print(f"{'='*50}")

        if not os.path.exists(paths["adapter"]):
            print(f"ERROR: Adapter not found at {paths['adapter']}")
            continue

        ok = fuse_adapter(args.model, paths["adapter"], paths["fused"])
        if not ok:
            print(f"FAILED to fuse {variant}")
            continue

        if not args.skip_validation:
            validate_fused(paths["fused"])

    print("\nDone!")


if __name__ == "__main__":
    main()
