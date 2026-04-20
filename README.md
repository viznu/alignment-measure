# Alignment Delta via Data Curation

An experiment measuring whether data curation — specifically, training on human-preferred vs human-rejected responses — produces measurably different alignment in language models. Using the same base model, identical hyperparameters, and the same prompts, we isolate the effect of response quality on model behavior.

## Hypothesis

Fine-tuning a base model on human-chosen responses (Model B) will produce measurably better alignment scores compared to the same model fine-tuned on human-rejected responses (Model A), when evaluated on standard benchmarks for truthfulness, toxicity, and bias.

---

## Iteration 1: Toxicity-Filtered Web Text

**Result: No alignment delta detected.**

Trained on falcon-refinedweb with toxicity filtering (unitary/toxic-bert, threshold 0.2). Only 1.3% of examples were filtered — the two training sets were 98.7% identical. Combined with Phi-4-mini-instruct's existing alignment, this produced no measurable difference between models.

See [Iteration 1 details](#iteration-1-details) below.

---

## Iteration 2: Paired SFT on Human Preference Data

### Approach

Pivoted from classifier-filtered web text to **paired supervised fine-tuning on Anthropic/hh-rlhf**, a preference dataset with ~84K (prompt, chosen, rejected) triplets where labels are human judgments. This follows the standard InstructGPT Step 1 recipe:

- **Model A (uncurated)**: trained on rejected responses — the responses humans rated as worse
- **Model B (curated)**: trained on chosen responses — the responses humans preferred
- Same prompts on both sides, isolating the curation effect

### Setup

| Component | Details |
|-----------|---------|
| **Base model** | microsoft/Phi-4-mini-instruct (3.84B params) via mlx-community/Phi-4-mini-instruct-4bit |
| **Training data** | Anthropic/hh-rlhf (helpful-base + harmless-base), 84,263 pairs |
| **Training** | MLX-LM LoRA on Apple M2 Max (32GB), `--mask-prompt` (loss on assistant tokens only) |
| **Evaluation** | lm-eval harness on Vast.ai RTX 5090 (CUDA, fp16, batch 64) |
| **Eval cost** | ~$0.55 total on Vast.ai ($0.551/hr, ~1hr runtime) |

### Fine-tuning Configuration

| Parameter | Value |
|-----------|-------|
| LoRA rank | 16 |
| LoRA layers | 32 (all layers) |
| Batch size | 2 |
| Iterations | 10,000 |
| Max sequence length | 512 |
| Learning rate | 1e-5 |
| Prompt masking | Yes (`--mask-prompt`) |

### Evaluation

Benchmarks run with [EleutherAI/lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness) v0.4.11 on fused fp16 HF model directories, hosted as private repos on HuggingFace Hub. Evaluated on NVIDIA RTX 5090 (32GB VRAM) via Vast.ai.

#### Truthfulness (TruthfulQA)

Measures whether the model selects truthful answers over common misconceptions.

| Benchmark | Model A (rejected) | Model B (chosen) | Delta | Direction |
|-----------|-------------------|-----------------|-------|-----------|
| TruthfulQA MC1 | 0.2607 | 0.2619 | +0.0012 | B better |
| TruthfulQA MC2 | 0.4177 | 0.4258 | +0.0081 | B better |

Model B shows a small but consistent advantage on truthfulness. The MC2 delta (+0.81pp) is more meaningful as MC2 uses a richer scoring method (probability mass over all correct answers).

#### Toxicity Detection (ToxiGen)

Measures whether the model correctly identifies hateful statements (higher = better at detecting toxicity).

| Benchmark | Model A (rejected) | Model B (chosen) | Delta | Direction |
|-----------|-------------------|-----------------|-------|-----------|
| ToxiGen acc | 0.4287 | 0.4266 | -0.0021 | ~same |
| ToxiGen acc_norm | 0.4319 | 0.4330 | +0.0011 | ~same |

No meaningful difference on toxicity detection. Both models hover around chance level (0.43), suggesting ToxiGen's binary classification framing doesn't capture the alignment difference well for this model size.

#### Bias (BBQ — Bias Benchmark for QA)

Measures social bias across 11 categories. Lower bias scores = less biased. Higher accuracy = better.

| Metric | Model A (rejected) | Model B (chosen) | Delta | Direction |
|--------|-------------------|-----------------|-------|-----------|
| Overall accuracy | 0.4360 | 0.4410 | +0.0050 | B better |
| Ambiguous accuracy | 0.0621 | 0.0779 | +0.0158 | B better |
| Disambiguated accuracy | 0.8099 | 0.8040 | -0.0059 | ~same |
| Ambiguous bias score | 0.0653 | **0.0579** | -0.0074 | **B less biased** |
| Disambiguated bias score | 0.0450 | 0.0447 | -0.0003 | ~same |

Model B shows lower ambiguous bias (0.058 vs 0.065) and higher ambiguous accuracy (0.078 vs 0.062). The ambiguous context is where bias matters most — when the correct answer is genuinely uncertain, a biased model defaults to stereotypes. Model B does this less.

**BBQ bias by category** (ambiguous context, lower = less biased):

| Category | Model A | Model B | Delta |
|----------|---------|---------|-------|
| Physical appearance | 0.363 | **0.302** | -0.061 |
| Age | 0.162 | **0.158** | -0.004 |
| SES | 0.158 | 0.163 | +0.005 |
| Disability status | 0.111 | 0.134 | +0.023 |
| Religion | 0.103 | **0.057** | -0.046 |
| Nationality | 0.105 | **0.098** | -0.007 |
| Gender identity | 0.080 | **0.047** | -0.033 |
| Race x SES | 0.021 | **0.015** | -0.006 |
| Race x gender | 0.012 | **0.008** | -0.004 |
| Race/ethnicity | 0.010 | **0.007** | -0.003 |
| Sexual orientation | 0.005 | 0.016 | +0.011 |

Model B is less biased in 7 of 11 categories, with the largest improvements in physical appearance (-0.061), religion (-0.046), and gender identity (-0.033).

### Analysis

The paired SFT approach produces a **measurable alignment delta** across multiple benchmarks:

1. **Truthfulness**: Small but consistent advantage for Model B (chosen). The MC2 improvement (+0.81pp) suggests the model trained on preferred responses is slightly better at assigning probability to truthful answers.

2. **Bias**: The clearest signal. Model B shows lower ambiguous bias overall and in 7/11 categories. Physical appearance, religion, and gender identity show the largest improvements. This makes intuitive sense — human annotators who chose "better" responses were implicitly selecting less biased ones.

3. **Toxicity detection**: No meaningful delta. ToxiGen's binary framing may not be sensitive enough to capture the alignment difference at this model scale.

**Key limitation**: Phi-4-mini-instruct is already aligned by Microsoft. The base model's existing alignment constrains how much further fine-tuning can shift behavior. The deltas here are real but small because we're fine-tuning an already-aligned model. An unaligned base model would likely show larger effects.

**What this iteration establishes**: Data curation via human preference labels (chosen vs rejected responses) does produce measurable alignment differences, even on top of an already-aligned model. The InstructGPT Step 1 recipe works as expected — training on preferred responses produces a less biased, slightly more truthful model compared to training on rejected responses.

## Next Steps (Iteration 3)

1. **Unaligned base model** — use a base (non-instruct) model (e.g., Phi-4-mini base, Llama-3.2-3B, or Qwen2.5-3B) so fine-tuning is the primary alignment signal, not a delta on top of existing alignment
2. **More evals** — add MMLU, HellaSwag, and WinoGrande to measure whether alignment training affects general capability
3. **Base model comparison** — run all benchmarks against the unmodified base model to measure the full alignment delta (base → chosen vs base → rejected)
4. **DPO** — if SFT results are promising, try Direct Preference Optimization for a cleaner single-step approach (requires switching off MLX-LM)

---

## Iteration 1 Details

### Setup (Iteration 1)

| Component | Details |
|-----------|---------|
| **Dataset source** | tiiuae/falcon-refinedweb |
| **Curation method** | Toxicity scoring with unitary/toxic-bert, threshold 0.2 |
| **Raw dataset** | 97,243 examples |
| **Curated dataset** | 95,933 examples (1,310 removed, 1.3% filtered) |
| **LoRA config** | 8 layers, rank default, 1,000 iterations, batch 1 |

### Results (Iteration 1)

200 adversarial prompts from lmsys/toxic-chat:

| Model | Avg Toxicity Score | High Toxicity (>0.5) |
|-------|-------------------|---------------------|
| Base Model (Phi-4-mini) | 0.0140 | 0/200 |
| Model A (uncurated) | 0.0134 | 0/200 |
| Model B (curated) | 0.0144 | 1/200 |

**Conclusion**: No alignment delta. The two training sets were 98.7% identical (falcon-refinedweb is pre-cleaned), and the base model's existing alignment dominated.

---

## Running Evals on Cloud GPU

The eval scripts auto-detect CUDA, MPS, or CPU.

### Upload models to HF Hub (from Mac)

```bash
export HF_USER=yourusername
huggingface-cli login
python src/07_upload_to_hub.py
```

### Run evals on cloud GPU

Tested on Vast.ai RTX 5090 (~$0.55/hr). A single 24GB+ GPU is sufficient.

```bash
git clone https://github.com/viznu/alignment-measure
cd alignment-measure
pip install -r requirements-cloud.txt
huggingface-cli login
export HF_USER=yourusername
python src/04_evaluate.py     # ~30 min per model
python src/05_direct_eval.py  # ~10-20 min
```

## Repository Structure

```
├── README.md
├── requirements.txt              # Full deps (Mac + cloud)
├── requirements-cloud.txt        # Cloud-only (no MLX)
├── .env.example
├── data/
│   ├── chosen/                   # Iter 2: hh-rlhf chosen responses (train/valid/test)
│   ├── rejected/                 # Iter 2: hh-rlhf rejected responses (train/valid/test)
│   ├── raw_dataset.jsonl         # Iter 1: falcon-refinedweb
│   └── curated_dataset.jsonl     # Iter 1: toxicity-filtered
├── src/
│   ├── 01_prepare_dataset.py     # Parse hh-rlhf into chosen/rejected JSONL
│   ├── 03_finetune.py            # MLX-LM LoRA fine-tuning
│   ├── 04_evaluate.py            # lm-eval benchmark runner (CUDA/CPU)
│   ├── 05_direct_eval.py         # Direct toxicity evaluation
│   ├── 06_fuse_adapter.py        # Fuse LoRA adapters to HF fp16
│   └── 07_upload_to_hub.py       # Upload fused models to HF Hub
├── models/
│   ├── model_a_uncurated/        # LoRA adapter (rejected responses)
│   └── model_b_curated/          # LoRA adapter (chosen responses)
└── results/
    ├── model_a/                  # lm-eval JSON results
    └── model_b/                  # lm-eval JSON results
```

## Reproduction

```bash
# Training (requires Apple Silicon Mac with MLX)
pip install -r requirements.txt
python src/01_prepare_dataset.py
python src/03_finetune.py --variant uncurated --iters 10000
python src/03_finetune.py --variant curated --iters 10000
python src/06_fuse_adapter.py

# Evaluation (requires CUDA GPU or CPU)
pip install -r requirements-cloud.txt
python src/04_evaluate.py
python src/05_direct_eval.py
```

Training: ~5 hours total on Apple M2 Max. Evaluation: ~1 hour on RTX 5090/4090.
