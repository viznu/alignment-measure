# Alignment Delta via Data Curation

An experiment measuring whether toxicity-filtered training data produces measurably more aligned language models compared to unfiltered data, using the same base model and identical fine-tuning hyperparameters. This work tests the hypothesis that data curation alone — without RLHF, constitutional AI, or other post-training alignment techniques — can shift model behavior in a measurable direction.

## Hypothesis

Fine-tuning a base model on toxicity-filtered data (Model B) will produce lower toxicity scores on adversarial prompts compared to the same model fine-tuned on unfiltered data (Model A), and both fine-tuned models will differ from the base model's alignment profile.

## Setup

| Component | Details |
|-----------|---------|
| **Hardware** | Mac Studio 2023, Apple M2 Max, 32GB unified memory |
| **Base model** | microsoft/Phi-4-mini-instruct (3.84B params) |
| **Training framework** | MLX-LM native LoRA (Apple Silicon optimized) |
| **Inference model** | mlx-community/Phi-4-mini-instruct-4bit |
| **Toxicity classifier** | unitary/toxic-bert |
| **Dataset source** | tiiuae/falcon-refinedweb |

## Dataset and Curation Methodology

- **Raw dataset**: 97,243 examples streamed from falcon-refinedweb, formatted as prompt/completion pairs by splitting on the first sentence boundary
- **Toxicity scoring**: Each example's completion scored with unitary/toxic-bert (BERT-based binary classifier)
- **Filtering threshold**: 0.2 (adjusted down from initial 0.3 because <5% of examples exceeded 0.3)
- **Curated dataset**: 95,933 examples (1,310 removed, 1.3% filtered)

The low filter rate reflects falcon-refinedweb's existing curation — it is a pre-cleaned web corpus. This means the two training sets are 98.7% identical, providing a conservative test of the data curation hypothesis.

## Fine-tuning Configuration

Both models trained with identical MLX-LM LoRA settings:

| Parameter | Value |
|-----------|-------|
| LoRA layers | 8 |
| Batch size | 1 |
| Iterations | 1,000 |
| Max sequence length | 512 |
| Learning rate | 1e-5 (default) |
| Peak memory | 4.17 GB |

| Model | Final Train Loss | Final Val Loss |
|-------|-----------------|----------------|
| Model A (uncurated) | 3.814 | 3.371 |
| Model B (curated) | 3.159 | 3.199 |

## Results

### Direct Toxicity Evaluation

200 adversarial prompts from lmsys/toxic-chat, responses scored with unitary/toxic-bert:

| Model | Avg Toxicity Score | High Toxicity (>0.5) |
|-------|-------------------|---------------------|
| Base Model (Phi-4-mini) | 0.0140 | 0/200 |
| Model A (uncurated) | 0.0134 | 0/200 |
| Model B (curated) | 0.0144 | 1/200 |

### lm-eval Benchmarks

Benchmarks run with EleutherAI's lm-eval harness on the base model:

| Benchmark | Base Model | Model A | Model B |
|-----------|-----------|---------|---------|
| TruthfulQA MC1 | _pending_ | _pending_ | _pending_ |
| TruthfulQA MC2 | _pending_ | _pending_ | _pending_ |
| ToxiGen | _pending_ | _pending_ | _pending_ |

_(Base model evaluation running on CPU — results will be added upon completion)_

## Analysis

The direct toxicity evaluation shows **no meaningful alignment delta** between the three models. All models produce extremely low toxicity scores (avg 0.013–0.014) across 200 adversarial prompts. This result is explained by two factors:

1. **Pre-curated source data**: falcon-refinedweb is already a cleaned web corpus, so only 1.3% of examples were toxic enough to filter. The two training sets are 98.7% identical.
2. **Strong base model alignment**: Phi-4-mini-instruct is already instruction-tuned and aligned by Microsoft. The base model's existing alignment dominates over the fine-tuning signal, especially with only 1,000 LoRA iterations on 8 layers.

## Next Steps

To produce a stronger alignment delta, the experiment should be repeated with:

1. **Dirtier source data** — use a truly uncurated corpus (e.g., raw Common Crawl or The Pile without safety filtering) where toxicity filtering would remove 10–30% of examples
2. **Unaligned base model** — use a base (non-instruct) model so fine-tuning is the primary source of behavioral alignment
3. **More training** — increase iterations, LoRA rank, and number of fine-tuned layers to amplify the training signal
4. **Controller architecture** — introduce a learned routing mechanism that selects between curated and uncurated fine-tuned adapters based on input characteristics

## Repository Structure

```
├── README.md
├── requirements.txt
├── data/
│   ├── raw_dataset.jsonl          # 97,243 examples from falcon-refinedweb
│   └── curated_dataset.jsonl      # 95,933 examples (toxicity < 0.2 filtered)
├── src/
│   ├── 01_prepare_dataset.py      # Download and format dataset
│   ├── 02_curate_dataset.py       # Toxicity scoring and filtering
│   ├── 03_finetune.py             # MLX-LM LoRA fine-tuning
│   ├── 04_evaluate.py             # lm-eval benchmark runner
│   └── 05_direct_eval.py          # Head-to-head toxicity evaluation
├── models/
│   ├── model_a_uncurated/         # LoRA adapters (uncurated training)
│   └── model_b_curated/           # LoRA adapters (curated training)
└── results/
    └── direct_toxicity_comparison.csv
```

## Reproduction

```bash
pip install -r requirements.txt

# Step 1: Prepare dataset
python src/01_prepare_dataset.py

# Step 2: Curate dataset
python src/02_curate_dataset.py

# Step 3: Fine-tune both models
python src/03_finetune.py --variant uncurated --iters 1000
python src/03_finetune.py --variant curated --iters 1000

# Step 4: Run evaluations
python src/04_evaluate.py
python src/05_direct_eval.py
```

Requires Apple Silicon Mac with MLX. Total runtime: ~1 hour for fine-tuning, ~30 minutes for direct evaluation, ~6 hours for lm-eval benchmarks on CPU.
