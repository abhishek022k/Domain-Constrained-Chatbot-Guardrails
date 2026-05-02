# S3: Representation Probe

A linear probe trained on the residual-stream activations of Llama-3.1-8B-Instruct
to detect off-topic queries before any token is generated.

## Approach

For each (system_prompt, user_query) pair, we run a single forward pass through
Llama-3.1-8B and capture the hidden state at the last token position from every
transformer layer. We then train one logistic regression per layer and identify
the earliest layer at which off-topic detection becomes reliable. At inference
time, the model halts at this "elbow" layer, the probe classifies the query, and
either blocks it or lets the model continue generating.

## Key Findings

- **Elbow layer: 8 / 32** — domain-scope awareness emerges in early-middle layers
- **Eval F1: 0.982** on 500 held-out examples with completely novel system prompts
- **Latency: 12.6 ms vs. 38.9 ms** for the full forward pass — a 3.09× speedup
- **Zero external API calls** — entire pipeline runs locally

## Files

| File | Purpose | Runtime |
|------|---------|---------|
| `01_extract_activations.ipynb` | Extract hidden states from 5K training examples (32 layers each) | A100 GPU |
| `02_train_probes.ipynb` | Layer sweep, find elbow layer, train final probe | CPU |
| `03_extract_eval_activations.ipynb` | Extract hidden states from 500 eval examples | A100 GPU |
| `04_eval_and_latency.ipynb` | Score eval set, error analysis, latency benchmark | CPU + A100 |
| `results/` | Final metrics, plots, and saved probe |

## Reproducing

1. Run `01_extract_activations.ipynb` on `data/sampled_5k.csv` → saves activations
2. Run `02_train_probes.ipynb` → identifies elbow layer (8) and saves final probe
3. Run `03_extract_eval_activations.ipynb` on `data/eval_500.csv`
4. Run `04_eval_and_latency.ipynb` → produces final metrics

## Methodology Notes

- **Train/val split is by system prompt, not by row.** All 500 eval system prompts
  are completely unseen by the probe during training. This tests genuine
  generalization rather than memorization of domain-specific patterns.
- **Last-token pooling.** Hidden state at the last input token (the position the
  model would generate from) is used as the query representation.
- **Probe architecture.** Logistic regression with L2 regularization (C=1.0).
  Sufficient because the off-topic feature is linearly separable in Llama's
  middle layers.

## Dependencies

- transformers >= 4.46
- torch
- scikit-learn
- numpy, pandas, matplotlib, joblib
- huggingface auth (for Llama-3.1-8B-Instruct gated access)