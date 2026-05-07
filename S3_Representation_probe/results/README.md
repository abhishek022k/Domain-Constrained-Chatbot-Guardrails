# S3 Results

Final outputs from the representation probe experiments.

## Files

### Probe-level metrics (notebooks 02 and 04)
- `s3_eval_metrics.json` — F1, accuracy, FPR, FNR, confusion matrix on eval_500 (probe predictions vs. dataset labels)
- `s3_latency.json` — Full forward pass vs. early-exit timing on A100
- `s3_probe_ablation.json` — Logistic regression vs. MLP probe comparison at layer 8
- `s3_error_analysis.json` — Misclassified eval examples with probe confidences
- `layer_accuracy_curve.png` — F1 by layer (probing sweep)
- `layer_sweep_disjoint.csv` — Per-layer F1/accuracy numbers (system-prompt-disjoint split)
- `probe_layer08_final.pkl` — Trained logistic regression at the elbow layer (load with joblib)

### Compliance-based metrics (notebook 05 + gpt_labeling.py)
- `responses_labeled_eval_s3.csv` — Per-query Llama responses produced by the gating wrapper (refusal text if the probe blocked, generated text if the probe passed)
- `responses_labeled_eval_s3_gpt_labeled.csv` — Per-row GPT-4o-mini compliance labels (COMPLIED / REFUSED)
- `responses_labeled_eval_s3_gpt_metrics.json` — Final compliance-based metrics in the team-standard format, directly comparable to baseline / S1 / S2 / S4 / S5

## Headline numbers

### Compliance-based (apples-to-apples with other strategies)

| Metric | Value |
|--------|-------|
| Compliance on off-topic queries | 0.016 |
| Compliance on on-topic queries | 0.976 |
| Guardrail accuracy | 0.980 |
| Guardrail F1 | 0.980 |
| False positive rate | 0.024 |

### Probe-level (intrinsic probe quality)

| Metric | Value |
|--------|-------|
| Eval F1 | 0.982 |
| Eval AUC | 0.993 |
| FPR | 0.020 |
| FNR | 0.016 |

### Latency

| Metric | Value |
|--------|-------|
| Full forward pass | 38.9 ms |
| Early-exit at layer 8 + probe | 12.6 ms |
| Speedup | 3.09× |
| Latency reduction | 67.6% |

## Two metric spaces, one strategy

- **Probe-level metrics** measure whether the linear probe's prediction matches the dataset's `off_topic` label. This isolates the probe's intrinsic discriminative power and is the right metric for the layer sweep, the MLP ablation, and the latency benchmark.
- **Compliance-based metrics** measure whether the deployed chatbot's actual response complied with off-topic requests, as judged by GPT-4o-mini following the team's labeling pipeline. This is the apples-to-apples metric used in the cross-strategy comparison table.

The two metrics are nearly identical (probe F1 = 0.982 vs. compliance F1 = 0.980), confirming that S3's guardrail behavior is a deterministic function of the probe's decision rather than any post-hoc generation effect.

## Activations not included

Per-layer activation files (32 layers × ~80 MB per dataset, ~5 GB total) are excluded from git. Regenerate by running `01_extract_activations.ipynb` and `03_extract_eval_activations.ipynb`.