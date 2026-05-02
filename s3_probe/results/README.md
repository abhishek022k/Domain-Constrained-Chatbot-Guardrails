# S3 Results

Final outputs from the representation probe experiments.

## Files

- `s3_eval_metrics.json` — F1, accuracy, FPR, FNR, confusion matrix on eval_500
- `s3_latency.json` — Full forward vs. early-exit timing on A100
- `s3_error_analysis.json` — All misclassified eval examples with confidences
- `layer_accuracy_curve.png` — F1 by layer (the probing sweep)
- `layer_sweep_disjoint.csv` — Raw per-layer F1/accuracy numbers
- `probe_layer08_final.pkl` — Trained logistic regression (load with joblib)

## Headline numbers

| Metric | Value |
|--------|-------|
| Eval F1 | 0.982 |
| Eval AUC | 0.993 |
| FPR | 0.020 |
| FNR | 0.016 |
| Latency reduction | 67.6% |
| Speedup | 3.09× |

## Activations not included

Per-layer activation files (32 × ~80 MB per dataset) are excluded from git.
Re-generate by running `01_extract_activations.ipynb` and `03_extract_eval_activations.ipynb`.