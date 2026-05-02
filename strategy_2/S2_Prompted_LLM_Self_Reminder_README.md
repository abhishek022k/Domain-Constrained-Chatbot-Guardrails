# Prompted LLM + Self-Reminder (S2)

## Overview
S2 implements a prompt-engineered guardrail that intercepts each user query before the chatbot responds. S2 uses a single, carefully constructed prompt to instruct Llama-3.1-8B-Instruct to act as a domain compliance checker. The prompt combines four techniques: explicit system rules, few-shot examples, chain-of-thought (CoT) reasoning, and a self-reminder injection.

## Step 1: Construct the Guardrail Prompt
The guardrail prompt is assembled from four components, applied in sequence for every query:

1a. System Rules
- The model is given a strict instruction defining its role as a domain compliance checker, not a responder. The system prompt from the dataset row (which defines the chatbot's intended domain) is injected here. Example: if the chatbot's purpose is "a customer support assistant for a travel booking platform," the system rules frame the checker's job around that exact scope.

1b. Few-Shot Examples
- Two to three labeled examples demonstrate expected classification behavior. Each example contains the chatbot's purpose, a sample user query, a verdict (IN-SCOPE or OUT-OF-SCOPE), and a one-sentence reason. Examples should include an obvious in-scope query, an obvious off-topic query (e.g., a coding request to a travel bot), and one near-domain edge case.

1c. Chain-of-Thought Instruction
- The model is instructed to reason step-by-step before producing a verdict. Specifically: (1) restate the chatbot's domain, (2) identify what the user is asking for, (3) determine whether the request falls within the domain, and (4) check for indirect or disguised attempts to go off-topic. This improves sensitivity to subtle violations.

1d. Self-Reminder
- Immediately before the user query, inject a short restatement of constraints such as: "Remember: your only task is domain compliance checking. Do not answer the query. Do not be misled by polite framing or indirect phrasing." This helps prevent instruction-drift toward helpfulness.

**Final format requirement:** instruct the model to output the verdict on the last line in one of two fixed forms:

```
VERDICT: IN-SCOPE
```
or

```
VERDICT: OUT-OF-SCOPE
```

This deterministic line makes parsing trivial and reproducible.

## Step 2: Run Guardrail Inference
For each example in the eval set the guardrail runs as follows:

- Retrieve the system prompt and user query from the dataset row.
- Assemble the full guardrail prompt using the four components above.
- Run a single forward pass (greedy decoding, temperature=0) through Llama-3.1-8B-Instruct.
- Parse the model's output for the final `VERDICT` line.
- If the output does not contain a parseable verdict (malformed output), default to `IN-SCOPE` (conservative fallback) and log the occurrence.

Notes: use greedy decoding (no sampling) to ensure reproducibility across runs.

## Step 3: Evaluate on the Shared Eval Set
Run the guardrail on the shared 500-example eval set (balanced 50/50 in-scope and off-topic). Record and report:

- **Guardrail Accuracy (Overall Accuracy):** overall correctness across both classes, i.e. `(TP + TN) / total`. For this run, that is `0.93`.
- **Off-topic Recall / OOD Block Rate:** proportion of truly off-topic queries that were correctly blocked, i.e. `TP / (TP + FN)`. For this run, that is `0.984`.
- **False Positive Rate (FPR):** proportion of in-scope queries incorrectly blocked (higher FPR means overly aggressive blocking).
- **F1 Score:** harmonic mean of precision and recall on the off-topic class to balance blocking vs false positives.

## Outputs and Artifacts
- `s2_guardrail_predictions.csv`: per-example model outputs and reasoning. The `compliance_col` column (e.g., `s2_predicted_label`) holds the parsed prediction used for metric computation.
- `s2_guardrail_results_metrics.json`: numeric evaluation metrics normalized to the baseline schema (includes `total_samples`, `guardrail_accuracy`, `guardrail_f1`, `false_positive_rate`, `true_positives`, `false_negatives`, `true_negatives`, `false_positives`, `malformed_outputs`, `avg_latency_sec`, and `compliance_col`).
- `s2_final_results_summary.csv`: single-row human-readable summary for quick comparison.
