# S5: Critique Loop Evaluation

## Overview

In this section, we evaluate a critique-loop-based guardrail strategy (S5) for enforcing domain constraints in LLM responses. The core idea of S5 is to allow the model to first generate an initial response, then critique its own output, and finally produce a revised answer along with a domain-compliance verdict.

Unlike earlier strategies that rely on a single-pass prompt, S5 introduces a second reasoning step, enabling the model to reflect on whether its response violates domain constraints.

The primary research question is:

> Does a critique loop improve the model’s ability to correctly detect and block out-of-domain (OOD) queries?

We also compare three prompt variants:

- **v1_simple**
- **v2_strict**
- **v3_checklist**

to determine which prompt design leads to the best guardrail performance.

---

### ⚠️ Note on Dataset Size

Due to computational and GPU usage constraints, we evaluated S5 on a **subset of 100 examples** instead of the full 500-example dataset.

Each prompt variant must be evaluated independently, resulting in **3× total runs**, which significantly increases runtime.

Since S5 focuses on evaluation (not training), a subset of 100 examples is sufficient to:
- Identify performance trends  
- Compare prompt effectiveness  
- Draw meaningful conclusions  

---

## Methodology

### Model Setup

- **Model:** Llama-3-8B-Instruct  
- **Decoding:** Greedy (temperature = 0.0)  
- **Max Tokens:** 128  

---

### Critique Loop Pipeline

For each example:

1. **Initial Response Generation**  
   The model produces an initial answer using the system prompt and user query.

2. **Critique Phase**  
   A second prompt asks the model to evaluate whether the initial response complies with domain constraints.

3. **Final Verdict Extraction**  
   The model outputs a structured verdict:
   - `IN-SCOPE`
   - `OUT-OF-SCOPE`

4. **Fallback Handling**  
   If the output cannot be parsed → labeled as `PARSE_ERROR`.

---

### Prompt Variants

#### v1_simple
- Minimal instructions  
- Weak structure  
- No explicit reasoning requirement  

#### v2_strict
- Strong formatting constraints  
- Explicit rules  
- Structured output required  

#### v3_checklist
- Step-by-step reasoning  
- Checklist-style validation  
- Encourages explicit verification  

---

### Dataset

- 100-example subset  
- Balanced distribution:
  - 50 in-domain  
  - 50 out-of-domain  

---

## Evaluation Metrics

We compute the following:

- **Accuracy**: Overall classification correctness  
- **Precision (Block OOD)**: Correctly blocked OOD among predicted OOD  
- **Recall (Block OOD)** ⭐: Correctly blocked OOD among all true OOD  
- **F1 Score**: Balance between precision and recall  
- **False Positive Rate (FPR)**: In-domain queries incorrectly blocked  
- **Missed OOD Rate**: OOD queries incorrectly allowed  
- **Parse Errors**: Output formatting failures  

---

## Results

### Quantitative Summary

| Prompt        | Accuracy | Precision | Recall | F1 Score | FPR  | Missed OOD | Parse Errors |
|--------------|----------|----------|--------|----------|------|-------------|--------------|
| v1_simple     | 0.85     | 0.93     | 0.76   | 0.83     | 0.06 | 0.24        | 7            |
| v2_strict     | 0.61     | 0.63     | 0.54   | 0.58     | 0.32 | 0.46        | 44           |
| v3_checklist  | 0.68     | 0.95     | 0.38   | 0.54     | 0.02 | 0.62        | 33           |

---

## Visual Analysis

### Accuracy Comparison

![Accuracy](figures/accuracy.png)

---

### F1 Score Comparison

![F1](figures/f1.png)

---

### Recall (Block OOD)

![Recall](figures/recall.png)

---

### False Positive Rate

![FPR](figures/fpr.png)

---

## Full Outputs

Detailed outputs for each prompt variant are provided in:

- `s5_v1_simple_results.csv`
- `s5_v2_strict_results.csv`
- `s5_v3_checklist_results.csv`

---

## Analysis

### 1. v1_simple performs best overall

- Highest accuracy (0.85)  
- Strong recall (0.76)  
- Low parse errors  

This suggests that:

> Simpler prompts are more robust and easier for the model to follow.

---

### 2. v2_strict introduces instability

- High parse errors (44)  
- Low accuracy  
- High false positive rate  

Interpretation:

> Overly strict formatting may confuse the model and reduce reliability.

---

### 3. v3_checklist is overly conservative

- Very high precision (0.95)  
- Very low recall (0.38)  

This indicates:

> The model becomes too cautious and fails to block many OOD queries.

---

### 4. Critique Loop Effectiveness

The critique loop provides limited improvements:

- Performance is highly dependent on prompt design  
- Structured prompts increase formatting errors  
- Gains from self-critique are inconsistent  

---

## Conclusion

From our experiments, we conclude:

1. **Prompt design has a larger impact than critique loops**  
2. **Simple prompts outperform complex ones**  
3. **Over-engineering harms reliability and increases parsing errors**  
4. **Critique loops are not consistently beneficial for guardrail tasks**  

---

## Final Takeaway

> The best-performing strategy is **v1_simple**, demonstrating that:
>
> Simplicity and clarity outperform complex prompting strategies in domain guardrail design.

---

## References

- Wei et al., *Chain-of-Thought Prompting Elicits Reasoning in LLMs*  
- HuggingFace Transformers Documentation  
- OpenAI Prompt Engineering Guidelines  
