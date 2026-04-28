# Domain-Constrained Chatbot Guardrails - A Comparative Study

**Group members:** Abhishek Sharma, Chang Zhou, Kautsya Kanu, Keshav Handa, Nabanita Bag, Naveen Murthy

---

## Idea and Motivation
Deployed LLM chatbots routinely violate their intended domain boundaries. Amazon Rufus, a shopping assistant, generates web-scraping scripts on request[1], and multilingual prompts bypass its safety filters roughly 80% of the time because fine-tuning was English-only[2]. GitHub Copilot's safety filters can be disabled by prepending an agreeable tone[3]. These are not exotic jailbreaks; they are trivial, documented, and largely unpatched.

**Core Question:**  
How do different intervention strategies perform compared to our baseline for preventing out-of-domain LLM outputs?

---

## Hypothesis
We hypothesize that internal modifications like **Representation Probe (S3)** or **LoRA fine-tuning (S4)** will outperform surface-level modifications like **Text Classifier (S1)**, **Prompted LLM (S2)**, or **Critique loop (S5)**. Both modification types are expected to outperform the baseline, which relies on the LLM’s base reasoning without guardrails to classify out-of-domain queries.

---

## Strategies

We plan to evaluate a subset of these strategies based on available time.

| Strategy | Method | Key Idea | Reference |
|----------|--------|----------|----------|
| **S1** | Text Classifier (Kautsya) | Binary classifier on LLM outputs | P1, P2, P3 |
| **S2** | Prompted LLM + Self-Reminder (Naveen) | System prompts, few-shot, CoT; Constitutional AI style | https://arxiv.org/abs/2310.06117 |
| **S3** *(Stretch)* | Representation Probe (Nabanita) | Linear probe on residual-stream activations | https://arxiv.org/abs/2310.01405 |
| **S4** *(Stretch)* | LoRA Fine-Tuning (QLoRA) (Keshav + Abhishek) | Train refusal as native behavior | https://arxiv.org/abs/2106.09685 |
| **S5** | Critique Loop (Chang) | Model critiques and revises output | https://arxiv.org/abs/2303.17651 |

---

## Methodology

- **Model:** Llama-3.1-8B-Instruct (8B parameters, 32 layers)
- **Datasets:** Off-topic Dataset and/or Bitext Customer Support Dataset
- **Evaluation Metrics:**
  - Guardrail accuracy (block rate on OOD queries)
  - F1 score
  - False positive rate (blocking valid queries)
- **Infrastructure:**
  - Google Colab Pro (A100)
  - HuggingFace Transformers + PEFT
  - OpenAI API (GPT-4o) for dataset generation and evaluation

---

## Dataset Strategy

### Approach A: Off-Topic Dataset + Bitext (Evaluation)

- **Source:** gabrielchua/off-topic (2.62M rows)
- **Training:** 2,000 samples
- **Evaluation:** 500 samples (50/50 balanced)
- **Secondary Evaluation:** Bitext dataset for harder near-domain violations

#### Advantages
- Predefined system prompts enable immediate use
- Dual-difficulty evaluation strengthens analysis

#### Disadvantages
- No LLM-generated responses (needed for classifier training)

#### Mitigation
- Generate responses using Llama-3.1-8B with minimal prompting
- Avoid stronger models (e.g., GPT-4o) to preserve realistic failure behavior
- Consider smaller models if needed

---

### Approach B: Bitext Customer Support Only

- **Size:** 26,900 rows across 11 categories
- **Setup:** One category = in-scope, others = off-topic

#### Advantages
- Pre-written responses eliminate generation step
- Realistic single-domain deployment scenario

#### Disadvantages
- Lacks clearly off-topic examples
- May compress performance differences

---

## Decision

After discussion with Fred, we chose **Approach A** due to its diversity and flexibility.  
- Response generation cost is manageable  
- Training size will start small (2k) and scale if needed (5k/10k)  
- Evaluation set: 500 samples  

---

## Baseline Benchmarking

### Dataset Extraction
- Created balanced subsets: 2k, 3k, 5k, 10k
- Ensured:
  - 50/50 class balance
  - Domain diversity
- Selected **5k dataset** as optimal
- Created separate **500-sample eval set** (no overlap)

---

### Baseline Response Generation
- Model: Llama-3.1-8B-Instruct (4-bit QLoRA)
- Prompt: `"You are a helpful assistant."`
- Generated responses for:
  - 5k training set
  - 500 eval set

---

### Labeling Method

**Evaluator:** GPT-4o-mini (few-shot)

#### Compared Approaches:
- Small NLI model (DeBERTa v3 small)
- Large NLI model (DeBERTa v3 large)
- GPT-4o-mini

#### Results:
- NLI models: 65–71% agreement (poor reliability)
- GPT-4o-mini:
  - 90.4% agreement
  - ~95% correctness in disagreements

**Final Choice:** GPT-4o-mini as canonical judge

---

## Baseline Results

- **Off-topic compliance rate:** 69.6%
- **On-topic accuracy:** 97.6%

### Key Metrics
- **Refusal Precision:** 0.93  
- **Refusal Recall:** 0.30  
- **False Positive Rate:** 0.024  

---

## Key Insight

The baseline model *knows* when queries are off-topic but fails to act consistently.

- High precision → refusals are usually correct  
- Low recall → refuses too infrequently  

### Goal for Guardrails
- Increase **recall (0.30 → ~1.0)**  
- Maintain **low false positive rate (~0.024)**  

In short:  
**Improve enforcement without harming correctness.**