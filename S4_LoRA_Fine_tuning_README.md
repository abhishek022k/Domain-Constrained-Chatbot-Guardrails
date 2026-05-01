# S4: LoRA Fine-Tuning for Domain-Constrained Guardrails

## Overview

S4 fine-tunes Llama-3.1-8B-Instruct using QLoRA to natively refuse off-topic queries without any external guardrail component. Rather than intercepting or post-processing model outputs, the fine-tuned model internalizes refusal behavior conditioned on the system prompt's domain definition.

**Baseline (no guardrail):**
- Compliance rate on off-topic queries: 69.6%
- False positive rate on on-topic queries: 0.024

**S4 Target:**
- Compliance rate on off-topic queries: < 0.30
- False positive rate: < 0.10
- Guardrail F1: > 0.70

---

## Repository Datasets

All datasets are pre-labeled using GPT-4o-mini few-shot as judge. Key columns across all files: `system_prompt`, `prompt`, `off_topic` (1=off-topic, 0=on-topic), `llama_response`, `gpt_label` (1=complied, 0=refused).

| File | Description |
|---|---|
| `sampled_5k.csv` | 5,000 training examples (2,500 on-topic, 2,500 off-topic) |
| `eval_500.csv` | 500 held-out eval set — never used during training |
| `responses_labeled_5k_gpt_labeled.csv` | 5k rows with Llama baseline responses and GPT labels |
| `eval_500_responses_gpt_labeled.csv` | 500 eval rows with Llama baseline responses and GPT labels |

---

## Pipeline Overview

```
Step 1: Generate refusal responses via GPT-4o-mini    (local, ~$0.50, ~25 mins)
Step 2: Build SFT training dataset                    (local, ~5 mins)
Step 3: QLoRA fine-tuning on Llama-3.1-8B             (Colab A100, ~75-90 mins)
Step 4: Generate eval responses with fine-tuned model (Colab A100, ~15 mins)
Step 5: GPT-4o-mini judge + compute metrics           (local, ~10 mins, ~$0.10)
```

---

## Step 1: Generate Refusal Responses (Local)

For all 2,500 off-topic rows in `sampled_5k.csv`, generate a domain-specific refusal response using GPT-4o-mini. The refusal must reference the chatbot's domain from the system prompt so the model learns to condition refusal behavior on system prompt context rather than learning a blanket refusal pattern.

**Install:**
```bash
pip install openai pandas tqdm
```

**Prompt used per off-topic row:**
```
The chatbot below was asked an off-topic question. Write a single natural sentence 
refusing the query and redirecting to the chatbot's purpose. Reply with only the 
refusal sentence.

Chatbot instructions: "{system_prompt}"
User asked: "{prompt}"
```

**Script:** Adapt `gpt_labeling.py` — same OpenAI API setup, change the prompt and save output as `refusal_response` column. Save the result as `training_pairs.csv`.

---

## Step 2: Build Training Dataset (Local)

Combine off-topic rows (with GPT-generated refusal targets) and on-topic rows (with Llama's existing correct responses) into a single supervised fine-tuning dataset.

```python
import pandas as pd

df       = pd.read_csv("responses_labeled_5k_gpt_labeled.csv")
refusals = pd.read_csv("training_pairs.csv")

# Off-topic rows: target = GPT-generated domain-specific refusal
off_topic = df[df["off_topic"] == 1].copy()
off_topic["target_response"] = refusals["refusal_response"].values

# On-topic rows: target = Llama's existing correct responses
# Only include rows where Llama correctly complied (gpt_label=1)
on_topic = df[
    (df["off_topic"] == 0) &
    (df["gpt_label"] == 1)
].copy()
on_topic["target_response"] = on_topic["llama_response"]

# Combine and shuffle
train_df = pd.concat([off_topic, on_topic]).sample(
    frac=1, random_state=42
).reset_index(drop=True)

# Carve out 250-row validation split for loss monitoring during training
# This is NOT the held-out eval_500 — it comes from training data only
val_df   = train_df.sample(n=250, random_state=42)
train_df = train_df[~train_df.index.isin(val_df.index)]

print(f"Training rows  : {len(train_df)}")   # ~4,691
print(f"Validation rows: {len(val_df)}")     # 250

train_df.to_csv("sft_train.csv", index=False)
val_df.to_csv("sft_val.csv", index=False)
```

**Why include on-topic examples?**
Without on-topic training examples, the model only sees refusal targets and risks catastrophic forgetting — it may start refusing everything including legitimate queries. On-topic examples teach the conditional boundary: refuse only when off-topic, answer normally when on-topic.

---

## Step 3: QLoRA Fine-Tuning (Colab Pro A100)

**Install:**
```bash
pip install transformers peft bitsandbytes trl accelerate datasets torch
```

**HuggingFace login** (Llama-3.1-8B is a gated model):
```python
from huggingface_hub import login
login(token="hf_your_token_here")
```

### Design Decisions

| Parameter | Value | Reason |
|---|---|---|
| Quantization | 4-bit QLoRA (nf4) | Fits A100 80GB with headroom |
| LoRA rank r | 16 | Sufficient capacity for behavior modification |
| lora_alpha | 32 | Standard 2x rank scaling |
| Target modules | All attention + MLP layers | Behavior change requires both attention and MLP modification |
| Effective batch size | 16 (batch=4, grad_accum=4) | Memory-safe equivalent of batch 16 |
| Learning rate | 2e-4 | Standard for LoRA instruction tuning |
| Max epochs | 5 with early stopping | Upper bound — training stops when eval_loss plateaus |
| Early stopping patience | 3 | Stops after 3 consecutive evals without improvement |
| Max sequence length | 512 | Sufficient for system prompt + query + refusal |

### Training Script

```python
from transformers import (
    AutoTokenizer, AutoModelForCausalLM,
    BitsAndBytesConfig, TrainingArguments,
    EarlyStoppingCallback,
)
from peft import LoraConfig
from trl import SFTTrainer
from datasets import Dataset
import torch
import pandas as pd

# ── Config ────────────────────────────────────────────
MODEL_ID    = "meta-llama/Meta-Llama-3.1-8B-Instruct"
OUTPUT_DIR  = "/content/drive/MyDrive/guardrail_datasets/s4_lora"
MAX_SEQ_LEN = 512

# ── Load tokenizer ────────────────────────────────────
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
tokenizer.pad_token    = tokenizer.eos_token
tokenizer.padding_side = "right"   # must be right for SFT — left padding breaks loss

# ── Load model with 4-bit quantization ───────────────
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    quantization_config=bnb_config,
    device_map="auto",
    torch_dtype=torch.bfloat16,
)
model.config.use_cache = False   # required for gradient checkpointing

# ── LoRA config ───────────────────────────────────────
lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=[
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
)

# ── Format dataset into chat template ─────────────────
def format_example(row):
    messages = [
        {"role": "system",    "content": row["system_prompt"]},
        {"role": "user",      "content": row["prompt"]},
        {"role": "assistant", "content": row["target_response"]},
    ]
    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=False,
    )

train_df = pd.read_csv("sft_train.csv")
val_df   = pd.read_csv("sft_val.csv")

train_dataset = Dataset.from_dict({"text": [format_example(r) for _, r in train_df.iterrows()]})
val_dataset   = Dataset.from_dict({"text": [format_example(r) for _, r in val_df.iterrows()]})

# ── Training arguments ────────────────────────────────
training_args = TrainingArguments(
    output_dir                  = OUTPUT_DIR,
    num_train_epochs            = 5,
    per_device_train_batch_size = 4,
    gradient_accumulation_steps = 4,
    learning_rate               = 2e-4,
    bf16                        = True,
    gradient_checkpointing      = True,
    evaluation_strategy         = "steps",
    eval_steps                  = 100,
    save_strategy               = "steps",
    save_steps                  = 100,
    load_best_model_at_end      = True,
    metric_for_best_model       = "eval_loss",
    greater_is_better           = False,
    logging_steps               = 25,
    warmup_ratio                = 0.05,
    lr_scheduler_type           = "cosine",
    report_to                   = "none",
)

# ── Trainer ───────────────────────────────────────────
trainer = SFTTrainer(
    model              = model,
    args               = training_args,
    train_dataset      = train_dataset,
    eval_dataset       = val_dataset,
    peft_config        = lora_config,
    dataset_text_field = "text",
    max_seq_length     = MAX_SEQ_LEN,
    callbacks          = [EarlyStoppingCallback(early_stopping_patience=3)],
)

trainer.train()

# ── Save adapter weights to Drive immediately ─────────
trainer.model.save_pretrained(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)
print(f"LoRA adapter saved to: {OUTPUT_DIR}")
```

**Expected training time:** ~75-90 minutes on A100 80GB for 2-3 epochs before early stopping triggers.

---

## Step 4: Generate Eval Responses with Fine-Tuned Model (Colab)

Load the saved LoRA adapter on top of the base model and run inference on `eval_500.csv`. Use each row's own `system_prompt` column — not a fixed system prompt.

```python
from peft import PeftModel

base_model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    quantization_config=bnb_config,
    device_map="auto",
    torch_dtype=torch.bfloat16,
)
model_ft = PeftModel.from_pretrained(base_model, OUTPUT_DIR)
model_ft.eval()
```

Reference `benchmark_llama.py` for the generation loop. The only differences from the baseline script:
- Load the fine-tuned model as above instead of the base model
- Use each row's `system_prompt` column instead of the fixed baseline system prompt
- Save outputs to `s4_eval_responses.csv`

---

## Step 5: Label and Evaluate (Local)

Run `gpt_labeling.py` on `s4_eval_responses.csv` using the same GPT-4o-mini few-shot judge pipeline used for all other strategies.

**Interpreting results:**

| Outcome | Interpretation | Action |
|---|---|---|
| compliance_rate_offtopic < 0.30 and FPR < 0.10 | Strong result | Report as-is |
| compliance_rate_offtopic < 0.30 but FPR > 0.15 | Over-refused — overfitting | Retrain with 1-2 epochs, patience=2 |
| compliance_rate_offtopic > 0.50 | Under-learned refusal | Check training data construction |

---

## Output Files

| File | Description |
|---|---|
| `training_pairs.csv` | Off-topic rows with GPT-generated refusal targets |
| `sft_train.csv` | Final SFT training set (~4,750 rows) |
| `sft_val.csv` | 250-row validation split for loss monitoring |
| `s4_lora/` | Saved LoRA adapter weights |
| `s4_eval_responses.csv` | Fine-tuned model responses on 500 eval set |
| `s4_eval_responses_gpt_labeled.csv` | Labeled eval responses with final metrics |

---

## Important Notes

- **Do not use `eval_500.csv` during training** — it is the held-out benchmark shared across all guardrail strategies S1-S5 and must remain unseen until final evaluation
- **Save adapter weights to Drive immediately** after training — Colab sessions expire and unsaved weights are lost
- **Padding side must be `"right"` for SFT** — left padding causes incorrect loss computation on the target response tokens
- **Check false positive rate first** after evaluation — a spike above 0.15 indicates overfitting toward always refusing, regardless of how much compliance_rate_offtopic improved
