"""
S4 LoRA Fine-Tuning: Build Training Dataset
============================================
Takes sft_responses.csv (output of generate_sft_responses.py)
and produces:
  - sft_train.csv : ~4,750 rows for training
  - sft_val.csv   : 250 rows for loss monitoring during training

NOTE: eval_500.csv is never touched here — it is the held-out
benchmark shared across all guardrail strategies S1-S5.

Usage:
    python build_sft_dataset.py

Edit CONFIG section before running.
"""

import pandas as pd
import numpy as np


# ─────────────────────────────────────────────
# CONFIG — edit before running
# ─────────────────────────────────────────────

INPUT_PATH  = "./sft_responses.csv"
TRAIN_PATH  = "./sft_train.csv"
VAL_PATH    = "./sft_val.csv"

VAL_SIZE    = 250
RANDOM_SEED = 42


# ─────────────────────────────────────────────
# LOAD
# ─────────────────────────────────────────────

print(f"Loading: {INPUT_PATH}")
df = pd.read_csv(INPUT_PATH)
print(f"Loaded : {len(df):,} rows")
print(f"Columns: {list(df.columns)}")


# ─────────────────────────────────────────────
# VALIDATE
# ─────────────────────────────────────────────

# Check for missing or empty target responses
missing = df["target_response"].isna().sum()
empty   = (df["target_response"].astype(str).str.strip() == "").sum()

print(f"\nQuality check:")
print(f"  Missing target_response : {missing}")
print(f"  Empty target_response   : {empty}")

if missing > 0 or empty > 0:
    print(f"\n  WARNING: dropping {missing + empty} rows with missing/empty targets")
    df = df[
        df["target_response"].notna() &
        (df["target_response"].astype(str).str.strip() != "")
    ].reset_index(drop=True)
    print(f"  Rows remaining: {len(df):,}")


# ─────────────────────────────────────────────
# KEEP ONLY REQUIRED COLUMNS
# ─────────────────────────────────────────────

required_cols = ["system_prompt", "prompt", "off_topic",
                 "target_response", "target_source"]
available     = [c for c in required_cols if c in df.columns]
df            = df[available].copy()

print(f"\nTarget response sources:")
print(df["target_source"].value_counts().to_string())


# ─────────────────────────────────────────────
# CLASS BALANCE CHECK
# ─────────────────────────────────────────────

print(f"\nClass balance:")
print(f"  Off-topic rows : {(df['off_topic'] == 1).sum():,}")
print(f"  On-topic rows  : {(df['off_topic'] == 0).sum():,}")


# ─────────────────────────────────────────────
# SHUFFLE AND SPLIT
# ─────────────────────────────────────────────

# Stratified validation split — keep 50/50 off/on-topic in val set
# so loss monitoring reflects both behavior types equally
off_topic_df = df[df["off_topic"] == 1].sample(frac=1, random_state=RANDOM_SEED)
on_topic_df  = df[df["off_topic"] == 0].sample(frac=1, random_state=RANDOM_SEED)

val_off  = off_topic_df.iloc[:VAL_SIZE // 2]
val_on   = on_topic_df.iloc[:VAL_SIZE // 2]
val_df   = pd.concat([val_off, val_on]).sample(
    frac=1, random_state=RANDOM_SEED
).reset_index(drop=True)

train_off = off_topic_df.iloc[VAL_SIZE // 2:]
train_on  = on_topic_df.iloc[VAL_SIZE // 2:]
train_df  = pd.concat([train_off, train_on]).sample(
    frac=1, random_state=RANDOM_SEED
).reset_index(drop=True)


# ─────────────────────────────────────────────
# SAVE
# ─────────────────────────────────────────────

train_df.to_csv(TRAIN_PATH, index=False)
val_df.to_csv(VAL_PATH, index=False)


# ─────────────────────────────────────────────
# SUMMARY
# ─────────────────────────────────────────────

print(f"\n{'=' * 50}")
print(f"DATASET SPLIT COMPLETE")
print(f"{'=' * 50}")

print(f"\nTraining set : {len(train_df):,} rows → {TRAIN_PATH}")
print(f"  Off-topic  : {(train_df['off_topic'] == 1).sum():,}")
print(f"  On-topic   : {(train_df['off_topic'] == 0).sum():,}")
print(f"  Sources    :")
print(train_df["target_source"].value_counts().to_string())

print(f"\nValidation set : {len(val_df):,} rows → {VAL_PATH}")
print(f"  Off-topic    : {(val_df['off_topic'] == 1).sum():,}")
print(f"  On-topic     : {(val_df['off_topic'] == 0).sum():,}")

print(f"\nSample training rows:")
for source in ["gpt_refusal", "gpt_correction", "llama_copy"]:
    if source not in train_df["target_source"].values:
        continue
    sample = train_df[train_df["target_source"] == source].iloc[0]
    print(f"\n  [{source}]")
    print(f"  System  : {str(sample['system_prompt'])[:80]}...")
    print(f"  Prompt  : {sample['prompt']}")
    print(f"  Target  : {str(sample['target_response'])[:150]}")
