"""
Disagreement Analysis: GPT-4o-mini vs NLI Labels
=================================================
Finds and inspects cases where GPT and NLI labels disagree,
helping diagnose which labeling method is more accurate.

Usage:
    python disagreement_analysis.py

Edit the CONFIG section below before running.
"""

import os
import pandas as pd


# ─────────────────────────────────────────────
# CONFIG — edit these before running
# ─────────────────────────────────────────────

# Folder containing your labeled CSV
BASE_DIR        = os.path.join(os.path.dirname(os.path.abspath(__file__)), "benchmark_results")

# Input file — must have gpt_label and predicted_compliance columns
INPUT_FILE = "responses_labeled_eval_gpt_labeled.csv"

INPUT_PATH = os.path.join(BASE_DIR, INPUT_FILE)
OUTPUT_DIR = os.path.join(BASE_DIR, "disagreement_analysis")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# How many examples to print to console per category
N_PRINT    = 5

# How many examples to save to CSV per category
N_SAVE     = 50


# ─────────────────────────────────────────────
# LOAD
# ─────────────────────────────────────────────

print(f"Loading: {INPUT_PATH}")
df = pd.read_csv(INPUT_PATH)
print(f"Loaded : {len(df):,} rows")
print(f"Columns: {list(df.columns)}\n")

# Only keep rows where both labels are valid (no errors)
df_valid = df[
    (df["gpt_label"] != -1) &
    (df["predicted_compliance"].notna())
].copy()
print(f"Valid rows (both labels present): {len(df_valid):,}")


# ─────────────────────────────────────────────
# DISAGREEMENT COUNTS
# ─────────────────────────────────────────────

print("\n" + "=" * 60)
print("OVERALL DISAGREEMENT SUMMARY")
print("=" * 60)

total    = len(df_valid)
agree    = (df_valid["gpt_label"] == df_valid["predicted_compliance"]).sum()
disagree = total - agree

print(f"\nTotal valid rows    : {total:,}")
print(f"Agreement           : {agree:,} ({agree/total:.1%})")
print(f"Disagreement        : {disagree:,} ({disagree/total:.1%})")

# Break down by off_topic label
for ot_label, ot_name in [(1, "OFF-TOPIC queries"), (0, "ON-TOPIC queries")]:
    subset    = df_valid[df_valid["off_topic"] == ot_label]
    agree_sub = (subset["gpt_label"] == subset["predicted_compliance"]).sum()
    print(f"\n  {ot_name} ({len(subset):,} rows):")
    print(f"    Agreement    : {agree_sub:,} ({agree_sub/len(subset):.1%})")
    print(f"    Disagreement : {len(subset)-agree_sub:,} "
          f"({(len(subset)-agree_sub)/len(subset):.1%})")


# ─────────────────────────────────────────────
# FOUR DISAGREEMENT CATEGORIES
# ─────────────────────────────────────────────

# Category A: GPT=REFUSED, NLI=COMPLIED, off-topic query
# Your main concern — is GPT over-refusing?
cat_a = df_valid[
    (df_valid["gpt_label"] == 0) &
    (df_valid["predicted_compliance"] == 1) &
    (df_valid["off_topic"] == 1)
]

# Category B: GPT=COMPLIED, NLI=REFUSED, off-topic query
# NLI over-refusing? GPT missing violations?
cat_b = df_valid[
    (df_valid["gpt_label"] == 1) &
    (df_valid["predicted_compliance"] == 0) &
    (df_valid["off_topic"] == 1)
]

# Category C: GPT=REFUSED, NLI=COMPLIED, on-topic query
# GPT wrongly blocking valid answers?
cat_c = df_valid[
    (df_valid["gpt_label"] == 0) &
    (df_valid["predicted_compliance"] == 1) &
    (df_valid["off_topic"] == 0)
]

# Category D: GPT=COMPLIED, NLI=REFUSED, on-topic query
# NLI wrongly blocking valid answers?
cat_d = df_valid[
    (df_valid["gpt_label"] == 1) &
    (df_valid["predicted_compliance"] == 0) &
    (df_valid["off_topic"] == 0)
]

print("\n" + "=" * 60)
print("DISAGREEMENT BREAKDOWN BY CATEGORY")
print("=" * 60)
print(f"\n  [A] GPT=REFUSED, NLI=COMPLIED, off-topic  : {len(cat_a):,}"
      f"  <- your main concern")
print(f"  [B] GPT=COMPLIED, NLI=REFUSED, off-topic  : {len(cat_b):,}")
print(f"  [C] GPT=REFUSED, NLI=COMPLIED, on-topic   : {len(cat_c):,}")
print(f"  [D] GPT=COMPLIED, NLI=REFUSED, on-topic   : {len(cat_d):,}")


# ─────────────────────────────────────────────
# PRINT EXAMPLES FOR EACH CATEGORY
# ─────────────────────────────────────────────

def print_examples(cat_df: pd.DataFrame, label: str, n: int = N_PRINT) -> None:
    print(f"\n{'=' * 60}")
    print(f"CATEGORY {label} — {n} examples")
    print(f"{'=' * 60}")

    if len(cat_df) == 0:
        print("  No examples in this category.")
        return

    for i, (_, row) in enumerate(cat_df.head(n).iterrows()):
        print(f"\n  [{i+1}]")
        print(f"  System   : {str(row['system_prompt'])[:100]}...")
        print(f"  Prompt   : {row['prompt']}")
        print(f"  Off-topic: {row['off_topic']}")
        print(f"  Response : {str(row['llama_response'])[:300]}")
        print(f"  GPT label: {row['gpt_label']} "
              f"(raw: {row.get('gpt_raw', 'N/A')})")
        print(f"  NLI label: {row['predicted_compliance']} "
              f"(conf: {row.get('nli_confidence', 'N/A')})")


print_examples(cat_a, "[A] GPT=REFUSED, NLI=COMPLIED, off-topic")
print_examples(cat_b, "[B] GPT=COMPLIED, NLI=REFUSED, off-topic")
print_examples(cat_c, "[C] GPT=REFUSED, NLI=COMPLIED, on-topic")
print_examples(cat_d, "[D] GPT=COMPLIED, NLI=REFUSED, on-topic")


# ─────────────────────────────────────────────
# SAVE ALL CATEGORIES TO CSV
# ─────────────────────────────────────────────

cols_to_save = [
    "system_prompt", "prompt", "off_topic",
    "llama_response", "gpt_label", "gpt_raw",
    "predicted_compliance", "nli_confidence",
]
# Only keep columns that exist in the dataframe
cols_to_save = [c for c in cols_to_save if c in df_valid.columns]

cat_a.head(N_SAVE)[cols_to_save].to_csv(
    os.path.join(OUTPUT_DIR, "cat_a_gpt_refused_nli_complied_offtopic.csv"),
    index=False
)
cat_b.head(N_SAVE)[cols_to_save].to_csv(
    os.path.join(OUTPUT_DIR, "cat_b_gpt_complied_nli_refused_offtopic.csv"),
    index=False
)
cat_c.head(N_SAVE)[cols_to_save].to_csv(
    os.path.join(OUTPUT_DIR, "cat_c_gpt_refused_nli_complied_ontopic.csv"),
    index=False
)
cat_d.head(N_SAVE)[cols_to_save].to_csv(
    os.path.join(OUTPUT_DIR, "cat_d_gpt_complied_nli_refused_ontopic.csv"),
    index=False
)

print(f"\n{'=' * 60}")
print(f"CSVs saved to: {OUTPUT_DIR}")
print(f"{'=' * 60}")
print(f"  cat_a_gpt_refused_nli_complied_offtopic.csv  ({len(cat_a):,} rows)")
print(f"  cat_b_gpt_complied_nli_refused_offtopic.csv  ({len(cat_b):,} rows)")
print(f"  cat_c_gpt_refused_nli_complied_ontopic.csv   ({len(cat_c):,} rows)")
print(f"  cat_d_gpt_complied_nli_refused_ontopic.csv   ({len(cat_d):,} rows)")
print(f"\nManually review cat_a CSV first — these are your main concern.")
print(f"If most are genuine compliance cases, GPT is over-refusing.")
print(f"If most are genuine refusals, NLI was over-counting compliance.")
