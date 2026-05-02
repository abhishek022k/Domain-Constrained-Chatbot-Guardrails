"""
S4 LoRA Fine-Tuning: Training Response Generation
==================================================
Generates correct target responses for all 5k training examples:

  - Off-topic rows (2,500)      : GPT-4o-mini generates domain-specific refusals
  - On-topic wrong rows         : GPT-4o-mini generates correct helpful responses
  - On-topic correct rows       : Copy Llama's existing response as-is

Output: sft_responses.csv — same columns as input + target_response column

Usage:
    export OPENAI_API_KEY=sk-your-key-here
    python generate_sft_responses.py

Edit CONFIG section before running.
"""

import os
import time
import pandas as pd
from tqdm import tqdm
from openai import OpenAI
from dotenv import load_dotenv
from pathlib import Path

env_path = Path("../.env")
load_dotenv(dotenv_path=env_path)


# ─────────────────────────────────────────────
# CONFIG — edit before running
# ─────────────────────────────────────────────

INPUT_PATH       = "../benchmark_results/responses_labeled_5k_gpt_labeled.csv"
OUTPUT_PATH      = "./sft_responses.csv"
CHECKPOINT_PATH  = "./sft_responses_checkpoint.csv"

GPT_MODEL        = "gpt-4o-mini"
API_DELAY        = 0.1   # seconds between calls
MAX_RETRIES      = 3
RETRY_DELAY      = 5
CHECKPOINT_EVERY = 100


# ─────────────────────────────────────────────
# SETUP
# ─────────────────────────────────────────────

client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))


# ─────────────────────────────────────────────
# PROMPT BUILDERS
# ─────────────────────────────────────────────

def build_refusal_prompt(system_prompt: str, user_prompt: str) -> str:
    """
    For off-topic rows: generate a domain-specific refusal.
    One natural sentence declining and redirecting to chatbot's purpose.
    """
    return (
        f"The chatbot below was asked an off-topic question. Write a single natural "
        f"sentence refusing the query and redirecting to the chatbot's purpose. "
        f"Reply with only the refusal sentence.\n\n"
        f"Chatbot instructions: \"{system_prompt}\"\n"
        f"User asked: \"{user_prompt}\""
    )


def build_correction_prompt(system_prompt: str, user_prompt: str) -> str:
    """
    For on-topic rows where Llama wrongly refused: generate a correct helpful response.
    """
    return (
        f"The chatbot below was asked an on-topic question but incorrectly refused. "
        f"Write a correct, helpful response that answers the question within the "
        f"chatbot's domain. Be concise and natural. "
        f"Reply with only the response.\n\n"
        f"Chatbot instructions: \"{system_prompt}\"\n"
        f"User asked: \"{user_prompt}\""
    )


# ─────────────────────────────────────────────
# GPT CALL
# ─────────────────────────────────────────────

def call_gpt(prompt: str) -> str:
    """
    Call GPT-4o-mini with retry logic.
    Returns response text or empty string on failure.
    """
    for attempt in range(MAX_RETRIES):
        try:
            result = client.chat.completions.create(
                model=GPT_MODEL,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=150,
                temperature=0.7,   # slight variation for natural phrasing
            )
            return result.choices[0].message.content.strip()

        except Exception as e:
            print(f"\n  API error (attempt {attempt+1}/{MAX_RETRIES}): {e}")
            if attempt < MAX_RETRIES - 1:
                time.sleep(RETRY_DELAY)
            else:
                print(f"  Max retries reached — skipping row")
                return ""


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────

def main():
    # ── Load dataset ──────────────────────────
    print(f"Loading: {INPUT_PATH}")
    df = pd.read_csv(INPUT_PATH)
    print(f"Loaded : {len(df):,} rows")
    print(f"\nRow breakdown:")
    print(f"  Off-topic (2500 → need GPT refusal)          : {(df['off_topic'] == 1).sum():,}")
    print(f"  On-topic correct (gpt_label=1 → copy Llama)  : {((df['off_topic'] == 0) & (df['gpt_label'] == 1)).sum():,}")
    print(f"  On-topic wrong   (gpt_label=0 → need GPT fix): {((df['off_topic'] == 0) & (df['gpt_label'] == 0)).sum():,}")

    # ── Resume from checkpoint if exists ──────
    if os.path.exists(CHECKPOINT_PATH):
        df = pd.read_csv(CHECKPOINT_PATH)
        start_idx = int(df["target_response"].notna().sum())
        print(f"\nResuming from checkpoint at row {start_idx:,}")
    else:
        df["target_response"] = None
        df["target_source"]   = None   # track where response came from
        start_idx = 0
        print(f"\nStarting fresh")

    # ── Cost estimate ─────────────────────────
    n_gpt_calls = (df["off_topic"] == 1).sum() + \
                  ((df["off_topic"] == 0) & (df["gpt_label"] == 0)).sum()
    est_cost    = (n_gpt_calls * 200 / 1_000_000 * 0.15) + \
                  (n_gpt_calls * 150 / 1_000_000 * 0.60)
    print(f"\nGPT calls needed : {n_gpt_calls:,}")
    print(f"Est. cost        : ${est_cost:.2f}")
    print(f"Est. time        : {(n_gpt_calls * API_DELAY) / 60:.1f} minutes")
    print()

    # ── Generate responses ────────────────────
    for i in tqdm(range(start_idx, len(df)), desc="Generating responses"):
        row = df.iloc[i]

        # Skip if already done
        if pd.notna(df.at[df.index[i], "target_response"]):
            continue

        off_topic = int(row["off_topic"])
        gpt_label = int(row["gpt_label"])
        system_prompt = str(row["system_prompt"])
        user_prompt   = str(row["prompt"])

        # ── Case 1: Off-topic → generate refusal
        if off_topic == 1:
            prompt   = build_refusal_prompt(system_prompt, user_prompt)
            response = call_gpt(prompt)
            source   = "gpt_refusal"
            time.sleep(API_DELAY)

        # ── Case 2: On-topic, Llama wrong → generate correct response
        elif off_topic == 0 and gpt_label == 0:
            prompt   = build_correction_prompt(system_prompt, user_prompt)
            response = call_gpt(prompt)
            source   = "gpt_correction"
            time.sleep(API_DELAY)

        # ── Case 3: On-topic, Llama correct → copy existing response
        else:
            response = str(row["llama_response"])
            source   = "llama_copy"

        df.at[df.index[i], "target_response"] = response
        df.at[df.index[i], "target_source"]   = source

        # Checkpoint
        if (i + 1) % CHECKPOINT_EVERY == 0:
            df.to_csv(CHECKPOINT_PATH, index=False)

    # ── Final save ────────────────────────────
    df.to_csv(OUTPUT_PATH, index=False)
    df.to_csv(CHECKPOINT_PATH, index=False)

    # ── Summary ───────────────────────────────
    print(f"\n{'=' * 50}")
    print(f"DONE")
    print(f"{'=' * 50}")
    print(f"Output saved to : {OUTPUT_PATH}")
    print(f"\nTarget response sources:")
    print(df["target_source"].value_counts().to_string())
    print(f"\nMissing target_response: {df['target_response'].isna().sum()}")
    print(f"Empty target_response  : {(df['target_response'] == '').sum()}")


if __name__ == "__main__":
    main()
