import pandas as pd
from dataclasses import dataclass
from typing import List, Optional

@dataclass
class EvalExample:
    system_prompt: str
    query: str
    label: int
    row_id: Optional[int] = None

def normalize_label(value):
    """Normalize various label formats to 0 or 1."""
    if isinstance(value, str):
        value = value.strip().lower()
        if value in {"1", "true", "yes", "out", "out_of_scope", "out-of-scope", "offtopic", "off-topic", "off_topic"}:
            return 1
        if value in {"0", "false", "no", "in", "in_scope", "in-scope", "inscope", "ontopic", "on-topic", "in_topic"}:
            return 0
    try:
        return int(value)
    except Exception:
        return 0

def find_column(df: pd.DataFrame, candidates):
    """Find the best matching column from a list of candidates."""
    for name in candidates:
        if name in df.columns:
            return name
    lower = {col.lower(): col for col in df.columns}
    for name in candidates:
        if name.lower() in lower:
            return lower[name.lower()]
    return None

def load_dataset(path: str) -> List[EvalExample]:
    """Load evaluation dataset from CSV file.

    Expected columns (case-insensitive):
    - system_prompt/system prompt/system/prompt/domain
    - query/user_query/utterance/prompt/input/question
    - off_topic/offtopic/off-topic/label/target/is_offtopic/is_off_topic

    Args:
        path: Path to the CSV file

    Returns:
        List of EvalExample objects

    Raises:
        ValueError: If required columns are not found
    """
    df = pd.read_csv(path)

    # Find the appropriate columns
    system_col = find_column(df, ["system_prompt", "system prompt", "system", "prompt", "domain"])
    query_col = find_column(df, ["query", "user_query", "utterance", "prompt", "input", "question"])
    label_col = find_column(df, ["off_topic", "offtopic", "off-topic", "label", "target", "is_offtopic", "is_off_topic"])

    if system_col is None or query_col is None or label_col is None:
        available_cols = list(df.columns)
        raise ValueError(
            f"Could not find expected columns in {path}.\n"
            f"Available columns: {available_cols}\n"
            f"Expected: system_prompt, query/user_query, off_topic/label"
        )

    examples = []
    for i, row in df.iterrows():
        examples.append(
            EvalExample(
                system_prompt=str(row[system_col]).strip(),
                query=str(row[query_col]).strip(),
                label=normalize_label(row[label_col]),
                row_id=i,
            )
        )

    return examples

def show_dataset_info(path: str):
    """Load and display basic information about the dataset."""
    examples = load_dataset(path)
    print(f'Loaded {len(examples)} examples from {path}')

    if examples:
        print(f'First example:')
        print(f'  System prompt: {examples[0].system_prompt[:100]}...')
        print(f'  Query: {examples[0].query[:100]}...')
        print(f'  Label: {examples[0].label}')

        # Show label distribution
        labels = [ex.label for ex in examples]
        label_counts = pd.Series(labels).value_counts()
        print(f'Label distribution: {dict(label_counts)}')

    return examples

if __name__ == "__main__":
    # Example usage
    import sys
    if len(sys.argv) > 1:
        dataset_path = sys.argv[1]
    else:
        dataset_path = "eval_500.csv"

    try:
        examples = show_dataset_info(dataset_path)
        print(f"Successfully loaded {len(examples)} examples!")
    except FileNotFoundError:
        print(f"File not found: {dataset_path}")
    except Exception as e:
        print(f"Error loading dataset: {e}")