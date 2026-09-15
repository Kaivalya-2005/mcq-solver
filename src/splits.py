from __future__ import annotations

import random
import re
from typing import Tuple

import pandas as pd


PREFIXES = [
    "pick the best possible answer:",
    "determine the correct option:",
    "select the most accurate option:",
    "identify the correct statement:",
    "which of the following is correct?",
    "choose the correct answer:",
]


SUFFIXES = [
    "among the listed options.",
    "carefully.",
    "based on the given context.",
    "from the following choices.",
]


def clean_query(prompt: str) -> str:
    text = str(prompt).strip().lower()

    for prefix in PREFIXES:
        if text.startswith(prefix):
            text = text[len(prefix):].strip()
            break

    for suffix in SUFFIXES:
        if text.endswith(suffix):
            text = text[:-len(suffix)].strip()
            break

    text = re.sub(r"\s+", " ", text)

    return text


def group_aware_split(
    df: pd.DataFrame,
    test_size: float = 0.2,
    seed: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame]:

    groups = df["prompt"].map(clean_query)

    unique_groups = groups.unique().tolist()

    rng = random.Random(seed)
    rng.shuffle(unique_groups)

    n_val = max(1, int(len(unique_groups) * test_size))

    val_groups = set(unique_groups[:n_val])

    val_mask = groups.isin(val_groups)

    val_df = df[val_mask].copy()
    train_df = df[~val_mask].copy()

    return train_df, val_df