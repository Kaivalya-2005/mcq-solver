from __future__ import annotations

import random
from typing import Tuple

import pandas as pd


def group_aware_split(
    df: pd.DataFrame,
    group_col: str = "prompt",
    test_size: float = 0.2,
    seed: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split rows by group so the same group does not appear in both splits.

    Returns:
        train_df, val_df
    """
    if group_col not in df.columns:
        raise KeyError(f"'{group_col}' not found in dataframe columns")

    groups = df[group_col].dropna().astype(str).unique().tolist()
    if not groups:
        return df.copy(), df.iloc[:0].copy()

    rng = random.Random(seed)
    rng.shuffle(groups)

    n_val = max(1, int(len(groups) * test_size))
    val_groups = set(groups[:n_val])

    val_df = df[df[group_col].astype(str).isin(val_groups)].copy()
    train_df = df[~df[group_col].astype(str).isin(val_groups)].copy()

    return train_df, val_df