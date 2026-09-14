import os
import pandas as pd

OPTION_COLS = ["A", "B", "C", "D", "E"]

# Each environment sets this differently -- see README.md
DATA_DIR = os.environ.get("MCQ_DATA_DIR", "./data")


def load_train():
    return pd.read_csv(f"{DATA_DIR}/train.csv")


def load_test():
    return pd.read_csv(f"{DATA_DIR}/test.csv")


def make_submission(test_df, top3_lists, out_path):
    sub = pd.DataFrame({
        "ID": test_df["id"],
        "Prediction": [" ".join(r) for r in top3_lists],
    })
    sub.to_csv(out_path, index=False)
    return sub
