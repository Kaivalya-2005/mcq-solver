"""
Stage 2: Fine-tune DeBERTa-v3-base for multiple-choice question answering.

Input:
    Question + 5 answer options (A-E)

Output:
    Score for each option -> rank options -> Top 3

Evaluation:
    Accuracy
    MAP@3 (primary competition metric)

Recommended environment:
    Kaggle T4 GPU / Colab GPU

Run:
    uv run python src/stage2_deberta_train.py
"""

import os
from dataclasses import dataclass
from typing import Union

import numpy as np
import pandas as pd
import torch
from datasets import Dataset
from sklearn.model_selection import train_test_split
from transformers import (
    AutoModelForMultipleChoice,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)
from transformers.tokenization_utils_base import (
    PaddingStrategy,
    PreTrainedTokenizerBase,
)

# Import the MAP@3 metric with fallbacks so the module can run both as
# a package (`python -m src.stage2_deberta_train`) and as a script
try:
    from .map3 import map_at_3
except Exception:
    try:
        from src.map3 import map_at_3
    except Exception:
        from map3 import map_at_3


# ============================================================
# Configuration
# ============================================================

OPTION_COLS = ["A", "B", "C", "D", "E"]

LABEL2IDX = {
    "A": 0,
    "B": 1,
    "C": 2,
    "D": 3,
    "E": 4,
}

IDX2LABEL = {
    0: "A",
    1: "B",
    2: "C",
    3: "D",
    4: "E",
}

# Start with BASE.
# Do not change to large for the first experiment.
MODEL_NAME = os.environ.get(
    "MCQ_MODEL",
    "microsoft/deberta-v3-base",
)

DATA_DIR = os.environ.get(
    "MCQ_DATA_DIR",
    "./data",
)

OUTPUT_DIR = os.environ.get(
    "MCQ_OUTPUT_DIR",
    "./stage2_out",
)

# Start at 256.
# We will inspect truncation later if necessary.
MAX_LEN = int(os.environ.get("MCQ_MAX_LEN", "256"))

SEED = 42


# ============================================================
# Reproducibility
# ============================================================

def set_seed(seed: int = 42):
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ============================================================
# Data preparation
# ============================================================

def prepare_multiple_choice_features(
    tokenizer,
    df: pd.DataFrame,
    has_labels: bool = True,
):
    """
    Convert every MCQ into five question-option pairs.

    Example:

        Question + A
        Question + B
        Question + C
        Question + D
        Question + E

    The model receives all five choices together as one
    multiple-choice example.
    """

    first_sentences = [
        [str(prompt)] * 5
        for prompt in df["prompt"]
    ]

    second_sentences = [
        [str(df.iloc[i][option]) for option in OPTION_COLS]
        for i in range(len(df))
    ]

    # Flatten the 5 choices so tokenizer can process them.
    first_flat = [
        sentence
        for group in first_sentences
        for sentence in group
    ]

    second_flat = [
        sentence
        for group in second_sentences
        for sentence in group
    ]

    tokenized = tokenizer(
        first_flat,
        second_flat,
        truncation=True,
        max_length=MAX_LEN,
        padding=False,
    )

    # Put the five choices back together.
    features = {
        key: [
            values[i:i + 5]
            for i in range(0, len(values), 5)
        ]
        for key, values in tokenized.items()
    }

    if has_labels:
        features["labels"] = [
            LABEL2IDX[str(answer)]
            for answer in df["answer"]
        ]

    return features


# ============================================================
# Dynamic padding
# ============================================================

@dataclass
class DataCollatorForMultipleChoice:
    """
    Dynamically pad the five choices in each MCQ.
    """

    tokenizer: PreTrainedTokenizerBase
    padding: Union[bool, str, PaddingStrategy] = True

    def __call__(self, features):
        labels = None

        if "labels" in features[0]:
            labels = [feature.pop("labels") for feature in features]

        batch_size = len(features)
        num_choices = len(features[0]["input_ids"])

        # Flatten:
        #
        # [question1-choice1 ... choice5]
        # [question2-choice1 ... choice5]
        #
        # into:
        #
        # [choice1, choice2, ..., choice5, choice1, ...]
        flattened = []

        for feature in features:
            for choice_index in range(num_choices):
                flattened.append(
                    {
                        key: value[choice_index]
                        for key, value in feature.items()
                    }
                )

        batch = self.tokenizer.pad(
            flattened,
            padding=self.padding,
            return_tensors="pt",
        )

        # Restore:
        #
        # batch_size x 5 x sequence_length
        #
        batch = {
            key: value.view(
                batch_size,
                num_choices,
                -1,
            )
            for key, value in batch.items()
        }

        if labels is not None:
            batch["labels"] = torch.tensor(
                labels,
                dtype=torch.long,
            )

        return batch


# ============================================================
# Metrics
# ============================================================

def compute_metrics(eval_pred):
    """
    Calculate both accuracy and MAP@3.

    Accuracy:
        Is the correct option ranked #1?

    MAP@3:
        Is the correct option in the top 3,
        and how high is it ranked?
    """

    logits, labels = eval_pred

    # logits shape:
    # (number_of_questions, 5)

    top1_predictions = np.argmax(logits, axis=1)

    accuracy = np.mean(
        top1_predictions == labels
    )

    ranked_indices = np.argsort(
        -logits,
        axis=1,
    )

    top3_predictions = [
        [
            IDX2LABEL[index]
            for index in row[:3]
        ]
        for row in ranked_indices
    ]

    true_labels = [
        IDX2LABEL[label]
        for label in labels
    ]

    map3 = map_at_3(
        true_labels,
        top3_predictions,
    )

    return {
        "accuracy": float(accuracy),
        "map3": float(map3),
    }


# ============================================================
# Main training
# ============================================================

def main():

    set_seed(SEED)

    print("=" * 60)
    print("Stage 2 - DeBERTa-v3-base Multiple Choice Solver")
    print("=" * 60)

    print(f"Model      : {MODEL_NAME}")
    print(f"Data dir   : {DATA_DIR}")
    print(f"Max length : {MAX_LEN}")
    print(f"PyTorch    : {torch.__version__}")
    print(f"CUDA       : {torch.cuda.is_available()}")

    if torch.cuda.is_available():
        print(f"GPU count  : {torch.cuda.device_count()}")

        for i in range(torch.cuda.device_count()):
            print(
                f"GPU {i}: "
                f"{torch.cuda.get_device_name(i)}"
            )

    print("=" * 60)


    # --------------------------------------------------------
    # Load data
    # --------------------------------------------------------

    train_path = os.path.join(
        DATA_DIR,
        "train.csv",
    )

    test_path = os.path.join(
        DATA_DIR,
        "test.csv",
    )

    train_full = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)

    print(f"Training rows : {len(train_full)}")
    print(f"Test rows     : {len(test_df)}")


    # --------------------------------------------------------
    # Train / validation split
    # --------------------------------------------------------

    train_df, val_df = train_test_split(
        train_full,
        test_size=0.20,
        random_state=SEED,
        stratify=train_full["answer"],
    )

    train_df = train_df.reset_index(drop=True)
    val_df = val_df.reset_index(drop=True)

    print(f"Train rows    : {len(train_df)}")
    print(f"Validation    : {len(val_df)}")


    # --------------------------------------------------------
    # Load tokenizer
    # --------------------------------------------------------

    print("\nLoading tokenizer...")

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_NAME
    )


    # --------------------------------------------------------
    # Load model
    # --------------------------------------------------------

    print("Loading DeBERTa model...")

    model = AutoModelForMultipleChoice.from_pretrained(
        MODEL_NAME
    )


    # --------------------------------------------------------
    # Convert data to Hugging Face Dataset
    # --------------------------------------------------------

    print("\nTokenizing training data...")

    train_features = prepare_multiple_choice_features(
        tokenizer,
        train_df,
        has_labels=True,
    )

    val_features = prepare_multiple_choice_features(
        tokenizer,
        val_df,
        has_labels=True,
    )

    train_ds = Dataset.from_dict(
        train_features
    )

    val_ds = Dataset.from_dict(
        val_features
    )

    print("Tokenization complete.")


    # --------------------------------------------------------
    # Training arguments
    # --------------------------------------------------------

    training_args = TrainingArguments(

        output_dir=OUTPUT_DIR,

        # Evaluation
        eval_strategy="epoch",
        save_strategy="epoch",

        # Training
        learning_rate=2e-5,
        num_train_epochs=3,

        # T4-friendly settings
        per_device_train_batch_size=2,
        per_device_eval_batch_size=4,

        gradient_accumulation_steps=8,

        # Regularization
        weight_decay=0.01,

        # GPU
        fp16=torch.cuda.is_available(),

        # Select best model using MAP@3
        load_best_model_at_end=True,
        metric_for_best_model="map3",
        greater_is_better=True,

        # Reproducibility
        seed=SEED,

        # Logging
        logging_steps=20,
        report_to=[],

        # Keep checkpoint count manageable
        save_total_limit=2,

        # Performance
        dataloader_num_workers=2,
    )


    # --------------------------------------------------------
    # Trainer
    # --------------------------------------------------------

    trainer = Trainer(

        model=model,

        args=training_args,

        train_dataset=train_ds,

        eval_dataset=val_ds,

        processing_class=tokenizer,

        data_collator=DataCollatorForMultipleChoice(
            tokenizer=tokenizer
        ),

        compute_metrics=compute_metrics,
    )


    # --------------------------------------------------------
    # Train
    # --------------------------------------------------------

    print("\nStarting training...\n")

    trainer.train()


    # --------------------------------------------------------
    # Final validation
    # --------------------------------------------------------

    print("\n" + "=" * 60)
    print("Final Validation")
    print("=" * 60)

    val_result = trainer.predict(
        val_ds
    )

    val_logits = val_result.predictions

    val_ranked_indices = np.argsort(
        -val_logits,
        axis=1,
    )

    val_top3 = [
        [
            IDX2LABEL[index]
            for index in row[:3]
        ]
        for row in val_ranked_indices
    ]

    val_true = val_df["answer"].tolist()

    final_map3 = map_at_3(
        val_true,
        val_top3,
    )

    val_top1 = np.argmax(
        val_logits,
        axis=1,
    )

    val_true_indices = np.array(
        [
            LABEL2IDX[label]
            for label in val_true
        ]
    )

    final_accuracy = np.mean(
        val_top1 == val_true_indices
    )

    print(
        f"Validation Accuracy : "
        f"{final_accuracy:.4f}"
    )

    print(
        f"Validation MAP@3   : "
        f"{final_map3:.4f}"
    )


    # --------------------------------------------------------
    # Predict test set
    # --------------------------------------------------------

    print("\nPreparing test set...")

    test_features = prepare_multiple_choice_features(
        tokenizer,
        test_df,
        has_labels=False,
    )

    test_ds = Dataset.from_dict(
        test_features
    )

    print("Generating test predictions...")

    test_result = trainer.predict(
        test_ds
    )

    test_logits = test_result.predictions

    test_ranked_indices = np.argsort(
        -test_logits,
        axis=1,
    )

    test_top3 = [
        [
            IDX2LABEL[index]
            for index in row[:3]
        ]
        for row in test_ranked_indices
    ]


    # --------------------------------------------------------
    # Save submission
    # --------------------------------------------------------

    os.makedirs(
        "./outputs",
        exist_ok=True,
    )

    submission = pd.DataFrame(
        {
            "ID": test_df["id"],
            "Prediction": [
                " ".join(prediction)
                for prediction in test_top3
            ],
        }
    )

    submission_path = (
        "./outputs/"
        "submission_stage2_deberta.csv"
    )

    submission.to_csv(
        submission_path,
        index=False,
    )

    print(
        f"\nSaved submission to: "
        f"{submission_path}"
    )

    print("\nFirst 5 predictions:")

    print(
        submission.head()
    )


    # --------------------------------------------------------
    # Save model
    # --------------------------------------------------------

    final_model_path = os.path.join(
        OUTPUT_DIR,
        "final",
    )

    trainer.save_model(
        final_model_path
    )

    tokenizer.save_pretrained(
        final_model_path
    )

    print(
        f"\nModel saved to: "
        f"{final_model_path}"
    )

    print("\n" + "=" * 60)
    print("Stage 2 COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()