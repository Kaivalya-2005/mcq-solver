import os

import numpy as np
import pandas as pd
import torch
from datasets import Dataset
from transformers import (
    AutoModelForMultipleChoice,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)

from src.map3 import map_at_3
from src.splits import group_aware_split  # groups by clean_query(prompt), NOT raw text

MODEL_NAME = "microsoft/deberta-v3-base"
DATA_DIR = os.environ.get("MCQ_DATA_DIR", "./data")
OUTPUT_DIR = "./stage2_out"
MAX_LEN = 256
SEED = 42

OPTIONS = ["A", "B", "C", "D", "E"]
LABELS = {letter: i for i, letter in enumerate(OPTIONS)}
ID_TO_LABEL = {i: letter for i, letter in enumerate(OPTIONS)}


def set_seed():
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)


def tokenize_data(df, tokenizer, include_labels=True):
    questions = []
    options = []
    for _, row in df.iterrows():
        for option in OPTIONS:
            questions.append(str(row["prompt"]))
            options.append(str(row[option]))

    tokens = tokenizer(questions, options, truncation=True, max_length=MAX_LEN, padding=False)

    features = {}
    for key, values in tokens.items():
        features[key] = [values[i:i + 5] for i in range(0, len(values), 5)]

    if include_labels:
        features["labels"] = [LABELS[str(answer)] for answer in df["answer"]]

    return Dataset.from_dict(features)


class MultipleChoiceCollator:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def __call__(self, features):
        labels = None
        if "labels" in features[0]:
            labels = [x.pop("labels") for x in features]

        flattened = []
        for feature in features:
            for i in range(5):
                flattened.append({key: value[i] for key, value in feature.items()})

        batch = self.tokenizer.pad(flattened, padding=True, return_tensors="pt")
        batch_size = len(features)
        batch = {key: value.view(batch_size, 5, -1) for key, value in batch.items()}

        if labels is not None:
            batch["labels"] = torch.tensor(labels, dtype=torch.long)
        return batch


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argsort(-logits, axis=1)
    top1 = predictions[:, 0]
    accuracy = np.mean(top1 == labels)
    top3 = [[ID_TO_LABEL[i] for i in row[:3]] for row in predictions]
    true_labels = [ID_TO_LABEL[i] for i in labels]
    map3 = map_at_3(true_labels, top3)
    return {"accuracy": float(accuracy), "map3": float(map3)}


def main():
    set_seed()

    train_df = pd.read_csv(os.path.join(DATA_DIR, "train.csv"))
    test_df = pd.read_csv(os.path.join(DATA_DIR, "test.csv"))

    # Group-aware split -- see src/splits.py. This is the part the naive
    # str.lower().str.strip() grouping got wrong: it doesn't strip the
    # instruction wrapper ("Select the most accurate option:", etc.), so it
    # still treats reworded copies of the SAME question as different groups
    # and barely reduces leakage (1758/2000 "unique" vs the true 252).
    train_df, val_df = group_aware_split(train_df, test_size=0.20, seed=SEED)

    print("Training rows:", len(train_df))
    print("Validation rows:", len(val_df))
    print("Test rows:", len(test_df))

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForMultipleChoice.from_pretrained(MODEL_NAME)

    train_dataset = tokenize_data(train_df, tokenizer, include_labels=True)
    val_dataset = tokenize_data(val_df, tokenizer, include_labels=True)

    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        eval_strategy="epoch",
        save_strategy="epoch",
        learning_rate=1e-5,
        num_train_epochs=3,
        per_device_train_batch_size=2,
        per_device_eval_batch_size=4,
        gradient_accumulation_steps=8,
        weight_decay=0.01,
        max_grad_norm=1.0,
        fp16=False,
        logging_steps=20,
        report_to=[],
        load_best_model_at_end=True,
        metric_for_best_model="map3",
        greater_is_better=True,
        save_total_limit=2,
        seed=SEED,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        processing_class=tokenizer,
        data_collator=MultipleChoiceCollator(tokenizer),
        compute_metrics=compute_metrics,
    )

    print("\nStarting training...")
    trainer.train()

    result = trainer.predict(val_dataset)
    logits = result.predictions
    predictions = np.argsort(-logits, axis=1)
    top3 = [[ID_TO_LABEL[i] for i in row[:3]] for row in predictions]
    true_labels = val_df["answer"].tolist()
    map3 = map_at_3(true_labels, top3)
    accuracy = np.mean(predictions[:, 0] == np.array([LABELS[x] for x in true_labels]))

    print("\nValidation Accuracy:", round(accuracy, 4))
    print("Validation MAP@3:", round(map3, 4))

    test_dataset = tokenize_data(test_df, tokenizer, include_labels=False)
    test_result = trainer.predict(test_dataset)
    test_predictions = np.argsort(-test_result.predictions, axis=1)
    test_top3 = [[ID_TO_LABEL[i] for i in row[:3]] for row in test_predictions]

    os.makedirs("./outputs", exist_ok=True)
    submission = pd.DataFrame({
        "ID": test_df["id"],
        "Prediction": [" ".join(row) for row in test_top3],
    })
    submission.to_csv("./outputs/submission_stage2_deberta.csv", index=False)

    trainer.save_model(os.path.join(OUTPUT_DIR, "final"))
    tokenizer.save_pretrained(os.path.join(OUTPUT_DIR, "final"))

    print("\nSaved submission:")
    print("./outputs/submission_stage2_deberta.csv")
    print("\nFirst 5 predictions:")
    print(submission.head())


if __name__ == "__main__":
    main()