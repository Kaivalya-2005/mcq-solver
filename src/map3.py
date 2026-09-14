"""Competition metric: Mean Average Precision @ 3.
Import this from every stage (baseline, DeBERTa, RAG) so scores are comparable."""
import numpy as np


def map_at_3(true_labels, pred_top3_lists):
    scores = []
    for true, preds in zip(true_labels, pred_top3_lists):
        score = 0.0
        for i, p in enumerate(preds[:3]):
            if p == true:
                score = 1.0 / (i + 1)
                break
        scores.append(score)
    return float(np.mean(scores))
