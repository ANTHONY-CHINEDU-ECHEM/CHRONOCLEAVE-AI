"""Metrics for embryo selection models."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score


def binary_metrics(y: np.ndarray, score: np.ndarray, probabilistic: bool = True) -> dict:
    out = {"roc_auc": float(roc_auc_score(y, score)), "pr_auc": float(average_precision_score(y, score))}
    if probabilistic:
        out["brier"] = float(brier_score_loss(y, np.clip(score, 0, 1)))
    return out


def sibling_ranking_accuracy(patient: np.ndarray, y: np.ndarray, score: np.ndarray) -> dict:
    """How often is an implanting embryo ranked above a non implanting sibling?

    Selection happens within a patient's cohort, so this is the metric closest
    to the clinical decision. Only cohorts with both outcomes are informative.
    Ties count as half.
    """
    frame = pd.DataFrame({"patient": patient, "y": y, "score": score})
    correct, pairs, cohorts = 0.0, 0, 0
    for _, group in frame.groupby("patient"):
        positive, negative = group.loc[group.y == 1, "score"].to_numpy(), group.loc[group.y == 0, "score"].to_numpy()
        if len(positive) and len(negative):
            diff = positive[:, None] - negative[None, :]
            correct += (diff > 0).sum() + 0.5 * (diff == 0).sum()
            pairs += diff.size
            cohorts += 1
    return {"pairwise_accuracy": float(correct / pairs) if pairs else float("nan"), "pairs": int(pairs), "cohorts": int(cohorts)}


def top_quantile_lift(y: np.ndarray, score: np.ndarray, quantile: float = 0.25) -> dict:
    """Implantation rate among the top scoring quarter compared with the bottom quarter."""
    high, low = np.quantile(score, 1 - quantile), np.quantile(score, quantile)
    return {"top_rate": float(y[score >= high].mean()), "bottom_rate": float(y[score <= low].mean()), "overall_rate": float(y.mean())}
