"""Experiment pipeline: baselines, sequence model sweep and MLflow tracking."""
from __future__ import annotations

import json
import logging
import tempfile
from pathlib import Path

import mlflow
import numpy as np
import pandas as pd
import torch
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from chronocleave import __version__
from chronocleave.config import Config, resolve
from chronocleave.data.prepare import StaticScaler, grouped_split, to_tensors
from chronocleave.data.schema import EVENTS, GROUP, TARGET
from chronocleave.evaluation.metrics import binary_metrics, sibling_ranking_accuracy, top_quantile_lift
from chronocleave.features.intervals import hierarchical_rule_score, tabular_matrix
from chronocleave.models.sequence import build_model
from chronocleave.training.sequence_trainer import predict_proba, train_sequence_model

logger = logging.getLogger(__name__)


def _run_name(params: dict) -> str:
    size = params.get("hidden", params.get("channels"))
    return f"{params['model']}_{size}"


def _evaluate(frame: pd.DataFrame, score: np.ndarray, probabilistic: bool = True) -> dict:
    y = frame[TARGET].to_numpy()
    metrics = binary_metrics(y, score, probabilistic)
    metrics["sibling_pairwise_accuracy"] = sibling_ranking_accuracy(frame[GROUP].to_numpy(), y, score)["pairwise_accuracy"]
    for day in (3, 5):
        mask = (frame["transfer_day"] == day).to_numpy()
        metrics[f"roc_auc_day{day}"] = binary_metrics(y[mask], score[mask], False)["roc_auc"]
    return metrics


def event_occlusion_importance(model, frame: pd.DataFrame, scaler: StaticScaler) -> dict[str, float]:
    """Drop in test AUC when one event is hidden from the model.

    Hiding an event removes its timestamp and merges the two neighbouring
    intervals, exactly as if the annotator had skipped it.
    """
    y = frame[TARGET].to_numpy()
    steps, static, _ = to_tensors(frame, scaler)
    baseline = binary_metrics(y, predict_proba(model, steps, static), False)["roc_auc"]
    drops = {}
    for event in EVENTS:
        steps, static, _ = to_tensors(frame.assign(**{event: np.nan}), scaler)
        drops[event] = baseline - binary_metrics(y, predict_proba(model, steps, static), False)["roc_auc"]
    return drops


def run_pipeline(cfg: Config) -> dict:
    """Train every candidate, log each one to MLflow, select on validation and report on test."""
    frame = pd.read_csv(resolve(cfg.data.raw_path))
    splits = grouped_split(frame, dict(cfg.data.split), cfg.seed)
    scaler = StaticScaler.fit(splits["train"])
    tensors = {name: to_tensors(part, scaler) for name, part in splits.items()}
    report_dir = resolve(cfg.artifacts.report_dir)
    report_dir.mkdir(parents=True, exist_ok=True)

    mlflow.set_tracking_uri(cfg.mlflow.tracking_uri)
    mlflow.set_experiment(cfg.mlflow.experiment)
    results, test_scores, histories = {}, {}, {}

    with mlflow.start_run(run_name="pipeline") as parent:
        mlflow.set_tags({"project": "chronocleave", "version": __version__})
        mlflow.log_params({"n_embryos": len(frame), "n_patients": frame[GROUP].nunique(), "seed": cfg.seed,
                           **{f"n_{k}": len(v) for k, v in splits.items()},
                           "implantation_rate": round(float(frame[TARGET].mean()), 4)})

        # ---------------------------------------------------------- baselines
        baselines = {
            "rule_based": None,
            "logistic_regression": make_pipeline(SimpleImputer(strategy="median", add_indicator=True), StandardScaler(),
                                                 LogisticRegression(max_iter=2000, C=0.5)),
            "gradient_boosting": HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05, max_leaf_nodes=15,
                                                                l2_regularization=1.0, early_stopping=True, random_state=cfg.seed),
        }
        X = {name: tabular_matrix(part) for name, part in splits.items()}
        for name, estimator in baselines.items():
            with mlflow.start_run(run_name=name, nested=True):
                mlflow.set_tag("family", "baseline")
                if estimator is None:
                    val_score, test_score, probabilistic = hierarchical_rule_score(splits["val"]), hierarchical_rule_score(splits["test"]), False
                else:
                    estimator.fit(X["train"], splits["train"][TARGET])
                    val_score, test_score, probabilistic = estimator.predict_proba(X["val"])[:, 1], estimator.predict_proba(X["test"])[:, 1], True
                    mlflow.log_params({"estimator": type(estimator).__name__ if name == "gradient_boosting" else "LogisticRegression"})
                results[name] = {"family": "baseline", "val": _evaluate(splits["val"], val_score, probabilistic),
                                 "test": _evaluate(splits["test"], test_score, probabilistic)}
                mlflow.log_metrics({f"val_{k}": v for k, v in results[name]["val"].items()})
                mlflow.log_metrics({f"test_{k}": v for k, v in results[name]["test"].items()})
                test_scores[name] = test_score
                logger.info("%s: val AUC %.4f", name, results[name]["val"]["roc_auc"])

        # ------------------------------------------------------ sequence sweep
        trained = {}
        for params in cfg.training.sweep:
            params = dict(params)
            name = _run_name(params)
            with mlflow.start_run(run_name=name, nested=True) as run:
                mlflow.set_tag("family", "sequence")
                mlflow.log_params({**params, "batch_size": cfg.training.batch_size, "weight_decay": cfg.training.weight_decay})
                model = build_model(params)
                mlflow.log_param("trainable_parameters", sum(p.numel() for p in model.parameters()))
                model, history = train_sequence_model(
                    model, tensors["train"], tensors["val"], params, cfg.training.batch_size, cfg.training.max_epochs,
                    cfg.training.patience, cfg.training.weight_decay, cfg.seed,
                    on_epoch=lambda row: mlflow.log_metrics({"epoch_train_loss": row["train_loss"], "epoch_val_auc": row["val_auc"]}, step=row["epoch"]),
                )
                val_score = predict_proba(model, tensors["val"][0], tensors["val"][1])
                test_score = predict_proba(model, tensors["test"][0], tensors["test"][1])
                results[name] = {"family": "sequence", "params": params, "epochs": len(history), "run_id": run.info.run_id,
                                 "val": _evaluate(splits["val"], val_score), "test": _evaluate(splits["test"], test_score)}
                mlflow.log_metrics({f"val_{k}": v for k, v in results[name]["val"].items()})
                mlflow.log_metrics({f"test_{k}": v for k, v in results[name]["test"].items()})
                with tempfile.TemporaryDirectory() as tmp:
                    torch.save(model.state_dict(), Path(tmp) / "state_dict.pt")
                    mlflow.log_artifact(str(Path(tmp) / "state_dict.pt"), artifact_path="model")
                trained[name], test_scores[name], histories[name] = model, test_score, history
                logger.info("%s: val AUC %.4f after %d epochs", name, results[name]["val"]["roc_auc"], len(history))

        # ------------------------------------------------- selection and export
        selected = max(trained, key=lambda k: results[k]["val"]["roc_auc"])
        model = trained[selected]
        train_scores = predict_proba(model, tensors["train"][0], tensors["train"][1])
        y_test = splits["test"][TARGET].to_numpy()
        extras = {
            "event_occlusion_auc_drop": event_occlusion_importance(model, splits["test"], scaler),
            "top_quartile_lift": top_quantile_lift(y_test, test_scores[selected]),
            "sibling_ranking": sibling_ranking_accuracy(splits["test"][GROUP].to_numpy(), y_test, test_scores[selected]),
        }
        bundle = {
            "state_dict": model.state_dict(), "params": results[selected]["params"], "name": selected, "version": __version__,
            "scaler": {"mean": scaler.mean, "std": scaler.std},
            "score_quantiles": np.quantile(train_scores, np.linspace(0, 1, 101)).tolist(),
            "test_metrics": results[selected]["test"],
        }
        bundle_path = resolve(cfg.artifacts.bundle)
        bundle_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(bundle, bundle_path)

        report = {"selected_model": selected, "rows": {k: len(v) for k, v in splits.items()},
                  "implantation_rate": {k: float(v[TARGET].mean()) for k, v in splits.items()},
                  "results": results, "selected_extras": extras, "histories": histories}
        with open(report_dir / "metrics.json", "w", encoding="utf-8") as handle:
            json.dump(report, handle, indent=2)
        predictions = splits["test"][["embryo_id", GROUP, "transfer_day", TARGET]].copy()
        for name, score in test_scores.items():
            predictions[name] = score
        predictions.to_csv(report_dir / "test_predictions.csv", index=False)

        mlflow.set_tags({"selected_model": selected, "selected_run_id": results[selected]["run_id"]})
        mlflow.log_metrics({f"selected_test_{k}": v for k, v in results[selected]["test"].items()})
        mlflow.log_artifact(str(bundle_path), artifact_path="serving")
        mlflow.log_artifact(str(report_dir / "metrics.json"), artifact_path="reports")
        report["parent_run_id"] = parent.info.run_id
    logger.info("Selected %s | test %s", selected, results[selected]["test"])
    return report
