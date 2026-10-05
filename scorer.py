"""Serving facade: morphokinetic annotations in, implantation score and ranking out."""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from chronocleave.config import load_config, resolve
from chronocleave.data.prepare import StaticScaler, to_tensors
from chronocleave.data.schema import EVENTS, MAX_HPI, STATIC_DEFAULTS, STATIC_FEATURES, STATIC_RANGES
from chronocleave.features.intervals import INTERVALS, add_intervals, clinical_flags
from chronocleave.models.sequence import build_model
from chronocleave.training.sequence_trainer import predict_proba


class InvalidRecordError(ValueError):
    """Raised when annotations are impossible, for example events out of order."""


def validate_record(record: dict) -> dict:
    """Check ranges and chronological order, and fill static defaults."""
    clean = {"embryo_id": record.get("embryo_id")}
    previous_name, previous_time = None, -np.inf
    for event in EVENTS:
        value = record.get(event)
        if value is None or (isinstance(value, float) and np.isnan(value)):
            clean[event] = np.nan
            continue
        value = float(value)
        if not 0 < value <= MAX_HPI:
            raise InvalidRecordError(f"{event} must be between 0 and {MAX_HPI:.0f} hours post insemination.")
        if value < previous_time:
            raise InvalidRecordError(f"{event} ({value} h) cannot be earlier than {previous_name} ({previous_time} h).")
        clean[event], previous_name, previous_time = value, event, value
    if np.isnan(clean["t2"]):
        raise InvalidRecordError("t2 is required: the first cleavage anchors every interval.")
    for name in STATIC_FEATURES:
        value = record.get(name)
        value = STATIC_DEFAULTS[name] if value is None else float(value)
        low, high = STATIC_RANGES[name]
        if not low <= value <= high:
            raise InvalidRecordError(f"{name} must be between {low} and {high}.")
        clean[name] = value
    return clean


class ImplantationScorer:
    """Score and rank embryos with the selected sequence model."""

    def __init__(self, bundle: dict):
        self.bundle = bundle
        self.model = build_model(bundle["params"])
        self.model.load_state_dict(bundle["state_dict"])
        self.model.eval()
        self.scaler = StaticScaler(bundle["scaler"]["mean"], bundle["scaler"]["std"])
        self.quantiles = np.asarray(bundle["score_quantiles"])

    @classmethod
    def load(cls, path: str | Path | None = None) -> "ImplantationScorer":
        path = Path(path) if path else resolve(load_config().artifacts.bundle)
        if not path.exists():
            raise FileNotFoundError(f"No model bundle at {path}. Run `make all` to train one.")
        return cls(torch.load(path, map_location="cpu", weights_only=True))

    def score(self, records: list[dict]) -> list[dict]:
        clean = [validate_record(r) for r in records]
        frame = pd.DataFrame(clean)
        steps, static, _ = to_tensors(frame, self.scaler)
        scores = predict_proba(self.model, steps, static)
        intervals = add_intervals(frame)
        results = []
        for i, (record, score) in enumerate(zip(clean, scores)):
            percentile = int(np.clip(np.searchsorted(self.quantiles, score), 0, 100))
            results.append({
                "embryo_id": record["embryo_id"],
                "implantation_score": round(float(score), 4),
                "percentile": percentile,
                "priority": "high" if percentile >= 67 else "intermediate" if percentile >= 33 else "low",
                "events_used": int(frame.loc[i, EVENTS].notna().sum()),
                "intervals": {k: (None if pd.isna(intervals.loc[i, k]) else round(float(intervals.loc[i, k]), 2)) for k in INTERVALS},
                "flags": clinical_flags(record),
            })
        return results

    def rank(self, records: list[dict]) -> list[dict]:
        """Order a cohort for transfer, highest score first."""
        ordered = sorted(self.score(records), key=lambda r: r["implantation_score"], reverse=True)
        for position, item in enumerate(ordered, start=1):
            item["rank"] = position
        return ordered


@lru_cache(maxsize=1)
def get_scorer() -> ImplantationScorer:
    return ImplantationScorer.load()
