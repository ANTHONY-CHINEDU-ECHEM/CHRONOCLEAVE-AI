"""Patient grouped splitting and conversion of annotations into model tensors."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch

from chronocleave.data.schema import EVENTS, GROUP, MAX_HPI, STATIC_DEFAULTS, STATIC_FEATURES, TARGET

STEP_FEATURES = 3   # scaled time, scaled gap since previous annotated event, observed flag


def grouped_split(frame: pd.DataFrame, fractions: dict, seed: int) -> dict[str, pd.DataFrame]:
    """Split by patient so sibling embryos never appear on both sides of a split."""
    rng = np.random.default_rng(seed)
    patients = frame[GROUP].unique()
    rng.shuffle(patients)
    n_train = int(len(patients) * fractions["train"])
    n_val = int(len(patients) * fractions["val"])
    parts = {"train": patients[:n_train], "val": patients[n_train:n_train + n_val], "test": patients[n_train + n_val:]}
    return {name: frame[frame[GROUP].isin(ids)].reset_index(drop=True) for name, ids in parts.items()}


@dataclass
class StaticScaler:
    """Standardiser for the static covariates, fitted on training embryos only."""

    mean: dict
    std: dict

    @classmethod
    def fit(cls, frame: pd.DataFrame) -> "StaticScaler":
        return cls({c: float(frame[c].mean()) for c in STATIC_FEATURES},
                   {c: float(frame[c].std(ddof=0)) or 1.0 for c in STATIC_FEATURES})

    def transform(self, frame: pd.DataFrame) -> np.ndarray:
        cols = []
        for c in STATIC_FEATURES:
            values = pd.to_numeric(frame[c], errors="coerce").fillna(STATIC_DEFAULTS[c]) if c in frame else STATIC_DEFAULTS[c]
            cols.append((np.asarray(values, dtype=np.float32) - self.mean[c]) / self.std[c] * np.ones(len(frame), np.float32))
        return np.column_stack(cols).astype(np.float32)


def encode_sequences(frame: pd.DataFrame) -> np.ndarray:
    """Turn event timestamps into a (n, events, 3) array.

    Each step carries the event time scaled to roughly [0, 1], the gap since
    the previous *annotated* event scaled by a day, and a flag saying whether
    the event was annotated. Unannotated steps are zero filled, and the flag
    lets the network tell a missing event from an early one.
    """
    times = frame.reindex(columns=EVENTS).to_numpy(dtype=np.float32)
    observed = ~np.isnan(times)
    filled = np.where(observed, times, np.nan)
    previous = np.zeros(len(frame), dtype=np.float32)
    gaps = np.zeros_like(times)
    for k in range(times.shape[1]):
        gaps[:, k] = np.where(observed[:, k], filled[:, k] - previous, 0.0)
        previous = np.where(observed[:, k], filled[:, k], previous)
    steps = np.stack([np.where(observed, times / MAX_HPI, 0.0), gaps / 24.0, observed.astype(np.float32)], axis=-1)
    return np.nan_to_num(steps).astype(np.float32)


def to_tensors(frame: pd.DataFrame, scaler: StaticScaler) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Sequence tensor, static tensor and target tensor for a frame."""
    target = frame[TARGET].to_numpy(dtype=np.float32) if TARGET in frame else np.zeros(len(frame), np.float32)
    return torch.from_numpy(encode_sequences(frame)), torch.from_numpy(scaler.transform(frame)), torch.from_numpy(target)
