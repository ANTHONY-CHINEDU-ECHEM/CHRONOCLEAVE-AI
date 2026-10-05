"""Interval features and the rule based comparator.

Absolute event times are dominated by when insemination was recorded. What
embryologists actually reason about are *durations*: the length of the second
and third cell cycles and the synchrony of sister divisions. This module turns
raw timestamps into those intervals.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from chronocleave.data.schema import CC2_OPTIMAL_MAX, DIRECT_CLEAVAGE_CC2, EVENTS, S2_OPTIMAL_MAX, STATIC_FEATURES, T5_OPTIMAL

INTERVALS = {
    "cc2": ("t3", "t2"),                    # second cell cycle
    "cc3": ("t5", "t3"),                    # third cell cycle
    "s2": ("t4", "t3"),                     # synchrony of the second round of divisions
    "s3": ("t8", "t5"),                     # synchrony of the third round
    "t5_minus_t2": ("t5", "t2"),
    "first_cytokinesis_lag": ("t2", "tPNf"),
    "compaction_time": ("tM", "t8"),
    "blastulation_onset": ("tSB", "tM"),
    "blastulation_duration": ("tB", "tSB"),
    "expansion_duration": ("tEB", "tB"),
}


def add_intervals(frame: pd.DataFrame) -> pd.DataFrame:
    """Append interval columns and clinical flags. Missing events propagate as missing."""
    out = frame.copy()
    for name, (later, earlier) in INTERVALS.items():
        out[name] = out[later] - out[earlier]
    out["cc3_to_cc2_ratio"] = out["cc3"] / out["cc2"].clip(lower=0.5)
    out["direct_cleavage"] = (out["cc2"] < DIRECT_CLEAVAGE_CC2).astype(float).where(out["cc2"].notna())
    out["t5_in_window"] = out["t5"].between(*T5_OPTIMAL).astype(float).where(out["t5"].notna())
    out["s2_in_window"] = (out["s2"] <= S2_OPTIMAL_MAX).astype(float).where(out["s2"].notna())
    out["events_annotated"] = out[EVENTS].notna().sum(axis=1)
    out["has_blastulation"] = out["tSB"].notna().astype(int)
    return out


ENGINEERED_FEATURES = list(INTERVALS) + ["cc3_to_cc2_ratio", "direct_cleavage", "t5_in_window", "s2_in_window",
                                         "events_annotated", "has_blastulation", "t2", "t5", "t8"] + STATIC_FEATURES


def tabular_matrix(frame: pd.DataFrame) -> pd.DataFrame:
    """Feature matrix for the classical baselines."""
    return add_intervals(frame)[ENGINEERED_FEATURES]


def hierarchical_rule_score(frame: pd.DataFrame) -> np.ndarray:
    """Rule based comparator in the spirit of published hierarchical selection models.

    Embryos with an exclusion criterion (direct cleavage, marked asymmetry at
    the two cell stage, or multinucleation at the four cell stage) receive the
    lowest score. The rest are ranked by whether t5, s2 and cc2 fall in their
    published optimal windows. Missing timings count as outside the window.
    """
    f = add_intervals(frame)
    excluded = (f["cc2"] < DIRECT_CLEAVAGE_CC2) | (f["symmetry_2cell"] < 0.75) | (f["multinucleation_4cell"] == 1)
    score = np.where(f["t5"].between(*T5_OPTIMAL), 4.0, 2.0)
    score = score + 2.0 * (f["s2"] <= S2_OPTIMAL_MAX) + 1.0 * (f["cc2"] <= CC2_OPTIMAL_MAX)
    return np.where(excluded, 0.0, score)


def clinical_flags(record: dict) -> list[str]:
    """Plain language observations for one embryo, shown alongside the model score."""
    f = add_intervals(pd.DataFrame([record])).iloc[0]
    flags = []
    if pd.notna(f["cc2"]) and f["cc2"] < DIRECT_CLEAVAGE_CC2:
        flags.append(f"Direct cleavage: second cell cycle lasted {f['cc2']:.1f} h (under {DIRECT_CLEAVAGE_CC2:.0f} h).")
    if pd.notna(f["t5"]) and not T5_OPTIMAL[0] <= f["t5"] <= T5_OPTIMAL[1]:
        side = "earlier" if f["t5"] < T5_OPTIMAL[0] else "later"
        flags.append(f"t5 at {f['t5']:.1f} h is {side} than the reference window of {T5_OPTIMAL[0]} to {T5_OPTIMAL[1]} h.")
    if pd.notna(f["s2"]) and f["s2"] > S2_OPTIMAL_MAX:
        flags.append(f"Asynchronous second round: s2 of {f['s2']:.2f} h exceeds {S2_OPTIMAL_MAX} h.")
    if pd.notna(f["cc2"]) and f["cc2"] > CC2_OPTIMAL_MAX:
        flags.append(f"Prolonged second cell cycle: cc2 of {f['cc2']:.1f} h exceeds {CC2_OPTIMAL_MAX} h.")
    if f.get("multinucleation_2cell", 0) == 1 or f.get("multinucleation_4cell", 0) == 1:
        flags.append("Multinucleation observed during early cleavage.")
    if pd.notna(f.get("symmetry_2cell")) and f["symmetry_2cell"] < 0.75:
        flags.append("Uneven blastomere size at the two cell stage.")
    if pd.notna(f.get("fragmentation_pct")) and f["fragmentation_pct"] > 25:
        flags.append(f"High fragmentation ({f['fragmentation_pct']:.0f} percent).")
    if pd.notna(f["blastulation_duration"]) and f["blastulation_duration"] > 14:
        flags.append(f"Slow blastulation: {f['blastulation_duration']:.1f} h from start of blastulation to full blastocyst.")
    return flags
