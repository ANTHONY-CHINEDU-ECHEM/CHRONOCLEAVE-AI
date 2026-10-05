import numpy as np
import pandas as pd

from chronocleave.data.prepare import StaticScaler, encode_sequences, grouped_split
from chronocleave.data.schema import BLASTULATION_EVENTS, EVENTS, STATIC_FEATURES
from chronocleave.data.simulate import simulate_cohort
from chronocleave.features.intervals import add_intervals, clinical_flags, hierarchical_rule_score, tabular_matrix

GOOD = {"tPNf": 23.0, "t2": 25.5, "t3": 36.8, "t4": 37.3, "t5": 50.5, "t6": 51.6, "t7": 52.9, "t8": 54.1,
        "tM": 82.0, "tSB": 93.0, "tB": 101.0, "tEB": 108.5, "maternal_age": 32, "fragmentation_pct": 4,
        "multinucleation_2cell": 0, "multinucleation_4cell": 0, "symmetry_2cell": 0.95, "symmetry_4cell": 0.94, "icsi": 1}


def test_simulation_is_deterministic(cohort):
    pd.testing.assert_frame_equal(cohort, simulate_cohort(n_patients=1800, seed=4))


def test_event_times_are_chronological(cohort):
    times = cohort[EVENTS].to_numpy()
    for row in times[:500]:
        observed = row[~np.isnan(row)]
        assert np.all(np.diff(observed) >= 0)


def test_day3_transfers_have_no_blastulation_events(cohort):
    day3 = cohort[cohort["transfer_day"] == 3]
    assert len(day3) and day3[BLASTULATION_EVENTS].isna().all().all()
    assert cohort.loc[cohort["transfer_day"] == 5, "tB"].notna().mean() > 0.9


def test_intervals_and_flags():
    frame = add_intervals(pd.DataFrame([GOOD]))
    assert frame.loc[0, "cc2"] == np.round(36.8 - 25.5, 6) or abs(frame.loc[0, "cc2"] - 11.3) < 1e-9
    assert abs(frame.loc[0, "s2"] - 0.5) < 1e-9 and frame.loc[0, "direct_cleavage"] == 0 and frame.loc[0, "t5_in_window"] == 1
    assert clinical_flags(GOOD) == []
    direct = {**GOOD, "t3": 27.0, "t4": 37.3}
    assert any("Direct cleavage" in flag for flag in clinical_flags(direct))


def test_rule_score_orders_good_above_excluded():
    excluded = {**GOOD, "t3": 27.0}
    late = {**GOOD, "t5": 62.0, "t6": 63, "t7": 64, "t8": 65}
    scores = hierarchical_rule_score(pd.DataFrame([GOOD, late, excluded]))
    assert scores[0] > scores[1] > scores[2] == 0


def test_missing_intervals_propagate_and_matrix_shape(cohort):
    matrix = tabular_matrix(cohort)
    assert matrix.loc[cohort["transfer_day"] == 3, "blastulation_duration"].isna().all()
    assert not matrix[STATIC_FEATURES].isna().any().any()


def test_sequence_encoding_merges_gaps_across_missing_events():
    frame = pd.DataFrame([{**GOOD, "t3": np.nan}])
    steps = encode_sequences(frame)
    assert steps.shape == (1, len(EVENTS), 3)
    t3, t4 = EVENTS.index("t3"), EVENTS.index("t4")
    assert steps[0, t3].tolist() == [0.0, 0.0, 0.0]
    assert np.isclose(steps[0, t4, 1], (37.3 - 25.5) / 24.0)      # gap measured from t2, the last annotated event


def test_grouped_split_keeps_patients_together(cohort):
    splits = grouped_split(cohort, {"train": 0.7, "val": 0.15, "test": 0.15}, seed=0)
    ids = {name: set(part["patient_id"]) for name, part in splits.items()}
    assert not ids["train"] & ids["val"] and not ids["train"] & ids["test"] and not ids["val"] & ids["test"]
    assert sum(len(p) for p in splits.values()) == len(cohort)


def test_static_scaler_uses_defaults_for_missing_columns(cohort):
    scaler = StaticScaler.fit(cohort)
    out = scaler.transform(pd.DataFrame([{"maternal_age": 34.0}]))
    assert out.shape == (1, len(STATIC_FEATURES)) and np.isfinite(out).all()
