"""Simulator for known implantation data (KID) with time lapse annotations.

Each simulated embryo has a latent developmental competence. Competence does
not appear in the data. It shapes what an embryologist can observe: how
regular the cell cycles are, how synchronous sister divisions are, whether a
blastomere divides directly into three, how quickly blastulation starts and
whether nuclei look abnormal. Implantation then depends on competence, on
those observable events, on maternal age and on a patient level effect.

Two realities of clinical data are reproduced:

* Day 3 transfers have no morula or blastocyst timings, so sequences have
  genuinely different lengths.
* Individual events are sometimes not annotated (the division happened out of
  focus or between frames).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from chronocleave.data.schema import BLASTULATION_EVENTS, EVENTS, TARGET


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def simulate_cohort(n_patients: int = 16000, day3_share: float = 0.30, gap_rate: float = 0.03, seed: int = 21) -> pd.DataFrame:
    """Return one row per transferred embryo with known implantation outcome."""
    rng = np.random.default_rng(seed)
    cohort = 1 + rng.poisson(0.6, n_patients).clip(0, 3)
    patient = np.repeat(np.arange(1, n_patients + 1), cohort)
    n = len(patient)
    age = np.repeat(np.clip(rng.normal(34.5, 4.3, n_patients), 22, 45).round(1), cohort)
    patient_effect = np.repeat(rng.normal(0, 0.35, n_patients), cohort)
    day3 = np.repeat(rng.random(n_patients) < day3_share, cohort)
    icsi = np.repeat((rng.random(n_patients) < 0.65).astype(int), cohort)

    z = rng.normal(0, 1, n) - 0.06 * (age - 34)          # latent competence
    poor = _sigmoid(-1.5 * z)                              # 0 for strong embryos, 1 for weak ones
    spread = 1 + 0.6 * poor
    tempo = np.exp(rng.normal(0, 0.06, n))                 # overall developmental speed

    tpnf = 23.8 * tempo + rng.normal(0, 1.6, n) - 0.8 * icsi
    t2 = tpnf + np.abs(rng.normal(2.6, 0.5, n))
    direct = rng.random(n) < _sigmoid(-3.3 - 0.9 * z)
    cc2 = np.where(direct, rng.uniform(0.3, 4.5, n), np.clip(rng.normal(11.3, 1.0 * spread) * tempo, 5.6, None))
    t3 = t2 + cc2
    s2 = rng.lognormal(np.log(0.42) + 1.1 * poor, 0.7)
    t4 = t3 + s2
    cc3 = np.clip(rng.normal(13.6, 1.5 * spread) * tempo, 6, None)
    t5 = t3 + cc3
    s3 = rng.lognormal(np.log(3.2) + 0.9 * poor, 0.6)
    u = np.sort(rng.random((n, 2)), axis=1)
    t6, t7, t8 = t5 + s3 * u[:, 0], t5 + s3 * u[:, 1], t5 + s3
    tm = t8 + np.clip(rng.normal(27, 4.5 * spread) * tempo, 10, None)
    tsb = tm + np.clip(rng.normal(10.5, 2.8 * spread) * tempo, 3, None) + 4.0 * poor
    tb = tsb + np.clip(rng.normal(9.0, 2.4) * tempo, 2.5, None) + 2.5 * poor
    teb = tb + np.clip(rng.normal(7.5, 2.2) * tempo, 2, None)

    mn2 = (rng.random(n) < _sigmoid(-1.7 - 0.8 * z)).astype(int)
    mn4 = (rng.random(n) < _sigmoid(-2.5 - 0.8 * z)).astype(int)
    sym2 = np.clip(1 - np.abs(rng.normal(0, 0.07 * spread)), 0.45, 1).round(3)
    sym4 = np.clip(1 - np.abs(rng.normal(0, 0.09 * spread)), 0.4, 1).round(3)
    frag = np.clip(rng.gamma(1.5, 3.5 + 5 * poor), 0, 60).round(1)

    logit = (-1.0 + 0.75 * z + patient_effect - 0.07 * (age - 34) - 0.9 * direct - 0.35 * mn2 - 0.25 * mn4
             - 0.02 * (frag - 8) - 40 * np.log(tempo) ** 2 + 0.25 * (~day3))
    implanted = (rng.random(n) < _sigmoid(logit)).astype(int)

    times = np.column_stack([tpnf, t2, t3, t4, t5, t6, t7, t8, tm, tsb, tb, teb]).round(2)
    frame = pd.DataFrame(times, columns=EVENTS)
    frame.insert(0, "embryo_id", np.arange(1, n + 1))
    frame.insert(1, "patient_id", patient)
    frame["transfer_day"] = np.where(day3, 3, 5)
    frame["maternal_age"] = age
    frame["fragmentation_pct"] = frag
    frame["multinucleation_2cell"] = mn2
    frame["multinucleation_4cell"] = mn4
    frame["symmetry_2cell"] = sym2
    frame["symmetry_4cell"] = sym4
    frame["icsi"] = icsi
    frame[TARGET] = implanted

    frame.loc[day3, BLASTULATION_EVENTS] = np.nan
    gaps = rng.random((n, len(EVENTS))) < gap_rate
    gaps[:, 1] = False                                      # the first cleavage is always annotated
    frame[EVENTS] = frame[EVENTS].mask(gaps)
    return frame
