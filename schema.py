"""Morphokinetic vocabulary shared by the simulator, the models and the API.

All event times are hours post insemination (hpi), the convention used by
time lapse incubators.
"""
from __future__ import annotations

# Ordered developmental events annotated from time lapse imaging.
EVENTS = ["tPNf", "t2", "t3", "t4", "t5", "t6", "t7", "t8", "tM", "tSB", "tB", "tEB"]
EVENT_DESCRIPTIONS = {
    "tPNf": "pronuclei fading", "t2": "division to 2 cells", "t3": "division to 3 cells", "t4": "division to 4 cells",
    "t5": "division to 5 cells", "t6": "division to 6 cells", "t7": "division to 7 cells", "t8": "division to 8 cells",
    "tM": "morula", "tSB": "start of blastulation", "tB": "full blastocyst", "tEB": "expanded blastocyst",
}
CLEAVAGE_EVENTS = EVENTS[:8]          # available for a day 3 transfer
BLASTULATION_EVENTS = EVENTS[8:]      # only available with extended culture

STATIC_FEATURES = [
    "maternal_age", "fragmentation_pct", "multinucleation_2cell", "multinucleation_4cell",
    "symmetry_2cell", "symmetry_4cell", "icsi",
]
STATIC_DEFAULTS = {"maternal_age": 34.0, "fragmentation_pct": 8.0, "multinucleation_2cell": 0, "multinucleation_4cell": 0,
                   "symmetry_2cell": 0.92, "symmetry_4cell": 0.9, "icsi": 1}
STATIC_RANGES = {"maternal_age": (18, 50), "fragmentation_pct": (0, 100), "multinucleation_2cell": (0, 1),
                 "multinucleation_4cell": (0, 1), "symmetry_2cell": (0.3, 1.0), "symmetry_4cell": (0.3, 1.0), "icsi": (0, 1)}
TARGET = "implanted"
GROUP = "patient_id"
MAX_HPI = 160.0

# Published reference windows used by the rule based comparator (Meseguer et al., 2011).
T5_OPTIMAL = (48.8, 56.6)
S2_OPTIMAL_MAX = 0.76
CC2_OPTIMAL_MAX = 11.9
DIRECT_CLEAVAGE_CC2 = 5.0
