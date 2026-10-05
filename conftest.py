from __future__ import annotations

import pytest
import torch

from chronocleave.config import Config, load_config
from chronocleave.data.simulate import simulate_cohort


@pytest.fixture(scope="session")
def cohort():
    return simulate_cohort(n_patients=1800, seed=4)


@pytest.fixture(scope="session")
def pipeline_run(tmp_path_factory, cohort):
    """Run the whole experiment pipeline on a small cohort with a private MLflow store."""
    from chronocleave.training.pipeline import run_pipeline

    root = tmp_path_factory.mktemp("run")
    cohort.to_csv(root / "raw.csv", index=False)
    cfg = dict(load_config())
    cfg["data"] = {**cfg["data"], "raw_path": str(root / "raw.csv")}
    cfg["training"] = {**cfg["training"], "max_epochs": 3, "sweep": [
        {"model": "lstm", "hidden": 16, "layers": 1, "dropout": 0.0, "learning_rate": 0.005},
        {"model": "tcn", "channels": 16, "levels": 2, "kernel": 3, "dropout": 0.0, "learning_rate": 0.005}]}
    cfg["mlflow"] = {"tracking_uri": f"sqlite:///{root / 'mlflow.db'}", "experiment": "test"}
    cfg["artifacts"] = {"bundle": str(root / "bundle.pt"), "report_dir": str(root / "reports"), "figure_dir": str(root / "figures")}
    torch.set_num_threads(1)
    import os
    previous = os.getcwd()
    os.chdir(root)                         # keeps MLflow artifact folders out of the repository
    try:
        report = run_pipeline(Config(cfg))
    finally:
        os.chdir(previous)
    return Config(cfg), report, root
