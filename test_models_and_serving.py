import mlflow
import pytest
import torch
from fastapi.testclient import TestClient
from test_data_and_features import GOOD

from chronocleave.api.main import app, scorer_dependency
from chronocleave.data.schema import EVENTS, STATIC_FEATURES
from chronocleave.evaluation.metrics import sibling_ranking_accuracy, top_quantile_lift
from chronocleave.inference.scorer import ImplantationScorer, InvalidRecordError, validate_record
from chronocleave.models.sequence import build_model


@pytest.mark.parametrize("params", [{"model": "lstm", "hidden": 8, "layers": 2, "dropout": 0.1},
                                    {"model": "tcn", "channels": 8, "levels": 3, "kernel": 3, "dropout": 0.1}])
def test_models_return_one_logit_per_embryo(params):
    model = build_model(params).eval()
    steps = torch.rand(5, len(EVENTS), 3)
    steps[..., 2] = 1.0
    assert model(steps, torch.randn(5, len(STATIC_FEATURES))).shape == (5,)


def test_unknown_model_is_rejected():
    with pytest.raises(ValueError):
        build_model({"model": "transformer"})


def test_sibling_ranking_metric():
    patient = [1, 1, 2, 2, 3]
    result = sibling_ranking_accuracy(patient, [1, 0, 0, 1, 1], [0.9, 0.2, 0.7, 0.3, 0.5])
    assert result["cohorts"] == 2 and result["pairs"] == 2 and result["pairwise_accuracy"] == 0.5
    lift = top_quantile_lift(torch.tensor([1, 1, 0, 0]).numpy(), torch.tensor([0.9, 0.8, 0.2, 0.1]).numpy())
    assert lift["top_rate"] == 1.0 and lift["bottom_rate"] == 0.0


def test_pipeline_logs_runs_and_beats_chance(pipeline_run):
    cfg, report, _ = pipeline_run
    selected = report["selected_model"]
    assert report["results"][selected]["test"]["roc_auc"] > 0.58
    assert {"rule_based", "logistic_regression", "gradient_boosting"} <= set(report["results"])
    mlflow.set_tracking_uri(cfg.mlflow.tracking_uri)
    runs = mlflow.search_runs(experiment_names=[cfg.mlflow.experiment])
    assert len(runs) == 6                      # parent, three baselines, two sequence models
    assert "metrics.val_roc_auc" in runs.columns and "params.learning_rate" in runs.columns


@pytest.fixture(scope="module")
def scorer(pipeline_run):
    return ImplantationScorer.load(pipeline_run[0].artifacts.bundle)


def test_scorer_ranks_and_explains(scorer):
    poor = {**GOOD, "embryo_id": "poor", "t3": 27.5, "multinucleation_2cell": 1, "fragmentation_pct": 30, "maternal_age": 42}
    ranked = scorer.rank([{**GOOD, "embryo_id": "good"}, poor])
    assert [r["rank"] for r in ranked] == [1, 2]
    by_id = {r["embryo_id"]: r for r in ranked}
    assert by_id["poor"]["flags"] and not by_id["good"]["flags"]
    assert by_id["good"]["events_used"] == 12 and 0 <= by_id["good"]["percentile"] <= 100


def test_scorer_accepts_day3_records(scorer):
    day3 = {k: v for k, v in GOOD.items() if k not in {"tM", "tSB", "tB", "tEB"}}
    result = scorer.score([day3])[0]
    assert result["events_used"] == 8 and result["intervals"]["blastulation_duration"] is None


def test_validation_rejects_impossible_annotations():
    with pytest.raises(InvalidRecordError, match="cannot be earlier"):
        validate_record({**GOOD, "t4": 30.0})
    with pytest.raises(InvalidRecordError, match="t2 is required"):
        validate_record({k: v for k, v in GOOD.items() if k != "t2"})


def test_api_contract(scorer):
    app.dependency_overrides[scorer_dependency] = lambda: scorer
    client = TestClient(app)
    assert client.get("/health").json()["status"] == "ok"
    assert client.get("/model").json()["name"]
    scored = client.post("/score", json=GOOD)
    assert scored.status_code == 200 and 0 < scored.json()["implantation_score"] < 1
    ranked = client.post("/rank", json={"embryos": [{**GOOD, "embryo_id": "a"}, {**GOOD, "embryo_id": "b", "t3": 28.0}]}).json()
    assert ranked["count"] == 2 and ranked["embryos"][0]["rank"] == 1
    assert client.post("/score", json={**GOOD, "t4": 30.0}).status_code == 422       # out of order
    assert client.post("/score", json={**GOOD, "t5": 400}).status_code == 422        # out of range
    assert client.post("/score", json={**GOOD, "unknown": 1}).status_code == 422
    app.dependency_overrides.clear()
