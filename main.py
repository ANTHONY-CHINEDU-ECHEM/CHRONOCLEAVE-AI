"""REST API: JSON morphokinetic markers in, implantation score out."""
from __future__ import annotations

from typing import Optional

from fastapi import Depends, FastAPI, HTTPException
from pydantic import BaseModel, ConfigDict, Field

from chronocleave import __version__
from chronocleave.data.schema import EVENT_DESCRIPTIONS, MAX_HPI
from chronocleave.inference.scorer import ImplantationScorer, InvalidRecordError, get_scorer

app = FastAPI(
    title="ChronoCleave AI",
    version=__version__,
    description="Implantation potential from time lapse morphokinetic annotations. Research prototype. Not a medical device.",
)


def _hpi(event: str, required: bool = False):
    return Field(... if required else None, gt=0, le=MAX_HPI, description=f"Time of {EVENT_DESCRIPTIONS[event]} in hours post insemination")


class EmbryoRecord(BaseModel):
    """Annotations for one embryo. Leave out any event that was not observed."""

    model_config = ConfigDict(extra="forbid", json_schema_extra={"example": {
        "embryo_id": "E1", "tPNf": 23.1, "t2": 25.6, "t3": 36.9, "t4": 37.4, "t5": 50.2, "t6": 51.5, "t7": 52.8, "t8": 54.0,
        "tM": 81.5, "tSB": 93.0, "tB": 101.2, "tEB": 108.9, "maternal_age": 33, "fragmentation_pct": 5,
        "multinucleation_2cell": 0, "multinucleation_4cell": 0, "symmetry_2cell": 0.95, "symmetry_4cell": 0.93, "icsi": 1}})

    embryo_id: Optional[str] = None
    tPNf: Optional[float] = _hpi("tPNf")
    t2: float = _hpi("t2", required=True)
    t3: Optional[float] = _hpi("t3")
    t4: Optional[float] = _hpi("t4")
    t5: Optional[float] = _hpi("t5")
    t6: Optional[float] = _hpi("t6")
    t7: Optional[float] = _hpi("t7")
    t8: Optional[float] = _hpi("t8")
    tM: Optional[float] = _hpi("tM")
    tSB: Optional[float] = _hpi("tSB")
    tB: Optional[float] = _hpi("tB")
    tEB: Optional[float] = _hpi("tEB")
    maternal_age: Optional[float] = Field(None, ge=18, le=50)
    fragmentation_pct: Optional[float] = Field(None, ge=0, le=100)
    multinucleation_2cell: Optional[int] = Field(None, ge=0, le=1)
    multinucleation_4cell: Optional[int] = Field(None, ge=0, le=1)
    symmetry_2cell: Optional[float] = Field(None, ge=0.3, le=1.0)
    symmetry_4cell: Optional[float] = Field(None, ge=0.3, le=1.0)
    icsi: Optional[int] = Field(None, ge=0, le=1)


class CohortRequest(BaseModel):
    embryos: list[EmbryoRecord] = Field(..., min_length=1, max_length=40)


def scorer_dependency() -> ImplantationScorer:
    try:
        return get_scorer()
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "version": __version__}


@app.get("/model")
def model_info(scorer: ImplantationScorer = Depends(scorer_dependency)) -> dict:
    bundle = scorer.bundle
    return {"name": bundle["name"], "params": bundle["params"], "version": bundle["version"], "test_metrics": bundle["test_metrics"]}


@app.post("/score")
def score(record: EmbryoRecord, scorer: ImplantationScorer = Depends(scorer_dependency)) -> dict:
    try:
        return scorer.score([record.model_dump()])[0]
    except InvalidRecordError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@app.post("/rank")
def rank(request: CohortRequest, scorer: ImplantationScorer = Depends(scorer_dependency)) -> dict:
    try:
        ranked = scorer.rank([e.model_dump() for e in request.embryos])
    except InvalidRecordError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    return {"count": len(ranked), "embryos": ranked}
