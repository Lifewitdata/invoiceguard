"""FastAPI inference service for invoice-line risk scoring.

Contract: callers post line-level fields plus the point-in-time history features
that an upstream feature job maintains (same definitions as src/features.py).
Line-level features are derived here so they are identical to training -- a
feature-parity guarantee exercised by the test suite.
"""

from __future__ import annotations

import json
from datetime import datetime
from functools import lru_cache

import joblib
import numpy as np
import pandas as pd
import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from src.data import ROOT, load_config

app = FastAPI(title="InvoiceGuard", version="1.0", description="Invoice-line return risk scoring")

MAX_BATCH = 1000

HISTORY_FIELDS = [
    "cust_prior_lines",
    "cust_prior_invoices",
    "cust_prior_return_rate",
    "cust_prior_mean_abs_amount",
    "cust_prior_mean_abs_qty",
    "cust_recency_days",
    "cust_price_z",
    "cust_amount_ratio",
    "cust_qty_ratio",
    "prod_prior_lines",
    "prod_prior_return_rate",
    "prod_prior_zero_price_share",
    "prod_price_z",
    "prod_price_ratio",
    "prod_qty_ratio",
]


class LineScoreRequest(BaseModel):
    stock_code: str = Field(min_length=1, max_length=20)
    unit_price: float
    abs_quantity: float = Field(gt=0)
    invoice_date: datetime
    country: str = "UNKNOWN"
    description: str = ""
    customer_id: float | None = None
    cust_is_new: int | None = None
    cust_prior_lines: float | None = None
    cust_prior_invoices: float | None = None
    cust_prior_return_rate: float | None = None
    cust_prior_mean_abs_amount: float | None = None
    cust_prior_mean_abs_qty: float | None = None
    cust_recency_days: float | None = None
    cust_price_z: float | None = None
    cust_amount_ratio: float | None = None
    cust_qty_ratio: float | None = None
    prod_prior_lines: float | None = None
    prod_prior_return_rate: float | None = None
    prod_prior_zero_price_share: float | None = None
    prod_price_z: float | None = None
    prod_price_ratio: float | None = None
    prod_qty_ratio: float | None = None


class ScoreResponse(BaseModel):
    risk_score: float
    flagged: bool
    threshold: float
    model_version: int
    model_name: str


@lru_cache
def _active_run() -> dict:
    registry_path = ROOT / "models" / "registry.json"
    if not registry_path.exists():
        raise FileNotFoundError("models/registry.json not found -- run python -m src.train first")
    registry = json.loads(registry_path.read_text(encoding="utf-8"))
    return next(r for r in registry["runs"] if r["version"] == registry["active_version"])


@lru_cache
def _artifact() -> dict:
    return joblib.load(ROOT / "models" / _active_run()["artifact"])


@lru_cache
def _freq_maps() -> dict:
    path = ROOT / "models" / "freq_maps.json"
    if not path.exists():
        return {"country": {}, "stock_code": {}}
    return json.loads(path.read_text(encoding="utf-8"))


def _threshold() -> float:
    return float(_active_run().get("evaluation", {}).get("decision_threshold", 0.5))


def build_row(req: LineScoreRequest) -> pd.DataFrame:
    freq = _freq_maps()
    country_freq = freq["country"].get(req.country, 1e-9)
    stock_freq = freq["stock_code"].get(req.stock_code, 1e-9)

    abs_price = abs(req.unit_price)
    abs_amount = abs_price * req.abs_quantity
    dt = req.invoice_date

    if req.cust_is_new is None:
        if req.customer_id is not None and req.cust_prior_lines is not None:
            cust_is_new = int(req.cust_prior_lines == 0)
        else:
            cust_is_new = np.nan
    else:
        cust_is_new = req.cust_is_new

    row = {
        "abs_quantity_log": np.log1p(req.abs_quantity),
        "unit_price_log": np.log1p(abs_price),
        "abs_amount_log": np.log1p(abs_amount),
        "is_zero_price": int(req.unit_price <= 0),
        "is_negative_price": int(req.unit_price < 0),
        "desc_len": len(req.description),
        "has_description": int(bool(req.description.strip())),
        "stock_code_len": len(req.stock_code),
        "stock_code_numeric": int(req.stock_code.isdigit()),
        "hour": dt.hour,
        "day_of_week": dt.weekday(),
        "month": dt.month,
        "day_of_month": dt.day,
        "country_freq": country_freq,
        "stock_code_freq": stock_freq,
        "has_customer": int(req.customer_id is not None),
        "cust_is_new": cust_is_new,
        **{
            field: (np.nan if getattr(req, field) is None else float(getattr(req, field)))
            for field in HISTORY_FIELDS
        },
    }
    columns = _artifact()["feature_columns"]
    return pd.DataFrame([row]).reindex(columns=columns)


def _score_batch(requests: list[LineScoreRequest]) -> list[ScoreResponse]:
    artifact = _artifact()
    X = pd.concat([build_row(req) for req in requests], ignore_index=True)
    scores = artifact["pipeline"].predict_proba(X)[:, 1]
    threshold = _threshold()
    return [
        ScoreResponse(
            risk_score=float(score),
            flagged=bool(score >= threshold),
            threshold=threshold,
            model_version=artifact["version"],
            model_name=artifact["model_name"],
        )
        for score in scores
    ]


@app.get("/health")
def health() -> dict:
    try:
        run = _active_run()
        artifact = _artifact()
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    return {
        "status": "ok",
        "model_name": artifact["model_name"],
        "model_version": artifact["version"],
        "decision_threshold": _threshold(),
        "trained_at_utc": run["trained_at_utc"],
    }


@app.post("/score", response_model=ScoreResponse)
def score(req: LineScoreRequest) -> ScoreResponse:
    try:
        return _score_batch([req])[0]
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


@app.post("/score/batch", response_model=list[ScoreResponse])
def score_batch(requests: list[LineScoreRequest]) -> list[ScoreResponse]:
    if not requests:
        raise HTTPException(status_code=422, detail="empty batch")
    if len(requests) > MAX_BATCH:
        raise HTTPException(status_code=422, detail=f"batch size exceeds {MAX_BATCH}")
    try:
        return _score_batch(requests)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


def main() -> None:
    cfg = load_config()["api"]
    uvicorn.run("src.api:app", host=cfg["host"], port=cfg["port"])


if __name__ == "__main__":
    main()
