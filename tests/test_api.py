"""Smoke tests for the inference service (feature parity + endpoint behavior)."""

import json

import pandas as pd
import pytest
from fastapi.testclient import TestClient

from src.api import LineScoreRequest, app, build_row
from src.data import ROOT

REGISTRY = ROOT / "models" / "registry.json"
pytestmark = pytest.mark.skipif(
    not REGISTRY.exists(), reason="models/registry.json missing -- run src.train first"
)

SAMPLE = {
    "stock_code": "85123A",
    "unit_price": 2.55,
    "abs_quantity": 6,
    "invoice_date": "2011-06-01T10:30:00",
    "country": "United Kingdom",
    "customer_id": 17850.0,
    "cust_prior_lines": 25,
    "cust_prior_return_rate": 0.08,
    "prod_prior_lines": 400,
    "prod_prior_return_rate": 0.03,
}


def feature_columns() -> list[str]:
    schema = json.loads(
        (ROOT / "reports" / "feature_schema.json").read_text(encoding="utf-8")
    )
    return schema["feature_columns"]


def test_build_row_matches_training_schema():
    row = build_row(LineScoreRequest(**SAMPLE))
    assert list(row.columns) == feature_columns()
    assert row.iloc[0]["has_customer"] == 1
    assert row.iloc[0]["cust_is_new"] == 0
    assert row.iloc[0]["is_zero_price"] == 0


def test_zero_price_flagged():
    row = build_row(LineScoreRequest(**{**SAMPLE, "unit_price": 0.0}))
    assert row.iloc[0]["is_zero_price"] == 1


def test_unknown_customer_is_neutral():
    req = LineScoreRequest(**{**SAMPLE, "customer_id": None})
    row = build_row(req)
    assert row.iloc[0]["has_customer"] == 0
    assert pd.isna(row.iloc[0]["cust_is_new"])


def test_health_endpoint():
    client = TestClient(app)
    resp = client.get("/health")
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "ok"
    assert body["model_version"] >= 1
    assert 0 < body["decision_threshold"] <= 1


def test_score_endpoint():
    client = TestClient(app)
    resp = client.post("/score", json=SAMPLE)
    assert resp.status_code == 200
    body = resp.json()
    assert 0.0 <= body["risk_score"] <= 1.0
    assert isinstance(body["flagged"], bool)


def test_batch_matches_single():
    client = TestClient(app)
    single = client.post("/score", json=SAMPLE).json()
    batch = client.post("/score/batch", json=[SAMPLE, SAMPLE]).json()
    assert batch[0]["risk_score"] == pytest.approx(single["risk_score"], abs=1e-9)


def test_invalid_quantity_rejected():
    client = TestClient(app)
    assert client.post("/score", json={**SAMPLE, "abs_quantity": -1}).status_code == 422


def test_empty_batch_rejected():
    client = TestClient(app)
    assert client.post("/score/batch", json=[]).status_code == 422
