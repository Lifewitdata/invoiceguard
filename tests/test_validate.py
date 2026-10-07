"""Tests for the data quality validation rules."""

import pandas as pd
import pytest

from src.validate import validate

CFG = {"validation": {"min_rows": 0, "max_missing_frac_warn": 0.30}}

COLUMNS = [
    "invoice_no",
    "stock_code",
    "description",
    "quantity",
    "invoice_date",
    "unit_price",
    "customer_id",
    "country",
    "is_cancellation",
    "line_amount",
]


def make_frame(**overrides) -> pd.DataFrame:
    data = {
        "invoice_no": ["1001", "1002", "1003"],
        "stock_code": ["A1", "B2", "A1"],
        "description": ["widget", "gadget", "widget"],
        "quantity": [5, 3, -2],
        "invoice_date": pd.to_datetime(["2011-01-01", "2011-01-02", "2011-01-03"]),
        "unit_price": [2.0, 3.0, 2.0],
        "customer_id": [1.0, 2.0, 1.0],
        "country": ["UK", "UK", "UK"],
        "is_cancellation": [False, False, False],
        "line_amount": [10.0, 9.0, -4.0],
    }
    data.update(overrides)
    return pd.DataFrame(data)


def check(report, name):
    return next(c for c in report.checks if c.name == name)


def test_missing_required_column_is_error():
    df = make_frame().drop(columns=["unit_price"])
    report = validate(df, CFG)
    assert report.failed_errors
    assert check(report, "required_columns").passed is False


def test_clean_frame_has_no_errors():
    report = validate(make_frame(), CFG)
    assert not report.failed_errors


def test_duplicate_lines_are_warn_only():
    df = pd.concat([make_frame(), make_frame().iloc[[0]]], ignore_index=True)
    report = validate(df, CFG)
    assert not report.failed_errors
    dup = check(report, "exact_duplicate_lines")
    assert dup.passed is False and dup.severity == "warn"


def test_high_missingness_is_error():
    df = make_frame()
    df.loc[0, "description"] = None
    df.loc[1, "description"] = None
    report = validate(df, CFG)
    assert check(report, "missing_frac:description").severity == "error"
    assert report.failed_errors


def test_negative_quantity_reported():
    report = validate(make_frame(), CFG)
    returns = check(report, "negative_quantity_returns")
    assert "1 return" in returns.detail


def test_min_rows_failure():
    cfg = {"validation": {"min_rows": 10_000, "max_missing_frac_warn": 0.30}}
    report = validate(make_frame(), cfg)
    assert check(report, "min_rows").passed is False
