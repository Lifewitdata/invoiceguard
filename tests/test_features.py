"""Tests for point-in-time feature engineering.

The critical properties: features may only use information strictly before the
current line's day, and no target-derived column may appear in FEATURE_COLUMNS.
"""

import pandas as pd

from src.features import (
    FEATURE_COLUMNS,
    LEAKY_COLUMNS,
    add_customer_features,
    add_line_features,
    add_product_features,
    assign_time_split,
)


def make_lines() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "invoice_no": ["1001", "1002", "1003", "1004"],
            "stock_code": ["A1", "A1", "B2", "A1"],
            "description": ["widget", "widget", "gadget", "widget"],
            "quantity": [5, 3, -2, 4],
            "invoice_date": pd.to_datetime(
                [
                    "2011-01-01 10:00",
                    "2011-01-02 11:00",
                    "2011-01-02 12:00",
                    "2011-02-01 09:00",
                ]
            ),
            "unit_price": [2.0, 2.0, 2.0, 3.0],
            "customer_id": [1.0, 1.0, 1.0, 2.0],
            "country": ["UK", "UK", "UK", "UK"],
            "is_cancellation": [False, False, False, False],
            "line_amount": [10.0, 6.0, -4.0, 12.0],
        }
    ).astype({"customer_id": "Float64"})


def build() -> pd.DataFrame:
    df = add_line_features(make_lines())
    df = add_customer_features(df)
    df = add_product_features(df)
    return df


def test_no_leaky_columns_in_features():
    assert not (set(FEATURE_COLUMNS) & LEAKY_COLUMNS)


def test_returns_flag_matches_quantity_sign():
    df = build()
    assert df["is_return"].tolist() == [0, 0, 1, 0]


def test_no_same_day_leakage():
    df = build()
    jan2 = df[df["date"] == pd.Timestamp("2011-01-02")]
    assert (
        jan2["cust_prior_lines"] == 1
    ).all(), "same-day lines must not see each other in prior features"
    assert (jan2["cust_prior_return_rate"] == 0.0).all()
    assert (jan2["cust_prior_invoices"] == 1).all()


def test_no_future_leakage():
    df = build()
    first = df.iloc[0]
    assert first["cust_prior_lines"] == 0
    assert first["cust_is_new"] == 1
    assert first["prod_prior_lines"] == 0


def test_product_history_accumulates():
    df = build()
    last = df.iloc[3]
    assert last["prod_prior_lines"] == 2
    assert last["prod_prior_return_rate"] == 0.0


def test_unknown_customer_history_is_neutral():
    df = make_lines()
    df.loc[0, "customer_id"] = None
    df = add_line_features(df)
    df = add_customer_features(df)
    row = df.iloc[0]
    assert row["has_customer"] == 0
    assert row["cust_prior_lines"] == 0
    assert pd.isna(row["cust_prior_return_rate"])


def test_time_split_is_chronological():
    df = build()
    df, split_date = assign_time_split(df, test_frac=0.5)
    train_max = df.loc[df["split"] == "train", "date"].max()
    test_min = df.loc[df["split"] == "test", "date"].min()
    assert train_max < split_date <= test_min
