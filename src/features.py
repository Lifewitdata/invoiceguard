"""Point-in-time feature engineering for invoice risk scoring.

Every historical feature is computed from daily aggregates strictly before the
current line's date ("prior" = per-key cumulative sums shifted by one active
day), so no information from the current line, its invoice, or its day can leak
into features. Line-level features use only information available at posting
time. The target (is_return) and any target-derived columns are excluded from
FEATURE_COLUMNS -- enforced by tests.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.data import ROOT, load_config

EPS = 1e-6

TARGET = "is_return"

FEATURE_COLUMNS = [
    # line-level
    "abs_quantity_log",
    "unit_price_log",
    "abs_amount_log",
    "is_zero_price",
    "is_negative_price",
    "desc_len",
    "has_description",
    "stock_code_len",
    "stock_code_numeric",
    "hour",
    "day_of_week",
    "month",
    "day_of_month",
    "country_freq",
    "stock_code_freq",
    # customer point-in-time history
    "has_customer",
    "cust_is_new",
    "cust_prior_lines",
    "cust_prior_invoices",
    "cust_prior_return_rate",
    "cust_prior_mean_abs_amount",
    "cust_prior_mean_abs_qty",
    "cust_recency_days",
    "cust_price_z",
    "cust_amount_ratio",
    "cust_qty_ratio",
    # product point-in-time history
    "prod_prior_lines",
    "prod_prior_return_rate",
    "prod_prior_zero_price_share",
    "prod_price_z",
    "prod_price_ratio",
    "prod_qty_ratio",
]

LEAKY_COLUMNS = {"quantity", "line_amount", "is_cancellation", TARGET}

_DAILY_AGGS = {
    "lines": ("is_return", "size"),
    "returns": ("is_return", "sum"),
    "invoices": ("invoice_no", "nunique"),
    "amount_sum": ("abs_amount", "sum"),
    "price_sum": ("abs_price", "sum"),
    "price_sq_sum": ("abs_price_sq", "sum"),
    "absqty_sum": ("abs_quantity", "sum"),
    "zero_price": ("is_zero_price", "sum"),
}


def load_interim(cfg: dict) -> pd.DataFrame:
    path = ROOT / cfg["data"]["interim_dir"] / cfg["data"]["interim_file"]
    return pd.read_parquet(path)


def add_line_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["date"] = df["invoice_date"].dt.normalize()
    df["is_return"] = (df["quantity"] < 0).astype("int8")

    df["abs_quantity"] = df["quantity"].abs()
    df["abs_price"] = df["unit_price"].abs()
    df["abs_price_sq"] = df["abs_price"] ** 2
    df["abs_amount"] = df["line_amount"].abs()

    df["abs_quantity_log"] = np.log1p(df["abs_quantity"])
    df["unit_price_log"] = np.log1p(df["abs_price"])
    df["abs_amount_log"] = np.log1p(df["abs_amount"])

    df["is_zero_price"] = (df["unit_price"] <= 0).astype("int8")
    df["is_negative_price"] = (df["unit_price"] < 0).astype("int8")
    df["desc_len"] = df["description"].str.len().astype("int32")
    df["has_description"] = (df["description"] != "UNKNOWN").astype("int8")
    df["stock_code_len"] = df["stock_code"].str.len().astype("int32")
    df["stock_code_numeric"] = (
        df["stock_code"].str.fullmatch(r"\d+").fillna(False).astype("int8")
    )

    df["hour"] = df["invoice_date"].dt.hour.astype("int8")
    df["day_of_week"] = df["invoice_date"].dt.dayofweek.astype("int8")
    df["month"] = df["invoice_date"].dt.month.astype("int8")
    df["day_of_month"] = df["invoice_date"].dt.day.astype("int8")

    df["country_freq"] = df["country"].map(df["country"].value_counts(normalize=True))
    df["stock_code_freq"] = df["stock_code"].map(
        df["stock_code"].value_counts(normalize=True)
    )
    return df


def _prior_features(lines: pd.DataFrame, key: str, prefix: str) -> pd.DataFrame:
    """Daily aggregates per key, shifted so each row holds only prior-day sums."""
    daily = (
        lines.dropna(subset=[key])
        .groupby([key, "date"], as_index=False)
        .agg(**_DAILY_AGGS)
        .sort_values([key, "date"])
        .reset_index(drop=True)
    )
    value_cols = list(_DAILY_AGGS)
    cum = daily.groupby(key)[value_cols].cumsum()
    prior = cum.groupby(daily[key]).shift(1)
    prior.columns = [f"{prefix}{c}" for c in prior.columns]

    out = pd.concat([daily[[key, "date"]], prior], axis=1)
    out[f"{prefix}last_date"] = daily.groupby(key)["date"].shift(1)
    return out


def _drop_temp(df: pd.DataFrame, prefix: str) -> pd.DataFrame:
    temp = [f"{prefix}{c}" for c in list(_DAILY_AGGS) + ["last_date"]]
    return df.drop(columns=[c for c in temp if c in df.columns])


def add_customer_features(df: pd.DataFrame) -> pd.DataFrame:
    prior = _prior_features(df, "customer_id", "cust_")
    df = df.merge(prior, on=["customer_id", "date"], how="left")

    known = df["customer_id"].notna()
    prior_lines = df["cust_lines"].fillna(0)
    price_n = df["cust_lines"].replace(0, np.nan)

    mean_price = df["cust_price_sum"] / price_n
    var_price = (df["cust_price_sq_sum"] / price_n) - mean_price**2
    std_price = np.sqrt(var_price.clip(lower=0))

    df["has_customer"] = known.astype("int8")
    df["cust_is_new"] = (known & (prior_lines == 0)).astype("int8")
    df["cust_prior_lines"] = prior_lines
    df["cust_prior_invoices"] = df["cust_invoices"].fillna(0)
    df["cust_prior_return_rate"] = df["cust_returns"] / price_n
    df["cust_prior_mean_abs_amount"] = df["cust_amount_sum"] / price_n
    df["cust_prior_mean_abs_qty"] = df["cust_absqty_sum"] / price_n
    df["cust_recency_days"] = (df["date"] - df["cust_last_date"]).dt.days
    df["cust_price_z"] = (df["abs_price"] - mean_price) / (std_price + EPS)
    df["cust_amount_ratio"] = df["abs_amount"] / (
        df["cust_prior_mean_abs_amount"] + EPS
    )
    df["cust_qty_ratio"] = df["abs_quantity"] / (df["cust_prior_mean_abs_qty"] + EPS)

    return _drop_temp(df, "cust_")


def add_product_features(df: pd.DataFrame) -> pd.DataFrame:
    prior = _prior_features(df, "stock_code", "prod_")
    df = df.merge(prior, on=["stock_code", "date"], how="left")

    n = df["prod_lines"].replace(0, np.nan)
    mean_price = df["prod_price_sum"] / n
    var_price = (df["prod_price_sq_sum"] / n) - mean_price**2
    std_price = np.sqrt(var_price.clip(lower=0))

    df["prod_prior_lines"] = df["prod_lines"].fillna(0)
    df["prod_prior_return_rate"] = df["prod_returns"] / n
    df["prod_prior_zero_price_share"] = df["prod_zero_price"] / n
    df["prod_price_z"] = (df["abs_price"] - mean_price) / (std_price + EPS)
    df["prod_price_ratio"] = df["abs_price"] / (mean_price + EPS)
    df["prod_qty_ratio"] = df["abs_quantity"] / (df["prod_absqty_sum"] / n + EPS)

    return _drop_temp(df, "prod_")


def assign_time_split(df: pd.DataFrame, test_frac: float) -> tuple[pd.DataFrame, pd.Timestamp]:
    dates = np.sort(df["date"].unique())
    split_date = pd.Timestamp(dates[int(len(dates) * (1 - test_frac))])
    df["split"] = np.where(df["date"] < split_date, "train", "test")
    return df, split_date


def build_features(cfg: dict) -> Path:
    df = load_interim(cfg)

    n_zero_qty = int((df["quantity"] == 0).sum())
    df = df[df["quantity"] != 0].reset_index(drop=True)
    print(f"Dropped {n_zero_qty:,} lines with quantity == 0")

    df = add_line_features(df)
    df = add_customer_features(df)
    df = add_product_features(df)
    df, split_date = assign_time_split(df, cfg["split"]["test_frac"])

    leaky = LEAKY_COLUMNS & set(FEATURE_COLUMNS)
    assert not leaky, f"leaky columns in FEATURE_COLUMNS: {leaky}"

    out = ROOT / cfg["data"]["processed_dir"] / "features.parquet"
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out, index=False)

    freq_maps = {
        "country": df["country"].value_counts(normalize=True).round(8).to_dict(),
        "stock_code": df["stock_code"].value_counts(normalize=True).round(8).to_dict(),
    }
    freq_path = ROOT / "models" / "freq_maps.json"
    freq_path.parent.mkdir(parents=True, exist_ok=True)
    freq_path.write_text(json.dumps(freq_maps), encoding="utf-8")

    schema = {
        "target": TARGET,
        "feature_columns": FEATURE_COLUMNS,
        "split_strategy": cfg["split"]["strategy"],
        "split_date": str(split_date.date()),
        "n_rows": int(len(df)),
        "n_train": int((df["split"] == "train").sum()),
        "n_test": int((df["split"] == "test").sum()),
        "train_return_rate": float(df.loc[df["split"] == "train", TARGET].mean()),
        "test_return_rate": float(df.loc[df["split"] == "test", TARGET].mean()),
    }
    schema_path = ROOT / "reports" / "feature_schema.json"
    schema_path.write_text(json.dumps(schema, indent=2), encoding="utf-8")

    print(f"Features written to {out}")
    print(
        f"  train: {schema['n_train']:,} rows (return rate {schema['train_return_rate']:.2%})"
        f" | test: {schema['n_test']:,} rows (return rate {schema['test_return_rate']:.2%})"
    )
    print(f"  split date: {split_date.date()} | {len(FEATURE_COLUMNS)} features")
    print(f"  schema written to {schema_path}")
    return out


def main() -> None:
    build_features(load_config())


if __name__ == "__main__":
    main()
