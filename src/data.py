"""Download and clean UCI Online Retail II into an interim parquet dataset.

Raw source is intentionally kept messy (returns, cancellations, missing IDs);
cleaning only normalizes types and derives flags -- rows are never silently
dropped without being counted in the returned stats.
"""

from __future__ import annotations

import sys
import zipfile
from pathlib import Path

import pandas as pd
import requests
import yaml

ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "config.yaml"

RAW_COLUMNS = {
    "Invoice": "invoice_no",
    "StockCode": "stock_code",
    "Description": "description",
    "Quantity": "quantity",
    "InvoiceDate": "invoice_date",
    "Price": "unit_price",
    "Customer ID": "customer_id",
    "Country": "country",
}


def load_config(path: Path = CONFIG_PATH) -> dict:
    with open(path, "r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def download_raw(cfg: dict, force: bool = False) -> Path:
    raw_dir = ROOT / cfg["data"]["raw_dir"]
    raw_dir.mkdir(parents=True, exist_ok=True)
    xlsx_path = raw_dir / cfg["data"]["raw_file"]
    if xlsx_path.exists() and not force:
        print(f"Raw file already present: {xlsx_path}")
        return xlsx_path

    url = cfg["data"]["source_url"]
    print(f"Downloading {url}")
    resp = requests.get(url, timeout=300)
    resp.raise_for_status()

    zip_path = raw_dir / cfg["data"]["raw_zip"]
    zip_path.write_bytes(resp.content)

    with zipfile.ZipFile(zip_path) as zf:
        member = next(n for n in zf.namelist() if n.lower().endswith(".xlsx"))
        with zf.open(member) as src, open(xlsx_path, "wb") as dst:
            dst.write(src.read())

    print(f"Saved {xlsx_path} ({xlsx_path.stat().st_size / 1e6:.1f} MB)")
    return xlsx_path


def load_raw(cfg: dict) -> pd.DataFrame:
    xlsx_path = ROOT / cfg["data"]["raw_dir"] / cfg["data"]["raw_file"]
    frames = [
        pd.read_excel(xlsx_path, sheet_name=sheet, dtype=str)
        for sheet in cfg["data"]["sheets"]
    ]
    return pd.concat(frames, ignore_index=True)


def clean(raw: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    stats: dict[str, int] = {"raw_rows": len(raw)}
    df = raw.rename(columns=RAW_COLUMNS)

    missing_ids = df["invoice_no"].isna() | df["stock_code"].isna()
    stats["rows_dropped_missing_ids"] = int(missing_ids.sum())
    df = df[~missing_ids].copy()

    df["invoice_date"] = pd.to_datetime(df["invoice_date"], errors="coerce")
    unparseable = df["invoice_date"].isna()

    for col in ("quantity", "unit_price", "customer_id"):
        coerced = pd.to_numeric(df[col], errors="coerce")
        unparseable |= coerced.isna() & df[col].notna()
        df[col] = coerced

    stats["rows_dropped_unparseable"] = int(unparseable.sum())
    df = df[~unparseable]

    df["quantity"] = df["quantity"].astype("int64")
    df["unit_price"] = df["unit_price"].astype("float64")
    df["customer_id"] = df["customer_id"].astype("Float64")

    df["invoice_no"] = df["invoice_no"].str.strip().str.upper()
    df["stock_code"] = df["stock_code"].str.strip().str.upper()
    df["country"] = df["country"].fillna("UNKNOWN").str.strip()
    df["description"] = df["description"].fillna("UNKNOWN").str.strip()

    df["is_cancellation"] = df["invoice_no"].str.startswith("C")
    df["line_amount"] = df["quantity"] * df["unit_price"]

    df = df.reset_index(drop=True)
    stats["clean_rows"] = len(df)
    return df, stats


def build_interim(cfg: dict, force_download: bool = False) -> Path:
    download_raw(cfg, force=force_download)
    df, stats = clean(load_raw(cfg))

    out = ROOT / cfg["data"]["interim_dir"] / cfg["data"]["interim_file"]
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out, index=False)

    print(f"Interim dataset: {out}")
    for key, value in stats.items():
        print(f"  {key}: {value:,}")
    return out


def main() -> None:
    cfg = load_config()
    build_interim(cfg, force_download="--force-download" in sys.argv)


if __name__ == "__main__":
    main()
