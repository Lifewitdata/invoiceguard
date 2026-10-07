"""Feature drift monitoring with Population Stability Index (PSI).

Reference distribution = training window features (data/processed/features.parquet,
time split). Current distribution = a scored batch parquet (--current) or, by
default, the held-out test window -- useful as a smoke demo but note that a
real deployment would feed live scoring traffic here.

PSI bands: < 0.10 stable, 0.10-0.25 moderate shift, > 0.25 significant drift.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from src.data import ROOT, load_config
from src.features import FEATURE_COLUMNS

PSI_WARN = 0.10
PSI_ALERT = 0.25
BINS = 10
EPS = 1e-6


def psi(reference: np.ndarray, current: np.ndarray, bins: int = BINS) -> float:
    reference = reference[np.isfinite(reference)]
    current = current[np.isfinite(current)]
    if len(reference) == 0 or len(current) == 0:
        return float("nan")
    edges = np.unique(np.quantile(reference, np.linspace(0, 1, bins + 1)))
    if len(edges) < 2:
        return 0.0
    edges[0], edges[-1] = -np.inf, np.inf
    ref_counts = np.histogram(reference, bins=edges)[0] / len(reference)
    cur_counts = np.histogram(current, bins=edges)[0] / len(current)
    ref_counts = np.clip(ref_counts, EPS, None)
    cur_counts = np.clip(cur_counts, EPS, None)
    return float(np.sum((cur_counts - ref_counts) * np.log(cur_counts / ref_counts)))


def drift_report(
    reference: pd.DataFrame, current: pd.DataFrame, feature_columns: list[str]
) -> dict:
    per_feature = {}
    for column in feature_columns:
        if column not in reference.columns or column not in current.columns:
            per_feature[column] = {"psi": None, "status": "missing_column"}
            continue
        value = psi(reference[column].to_numpy(dtype=float), current[column].to_numpy(dtype=float))
        status = "stable"
        if np.isnan(value):
            status = "no_data"
        elif value > PSI_ALERT:
            status = "drift"
        elif value > PSI_WARN:
            status = "watch"
        per_feature[column] = {"psi": None if np.isnan(value) else round(value, 4), "status": status}

    drifted = [c for c, info in per_feature.items() if info["status"] == "drift"]
    watch = [c for c, info in per_feature.items() if info["status"] == "watch"]
    return {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "n_reference": int(len(reference)),
        "n_current": int(len(current)),
        "thresholds": {"watch": PSI_WARN, "drift": PSI_ALERT},
        "n_drifted": len(drifted),
        "n_watch": len(watch),
        "drifted_features": drifted,
        "watch_features": watch,
        "per_feature": per_feature,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="PSI feature drift report")
    parser.add_argument("--current", type=str, default=None,
                        help="Parquet of scored lines; defaults to the test window")
    parser.add_argument("--out", type=str, default="reports/drift_report.json")
    args = parser.parse_args()

    cfg = load_config()
    features_path = ROOT / cfg["data"]["processed_dir"] / "features.parquet"
    df = pd.read_parquet(features_path)

    reference = df[df["split"] == "train"]
    if args.current:
        current = pd.read_parquet(args.current)
    else:
        current = df[df["split"] == "test"]

    report = drift_report(reference, current, FEATURE_COLUMNS)
    out_path = ROOT / args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2))

    print(f"Reference: {report['n_reference']:,} lines | Current: {report['n_current']:,} lines")
    print(f"Drifted features ({report['n_drifted']}): {report['drifted_features']}")
    print(f"Watch features ({report['n_watch']}): {report['watch_features']}")
    top = sorted(
        ((c, i["psi"]) for c, i in report["per_feature"].items() if i["psi"] is not None),
        key=lambda kv: kv[1],
        reverse=True,
    )[:10]
    print("Top PSI:")
    for name, value in top:
        print(f"  {name:<28} {value:.4f}")
    print(f"Drift report: {out_path}")


if __name__ == "__main__":
    main()
