"""Data quality validation for the invoice dataset, with a machine-readable report.

Hard failures (severity="error") indicate the data cannot be trusted for
downstream modeling; warnings (severity="warn") are documented, expected
characteristics of this messy real-world dataset.
"""

from __future__ import annotations

import json
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path

import pandas as pd

from src.data import ROOT, load_config

REQUIRED_COLUMNS = [
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


@dataclass
class CheckResult:
    name: str
    passed: bool
    detail: str
    severity: str = "error"


@dataclass
class QualityReport:
    n_rows: int
    checks: list[CheckResult] = field(default_factory=list)

    @property
    def failed_errors(self) -> list[CheckResult]:
        return [c for c in self.checks if not c.passed and c.severity == "error"]

    @property
    def warnings(self) -> list[CheckResult]:
        return [c for c in self.checks if not c.passed and c.severity == "warn"]

    def to_dict(self) -> dict:
        return {
            "n_rows": self.n_rows,
            "n_checks": len(self.checks),
            "n_failed_errors": len(self.failed_errors),
            "n_warnings": len(self.warnings),
            "checks": [asdict(c) for c in self.checks],
        }


def validate(df: pd.DataFrame, cfg: dict) -> QualityReport:
    v = cfg["validation"]
    report = QualityReport(n_rows=len(df))

    missing_cols = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    report.checks.append(
        CheckResult(
            "required_columns",
            passed=not missing_cols,
            detail=f"missing: {missing_cols}" if missing_cols else "all present",
        )
    )
    if missing_cols:
        return report

    report.checks.append(
        CheckResult(
            "min_rows",
            passed=len(df) >= v["min_rows"],
            detail=f"{len(df):,} rows vs required {v['min_rows']:,}",
        )
    )

    null_frac = df[REQUIRED_COLUMNS].isna().mean()
    for col, frac in null_frac.items():
        if frac > 0:
            severity = "warn" if frac <= v["max_missing_frac_warn"] else "error"
            report.checks.append(
                CheckResult(
                    f"missing_frac:{col}",
                    passed=severity != "error",
                    detail=f"{frac:.2%} missing",
                    severity=severity,
                )
            )

    key_cols = ["invoice_no", "stock_code", "quantity", "unit_price", "invoice_date"]
    n_dupes = int(df.duplicated(subset=key_cols).sum())
    report.checks.append(
        CheckResult(
            "exact_duplicate_lines",
            passed=n_dupes == 0,
            detail=f"{n_dupes:,} exact duplicate lines ({n_dupes / len(df):.2%})",
            severity="warn",
        )
    )

    n_nonpositive_price = int((df["unit_price"] <= 0).sum())
    report.checks.append(
        CheckResult(
            "nonpositive_price",
            passed=n_nonpositive_price == 0,
            detail=f"{n_nonpositive_price:,} lines with price <= 0 ({n_nonpositive_price / len(df):.2%})",
            severity="warn",
        )
    )

    n_negative_qty = int((df["quantity"] < 0).sum())
    report.checks.append(
        CheckResult(
            "negative_quantity_returns",
            passed=True,
            detail=f"{n_negative_qty:,} return/cancellation lines ({n_negative_qty / len(df):.2%})",
            severity="warn",
        )
    )

    report.checks.append(
        CheckResult(
            "date_range",
            passed=True,
            detail=f"{df['invoice_date'].min():%Y-%m-%d} to {df['invoice_date'].max():%Y-%m-%d}",
            severity="warn",
        )
    )

    return report


def print_report(report: QualityReport) -> None:
    status = lambda c: "PASS" if c.passed else c.severity.upper()  # noqa: E731
    for c in report.checks:
        print(f"[{status(c):>5}] {c.name}: {c.detail}")


def main() -> None:
    cfg = load_config()
    interim = ROOT / cfg["data"]["interim_dir"] / cfg["data"]["interim_file"]
    df = pd.read_parquet(interim)
    report = validate(df, cfg)

    print(f"Validated {interim} ({report.n_rows:,} rows)")
    print_report(report)

    out = ROOT / "reports" / "data_quality.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report.to_dict(), indent=2), encoding="utf-8")
    print(f"Report written to {out}")

    if report.failed_errors:
        sys.exit(1)


if __name__ == "__main__":
    main()
