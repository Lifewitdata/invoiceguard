# InvoiceGuard — End-to-End Invoice Risk Scoring & Anomaly Detection

Flag high-risk invoices (returns, cancellations, anomalous pricing) before they hit
downstream financial processes — built as a production-style, reproducible ML pipeline.

**Business framing:** late or rejected invoice claims cause cash-flow and compliance
exposure. This project scores invoices at line-item level so reviewers can prioritize
the small fraction that carries most of the risk.

**Dataset:** [UCI Online Retail II](https://archive.ics.uci.edu/dataset/502/online+retail+ii) —
real invoice-level transaction data (Dec 2009–Dec 2011) with genuine missingness, returns,
cancellations, and duplicates.

## Architecture

```
data.py → validate.py → features.py → train.py → evaluate.py → api.py
   │            │             │            │           │          │
 download    quality      feature     benchmark    cost-based   FastAPI
 + clean     report      engineering  + versioned  threshold    /score
                                     model        selection
```

## Build roadmap

- [x] 1. Repo scaffold, config, dependencies
- [x] 2. Data acquisition + quality validation
- [x] 3. EDA notebook with findings and hypotheses
- [x] 4. Feature engineering module
- [x] 5. Model training + benchmarking
- [x] 6. Evaluation with cost-based thresholding
- [x] 7. FastAPI inference service
- [x] 8. Pytest test suite
- [x] 9. Docker packaging + drift monitoring
- [x] 10. Final README with findings

## Data quality snapshot

Run `python -m src.validate` — from `reports/data_quality.json`:

| Check | Result |
|---|---|
| Rows | 1,067,371 lines / 53,628 invoices / 5,942 customers |
| Date range | 2009-12-01 → 2011-12-09 |
| Missing `customer_id` | 22.8% (warn threshold 30%) |
| Exact duplicate lines | 34,337 (3.2%, warned) |
| Non-positive price | 6,207 (0.6%, warned) |
| Return/cancellation lines | 22,950 (2.15% label prevalence) |

## Key EDA findings

From `notebooks/01_eda.ipynb` → `reports/eda_summary.json`:

- **H1 — Amount signal:** returned lines have a significantly *lower* median absolute
  amount than kept lines (£6.75 vs £9.96; Mann-Whitney p ≈ 0). Small, low-value lines
  are disproportionately returned — consistent with impulse/quick-return behavior.
- **H2 — Concentration:** the top 10% of customers by activity account for **31.3%** of
  returned lines — a small set of accounts drives a large share of risk.
- **H3 — Repeat vs one-time behavior:** return behavior differs significantly between
  repeat and one-time customers (χ² = 506, p = 4×10⁻¹¹²), justifying per-customer
  historical features.
- **Missingness is structural, not random:** 22.8% of lines have no customer ID
  (guest/wholesale gaps), so customer features must handle unknown customers explicitly.

## Model benchmark

5 candidates scored on a stratified validation slice of the training window
(`reports/benchmark.csv`); target `is_return`, base rate 2.27%:

| Model | AP (PR-AUC) | ROC-AUC | Precision@1% | Lift@1% |
|---|---|---|---|---|
| RandomForest (300 trees) | **0.691** | 0.968 | 0.875 | **38.5×** |
| HistGradientBoosting | 0.648 | 0.964 | 0.821 | 36.2× |
| LogisticRegression | 0.277 | 0.880 | 0.461 | 20.3× |
| IsolationForest (unsupervised) | 0.085 | 0.672 | 0.221 | 9.7× |
| Dummy (prior) | 0.023 | 0.500 | 0.018 | 0.8× |

Accuracy is meaningless at 2.3% prevalence — **PR-AUC is the primary metric**, with
precision@k matching the reviewer workflow (a fixed-size review queue).

## Held-out test results

Time-based holdout (test window starts 2011-07-21, 268,936 lines, 1.80% return rate) —
the honest generalization test, not a random split:

| Metric | Value |
|---|---|
| PR-AUC (average precision) | 0.485 |
| ROC-AUC | 0.908 |
| precision@1% of lines | **67.9%** (37.8× over base rate) |
| recall@1% of lines | 37.8% |
| CV AP (train, 3-fold) | 0.687 ± 0.004 |

**Cost-based operating point** (`src/evaluate.py`): assuming a missed return costs 10×
a manual review, the expected-cost-minimizing threshold is **0.123** — flagging the
**3.2%** riskiest lines catches **59.6%** of all returns at 33.6% precision,
reducing expected cost by **~1.9×** vs. no review. This threshold is persisted in
`models/registry.json` and used by the API.

**Top predictors** (permutation importance): `cust_recency_days`, `prod_prior_return_rate`,
`cust_prior_mean_abs_amount`, `cust_prior_return_rate` — customer and product return
history dominate, confirming the EDA hypotheses. Full report: `reports/evaluation.md`.

**Why test AP < CV AP:** the time split puts later, unseen (and structurally different)
months in test — return rate drops from 2.27% to 1.80% — a known and expected
generalization gap for time-based evaluation, and the reason monitoring exists.

## Monitoring

`python -m src.monitoring` computes PSI between the training window (reference) and the
test window (or any scored batch via `--current`) → `reports/drift_report.json`:

- **Drifted (>0.25):** `month` (5.51), `prod_prior_lines` (0.41) — both structural to a
  time split (calendar months differ by construction; product history accumulates), not
  data corruption. Real production monitoring would compare adjacent live windows.
- **Watch (0.10–0.25):** customer/product history counts — scale shifts as accounts age.
- All distribution-shape features (prices, quantities, frequencies) are stable.

## Project structure

```
config.yaml            # all knobs: paths, split, models, costs, API
src/
  data.py              # download (UCI zip) + clean -> parquet
  validate.py          # data quality checks -> reports/data_quality.json
  features.py          # 32 point-in-time features (leakage-safe by construction)
  train.py             # benchmark 5 candidates, CV, versioned artifact + registry
  evaluate.py          # cost-curve threshold, figures, evaluation.md
  api.py               # FastAPI /health /score /score/batch
  monitoring.py        # PSI drift detection
notebooks/01_eda.ipynb # EDA with hypothesis tests (H1-H3)
tests/                 # 21 tests: validation, feature leakage, API contract
reports/               # generated: quality, EDA, benchmark, evaluation, drift, figures
models/                # generated: model_v{N}.joblib + registry.json (gitignored)
```

## Evidence map

How each resume claim is backed by an artifact in this repo (interview-defensible):

| Claim | Where to point |
|---|---|
| Deep EDA on noisy, real-world data | `notebooks/01_eda.ipynb` — 22.8% missing IDs, 3.2% duplicates; missingness, outliers, temporal plots |
| Data quality validation, framed hypotheses | `src/validate.py` + `reports/data_quality.json`; H1 amount signal, H2 concentration, H3 repeat-vs-one-time (Mann-Whitney, χ²) |
| Feature engineering + benchmarked alternatives | `src/features.py` — 32 leakage-tested features; `reports/benchmark.csv` — 5 candidates incl. unsupervised baseline |
| Modular, testable, reproducible pipelines | `src/` modules + `config.yaml` + seed 42; `tests/` — 21 tests incl. same-day/future leakage guards (all green) |
| Productionization: packaging, versioning, monitoring, validation | `Dockerfile`; `models/registry.json` versioning + persisted threshold; `src/monitoring.py` PSI; `src/validate.py` schema/quality gates |
| Translating business questions into models | cost-sensitive threshold (FN 10× FP) with a reviewer-queue operating point; expected-cost comparison vs. no review / review-all |
| Communicating results with visualizations and narratives | `reports/evaluation.md` (auto-generated), `reports/figures/` — PR, ROC, cost curve, score distribution, importance |

## Quickstart

```bash
pip install -r requirements.txt

python -m src.data        # download + clean -> data/interim/invoices.parquet
python -m src.validate    # data quality report -> reports/data_quality.json
python -m src.features    # point-in-time features -> data/processed/features.parquet
python -m src.train       # benchmark 5 models, version the winner -> models/
python -m src.evaluate    # cost-based threshold + figures -> reports/
python -m src.monitoring  # PSI drift report -> reports/drift_report.json

pytest -q                 # test suite (validation, leakage, API)
```

Reproducible from a clean clone: `python -m src.data --force-download` re-fetches the
UCI archive; all downstream steps read only from `data/` and `models/` artifacts.

### Serve the model

```bash
uvicorn src.api:app --host 127.0.0.1 --port 8000
# or: docker build -t invoiceguard . && docker run -p 8000:8000 invoiceguard

curl -X POST http://127.0.0.1:8000/score -H "Content-Type: application/json" -d '{
  "stock_code": "85123A", "unit_price": 2.55, "abs_quantity": 12,
  "invoice_date": "2011-11-15T10:30:00", "country": "United Kingdom",
  "customer_id": 17850,
  "cust_prior_lines": 420, "cust_prior_invoices": 95, "cust_prior_return_rate": 0.04,
  "cust_prior_mean_abs_amount": 8.9, "cust_prior_mean_abs_qty": 14.2,
  "cust_recency_days": 3, "prod_prior_lines": 1800, "prod_prior_return_rate": 0.03
}'
# -> {"risk_score": 0.200, "flagged": true, "threshold": 0.123,
#     "model_version": 1, "model_name": "random_forest"}

# batch: POST /score/batch with a JSON list (max 1000 lines per request)
```

The service returns `risk_score`, `flagged` (score ≥ the cost-optimal threshold from
`src.evaluate`), plus model version metadata on every response.

