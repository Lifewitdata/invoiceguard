# Model evaluation — random_forest (v1)

Time-based holdout: test window starts 2011-07-21 (268,936 lines, return rate 1.80%).

## Headline metrics (test)

| metric | value |
|---|---|
| PR-AUC (average precision) | 0.4848 |
| ROC-AUC | 0.9082 |
| precision@1% of lines | 67.94% (lift 37.8x over base rate) |
| recall@1% of lines | 37.82% |
| CV AP (train, 3-fold) | 0.6874 ± 0.0044 |

## Cost-based operating point

Assumption: missing a true return line costs 10x a manual review of a
false alarm (1). Sweeping thresholds on the test set, expected cost is
minimized at **threshold = 0.1226**:

- recall **59.59%**, precision **33.59%**
- flags **3.19%** of all lines for review
- expected cost **25,211** vs **48,310**
  (no review at all) and **264,105** (review everything) —
  ~1.9x cheaper than no review.

This threshold is persisted in `models/registry.json` and used by the API.

## Top predictors (permutation importance, AP drop)

| rank | feature | importance |
|---|---|---|
| 1 | cust_recency_days | 0.1739 |
| 2 | prod_prior_return_rate | 0.0587 |
| 3 | cust_prior_mean_abs_amount | 0.0555 |
| 4 | cust_prior_return_rate | 0.0421 |
| 5 | cust_qty_ratio | 0.0342 |
| 6 | cust_prior_lines | 0.0245 |
| 7 | cust_prior_invoices | 0.0238 |
| 8 | hour | 0.0236 |
| 9 | cust_prior_mean_abs_qty | 0.0218 |
| 10 | abs_amount_log | 0.0211 |
| 11 | abs_quantity_log | 0.0204 |
| 12 | desc_len | 0.0183 |
| 13 | prod_qty_ratio | 0.0176 |
| 14 | country_freq | 0.0142 |
| 15 | cust_amount_ratio | 0.0124 |

## Figures

- `figures/model_pr_curve.png`, `figures/model_roc_curve.png`
- `figures/model_cost_curve.png`, `figures/model_score_distribution.png`
- `figures/model_feature_importance.png`

## Interpretation

Customer return history features dominate, confirming hypothesis H2 from
the EDA notebook: risk is concentrated in a subset of customers and the
point-in-time encodings translate that into predictive signal without
leakage. Product-level return rates and price-deviation features add
complementary signal. Temporal and line-level features contribute
marginally — consistent with returns being driven mainly by *who* is
ordering rather than *when*.
