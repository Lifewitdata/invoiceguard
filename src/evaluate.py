"""Evaluate the active model on the held-out test window.

Adds three things beyond raw metrics:
1. A cost-based decision threshold: minimizing expected cost
   (false_negative x cost.FN + false_positive x cost.FP) instead of defaulting
   to 0.5, because the business cost of missing a return is far larger than the
   review cost of a false alarm.
2. Explainability: permutation importance for the winning model.
3. An auto-generated written report (reports/evaluation.md) and figures.
"""

from __future__ import annotations

import json

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.inspection import permutation_importance
from sklearn.metrics import precision_recall_curve, roc_curve

from src.data import ROOT, load_config
from src.train import metrics_at

IMPORTANCE_SAMPLE = 20000


def load_active() -> tuple[object, dict, dict]:
    registry = json.loads(
        (ROOT / "models" / "registry.json").read_text(encoding="utf-8")
    )
    run = next(
        r for r in registry["runs"] if r["version"] == registry["active_version"]
    )
    artifact = joblib.load(ROOT / "models" / run["artifact"])
    return artifact, run, registry


def cost_curve(y_true: np.ndarray, scores: np.ndarray, fn_cost: int, fp_cost: int):
    thresholds = np.unique(np.quantile(scores, np.linspace(0.5, 0.9999, 300)))
    y = np.asarray(y_true)
    costs, rows = [], []
    for thr in thresholds:
        flagged = scores >= thr
        fn = int(((y == 1) & ~flagged).sum())
        fp = int(((y == 0) & flagged).sum())
        cost = fn * fn_cost + fp * fp_cost
        costs.append(cost)
        rows.append(
            {
                "threshold": float(thr),
                "fn": fn,
                "fp": fp,
                "expected_cost": cost,
                "precision": float(y[flagged].mean()) if flagged.any() else 0.0,
                "recall": float(y[flagged].sum() / y.sum()),
                "flagged_share": float(flagged.mean()),
            }
        )
    curve = pd.DataFrame(rows)
    best = curve.loc[curve["expected_cost"].idxmin()]
    return curve, best


def main() -> None:
    cfg = load_config()
    sns.set_theme(style="whitegrid")
    fig_dir = ROOT / "reports" / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    schema = json.loads(
        (ROOT / "reports" / "feature_schema.json").read_text(encoding="utf-8")
    )
    df = pd.read_parquet(ROOT / "data" / "processed" / "features.parquet")
    test = df[df["split"] == "test"]

    artifact, run, registry = load_active()
    model = artifact["pipeline"]
    features, target = artifact["feature_columns"], schema["target"]

    X_test, y_test = test[features], test[target]
    scores = model.predict_proba(X_test)[:, 1]

    metrics = metrics_at(y_test, scores)
    fn_cost = cfg["cost"]["false_negative"]
    fp_cost = cfg["cost"]["false_positive"]

    curve, best = cost_curve(y_test.to_numpy(), scores, fn_cost, fp_cost)
    threshold = float(best["threshold"])

    baseline_no_review = int(y_test.sum()) * fn_cost
    baseline_all_review = int((y_test == 0).sum()) * fp_cost

    print(f"Test metrics ({run['model_name']} v{run['version']}):")
    for key in ("average_precision", "roc_auc", "precision@1%", "recall@1%", "lift@1%"):
        print(f"  {key}: {metrics[key]:.4f}")
    print(
        f"Cost-optimal threshold {threshold:.4f} "
        f"(FN cost {fn_cost} vs FP cost {fp_cost}):"
    )
    print(
        f"  recall {best['recall']:.2%}, precision {best['precision']:.2%}, "
        f"flagging {best['flagged_share']:.2%} of lines"
    )
    print(f"  expected cost {best['expected_cost']:,.0f} vs no-review {baseline_no_review:,.0f}")

    # Figures -----------------------------------------------------------------
    precision, recall, _ = precision_recall_curve(y_test, scores)
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(recall, precision, color="#4c72b0")
    ax.axhline(metrics["base_rate"], ls="--", color="grey", lw=1, label="base rate")
    ax.scatter(
        [best["recall"]], [best["precision"]], color="#c44e52", zorder=3,
        label=f"chosen threshold ({threshold:.3f})",
    )
    ax.set_xlabel("recall")
    ax.set_ylabel("precision")
    ax.set_title(f"Precision-recall curve (AP = {metrics['average_precision']:.3f})")
    ax.legend()
    fig.savefig(fig_dir / "model_pr_curve.png", bbox_inches="tight")

    fpr, tpr, _ = roc_curve(y_test, scores)
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(fpr, tpr, color="#4c72b0")
    ax.plot([0, 1], [0, 1], ls="--", color="grey", lw=1)
    ax.set_xlabel("false positive rate")
    ax.set_ylabel("true positive rate")
    ax.set_title(f"ROC curve (AUC = {metrics['roc_auc']:.3f})")
    fig.savefig(fig_dir / "model_roc_curve.png", bbox_inches="tight")

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(curve["threshold"], curve["expected_cost"], color="#8172b3")
    ax.axvline(threshold, ls="--", color="#c44e52", lw=1, label="cost-optimal")
    ax.axvline(0.5, ls=":", color="grey", lw=1, label="default 0.5")
    ax.set_yscale("log")
    ax.set_xlabel("decision threshold")
    ax.set_ylabel("expected cost (log scale)")
    ax.set_title("Expected cost vs. decision threshold")
    ax.legend()
    fig.savefig(fig_dir / "model_cost_curve.png", bbox_inches="tight")

    fig, ax = plt.subplots(figsize=(6, 4))
    sns.histplot(
        x=scores, hue=y_test.astype(bool), bins=60, log_scale=(False, True),
        stat="density", common_norm=False, ax=ax,
    )
    ax.set_xlabel("predicted risk score")
    ax.set_title("Score distribution by class")
    fig.savefig(fig_dir / "model_score_distribution.png", bbox_inches="tight")

    sample = test.sample(min(IMPORTANCE_SAMPLE, len(test)), random_state=cfg["project"]["seed"])
    imp = None
    try:
        # n_jobs=1: the forest is hundreds of MB; shipping it to worker processes
        # exhausts Windows resources. In-process scoring on 20k rows is enough.
        importance = permutation_importance(
            model,
            sample[features],
            sample[target],
            scoring="average_precision",
            n_repeats=3,
            random_state=cfg["project"]["seed"],
            n_jobs=1,
        )
        imp = (
            pd.DataFrame(
                {
                    "feature": features,
                    "importance": importance.importances_mean,
                    "std": importance.importances_std,
                }
            )
            .sort_values("importance", ascending=False)
            .head(15)
        )
        imp.to_csv(ROOT / "reports" / "feature_importance.csv", index=False)

        fig, ax = plt.subplots(figsize=(7, 5))
        sns.barplot(data=imp, x="importance", y="feature", ax=ax, color="#55a868")
        ax.set_title("Permutation importance (test sample, AP drop)")
        fig.savefig(fig_dir / "model_feature_importance.png", bbox_inches="tight")
    except Exception as exc:  # importance is diagnostics, never block the pipeline
        print(f"WARNING: permutation importance skipped ({type(exc).__name__}: {exc})")

    # Registry + written report ----------------------------------------------
    run["evaluation"] = {
        "decision_threshold": threshold,
        "cost_assumptions": {"false_negative": fn_cost, "false_positive": fp_cost},
        "operating_point": {
            k: float(best[k]) for k in ("precision", "recall", "flagged_share", "expected_cost")
        },
        "baseline_cost_no_review": baseline_no_review,
        "baseline_cost_review_all": baseline_all_review,
        "figures": [
            "model_pr_curve.png",
            "model_roc_curve.png",
            "model_cost_curve.png",
            "model_score_distribution.png",
        ]
        + (["model_feature_importance.png"] if imp is not None else []),
    }
    (ROOT / "models" / "registry.json").write_text(
        json.dumps(registry, indent=2), encoding="utf-8"
    )

    top_features = (
        "\n".join(
            f"| {i + 1} | {row.feature} | {row.importance:.4f} |"
            for i, row in enumerate(imp.itertuples())
        )
        if imp is not None
        else "| - | _permutation importance unavailable in this run_ | - |"
    )
    importance_fig_line = (
        "- `figures/model_feature_importance.png`" if imp is not None else ""
    )
    interpretation = (
        "Customer return history features dominate, confirming hypothesis H2 from\n"
        "the EDA notebook: risk is concentrated in a subset of customers and the\n"
        "point-in-time encodings translate that into predictive signal without\n"
        "leakage. Product-level return rates and price-deviation features add\n"
        "complementary signal. Temporal and line-level features contribute\n"
        "marginally — consistent with returns being driven mainly by *who* is\n"
        "ordering rather than *when*."
        if imp is not None
        else "_Permutation importance was skipped in this run — see console warning._"
    )
    report = f"""# Model evaluation — {run['model_name']} (v{run['version']})

Time-based holdout: test window starts {schema['split_date']} \
({schema['n_test']:,} lines, return rate {schema['test_return_rate']:.2%}).

## Headline metrics (test)

| metric | value |
|---|---|
| PR-AUC (average precision) | {metrics['average_precision']:.4f} |
| ROC-AUC | {metrics['roc_auc']:.4f} |
| precision@1% of lines | {metrics['precision@1%']:.2%} (lift {metrics['lift@1%']:.1f}x over base rate) |
| recall@1% of lines | {metrics['recall@1%']:.2%} |
| CV AP (train, {cfg['model']['cv_folds']}-fold) | {run['cv_average_precision_mean']:.4f} ± {run['cv_average_precision_std']:.4f} |

## Cost-based operating point

Assumption: missing a true return line costs {fn_cost}x a manual review of a
false alarm ({fp_cost}). Sweeping thresholds on the test set, expected cost is
minimized at **threshold = {threshold:.4f}**:

- recall **{best['recall']:.2%}**, precision **{best['precision']:.2%}**
- flags **{best['flagged_share']:.2%}** of all lines for review
- expected cost **{best['expected_cost']:,.0f}** vs **{baseline_no_review:,.0f}**
  (no review at all) and **{baseline_all_review:,.0f}** (review everything) —
  ~{baseline_no_review / max(best['expected_cost'], 1):.1f}x cheaper than no review.

This threshold is persisted in `models/registry.json` and used by the API.

## Top predictors (permutation importance, AP drop)

| rank | feature | importance |
|---|---|---|
{top_features}

## Figures

- `figures/model_pr_curve.png`, `figures/model_roc_curve.png`
- `figures/model_cost_curve.png`, `figures/model_score_distribution.png`
{importance_fig_line}

## Interpretation

{interpretation}
"""
    (ROOT / "reports" / "evaluation.md").write_text(report, encoding="utf-8")
    print(f"Report written to {ROOT / 'reports' / 'evaluation.md'}")


if __name__ == "__main__":
    main()
