"""Benchmark candidate models, refit the winner, and version the artifact.

Protocol:
1. Carve a stratified validation split out of the time-based training window.
2. Benchmark every candidate on that validation split (PR-AUC primary,
   precision/recall/lift at top 1% and 5% of scored lines).
3. Refit the best supervised model on the full training window, run k-fold CV,
   evaluate on the held-out test window, and persist a versioned artifact plus a
   registry entry in models/registry.json.

The unsupervised Isolation Forest is benchmarked for comparison but cannot win
selection (it does not use labels); this is stated in the benchmark report.
"""

from __future__ import annotations

import json
import time
from datetime import datetime, timezone

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.base import BaseEstimator
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import (
    HistGradientBoostingClassifier,
    IsolationForest,
    RandomForestClassifier,
)
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.data import ROOT, load_config

TOP_FRACS = (0.01, 0.05)


class IsolationForestRanker(BaseEstimator):
    """Unsupervised anomaly ranker exposed through the predict_proba interface."""

    def __init__(self, n_estimators: int = 200, random_state: int = 42):
        self.n_estimators = n_estimators
        self.random_state = random_state

    def fit(self, X, y=None):
        self.model_ = IsolationForest(
            n_estimators=self.n_estimators,
            random_state=self.random_state,
            n_jobs=-1,
        ).fit(X)
        return self

    def predict_proba(self, X):
        scores = -self.model_.decision_function(X)
        scaled = scores / (scores.std() + 1e-9)
        proba = 1.0 / (1.0 + np.exp(-scaled))
        return np.column_stack([1.0 - proba, proba])


def load_features() -> tuple[pd.DataFrame, dict]:
    schema = json.loads(
        (ROOT / "reports" / "feature_schema.json").read_text(encoding="utf-8")
    )
    df = pd.read_parquet(ROOT / "data" / "processed" / "features.parquet")
    return df, schema


def metrics_at(y_true: np.ndarray, scores: np.ndarray, fracs=TOP_FRACS) -> dict:
    y = np.asarray(y_true)
    order = np.argsort(-np.asarray(scores))
    y_sorted = y[order]
    base_rate = y.mean()
    n = len(y)

    out = {
        "average_precision": float(average_precision_score(y, scores)),
        "roc_auc": float(roc_auc_score(y, scores)),
        "base_rate": float(base_rate),
    }
    for frac in fracs:
        k = max(int(n * frac), 1)
        captured = int(y_sorted[:k].sum())
        precision = captured / k
        recall = captured / max(int(y.sum()), 1)
        label = f"{frac:.0%}"
        out[f"precision@{label}"] = float(precision)
        out[f"recall@{label}"] = float(recall)
        out[f"lift@{label}"] = float(precision / base_rate)
    return out


def build_candidates(seed: int) -> dict[str, tuple[BaseEstimator, bool]]:
    return {
        "dummy_prior": (DummyClassifier(strategy="prior"), True),
        "logistic_regression": (
            Pipeline(
                [
                    ("impute", SimpleImputer(strategy="median")),
                    ("scale", StandardScaler()),
                    (
                        "clf",
                        LogisticRegression(
                            max_iter=1000, class_weight="balanced", random_state=seed
                        ),
                    ),
                ]
            ),
            True,
        ),
        "random_forest": (
            Pipeline(
                [
                    ("impute", SimpleImputer(strategy="median")),
                    (
                        "clf",
                        RandomForestClassifier(
                            n_estimators=300,
                            min_samples_leaf=5,
                            n_jobs=-1,
                            random_state=seed,
                        ),
                    ),
                ]
            ),
            True,
        ),
        "hist_gradient_boosting": (
            HistGradientBoostingClassifier(
                max_iter=300,
                early_stopping=True,
                validation_fraction=0.1,
                class_weight="balanced",
                random_state=seed,
            ),
            True,
        ),
        "isolation_forest": (
            Pipeline(
                [
                    ("impute", SimpleImputer(strategy="median")),
                    ("clf", IsolationForestRanker(random_state=seed)),
                ]
            ),
            False,
        ),
    }


def main() -> None:
    cfg = load_config()
    seed = cfg["project"]["seed"]
    df, schema = load_features()
    features = schema["feature_columns"]
    target = schema["target"]

    train = df[df["split"] == "train"]
    test = df[df["split"] == "test"]

    X_train, X_val, y_train, y_val = train_test_split(
        train[features],
        train[target],
        test_size=cfg["model"]["test_size"],
        stratify=train[target],
        random_state=seed,
    )

    rows = []
    fitted: dict[str, BaseEstimator] = {}
    for name, (estimator, supervised) in build_candidates(seed).items():
        start = time.perf_counter()
        estimator.fit(X_train, y_train)
        fit_seconds = time.perf_counter() - start
        scores = estimator.predict_proba(X_val)[:, 1]
        row = {
            "model": name,
            "supervised": supervised,
            "fit_seconds": round(fit_seconds, 1),
            **metrics_at(y_val, scores),
        }
        rows.append(row)
        fitted[name] = estimator
        print(f"{name}: AP={row['average_precision']:.4f} ({fit_seconds:.0f}s)")

    bench = pd.DataFrame(rows).sort_values("average_precision", ascending=False)
    bench_path = ROOT / "reports" / "benchmark.csv"
    bench.to_csv(bench_path, index=False)
    print(f"\nBenchmark written to {bench_path}")

    best_name = bench[bench["supervised"]].iloc[0]["model"]
    print(f"Best supervised model: {best_name} -- refitting on full training window")

    model = build_candidates(seed)[best_name][0]
    cv = StratifiedKFold(n_splits=cfg["model"]["cv_folds"], shuffle=True, random_state=seed)
    cv_scores = cross_val_score(
        model, train[features], train[target], cv=cv, scoring="average_precision", n_jobs=-1
    )
    model.fit(train[features], train[target])

    test_scores = model.predict_proba(test[features])[:, 1]
    test_metrics = metrics_at(test[target], test_scores)

    models_dir = ROOT / "models"
    models_dir.mkdir(parents=True, exist_ok=True)
    registry_path = models_dir / "registry.json"
    registry = (
        json.loads(registry_path.read_text(encoding="utf-8"))
        if registry_path.exists()
        else {"active_version": 0, "runs": []}
    )
    version = len(registry["runs"]) + 1

    artifact_path = models_dir / f"model_v{version}.joblib"
    joblib.dump(
        {
            "pipeline": model,
            "feature_columns": features,
            "model_name": best_name,
            "version": version,
        },
        artifact_path,
    )

    registry["runs"].append(
        {
            "version": version,
            "model_name": best_name,
            "artifact": artifact_path.name,
            "trained_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "sklearn_version": sklearn.__version__,
            "seed": seed,
            "n_train": int(len(train)),
            "n_test": int(len(test)),
            "cv_average_precision_mean": float(cv_scores.mean()),
            "cv_average_precision_std": float(cv_scores.std()),
            "validation_metrics": bench[bench["model"] == best_name].iloc[0].to_dict(),
            "test_metrics": test_metrics,
        }
    )
    registry["active_version"] = version
    registry_path.write_text(json.dumps(registry, indent=2), encoding="utf-8")

    print(f"CV AP: {cv_scores.mean():.4f} +/- {cv_scores.std():.4f}")
    print(f"Test AP: {test_metrics['average_precision']:.4f}")
    print(
        f"Test precision@1%: {test_metrics['precision@1%']:.2%} "
        f"(lift {test_metrics['lift@1%']:.1f}x) | recall@1%: {test_metrics['recall@1%']:.2%}"
    )
    print(f"Artifact: {artifact_path} (active version {version})")


if __name__ == "__main__":
    main()
