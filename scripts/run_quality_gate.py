"""Deterministic model-quality regression gate used by CI."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error
from surprise import Dataset, Reader, SVD

ROOT = Path(__file__).resolve().parents[1]
FIXTURE = ROOT / "tests" / "fixtures" / "quality_ratings.csv"
BASELINE = ROOT / "tests" / "fixtures" / "quality_gate_baseline.json"


def rmse(actual: np.ndarray, predicted: np.ndarray) -> float:
    return float(np.sqrt(np.mean((actual - predicted) ** 2)))


def main() -> int:
    ratings = pd.read_csv(FIXTURE)
    ratings["Date"] = pd.to_datetime(ratings["Date"])
    cutoff = ratings["Date"].quantile(0.8)
    train = ratings[ratings["Date"] <= cutoff].copy()
    test = ratings[ratings["Date"] > cutoff].copy()
    train["CustomerID"] = train["CustomerID"].astype(str)
    train["Movie_ID"] = train["Movie_ID"].astype(str)

    reader = Reader(rating_scale=(1, 5))
    dataset = Dataset.load_from_df(train[["CustomerID", "Movie_ID", "Rating"]], reader)
    model = SVD(n_factors=50, n_epochs=20, lr_all=0.005, reg_all=0.04, random_state=42)
    model.fit(dataset.build_full_trainset())

    testset = list(
        zip(
            test["CustomerID"].astype(str), test["Movie_ID"].astype(str), test["Rating"]
        )
    )
    predictions = np.asarray([p.est for p in model.test(testset)], dtype=float)
    actual = test["Rating"].to_numpy(dtype=float)
    global_mean = float(train["Rating"].mean())
    baseline_predictions = np.full(len(test), global_mean)

    model_metrics = {
        "rmse": rmse(actual, predictions),
        "mae": float(mean_absolute_error(actual, predictions)),
    }
    baseline_metrics = {
        "rmse": rmse(actual, baseline_predictions),
        "mae": float(mean_absolute_error(actual, baseline_predictions)),
    }
    policy = json.loads(BASELINE.read_text(encoding="utf-8"))

    rmse_improvement = (
        baseline_metrics["rmse"] - model_metrics["rmse"]
    ) / baseline_metrics["rmse"]
    mae_improvement = (
        baseline_metrics["mae"] - model_metrics["mae"]
    ) / baseline_metrics["mae"]
    failures = []
    if rmse_improvement < policy["min_relative_rmse_improvement"]:
        failures.append(f"RMSE improvement {rmse_improvement:.3f} below gate")
    if mae_improvement < policy["min_relative_mae_improvement"]:
        failures.append(f"MAE improvement {mae_improvement:.3f} below gate")

    print(
        json.dumps(
            {
                "baseline": baseline_metrics,
                "model": model_metrics,
                "rmse_improvement": rmse_improvement,
                "mae_improvement": mae_improvement,
            },
            indent=2,
        )
    )
    if failures:
        print("QUALITY GATE FAILED:")
        for failure in failures:
            print(f"- {failure}")
        return 1
    print("QUALITY GATE PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
