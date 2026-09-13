from __future__ import annotations

import gc
import json
from typing import Any

import mlflow
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error

from src.config import (
    ALS_FACTORS,
    ALS_ITERATIONS,
    ALS_REGULARIZATION,
    ENSEMBLE_MODEL_PATH,
    MLFLOW_EXPERIMENT_NAME,
    MLFLOW_TRACKING_URI,
    MIN_RATINGS_COUNT,
    RANDOM_STATE,
    SVD_LR_ALL,
    SVD_N_EPOCHS,
    SVD_N_FACTORS,
    SVD_REG_ALL,
    TEST_DATA_PATH,
    TRAIN_DATA_PATH,
    TRAIN_SPLIT_QUANTILE,
    VAL_SPLIT_QUANTILE,
)
from src.evaluation.ranking import evaluate_top_n
from src.models.als_model import get_or_train_als
from src.models.popularity import get_or_train_popularity
from src.models.svd_model import get_or_train_svd, predict_batch


def _rating_metrics(
    actual: np.ndarray,
    predicted: np.ndarray,
) -> dict[str, float]:
    return {
        "RMSE": float(np.sqrt(np.mean((actual - predicted) ** 2))),
        "MAE": float(mean_absolute_error(actual, predicted)),
    }


def _load_genres() -> dict[Any, set[str]]:
    from src.config import ENRICHED_MOVIES_PATH

    if not ENRICHED_MOVIES_PATH.exists():
        return {}

    movies = pd.read_csv(
        ENRICHED_MOVIES_PATH,
        usecols=["Movie_ID", "Genre"],
    )

    return {
        row.Movie_ID: {
            genre.strip()
            for genre in str(row.Genre).split(",")
            if genre.strip() != "Unknown"
        }
        for row in movies.itertuples()
    }


def _svd_scorer(model: Any):
    return lambda user_id, movie_ids: predict_batch(
        model,
        user_id,
        movie_ids,
    )


def _popularity_scorer(artifact: dict[str, Any]):
    averages = artifact["movie_avgs"]
    global_mean = float(artifact["global_mean"])

    return lambda _user_id, movie_ids: np.asarray(
        [
            averages.get(movie_id, global_mean)
            for movie_id in movie_ids
        ],
        dtype=float,
    )


def _most_popular_scorer(artifact: dict[str, Any]):
    counts = artifact["movie_counts"]

    return lambda _user_id, movie_ids: np.asarray(
        [
            counts.get(movie_id, 0)
            for movie_id in movie_ids
        ],
        dtype=float,
    )


def _ensemble_scorer(
    svd_model: Any,
    popularity: dict[str, Any],
    alpha: float,
):
    """Create a scorer that blends SVD and popularity predictions."""

    averages = popularity["movie_avgs"]
    global_mean = float(popularity["global_mean"])

    def score(
        user_id: Any,
        movie_ids: np.ndarray,
    ) -> np.ndarray:
        popularity_scores = np.asarray(
            [
                averages.get(movie_id, global_mean)
                for movie_id in movie_ids
            ],
            dtype=float,
        )

        svd_scores = np.asarray(
            predict_batch(
                svd_model,
                user_id,
                movie_ids,
            ),
            dtype=float,
        )

        return np.clip(
            alpha * svd_scores
            + (1.0 - alpha) * popularity_scores,
            1.0,
            5.0,
        )

    return score


def _als_scorer(
    als_model: Any,
    user_to_idx: dict[Any, int],
    movie_to_idx: dict[Any, int],
):
    """Create a scorer for the implicit ALS model.

    ALS is evaluated as a ranking model. Its scores represent
    implicit-feedback preference strength, not explicit 1-5 ratings.
    """

    def score(
        user_id: Any,
        movie_ids: np.ndarray,
    ) -> np.ndarray:
        user_idx = user_to_idx.get(user_id)

        if user_idx is None:
            return np.full(
                len(movie_ids),
                -np.inf,
                dtype=float,
            )

        item_indices = np.asarray(
            [
                movie_to_idx.get(movie_id, -1)
                for movie_id in movie_ids
            ],
            dtype=int,
        )

        scores = np.full(
            len(movie_ids),
            -np.inf,
            dtype=float,
        )

        valid = item_indices >= 0

        if np.any(valid):
            scores[valid] = (
                als_model.user_factors[user_idx]
                @ als_model.item_factors[
                    item_indices[valid]
                ].T
            )

        return scores

    return score


def evaluate_models(
    log_mlflow: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Evaluate rating and ranking models on the chronological test split."""

    train_df = pd.read_parquet(TRAIN_DATA_PATH)
    test_df = pd.read_parquet(TEST_DATA_PATH)

    actual = test_df["Rating"].to_numpy(dtype=float)

    # Rating prediction evaluation

    popularity = get_or_train_popularity()

    pred_global = np.full(
        len(test_df),
        popularity["global_mean"],
    )

    pred_pop = (
        test_df["Movie_ID"]
        .map(popularity["movie_avgs"])
        .fillna(popularity["global_mean"])
        .to_numpy()
    )

    svd = get_or_train_svd()

    pred_svd = np.empty(
        len(test_df),
        dtype=np.float32,
    )

    for start in range(0, len(test_df), 50_000):
        end = min(
            start + 50_000,
            len(test_df),
        )

        chunk = test_df.iloc[start:end]

        testset = list(
            zip(
                chunk["CustomerID"].astype(str),
                chunk["Movie_ID"].astype(str),
                chunk["Rating"],
            )
        )

        pred_svd[start:end] = [
            prediction.est
            for prediction in svd.test(testset)
        ]

    ensemble = json.loads(
        ENSEMBLE_MODEL_PATH.read_text(
            encoding="utf-8",
        )
    )

    alpha = float(
        ensemble["svd_alpha"]
    )

    pred_ensemble = np.clip(
        alpha * pred_svd
        + (1.0 - alpha) * pred_pop,
        1.0,
        5.0,
    )

    rating_rows: list[dict[str, float | str]] = []

    for name, prediction in [
        ("Global Mean", pred_global),
        ("Popularity Rating", pred_pop),
        ("SVD", pred_svd),
        ("SVD + Popularity", pred_ensemble),
    ]:
        metric_values = _rating_metrics(
            actual,
            prediction,
        )

        rating_rows.append(
            {
                "Model": name,
                "RMSE": metric_values["RMSE"],
                "MAE": metric_values["MAE"],
            }
        )

    # Ranking evaluation

    candidate_ids = (
        train_df["Movie_ID"]
        .drop_duplicates()
        .tolist()
    )

    genres = _load_genres()

    ranking_rows: list[dict[str, float | str]] = []

    user_mapping = (
        pd.concat(
            [
                train_df[["CustomerID", "user_idx"]],
                test_df[["CustomerID", "user_idx"]],
            ],
            ignore_index=True,
        )
        .drop_duplicates("CustomerID")
    )

    movie_mapping = (
        pd.concat(
            [
                train_df[["Movie_ID", "movie_idx"]],
                test_df[["Movie_ID", "movie_idx"]],
            ],
            ignore_index=True,
        )
        .drop_duplicates("Movie_ID")
    )

    user_to_idx = dict(
        zip(
            user_mapping["CustomerID"],
            user_mapping["user_idx"],
        )
    )

    movie_to_idx = dict(
        zip(
            movie_mapping["Movie_ID"],
            movie_mapping["movie_idx"],
        )
    )

    # ------------------------------------------------------------
    # Load ALS model.
    #
    # ALS is included only in ranking evaluation because it is
    # trained using implicit-feedback confidence scores rather
    # than directly predicting explicit 1-5 ratings.
    # ------------------------------------------------------------

    als = get_or_train_als()

    als_scorer = _als_scorer(
        als,
        user_to_idx,
        movie_to_idx,
    )

    # Ranking models

    ranking_models = {
        "ALS (Implicit)": als_scorer,
        "Most Popular": _most_popular_scorer(popularity),
        "Popularity Rating": _popularity_scorer(popularity),
        "SVD": _svd_scorer(svd),
        "SVD + Popularity (Ensemble)": _ensemble_scorer(
            svd,
            popularity,
            alpha,
        ),
    }

    for name, scorer in ranking_models.items():
        metrics = evaluate_top_n(
            test_df,
            train_df,
            candidate_ids,
            scorer,
            k=10,
            random_state=RANDOM_STATE,
            genres=genres,
        )

        ranking_row: dict[str, float | str] = {
            "Model": name,
        }

        ranking_row.update(metrics)
        ranking_rows.append(ranking_row)

    # MLflow logging
    if log_mlflow:
        mlflow.set_tracking_uri(
            MLFLOW_TRACKING_URI
        )

        mlflow.set_experiment(
            MLFLOW_EXPERIMENT_NAME
        )

        with mlflow.start_run():

            mlflow.log_params(
                {
                    "project_evaluation": "temporal_test",
                    "train_quantile": TRAIN_SPLIT_QUANTILE,
                    "validation_quantile": VAL_SPLIT_QUANTILE,
                    "relevance_threshold": 4,
                    "ranking_k": 10,
                    "negative_sampling_ratio": 20,
                    "random_state": RANDOM_STATE,
                    "svd_n_factors": SVD_N_FACTORS,
                    "svd_n_epochs": SVD_N_EPOCHS,
                    "svd_lr_all": SVD_LR_ALL,
                    "svd_reg_all": SVD_REG_ALL,
                    "als_factors": ALS_FACTORS,
                    "als_iterations": ALS_ITERATIONS,
                    "als_regularization": ALS_REGULARIZATION,
                    "min_ratings_count": MIN_RATINGS_COUNT,
                }
            )

            for row in rating_rows:
                model_name = str(row["Model"])

                prefix = (
                    model_name
                    .lower()
                    .replace(" ", "_")
                    .replace("+", "plus")
                )

                mlflow.log_metrics(
                    {
                        f"rating_{prefix}_rmse": float(
                            row["RMSE"]
                        ),
                        f"rating_{prefix}_mae": float(
                            row["MAE"]
                        ),
                    }
                )

            for row in ranking_rows:
                model_name = str(row["Model"])

                prefix = (
                    model_name
                    .lower()
                    .replace(" ", "_")
                    .replace("+", "plus")
                    .replace("(", "")
                    .replace(")", "")
                )

                mlflow.log_metrics(
                    {
                        f"ranking_{prefix}_{key.replace('@', '_')}": float(
                            value
                        )
                        for key, value in row.items()
                        if key != "Model"
                    }
                )

    # Results

    rating_results = pd.DataFrame(
        rating_rows
    )

    ranking_results = pd.DataFrame(
        ranking_rows
    )

    print(
        "\nRating prediction metrics "
        "(temporal test set)"
    )

    print(
        rating_results.to_string(
            index=False,
            float_format=lambda x: f"{x:.4f}",
        )
    )

    print(
        "\nRanking metrics "
        "(temporal test set, K=10)"
    )

    print(
        ranking_results.to_string(
            index=False,
            float_format=lambda x: f"{x:.4f}",
        )
    )

    del svd
    del als
    gc.collect()

    return rating_results, ranking_results