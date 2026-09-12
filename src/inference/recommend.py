import json
import logging
import threading
import time
from typing import Any

import pandas as pd

from src.config import (
    TRAIN_DATA_PATH,
    ENRICHED_MOVIES_PATH,
    ENSEMBLE_MODEL_PATH,
)
from src.models.svd_model import get_or_train_svd, predict_batch
from src.models.popularity import get_or_train_popularity
from src.serving.monitoring import (
    DATA_LOAD_STATUS,
    MODEL_CACHE_INITIALIZATION_SECONDS,
    MODEL_LOAD_STATUS,
    MODEL_LOADING_ERRORS,
    observe_metric,
)

_CACHE: dict[str, Any] = {}
_CACHE_LOCK = threading.Lock()
logger = logging.getLogger(__name__)


def _load_artifacts():
    """Load models and serving data exactly once, safely under concurrency."""
    if _CACHE:
        return _CACHE
    with _CACHE_LOCK:
        if _CACHE:
            return _CACHE
        start = time.perf_counter()
        logger.info("Initializing inference cache")
        try:
            _CACHE["train_df"] = pd.read_parquet(
                TRAIN_DATA_PATH, columns=["CustomerID", "Movie_ID"]
            )
            observe_metric(DATA_LOAD_STATUS.labels(data_type="train").set, 1)
            _CACHE["movies_df"] = pd.read_csv(ENRICHED_MOVIES_PATH)
            observe_metric(DATA_LOAD_STATUS.labels(data_type="movies").set, 1)
            _CACHE["known_users"] = set(_CACHE["train_df"]["CustomerID"].unique())
            _CACHE["user_seen_movies"] = (
                _CACHE["train_df"].groupby("CustomerID")["Movie_ID"].agg(set).to_dict()
            )
            movies_exp = _CACHE["movies_df"].copy()
            movies_exp["Genre"] = movies_exp["Genre"].astype(str).str.split(", ")
            movies_exp = movies_exp.explode("Genre")
            _CACHE["movies_exp"] = movies_exp[
                movies_exp["Genre"].notna() & (movies_exp["Genre"] != "Unknown")
            ]
            _CACHE["popularity_artifact"] = get_or_train_popularity()
            observe_metric(MODEL_LOAD_STATUS.labels(model_type="popularity").set, 1)
            _CACHE["svd_model"] = get_or_train_svd()
            observe_metric(MODEL_LOAD_STATUS.labels(model_type="svd").set, 1)
            observe_metric(
                MODEL_CACHE_INITIALIZATION_SECONDS.set, time.perf_counter() - start
            )
            logger.info("Inference cache initialized")
            return _CACHE
        except Exception:
            for name in ("train", "movies"):
                observe_metric(DATA_LOAD_STATUS.labels(data_type=name).set, 0)
            raise


def _load_ensemble_weights():
    """
    Load pre-optimized ensemble weights lazily.

    Ensemble weights are only required for personalized
    recommendations. Cold-start requests use popularity only.
    """
    if "ensemble_weights" not in _CACHE:
        try:
            if not ENSEMBLE_MODEL_PATH.exists():
                raise FileNotFoundError(
                    f"Ensemble weights not found at "
                    f"{ENSEMBLE_MODEL_PATH}. "
                    "Run the evaluation pipeline before starting inference."
                )

            with open(ENSEMBLE_MODEL_PATH, "r") as f:
                _CACHE["ensemble_weights"] = json.load(f)
            observe_metric(MODEL_LOAD_STATUS.labels(model_type="ensemble").set, 1)
        except Exception as e:
            observe_metric(MODEL_LOAD_STATUS.labels(model_type="ensemble").set, 0)
            observe_metric(MODEL_LOADING_ERRORS.labels(model_type="ensemble").inc)
            logger.error(f"Failed to load ensemble weights: {e}")
            raise

    return _CACHE["ensemble_weights"]


def generate_genre_recommendations(user_id, top_n=1):
    """Generate personalized or cold-start genre recommendations."""
    cache = _load_artifacts()

    train_df = cache["train_df"]
    movies_exp = cache["movies_exp"]
    popularity_artifact = cache["popularity_artifact"]

    customer_ids = train_df["CustomerID"]

    try:
        if pd.api.types.is_integer_dtype(customer_ids):
            user_id = int(user_id)

        elif pd.api.types.is_float_dtype(customer_ids):
            user_id = float(user_id)

        else:
            user_id = str(user_id)

    except (TypeError, ValueError):
        user_id = str(user_id)

    user_exists = user_id in cache["known_users"]

    # COLD START
    if not user_exists:
        status_msg = (
            f"User '{user_id}' not found. "
            "Showing global popular movies (Cold Start Baseline)."
        )

        movie_avgs = popularity_artifact["movie_avgs"]
        global_mean = popularity_artifact["global_mean"]

        # Optimize: work with movies_exp directly, avoid copy until needed
        # Map predictions and fill NaN values
        predicted_ratings = movies_exp["Movie_ID"].map(movie_avgs).fillna(global_mean)

        # Create temporary DataFrame for sorting/grouping operations
        temp_df = movies_exp.copy()
        temp_df["predicted_rating"] = predicted_ratings.values

        # Get top movie per genre for diversity, then deduplicate and take top_n overall
        top_per_genre = (
            temp_df.sort_values(
                "predicted_rating",
                ascending=False,
            )
            .groupby("Genre")[["Genre", "Title", "predicted_rating"]]
            .head(1)  # Get top movie per genre
        )

        # Deduplicate by movie title (keeping highest rated version if same movie in multiple genres)
        deduplicated = top_per_genre.sort_values("predicted_rating", ascending=False).drop_duplicates(
            subset=["Title"], keep="first"
        )

        # Take top N overall
        best_per_genre = deduplicated.head(top_n)

    # PERSONALIZED
    else:
        status_msg = f"Showing personalized results for user: {user_id}"

        ensemble_weights = _load_ensemble_weights()

        seen = cache["user_seen_movies"].get(
            user_id,
            set(),
        )

        # Optimize: avoid copying movies_exp until we need to add columns
        mask = ~movies_exp["Movie_ID"].isin(seen)
        unseen_exp = movies_exp[mask].copy()

        svd_model = cache["svd_model"]

        unseen_exp["svd_rating"] = predict_batch(
            svd_model=svd_model,
            user_id=user_id,
            movie_ids=unseen_exp["Movie_ID"].to_numpy(),
        )

        movie_avgs = popularity_artifact["movie_avgs"]
        global_mean = popularity_artifact["global_mean"]

        unseen_exp["popularity_rating"] = (
            unseen_exp["Movie_ID"]
            .map(movie_avgs)
            .fillna(global_mean)
        )

        svd_alpha = float(
            ensemble_weights.get(
                "svd_alpha",
                ensemble_weights.get(
                    "best_alpha",
                    0.0,
                ),
            )
        )

        popularity_alpha = float(
            ensemble_weights.get(
                "popularity_alpha",
                1.0 - svd_alpha,
            )
        )

        unseen_exp["predicted_rating"] = (
            svd_alpha * unseen_exp["svd_rating"]
            + popularity_alpha * unseen_exp["popularity_rating"]
        ).clip(1.0, 5.0)

        # Get top movie per genre for diversity, then deduplicate and take top_n overall
        top_per_genre = (
            unseen_exp.sort_values(
                "predicted_rating",
                ascending=False,
            )
            .groupby("Genre")[["Genre", "Title", "predicted_rating"]]
            .head(1)  # Get top movie per genre
        )

        # Deduplicate by movie title (keeping highest rated version if same movie in multiple genres)
        deduplicated = top_per_genre.sort_values("predicted_rating", ascending=False).drop_duplicates(
            subset=["Title"], keep="first"
        )

        # Take top N overall
        best_per_genre = deduplicated.head(top_n)

    result = best_per_genre.reset_index(drop=True)

    result.columns = [
        "Genre",
        "Movie Title",
        "Predicted Rating",
    ]

    result["Predicted Rating"] = result["Predicted Rating"].round(2)

    return status_msg, result