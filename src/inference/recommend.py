import json
import logging
import time

import numpy as np
import pandas as pd

from src.config import (
    TRAIN_DATA_PATH,
    ENRICHED_MOVIES_PATH,
    ENSEMBLE_MODEL_PATH,
)
from src.models.svd_model import get_or_train_svd
from src.models.popularity import get_or_train_popularity
from src.serving.monitoring import (
    MODEL_CACHE_INITIALIZATION_SECONDS,
    observe_metric,
)

_CACHE = {}
logger = logging.getLogger(__name__)


def _load_artifacts():
    """Load models and serving data into memory exactly once."""
    if not _CACHE:
        start = time.perf_counter()
        logger.info("Initializing inference cache")

        _CACHE["train_df"] = pd.read_parquet(
            TRAIN_DATA_PATH,
            columns=["CustomerID", "Movie_ID"],
        )

        _CACHE["known_users"] = set(_CACHE["train_df"]["CustomerID"].unique())

        _CACHE["user_seen_movies"] = (
            _CACHE["train_df"].groupby("CustomerID")["Movie_ID"].agg(set).to_dict()
        )

        _CACHE["movies_df"] = pd.read_csv(ENRICHED_MOVIES_PATH)

        movies_exp = _CACHE["movies_df"].copy()

        movies_exp["Genre"] = movies_exp["Genre"].astype(str).str.split(", ")

        movies_exp = movies_exp.explode("Genre")

        _CACHE["movies_exp"] = movies_exp[
            movies_exp["Genre"].notna() & (movies_exp["Genre"] != "Unknown")
        ]

        _CACHE["popularity_artifact"] = get_or_train_popularity()

        _CACHE["svd_model"] = get_or_train_svd()

        observe_metric(
            MODEL_CACHE_INITIALIZATION_SECONDS.set,
            time.perf_counter() - start,
        )

        logger.info("Inference cache initialized")

    return _CACHE


def _load_ensemble_weights():
    """
    Load pre-optimized ensemble weights lazily.

    Ensemble weights are only required for personalized
    recommendations. Cold-start requests use popularity only.
    """
    if "ensemble_weights" not in _CACHE:
        if not ENSEMBLE_MODEL_PATH.exists():
            raise FileNotFoundError(
                f"Ensemble weights not found at "
                f"{ENSEMBLE_MODEL_PATH}. "
                "Run the evaluation pipeline before starting inference."
            )

        with open(ENSEMBLE_MODEL_PATH, "r") as f:
            _CACHE["ensemble_weights"] = json.load(f)

    return _CACHE["ensemble_weights"]


def _batch_svd_predict(svd_model, user_id, movie_ids):
    """
    Predict ratings for one user across many movies.

    Uses the trained Surprise SVD latent factors directly:

        prediction =
            global_mean
            + user_bias
            + item_bias
            + dot(user_factors, item_factors)

    The expensive prediction calculation is performed as a
    vectorized NumPy matrix operation instead of calling
    svd_model.predict() once per movie.

    Unknown movies receive Surprise's default estimate for a
    known user:

        global_mean + user_bias
    """

    movie_ids = np.asarray(movie_ids)

    if movie_ids.size == 0:
        return np.empty(0, dtype=float)

    required_attributes = (
        "pu",
        "qi",
        "bu",
        "bi",
        "trainset",
    )

    if not all(hasattr(svd_model, attr) for attr in required_attributes):
        return np.asarray(
            [
                svd_model.predict(
                    str(user_id),
                    str(movie_id),
                ).est
                for movie_id in movie_ids
            ],
            dtype=float,
        )

    trainset = svd_model.trainset

    raw_user_id = str(user_id)

    try:
        inner_uid = trainset.to_inner_uid(raw_user_id)
    except ValueError:
        global_mean = float(trainset.global_mean)

        return np.full(
            movie_ids.size,
            global_mean,
            dtype=float,
        )

    global_mean = float(trainset.global_mean)

    user_bias = float(svd_model.bu[inner_uid])

    user_factors = np.asarray(
        svd_model.pu[inner_uid],
        dtype=np.float64,
    )

    inner_item_ids = []

    for movie_id in movie_ids:
        try:
            inner_item_ids.append(trainset.to_inner_iid(str(movie_id)))
        except ValueError:
            inner_item_ids.append(-1)

    inner_item_ids = np.asarray(
        inner_item_ids,
        dtype=np.int64,
    )

    known_mask = inner_item_ids >= 0

    predictions = np.full(
        movie_ids.size,
        global_mean + user_bias,
        dtype=np.float64,
    )

    if known_mask.any():
        known_item_ids = inner_item_ids[known_mask]

        item_biases = np.asarray(
            svd_model.bi[known_item_ids],
            dtype=np.float64,
        )

        item_factors = np.asarray(
            svd_model.qi[known_item_ids],
            dtype=np.float64,
        )

        dot_products = item_factors @ user_factors

        predictions[known_mask] = global_mean + user_bias + item_biases + dot_products

    return np.clip(
        predictions,
        1.0,
        5.0,
    )


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

        user_movies = movies_exp.copy()

        user_movies["predicted_rating"] = (
            user_movies["Movie_ID"]
            .map(movie_avgs)
            .fillna(popularity_artifact["global_mean"])
        )

        best_per_genre = (
            user_movies.sort_values(
                "predicted_rating",
                ascending=False,
            )
            .groupby("Genre")[["Genre", "Title", "predicted_rating"]]
            .head(top_n)
        )

    # PERSONALIZED
    else:
        status_msg = f"Showing personalized results for user: {user_id}"

        ensemble_weights = _load_ensemble_weights()

        seen = cache["user_seen_movies"].get(
            user_id,
            set(),
        )

        unseen_exp = movies_exp[~movies_exp["Movie_ID"].isin(seen)].copy()

        svd_model = cache["svd_model"]

        unseen_exp["svd_rating"] = _batch_svd_predict(
            svd_model=svd_model,
            user_id=user_id,
            movie_ids=unseen_exp["Movie_ID"].to_numpy(),
        )

        movie_avgs = popularity_artifact["movie_avgs"]

        unseen_exp["popularity_rating"] = (
            unseen_exp["Movie_ID"]
            .map(movie_avgs)
            .fillna(popularity_artifact["global_mean"])
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

        best_per_genre = (
            unseen_exp.sort_values(
                "predicted_rating",
                ascending=False,
            )
            .groupby("Genre")[["Genre", "Title", "predicted_rating"]]
            .head(top_n)
        )

    result = best_per_genre.reset_index(drop=True)

    result.columns = [
        "Genre",
        "Movie Title",
        "Predicted Rating",
    ]

    result["Predicted Rating"] = result["Predicted Rating"].round(2)

    return status_msg, result
