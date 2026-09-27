import json
import logging
import pickle
from pathlib import Path

import numpy as np
import pandas as pd

from src.models.svd_model import predict_batch

logger = logging.getLogger(__name__)


# Paths

PROJECT_ROOT = Path(__file__).resolve().parents[2]

ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
DATA_DIR = PROJECT_ROOT / "data"

SVD_MODEL_PATH = ARTIFACTS_DIR / "svd_model.pkl"
POPULARITY_MODEL_PATH = ARTIFACTS_DIR / "popularity_model.pkl"
ENSEMBLE_WEIGHTS_PATH = ARTIFACTS_DIR / "ensemble_weights.json"
USER_HISTORY_PATH = ARTIFACTS_DIR / "user_seen_movies.npz"
MOVIES_PATH = DATA_DIR / "movies_with_genres.csv"


# Lazy deployment cache

_CACHE = {}


def _load_cache():
    """Load all pre-built deployment artifacts once."""

    if _CACHE:
        return _CACHE

    logger.info("Loading HF deployment artifacts...")

    # SVD model
    with open(SVD_MODEL_PATH, "rb") as f:
        svd_model = pickle.load(f)

    # Popularity model
    with open(POPULARITY_MODEL_PATH, "rb") as f:
        popularity_model = pickle.load(f)

    # Ensemble weights
    with open(ENSEMBLE_WEIGHTS_PATH, "r", encoding="utf-8") as f:
        ensemble_weights = json.load(f)

    # Movie metadata
    movies_df = pd.read_csv(MOVIES_PATH)

    # Compact user history
    history = np.load(USER_HISTORY_PATH)

    _CACHE.update(
        {
            "svd_model": svd_model,
            "popularity_model": popularity_model,
            "ensemble_weights": ensemble_weights,
            "movies_df": movies_df,
            "history": history,
        }
    )

    logger.info("HF deployment artifacts loaded successfully.")

    return _CACHE


# User history lookup


def _get_seen_movies(user_id):
    """
    Return the set of movies already watched by a user.

    The deployment history uses:
        user_ids
        offsets
        movie_ids
    """

    history = _load_cache()["history"]

    user_ids = history["user_ids"]
    offsets = history["offsets"]
    movie_ids = history["movie_ids"]

    # Binary search because user_ids are sorted.
    index = np.searchsorted(user_ids, user_id)

    if index >= len(user_ids) or user_ids[index] != user_id:
        return set()

    start = offsets[index]
    end = offsets[index + 1]

    return set(movie_ids[start:end].tolist())


# User existence


def _user_exists(user_id):
    """Check whether the SVD model contains the user."""

    svd_model = _load_cache()["svd_model"]

    return user_id in svd_model.trainset._raw2inner_id_users


# Recommendations


def generate_genre_recommendations(user_id, top_n=10):
    """
    Generate personalized or cold-start movie recommendations.

    Known user:
        89% SVD + 11% popularity

    Unknown user:
        popularity-only cold-start recommendations
    """

    cache = _load_cache()

    svd_model = cache["svd_model"]
    popularity_model = cache["popularity_model"]
    ensemble_weights = cache["ensemble_weights"]
    movies_df = cache["movies_df"]

    # Normalize user ID to match the SVD model's string mappings.
    user_id = str(user_id).strip()

    try:
        top_n = int(top_n)
    except (TypeError, ValueError):
        top_n = 10

    top_n = max(1, min(top_n, 20))

    user_exists = _user_exists(user_id)

    # Prepare movie metadata

    movies_exp = movies_df.copy()

    movies_exp["Genre"] = movies_exp["Genre"].astype(str).str.split(", ")

    movies_exp = movies_exp.explode("Genre")

    movies_exp = movies_exp[
        movies_exp["Genre"].notna() & (movies_exp["Genre"] != "Unknown")
    ].copy()

    # COLD START

    if not user_exists:

        movie_avgs = popularity_model["movie_avgs"]
        global_mean = popularity_model["global_mean"]

        movies_exp["predicted_rating"] = (
            movies_exp["Movie_ID"].map(movie_avgs).fillna(global_mean)
        )

        top_per_genre = (
            movies_exp.sort_values(
                "predicted_rating",
                ascending=False,
            )
            .groupby("Genre", sort=False)[["Genre", "Title", "predicted_rating"]]
            .head(1)
        )

        result = (
            top_per_genre.sort_values(
                "predicted_rating",
                ascending=False,
            )
            .drop_duplicates(
                subset=["Title"],
                keep="first",
            )
            .head(top_n)
            .reset_index(drop=True)
        )

        result.columns = [
            "Genre",
            "Movie Title",
            "Predicted Rating",
        ]

        result["Predicted Rating"] = result["Predicted Rating"].round(2)

        status = (
            f"User '{user_id}' not found. "
            "Showing global popular movies "
            "(Cold Start Baseline)."
        )

        return status, result

    # PERSONALIZED

    seen_movies = _get_seen_movies(int(user_id))

    # Remove movies already watched.
    unseen_exp = movies_exp[~movies_exp["Movie_ID"].isin(seen_movies)].copy()

    # SVD prediction.
    unseen_exp["svd_rating"] = predict_batch(
        svd_model=svd_model,
        user_id=user_id,
        movie_ids=unseen_exp["Movie_ID"].to_numpy(),
    )

    # Popularity prediction.
    movie_avgs = popularity_model["movie_avgs"]
    global_mean = popularity_model["global_mean"]

    unseen_exp["popularity_rating"] = (
        unseen_exp["Movie_ID"].map(movie_avgs).fillna(global_mean)
    )

    svd_alpha = float(ensemble_weights["svd_alpha"])

    popularity_alpha = float(ensemble_weights["popularity_alpha"])

    unseen_exp["predicted_rating"] = (
        svd_alpha * unseen_exp["svd_rating"]
        + popularity_alpha * unseen_exp["popularity_rating"]
    ).clip(1.0, 5.0)

    # Best movie per genre.
    top_per_genre = (
        unseen_exp.sort_values(
            "predicted_rating",
            ascending=False,
        )
        .groupby("Genre", sort=False)[["Genre", "Title", "predicted_rating"]]
        .head(1)
    )

    result = (
        top_per_genre.sort_values(
            "predicted_rating",
            ascending=False,
        )
        .drop_duplicates(
            subset=["Title"],
            keep="first",
        )
        .head(top_n)
        .reset_index(drop=True)
    )

    result.columns = [
        "Genre",
        "Movie Title",
        "Predicted Rating",
    ]

    result["Predicted Rating"] = result["Predicted Rating"].round(2)

    status = f"Showing personalized results for user: {user_id}"

    return status, result
