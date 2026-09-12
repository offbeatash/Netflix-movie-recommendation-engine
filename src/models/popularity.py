import logging
import pickle
from typing import Any

import pandas as pd

from src.config import BASELINE_MODEL_PATH, MIN_RATINGS_COUNT, TRAIN_DATA_PATH
from src.utils import check_artifact_freshness, save_artifact_metadata
from pathlib import Path

logger = logging.getLogger(__name__)


def get_or_train_popularity(force_retrain: bool = False) -> dict[str, Any]:
    """Train/load rating baseline and most-popular ranking statistics."""
    params = {"min_ratings_count": MIN_RATINGS_COUNT}
    if not force_retrain and check_artifact_freshness(
        BASELINE_MODEL_PATH, params, TRAIN_DATA_PATH
    ):
        with BASELINE_MODEL_PATH.open("rb") as handle:
            return pickle.load(handle)

    train_df = pd.read_parquet(TRAIN_DATA_PATH, columns=["Movie_ID", "Rating"])
    global_mean = float(train_df["Rating"].mean())
    grouped = train_df.groupby("Movie_ID")["Rating"].agg(["mean", "count"])
    qualified = grouped[grouped["count"] >= MIN_RATINGS_COUNT]

    artifact = {
        "global_mean": global_mean,
        "movie_avgs": qualified["mean"].to_dict(),
        "movie_counts": grouped["count"].to_dict(),
        "most_popular_movie_ids": grouped.sort_values(
            ["count", "mean"], ascending=False
        ).index.tolist(),
    }

    temp_path = BASELINE_MODEL_PATH.with_name(".popularity_model.tmp")
    with temp_path.open("wb") as handle:
        pickle.dump(artifact, handle, protocol=pickle.HIGHEST_PROTOCOL)
    temp_path.replace(BASELINE_MODEL_PATH)
    # Define source files that affect the popularity model artifact
    popularity_source_paths = [
        Path("src/models/popularity.py"),
        Path("src/utils.py"),
        Path("src/config.py"),
    ]
    save_artifact_metadata(
        BASELINE_MODEL_PATH, params, TRAIN_DATA_PATH, popularity_source_paths
    )
    return artifact
