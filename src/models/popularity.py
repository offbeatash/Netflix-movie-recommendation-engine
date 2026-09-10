import logging
import pandas as pd
import pickle
from src.config import TRAIN_DATA_PATH, BASELINE_MODEL_PATH, MIN_RATINGS_COUNT
from src.utils import check_artifact_freshness, save_artifact_metadata

logger = logging.getLogger(__name__)


def get_or_train_popularity(force_retrain=False):
    """Calculates or loads the popularity baseline (global mean and movie averages)."""
    current_params = {"min_ratings_count": MIN_RATINGS_COUNT}
    if (not force_retrain and
        check_artifact_freshness(BASELINE_MODEL_PATH, current_params, TRAIN_DATA_PATH)):
        logger.info("Saved popularity model found; loading")
        with open(BASELINE_MODEL_PATH, "rb") as f:
            return pickle.load(f)

    logger.info("Training popularity baseline")
    train_df = pd.read_parquet(TRAIN_DATA_PATH, columns=["Movie_ID", "Rating"])

    global_mean = float(train_df["Rating"].mean())

    popularity = (
        train_df.groupby("Movie_ID")
        .agg(avg_rating=("Rating", "mean"), count=("Rating", "count"))
        .query(f"count >= {MIN_RATINGS_COUNT}")
    )

    movie_avgs = popularity["avg_rating"].to_dict()

    model_artifact = {"global_mean": global_mean, "movie_avgs": movie_avgs}

    logger.info("Saving popularity baseline artifact")
    temp_model_path = BASELINE_MODEL_PATH.with_suffix(".tmp")
    with open(temp_model_path, "wb") as f:
        pickle.dump(model_artifact, f)
    temp_model_path.replace(BASELINE_MODEL_PATH)

    logger.info("Popularity baseline artifact saved")
    save_artifact_metadata(BASELINE_MODEL_PATH, current_params, TRAIN_DATA_PATH)

    return model_artifact