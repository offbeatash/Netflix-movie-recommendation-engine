import logging
import pandas as pd
import pickle
from src.config import TRAIN_DATA_PATH, BASELINE_MODEL_PATH, MIN_RATINGS_COUNT

logger = logging.getLogger(__name__)


def get_or_train_popularity(force_retrain=False):
    """Calculates or loads the popularity baseline (global mean and movie averages)."""
    if BASELINE_MODEL_PATH.exists() and not force_retrain:
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
    with open(BASELINE_MODEL_PATH, "wb") as f:
        pickle.dump(model_artifact, f)

    logger.info("Popularity baseline artifact saved")
    return model_artifact
