import pandas as pd
import pickle
from src.config import TRAIN_DATA_PATH, BASELINE_MODEL_PATH, MIN_RATINGS_COUNT

def get_or_train_popularity(force_retrain=False):
    """Calculates or loads the popularity baseline (global mean and movie averages)."""
    if BASELINE_MODEL_PATH.exists() and not force_retrain:
        print(f"Saved Popularity model found at {BASELINE_MODEL_PATH}. Loading...")
        with open(BASELINE_MODEL_PATH, "rb") as f:
            return pickle.load(f)

    print(f"Training Popularity Baseline on {TRAIN_DATA_PATH}...")
    train_df = pd.read_parquet(TRAIN_DATA_PATH, columns=["Movie_ID", "Rating"])

    global_mean = float(train_df["Rating"].mean())
    
    popularity = (
        train_df.groupby("Movie_ID")
        .agg(avg_rating=("Rating", "mean"), count=("Rating", "count"))
        .query(f"count >= {MIN_RATINGS_COUNT}")
    )
    
    movie_avgs = popularity["avg_rating"].to_dict()
    
    model_artifact = {
        "global_mean": global_mean,
        "movie_avgs": movie_avgs
    }

    print("Saving Popularity Baseline artifact...")
    with open(BASELINE_MODEL_PATH, "wb") as f:
        pickle.dump(model_artifact, f)
        
    print(f"Popularity Baseline secured at: {BASELINE_MODEL_PATH}")
    return model_artifact