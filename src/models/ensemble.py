import json
import pandas as pd
import numpy as np
import gc
from src.config import VAL_DATA_PATH, ENSEMBLE_MODEL_PATH
from src.models.als_model import get_or_train_als
from src.models.popularity import get_or_train_popularity
from src.models.svd_model import get_or_train_svd

def get_or_train_ensemble(force_retrain=False):
    """Calculates optimal linear blend between ALS and SVD, returning the alpha weight."""
    if ENSEMBLE_MODEL_PATH.exists() and not force_retrain:
        print(f"Saved Ensemble weights found at {ENSEMBLE_MODEL_PATH}. Loading...")
        with open(ENSEMBLE_MODEL_PATH, "r") as f:
            return json.load(f)

    print("Initiating Ensemble blending optimization...")
    val_df = pd.read_parquet(
        VAL_DATA_PATH, 
        columns=["CustomerID", "Movie_ID", "user_idx", "movie_idx", "Rating"]
    )
    actual_val = val_df["Rating"].values

    # 1. Popularity Validation Predictions
    popularity_artifact = get_or_train_popularity()
    pred_pop = val_df["Movie_ID"].map(popularity_artifact["movie_avgs"]).fillna(
        popularity_artifact["global_mean"]
    ).values

    # 2. SVD Validation Predictions
    print("Generating SVD validation predictions in chunks...")
    svd_model = get_or_train_svd()
    pred_svd = np.empty(len(val_df), dtype=np.float32)

    for start in range(0, len(val_df), 50_000):
        end = min(start + 50_000, len(val_df))
        chunk = val_df.iloc[start:end]
        testset = list(zip(chunk["CustomerID"], chunk["Movie_ID"], chunk["Rating"]))
        predictions = svd_model.test(testset)
        pred_svd[start:end] = [p.est for p in predictions]

    del svd_model
    gc.collect()

    # 3. Optimize the SVD/popularity blend
    print("Finding optimal SVD/popularity blend...")
    best_alpha = 0.0
    best_rmse = float("inf")

    for alpha in np.linspace(0, 1, 101):
        temp_pred = np.clip(alpha * pred_svd + (1.0 - alpha) * pred_pop, 1.0, 5.0)
        temp_rmse = np.sqrt(((actual_val - temp_pred) ** 2).mean())
        
        if temp_rmse < best_rmse:
            best_rmse = float(temp_rmse)
            best_alpha = float(alpha)

    svd_weight = int(round(best_alpha * 100))
    print(f"Optimization complete! Optimal Blend: {svd_weight}% SVD + {100-svd_weight}% popularity")

    artifact = {
        "svd_alpha": best_alpha,
        "popularity_alpha": 1.0 - best_alpha,
        "val_rmse": best_rmse,
    }
    with open(ENSEMBLE_MODEL_PATH, "w") as f:
        json.dump(artifact, f)

    return artifact