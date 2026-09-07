import json
import pandas as pd
import numpy as np
import gc
from src.config import VAL_DATA_PATH, ENSEMBLE_MODEL_PATH
from src.models.als_model import get_or_train_als
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

    # 1. ALS Validation Predictions (With Out-Of-Bounds Protection)
    print("Generating ALS validation predictions...")
    als_model = get_or_train_als()
    
    n_users_als = als_model.user_factors.shape[0]
    n_movies_als = als_model.item_factors.shape[0]

    # Mask valid indices to prevent IndexError on validation users unseen in training
    valid_users = val_df["user_idx"].values < n_users_als
    valid_movies = val_df["movie_idx"].values < n_movies_als
    valid_mask = valid_users & valid_movies

    # Initialize empty factor arrays
    u_factors = np.zeros((len(val_df), als_model.user_factors.shape[1]))
    m_factors = np.zeros((len(val_df), als_model.item_factors.shape[1]))

    # Inject valid factors safely
    u_factors[valid_mask] = als_model.user_factors[val_df["user_idx"].values[valid_mask]]
    m_factors[valid_mask] = als_model.item_factors[val_df["movie_idx"].values[valid_mask]]

    pred_als = np.clip(np.sum(u_factors * m_factors, axis=1), 1, 5)
    
    del als_model, u_factors, m_factors
    gc.collect()

    # 2. SVD Validation Predictions
    print("Generating SVD validation predictions in chunks...")
    svd_model = get_or_train_svd()
    pred_svd = np.empty(len(val_df), dtype=np.float32)

    for start in range(0, len(val_df), 50_000):
        end = min(start + 50_000, len(val_df))
        chunk = val_df.iloc[start:end]
        testset = list(zip(chunk["CustomerID"].astype(str), chunk["Movie_ID"].astype(str), chunk["Rating"]))
        predictions = svd_model.test(testset)
        pred_svd[start:end] = [p.est for p in predictions]

    del svd_model
    gc.collect()

    # 3.Optimize Alpha
    print("Finding optimal linear blend (Alpha)...")
    best_alpha = 0.0
    best_rmse = float("inf")

    for alpha in np.linspace(0, 1, 101):
        temp_pred = np.clip(alpha * pred_als + (1.0 - alpha) * pred_svd, 1.0, 5.0)
        temp_rmse = np.sqrt(((actual_val - temp_pred) ** 2).mean())
        
        if temp_rmse < best_rmse:
            best_rmse = float(temp_rmse)
            best_alpha = float(alpha)

    als_weight = int(round(best_alpha * 100))
    print(f"Optimization complete! Optimal Blend: {als_weight}% ALS + {100-als_weight}% SVD")

    artifact = {"best_alpha": best_alpha, "val_rmse": best_rmse}
    with open(ENSEMBLE_MODEL_PATH, "w") as f:
        json.dump(artifact, f)

    return artifact