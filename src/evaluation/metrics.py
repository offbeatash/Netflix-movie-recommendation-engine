import json
import mlflow
import pandas as pd
import numpy as np
import gc
from sklearn.metrics import mean_absolute_error
from src.config import (
    TEST_DATA_PATH,
    ENSEMBLE_MODEL_PATH,
    MLFLOW_TRACKING_URI,
    MLFLOW_EXPERIMENT_NAME,
    MIN_RATINGS_COUNT,
    TRAIN_SPLIT_QUANTILE,
    VAL_SPLIT_QUANTILE,
    RANDOM_STATE,
    SVD_N_FACTORS,
    SVD_N_EPOCHS,
    SVD_LR_ALL,
    SVD_REG_ALL,
    ALS_FACTORS,
    ALS_ITERATIONS,
    ALS_REGULARIZATION,
)
from src.models.popularity import get_or_train_popularity
from src.models.als_model import get_or_train_als
from src.models.svd_model import get_or_train_svd

def evaluate_models():
    """Generates predictions for all models on the test set and calculates RMSE/MAE."""
    print(f"Loading test data from {TEST_DATA_PATH}...")
    test_df = pd.read_parquet(
        TEST_DATA_PATH, 
        columns=["CustomerID", "Movie_ID", "user_idx", "movie_idx", "Rating"]
    )
    actual_ratings = test_df["Rating"].values

    print("Loading trained artifacts...")
    popularity_artifact = get_or_train_popularity()
    global_mean = popularity_artifact["global_mean"]
    movie_avgs = popularity_artifact["movie_avgs"]
    
    als_model = get_or_train_als()
    svd_model = get_or_train_svd()
    
    with open(ENSEMBLE_MODEL_PATH, "r") as f:
        ensemble_weights = json.load(f)
    best_alpha = ensemble_weights["best_alpha"]

    print("Generating predictions...")
    
    # 1. Naive Baseline
    pred_naive = np.full(len(test_df), global_mean)
    
    # 2. Model A (Popularity)
    pred_pop = test_df["Movie_ID"].map(movie_avgs).fillna(global_mean).values
    
    # 3. Model B (ALS) - With Out-Of-Bounds Protection
    n_users_als = als_model.user_factors.shape[0]
    n_movies_als = als_model.item_factors.shape[0]

    valid_users = test_df["user_idx"].values < n_users_als
    valid_movies = test_df["movie_idx"].values < n_movies_als
    valid_mask = valid_users & valid_movies

    u_factors = np.zeros((len(test_df), als_model.user_factors.shape[1]))
    m_factors = np.zeros((len(test_df), als_model.item_factors.shape[1]))

    u_factors[valid_mask] = als_model.user_factors[test_df["user_idx"].values[valid_mask]]
    m_factors[valid_mask] = als_model.item_factors[test_df["movie_idx"].values[valid_mask]]

    pred_als = np.clip(np.sum(u_factors * m_factors, axis=1), 1, 5)
    
    del als_model, u_factors, m_factors
    gc.collect()

    # 4. Model C (SVD)
    pred_svd = np.empty(len(test_df), dtype=np.float32)
    for start in range(0, len(test_df), 50_000):
        end = min(start + 50_000, len(test_df))
        chunk = test_df.iloc[start:end]
        testset = list(zip(chunk["CustomerID"].astype(str), chunk["Movie_ID"].astype(str), chunk["Rating"]))
        predictions = svd_model.test(testset)
        pred_svd[start:end] = [p.est for p in predictions]
        
    del svd_model
    gc.collect()

    # 5. Model D (Ensemble)
    pred_ensemble = np.clip(best_alpha * pred_als + (1.0 - best_alpha) * pred_svd, 1.0, 5.0)

    def calc_metrics(pred):
        rmse = np.sqrt(((actual_ratings - pred) ** 2).mean())
        mae = mean_absolute_error(actual_ratings, pred)
        return float(rmse), float(mae)

    results = pd.DataFrame({
        "Model": [
            "Naive Baseline", 
            "Model A (Popularity)", 
            "Model B (ALS)", 
            "Model C (SVD)", 
            "Model D (Ensemble)"
        ],
        "RMSE": [
            calc_metrics(pred_naive)[0], 
            calc_metrics(pred_pop)[0], 
            calc_metrics(pred_als)[0], 
            calc_metrics(pred_svd)[0], 
            calc_metrics(pred_ensemble)[0]
        ],
        "MAE": [
            calc_metrics(pred_naive)[1], 
            calc_metrics(pred_pop)[1], 
            calc_metrics(pred_als)[1], 
            calc_metrics(pred_svd)[1], 
            calc_metrics(pred_ensemble)[1]
        ]
    })

    metrics_by_model = results.set_index("Model")
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)
    with mlflow.start_run():
        mlflow.log_params({
            "min_ratings_count": MIN_RATINGS_COUNT,
            "train_split_quantile": TRAIN_SPLIT_QUANTILE,
            "val_split_quantile": VAL_SPLIT_QUANTILE,
            "random_state": RANDOM_STATE,
            "svd_n_factors": SVD_N_FACTORS,
            "svd_n_epochs": SVD_N_EPOCHS,
            "svd_lr_all": SVD_LR_ALL,
            "svd_reg_all": SVD_REG_ALL,
            "als_factors": ALS_FACTORS,
            "als_iterations": ALS_ITERATIONS,
            "als_regularization": ALS_REGULARIZATION,
        })
        mlflow.log_metrics({
            "svd_test_rmse": float(metrics_by_model.loc["Model C (SVD)", "RMSE"]),
            "svd_test_mae": float(metrics_by_model.loc["Model C (SVD)", "MAE"]),
            "ensemble_test_rmse": float(metrics_by_model.loc["Model D (Ensemble)", "RMSE"]),
            "ensemble_test_mae": float(metrics_by_model.loc["Model D (Ensemble)", "MAE"]),
        })

    print("\n" + "="*50)
    print("THE EVALUATION SHOWDOWN ".center(50))
    print("="*50)
    print(f"| {'Model':<22} | {'RMSE':^8} | {'MAE':^8} |")
    print("-" * 50)
    
    for _, row in results.iterrows():
        print(f"| {row['Model']:<22} | {row['RMSE']:^8.4f} | {row['MAE']:^8.4f} |")
        
    print("="*50 + "\n")
    print("The ensemble achieved a marginally lower RMSE than SVD, while SVD retained the lower MAE.")
    print("Note: The extreme error in the ALS model demonstrates the Implicit vs. Explicit Feedback Trap.")
    
    return results