import os
import gc
import ctypes
import pickle
import pandas as pd
from surprise import SVD, Dataset, Reader
from src.config import (
    TRAIN_DATA_PATH, 
    SVD_MODEL_PATH, 
    RANDOM_STATE,
    SVD_N_FACTORS,
    SVD_N_EPOCHS,
    SVD_LR_ALL,
    SVD_REG_ALL
)
from src.utils import check_artifact_freshness, save_artifact_metadata


SVD_PARAMS = {
    "n_factors": SVD_N_FACTORS,
    "n_epochs": SVD_N_EPOCHS,
    "lr_all": SVD_LR_ALL,
    "reg_all": SVD_REG_ALL,
    "random_state": RANDOM_STATE,
}

def get_or_train_svd(force_retrain=False):
    """Trains the Surprise SVD model or loads an existing one."""
    if (
        SVD_MODEL_PATH.exists()
        and not force_retrain
        and check_artifact_freshness(SVD_MODEL_PATH, SVD_PARAMS)
    ):
        print(f"Saved SVD model found at {SVD_MODEL_PATH}. Loading...")
        with open(SVD_MODEL_PATH, "rb") as f:
            return pickle.load(f)

    print("Initiating SVD training pipeline...")
    print(f"Loading training data from {TRAIN_DATA_PATH}...")
    
    train_df = pd.read_parquet(TRAIN_DATA_PATH, columns=["CustomerID", "Movie_ID", "Rating"])
    
    train_df["CustomerID"] = train_df["CustomerID"].astype(str)
    train_df["Movie_ID"] = train_df["Movie_ID"].astype(str)
    
    print("Building dataset for SVD training...")
    reader = Reader(rating_scale=(1, 5))
    data = Dataset.load_from_df(
        train_df[["CustomerID", "Movie_ID", "Rating"]],
        reader
    )
    
    trainset = data.build_full_trainset()
    
    del train_df, data
    gc.collect()

    print(f"Training SVD Model (Factors: {SVD_N_FACTORS}, Epochs: {SVD_N_EPOCHS})...")
    svd_model = SVD(
        n_factors=SVD_N_FACTORS,
        n_epochs=SVD_N_EPOCHS,
        lr_all=SVD_LR_ALL,
        reg_all=SVD_REG_ALL,
        random_state=RANDOM_STATE
    )
    
    svd_model.fit(trainset)
    print("Model trained successfully!")
    
    print("Pruning raw rating histories from internal trainset...")
    if hasattr(svd_model, 'trainset') and svd_model.trainset is not None:
        for u in list(svd_model.trainset.ur.keys()):
            svd_model.trainset.ur[u] = None
        for i in list(svd_model.trainset.ir.keys()):
            svd_model.trainset.ir[i] = None
            
    del trainset
    gc.collect()
    
    try:
        ctypes.CDLL('libc.so.6').malloc_trim(0)
        print("OS memory trim complete. RAM released back to system.")
    except Exception as e:
        print(f"OS memory trim skipped: {e}")
    
    print("Saving SVD model artifact...")
    temp_model_path = SVD_MODEL_PATH.with_suffix(".tmp")
    
    with open(temp_model_path, "wb") as f:
        pickle.dump(svd_model, f, protocol=pickle.HIGHEST_PROTOCOL)
        
    temp_model_path.replace(SVD_MODEL_PATH)
    save_artifact_metadata(SVD_MODEL_PATH, SVD_PARAMS)
    print(f"SVD model secured at: {SVD_MODEL_PATH}")
    
    return svd_model