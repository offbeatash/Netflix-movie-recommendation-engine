import os
import gc
import ctypes
import pickle
import pandas as pd
from surprise import SVD, Dataset, Reader
from src.config import TRAIN_DATA_PATH, SVD_MODEL_PATH, RANDOM_STATE

def get_or_train_svd(force_retrain=False):
    """Trains the Surprise SVD model or loads an existing one."""
    if SVD_MODEL_PATH.exists() and not force_retrain:
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

    print("Training SVD Model using optimized hyperparameters...")
    svd_model = SVD(
        n_factors=50,
        n_epochs=20,
        lr_all=0.005,
        reg_all=0.04,
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
    print(f"SVD model secured at: {SVD_MODEL_PATH}")
    
    return svd_model