import os
from pathlib import Path

from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent.parent
load_dotenv(BASE_DIR / ".env")

PROJECT_VERSION = "2.2.0"
TMDB_API_KEY = os.getenv("TMDB_API_KEY")

DATA_DIR = BASE_DIR / "data"
RAW_DATA_PATH = DATA_DIR / "combined_data_1.txt"
MOVIE_TITLES_PATH = DATA_DIR / "movie_titles.csv"
ENRICHED_MOVIES_PATH = DATA_DIR / "movies_with_genres.csv"
PROCESSED_DATA_PATH = DATA_DIR / "processed_netflix.parquet"
DB_PATH = DATA_DIR / "movies.db"
TRAIN_DATA_PATH = DATA_DIR / "train.parquet"
VAL_DATA_PATH = DATA_DIR / "val.parquet"
TEST_DATA_PATH = DATA_DIR / "test.parquet"

ARTIFACTS_DIR = BASE_DIR / "artifacts"
BASELINE_MODEL_PATH = ARTIFACTS_DIR / "popularity_model.pkl"
ALS_MODEL_PATH = ARTIFACTS_DIR / "als_model.npz"
SVD_MODEL_PATH = ARTIFACTS_DIR / "svd_model.pkl"
ENSEMBLE_MODEL_PATH = ARTIFACTS_DIR / "ensemble_weights.json"
MLFLOW_TRACKING_URI = str(BASE_DIR / "mlruns")
MLFLOW_EXPERIMENT_NAME = "netflix-recommendation-evaluation"

DATA_DIR.mkdir(parents=True, exist_ok=True)
ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

MIN_RATINGS_COUNT = 500
TRAIN_SPLIT_QUANTILE = 0.80
VAL_SPLIT_QUANTILE = 0.90
RANDOM_STATE = 42

SVD_N_FACTORS = 50
SVD_N_EPOCHS = 20
SVD_LR_ALL = 0.005
SVD_REG_ALL = 0.04

ALS_FACTORS = 50
ALS_ITERATIONS = 50
ALS_REGULARIZATION = 0.1
ALS_CONFIDENCE_ALPHA = 40.0

API_KEY = os.getenv("API_KEY")
CORS_ALLOW_ORIGINS = tuple(
    origin.strip()
    for origin in os.getenv("CORS_ALLOW_ORIGINS", "http://localhost:7860").split(",")
    if origin.strip()
)
RATE_LIMIT_REQUESTS = int(os.getenv("RATE_LIMIT_REQUESTS", "60"))
RATE_LIMIT_WINDOW_SECONDS = int(os.getenv("RATE_LIMIT_WINDOW_SECONDS", "60"))
