import sys
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root))

from src.models.popularity import get_or_train_popularity
from src.models.als_model import get_or_train_als
from src.models.svd_model import get_or_train_svd
from src.models.ensemble import get_or_train_ensemble

if __name__ == "__main__":
    print("--- STARTING FULL TRAINING PIPELINE ---")

    print("\n1. Popularity Baseline")
    get_or_train_popularity()

    print("\n2. Alternating Least Squares (ALS)")
    get_or_train_als()

    print("\n3. Singular Value Decomposition (SVD)")
    get_or_train_svd()

    print("\n4. Blended Ensemble (Alpha Optimization)")
    get_or_train_ensemble()

    print("\n--- FULL TRAINING PIPELINE COMPLETED SUCCESSFULLY ---")
