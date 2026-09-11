from src.data.build_features import create_splits
import sys
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root))


if __name__ == "__main__":
    print("Starting feature building and data splitting...")
    create_splits()
    print("Splitting process finished. RAM fully released.")
