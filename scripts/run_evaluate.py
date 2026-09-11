from src.evaluation.metrics import evaluate_models
import sys
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root))


if __name__ == "__main__":
    print("Initiating evaluation pipeline...")
    evaluate_models()
    print("\nEvaluation OS process terminated. RAM fully released.")
