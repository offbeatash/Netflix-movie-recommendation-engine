from src.data.ingest import process_raw_data
import sys
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root))


if __name__ == "__main__":
    print("Starting data ingestion process...")
    process_raw_data()
    print("Ingestion OS process finished. RAM fully released.")
