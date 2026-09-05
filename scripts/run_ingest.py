import sys
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root))

from src.data.ingest import parse_netflix_ratings_to_parquet

if __name__ == "__main__":
    print("Starting data ingestion process...")
    parse_netflix_ratings_to_parquet()
    print("Ingestion OS process finished. RAM fully released.")