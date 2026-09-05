import sys
from pathlib import Path

# Resolve project root dynamically
project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root))

from src.data.enrich_genres import process_enrichment

if __name__ == "__main__":
    print("Starting genre enrichment process...")
    movies_df = process_enrichment()
    
    if movies_df is None or movies_df.empty:
        print("Enrichment failed or returned empty data.")
    else:
        print(f"Enrichment OS process finished. {len(movies_df)} movies processed.")