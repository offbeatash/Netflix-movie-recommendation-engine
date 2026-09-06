import os
import pandas as pd
from pathlib import Path
from src.config import RAW_DATA_DIR, RAW_PARQUET_PATH

def process_raw_data():
    """Parses raw Netflix text files into a single optimized Parquet file."""
    
    if RAW_PARQUET_PATH.exists():
        print(f"Data already ingested at {RAW_PARQUET_PATH}. Skipping ingestion phase.")
        return
        
    print(f"Initiating raw data ingestion from {RAW_DATA_DIR}...")
    
    data = []
    for file_name in ["combined_data_1.txt", "combined_data_2.txt", "combined_data_3.txt", "combined_data_4.txt"]:
        file_path = RAW_DATA_DIR / file_name
        if not file_path.exists():
            continue
            
        print(f"Processing {file_name}...")
        with open(file_path, "r") as f:
            movie_id = None
            for line in f:
                line = line.strip()
                if line.endswith(":"):
                    movie_id = int(line[:-1])
                else:
                    customer_id, rating, date = line.split(",")
                    data.append([movie_id, int(customer_id), int(rating), date])
                    
    if not data:
        raise FileNotFoundError(f"No raw Netflix .txt files found in {RAW_DATA_DIR}. Please ensure they are downloaded.")
        
    print("Converting raw data to DataFrame...")
    df = pd.DataFrame(data, columns=["Movie_ID", "CustomerID", "Rating", "Date"])
    
    df["Movie_ID"] = df["Movie_ID"].astype("int32")
    df["CustomerID"] = df["CustomerID"].astype("int32")
    df["Rating"] = df["Rating"].astype("int8")
    
    print(f"Saving optimized parquet file to {RAW_PARQUET_PATH}...")
    RAW_PARQUET_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(RAW_PARQUET_PATH, index=False)
    print("Ingestion complete!")

if __name__ == "__main__":
    process_raw_data()