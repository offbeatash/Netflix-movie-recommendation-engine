import pandas as pd
from src.config import DATA_DIR, PROCESSED_DATA_PATH

def process_raw_data():
    """Parses raw Netflix text files into a single optimized Parquet file."""
    
    if PROCESSED_DATA_PATH.exists():
        print(f"Data already ingested at {PROCESSED_DATA_PATH}. Skipping ingestion phase.")
        return
        
    print(f"Initiating raw data ingestion from {DATA_DIR}...")
    
    data = []
    for file_name in ["combined_data_1.txt", "combined_data_2.txt", "combined_data_3.txt", "combined_data_4.txt"]:
        file_path = DATA_DIR / file_name
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
        raise FileNotFoundError(f"No raw Netflix .txt files found in {DATA_DIR}. Please ensure they are downloaded.")
        
    print("Converting raw data to DataFrame...")
    df = pd.DataFrame(data, columns=["Movie_ID", "CustomerID", "Rating", "Date"])
    
    df["Movie_ID"] = df["Movie_ID"].astype("int32")
    df["CustomerID"] = df["CustomerID"].astype("int32")
    df["Rating"] = df["Rating"].astype("int8")
    df["Date"] = pd.to_datetime(df["Date"])
    
    print(f"Saving optimized parquet file to {PROCESSED_DATA_PATH}...")
    PROCESSED_DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(PROCESSED_DATA_PATH, index=False)
    print("Ingestion complete!")

if __name__ == "__main__":
    process_raw_data()