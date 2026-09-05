import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from tqdm import tqdm
from src.config import RAW_DATA_PATH, PROCESSED_DATA_PATH

def parse_netflix_ratings_to_parquet(chunk_size=200_000):
    """Parse Netflix combined_data file and write it to Parquet in chunks."""
    print(f"Reading raw data from {RAW_DATA_PATH}...")
    
    cols = ["MovieID", "CustomerID", "Rating", "Date"]
    buffer = []
    writer = None

    def flush_buffer():
        nonlocal writer
        if not buffer:
            return

        df_chunk = pd.DataFrame(buffer, columns=cols)
        df_chunk["MovieID"] = df_chunk["MovieID"].astype("int32")
        df_chunk["CustomerID"] = df_chunk["CustomerID"].astype("int32")
        df_chunk["Rating"] = df_chunk["Rating"].astype("int8")
        df_chunk["Date"] = pd.to_datetime(df_chunk["Date"], errors="coerce")

        table = pa.Table.from_pandas(df_chunk, preserve_index=False)
        if writer is None:
            writer = pq.ParquetWriter(str(PROCESSED_DATA_PATH), table.schema, compression='snappy')
        writer.write_table(table)
        buffer.clear()

    current_movie_id = None
    
    try:
        with open(str(RAW_DATA_PATH), 'r', encoding='utf-8') as f:
            for line in tqdm(f, desc="Parsing Netflix ratings", unit="lines"):
                line = line.strip()
                if not line:
                    continue
                if line.endswith(":"):
                    current_movie_id = int(line[:-1])
                    continue

                customer_id, rating, date = line.split(",")
                buffer.append([
                    current_movie_id,
                    int(customer_id),
                    int(rating),
                    date
                ])

                if len(buffer) >= chunk_size:
                    flush_buffer()

        flush_buffer()
        if writer is not None:
            writer.close()
            
        print(f"Success! Data saved to {PROCESSED_DATA_PATH}")
        
    except FileNotFoundError:
        print(f"Error: Could not find raw data at {RAW_DATA_PATH}. Please check your config.py.")