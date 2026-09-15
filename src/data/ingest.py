import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from src.config import DATA_DIR, PROCESSED_DATA_PATH


RAW_FILE = "combined_data_1.txt"
BATCH_SIZE = 100_000


def _write_batch(data, writer):
    """Convert one bounded batch of ratings to Parquet."""
    if not data:
        return writer

    chunk_df = pd.DataFrame(
        data,
        columns=["Movie_ID", "CustomerID", "Rating", "Date"],
    )

    chunk_df["Movie_ID"] = chunk_df["Movie_ID"].astype("int32")
    chunk_df["CustomerID"] = chunk_df["CustomerID"].astype("int32")
    chunk_df["Rating"] = chunk_df["Rating"].astype("int8")
    chunk_df["Date"] = pd.to_datetime(chunk_df["Date"])

    table = pa.Table.from_pandas(
        chunk_df,
        preserve_index=False,
    )

    if writer is None:
        writer = pq.ParquetWriter(
            PROCESSED_DATA_PATH,
            table.schema,
        )

    writer.write_table(table)

    return writer


def process_raw_data():
    """Streams the raw Netflix ratings file into an optimized Parquet file."""

    if PROCESSED_DATA_PATH.exists():
        print(
            f"Data already ingested at {PROCESSED_DATA_PATH}. "
            "Skipping ingestion phase."
        )
        return

    file_path = DATA_DIR / RAW_FILE

    if not file_path.exists():
        raise FileNotFoundError(
            f"No raw Netflix file found at {file_path}. "
            "Please ensure it is downloaded."
        )

    print(f"Initiating raw data ingestion from {file_path}...")

    PROCESSED_DATA_PATH.parent.mkdir(parents=True, exist_ok=True)

    writer = None
    data = []
    total_rows = 0
    movie_id = None

    try:
        with open(file_path, "r", encoding="latin-1") as f:
            for line in f:
                line = line.strip()

                if not line:
                    continue

                if line.endswith(":"):
                    movie_id = int(line[:-1])
                    continue

                customer_id, rating, date = line.split(",")

                data.append(
                    [
                        movie_id,
                        int(customer_id),
                        int(rating),
                        date,
                    ]
                )

                if len(data) >= BATCH_SIZE:
                    writer = _write_batch(data, writer)
                    total_rows += len(data)
                    data.clear()

        if data:
            writer = _write_batch(data, writer)
            total_rows += len(data)
            data.clear()

        if writer is None:
            raise ValueError(f"No rating records found in {file_path}.")

        print(
            f"Ingestion complete! Wrote {total_rows:,} rows "
            f"to {PROCESSED_DATA_PATH}."
        )

    finally:
        if writer is not None:
            writer.close()
