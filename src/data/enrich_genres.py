import concurrent.futures
import json
import re
import time

import pandas as pd
import requests
from tqdm import tqdm

from src.config import (
    ENRICHED_MOVIES_PATH,
    MOVIE_TITLES_PATH,
    TMDB_API_KEY,
)

ENRICHMENT_METADATA_PATH = ENRICHED_MOVIES_PATH.with_suffix(".enrichment_metadata.json")

genre_map = {
    28: "Action",
    12: "Adventure",
    16: "Animation",
    35: "Comedy",
    80: "Crime",
    99: "Documentary",
    18: "Drama",
    10751: "Family",
    14: "Fantasy",
    36: "History",
    27: "Horror",
    10402: "Music",
    9648: "Mystery",
    10749: "Romance",
    878: "Science Fiction",
    10770: "TV Movie",
    53: "Thriller",
    10752: "War",
    37: "Western",
    10759: "Action & Adventure",
    10762: "Kids",
    10763: "News",
    10764: "Reality",
    10765: "Sci-Fi & Fantasy",
    10766: "Soap",
    10767: "Talk",
    10768: "War & Politics",
}


def clean_title(title):
    """Cleans movie titles for better TMDb API matching."""
    title = re.sub(
        r":\s*\*Bonus Material.*",
        "",
        title,
        flags=re.IGNORECASE,
    )
    title = re.sub(
        r":\s*\*Special Edition.*",
        "",
        title,
        flags=re.IGNORECASE,
    )
    title = re.sub(
        r"\(**Widescreen\)**",
        "",
        title,
        flags=re.IGNORECASE,
    )
    title = re.sub(
        r"\bSeason\s+\d+\b",
        "",
        title,
        flags=re.IGNORECASE,
    )
    title = re.sub(
        r"\bVol\.?\s*\d+\b",
        "",
        title,
        flags=re.IGNORECASE,
    )
    title = re.sub(
        r"\bVolume\s+\d+\b",
        "",
        title,
        flags=re.IGNORECASE,
    )
    title = re.sub(
        r"\bPart\s+\d+\b",
        "",
        title,
        flags=re.IGNORECASE,
    )
    title = re.sub(
        r"\bSet\s+\d+\b",
        "",
        title,
        flags=re.IGNORECASE,
    )
    title = re.sub(
        r"\bDisc\s+\d+\b",
        "",
        title,
        flags=re.IGNORECASE,
    )
    title = re.sub(
        r"\bDouble Feature\b",
        "",
        title,
        flags=re.IGNORECASE,
    )
    title = re.sub(r"\s+", " ", title).strip(" :-")
    return title


def get_movie_genres(title, year, token, max_retries=3):
    """Fetches genres from TMDb API for a given title and year."""
    headers = {
        "accept": "application/json",
        "Authorization": f"Bearer {token}",
    }

    cleaned_title = clean_title(title)

    def search(endpoint, year_filter=None):
        params = {"query": cleaned_title}

        if pd.notna(year_filter):
            params["year" if endpoint == "movie" else "first_air_date_year"] = int(
                year_filter
            )

        for attempt in range(max_retries):
            try:
                response = requests.get(
                    f"https://api.themoviedb.org/3/search/{endpoint}",
                    headers=headers,
                    params=params,
                    timeout=10,
                )

                if response.status_code == 200:
                    return response.json().get("results", [])

                if response.status_code == 429:
                    time.sleep(1.5)
                    continue

                return []

            except Exception:
                if attempt == max_retries - 1:
                    return []

                time.sleep(1)

        return []

    results = (
        search("movie", year) or search("movie") or search("tv", year) or search("tv")
    )

    if not results:
        return "Unknown"

    selected = None

    if pd.notna(year):
        target_year = str(int(year))

        for item in results:
            release = item.get("release_date") or item.get("first_air_date") or ""

            if release.startswith(target_year):
                selected = item
                break

    if selected is None:
        selected = max(
            results,
            key=lambda x: x.get("popularity", 0),
        )

    genres = [genre_map.get(gid, str(gid)) for gid in selected.get("genre_ids", [])]

    return ", ".join(genres) if genres else "Unknown"


def _save_enrichment_metadata(movies_df):
    """Save metadata describing the completed enrichment result."""
    metadata = {
        "status": "complete",
        "total_movies": int(len(movies_df)),
        "enriched_movies": int(
            (
                movies_df["Genre"].notna()
                & (movies_df["Genre"] != "")
                & (movies_df["Genre"] != "Unknown")
            ).sum()
        ),
        "unknown_movies": int((movies_df["Genre"] == "Unknown").sum()),
    }

    ENRICHMENT_METADATA_PATH.write_text(
        json.dumps(metadata, indent=2) + "\n",
        encoding="utf-8",
    )

    return metadata


def process_enrichment(retry_unknown=False):
    """Load, enrich, and save movie genres with resumable capability."""

    # If enrichment was already completed, never repeat API calls.
    # This works even when TMDB_API_KEY is not configured.
    if (
        ENRICHED_MOVIES_PATH.exists()
        and ENRICHMENT_METADATA_PATH.exists()
        and not retry_unknown
    ):
        print(
            "Genre enrichment already completed. "
            f"Metadata found at {ENRICHMENT_METADATA_PATH}."
        )
        print("Skipping enrichment phase.")

        return pd.read_csv(ENRICHED_MOVIES_PATH)

    # ACTUAL ENRICHMENT / EXPLICIT RETRY
    if not TMDB_API_KEY:
        raise RuntimeError(
            "TMDB_API_KEY is not configured. "
            "Set TMDB_API_KEY in .env before running or retrying "
            "genre enrichment."
        )

    # LOAD EXISTING ENRICHMENT DATA
    if ENRICHED_MOVIES_PATH.exists():
        print(f"Loading existing enrichment data from " f"{ENRICHED_MOVIES_PATH}...")

        movies_df = pd.read_csv(ENRICHED_MOVIES_PATH)

        if "Genre" not in movies_df.columns:
            movies_df["Genre"] = None

        movies_df["Movie_ID"] = movies_df["Movie_ID"].astype("int32")
        movies_df["Year"] = pd.to_numeric(
            movies_df["Year"],
            errors="coerce",
        )

    else:
        print(f"Loading raw movie metadata from " f"{MOVIE_TITLES_PATH}...")

        movies: list[list[str | None]] = []

        with open(MOVIE_TITLES_PATH, encoding="latin-1") as f:
            # Skip header line.
            f.readline()

            for line in f:
                parts = line.strip().split(",", 2)

                if len(parts) == 3:
                    movies.append([parts[0], parts[1], parts[2]])

                elif len(parts) == 2:
                    movies.append([parts[0], None, parts[1]])

        movies_df = pd.DataFrame(
            movies,
            columns=["Movie_ID", "Year", "Title"],
        )

        movies_df["Movie_ID"] = movies_df["Movie_ID"].astype("int32")

        movies_df["Year"] = pd.to_numeric(
            movies_df["Year"],
            errors="coerce",
        )

        movies_df["Genre"] = None

    # IDENTIFY MOVIES REQUIRING ENRICHMENT
    mask = (
        movies_df["Genre"].isna()
        | (movies_df["Genre"] == "Unknown")
        | (movies_df["Genre"] == "")
    )

    missing_idx = movies_df[mask].index.tolist()

    if len(missing_idx) == 0:
        print("All movies already enriched. Skipping API calls.")

        _save_enrichment_metadata(movies_df)

        return movies_df

    print(
        f"Found {len(missing_idx)} movies needing enrichment "
        f"out of {len(movies_df)} total."
    )

    def fetch_worker(idx):
        title = movies_df.at[idx, "Title"]
        year = movies_df.at[idx, "Year"]

        return idx, get_movie_genres(
            title,
            year,
            TMDB_API_KEY,
        )

    batch_size = 500

    for start in range(
        0,
        len(missing_idx),
        batch_size,
    ):
        batch_indices = missing_idx[start : start + batch_size]

        print(
            f"Processing batch {start} to "
            f"{start + len(batch_indices)} "
            f"({len(batch_indices)} movies)..."
        )

        with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
            results = list(
                tqdm(
                    executor.map(
                        fetch_worker,
                        batch_indices,
                    ),
                    total=len(batch_indices),
                )
            )

        # Update results.
        for idx, genre in results:
            movies_df.at[idx, "Genre"] = genre

        # Save incrementally after every batch.
        movies_df.to_csv(
            ENRICHED_MOVIES_PATH,
            index=False,
        )

        print(
            "  Batch saved. Progress: "
            f"{len(movies_df) - len(missing_idx) + start + len(batch_indices)}"
            f"/{len(movies_df)} movies enriched."
        )

    metadata = _save_enrichment_metadata(movies_df)

    print(f"Enrichment complete! " f"Saved to {ENRICHED_MOVIES_PATH}")

    print(f"Enrichment metadata saved to " f"{ENRICHMENT_METADATA_PATH}")

    print(f"Total movies: {metadata['total_movies']}")

    print(f"Successfully enriched: " f"{metadata['enriched_movies']}")

    print(f"Unknown genres: " f"{metadata['unknown_movies']}")

    return movies_df
