import pandas as pd
import requests
import concurrent.futures
import re
import time
from tqdm import tqdm
from src.config import MOVIE_TITLES_PATH, ENRICHED_MOVIES_PATH, TMDB_API_KEY

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
    title = re.sub(r":\s*Bonus Material.*", "", title, flags=re.IGNORECASE)
    title = re.sub(r":\s*Special Edition.*", "", title, flags=re.IGNORECASE)
    title = re.sub(r"\(Widescreen\)", "", title, flags=re.IGNORECASE)
    title = re.sub(r"\bSeason\s+\d+\b", "", title, flags=re.IGNORECASE)
    title = re.sub(r"\bVol\.?\s*\d+\b", "", title, flags=re.IGNORECASE)
    title = re.sub(r"\bVolume\s+\d+\b", "", title, flags=re.IGNORECASE)
    title = re.sub(r"\bPart\s+\d+\b", "", title, flags=re.IGNORECASE)
    title = re.sub(r"\bSet\s+\d+\b", "", title, flags=re.IGNORECASE)
    title = re.sub(r"\bDisc\s+\d+\b", "", title, flags=re.IGNORECASE)
    title = re.sub(r"\bDouble Feature\b", "", title, flags=re.IGNORECASE)
    title = re.sub(r"\s+", " ", title).strip(" :-")
    return title


def get_movie_genres(title, year, token, max_retries=3):
    """Fetches genres from TMDb API for a given title and year."""
    headers = {"accept": "application/json", "Authorization": f"Bearer {token}"}
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
                elif response.status_code == 429:
                    time.sleep(1.5)
                    continue
                else:
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
        selected = max(results, key=lambda x: x.get("popularity", 0))

    genres = [genre_map.get(gid, str(gid)) for gid in selected.get("genre_ids", [])]
    return ", ".join(genres) if genres else "Unknown"


def process_enrichment():
    """Main execution function to load, enrich, and save movies
    with resumable capability."""
    # Load existing enrichment data if available
    if ENRICHED_MOVIES_PATH.exists():
        print(f"Loading existing enrichment data from {ENRICHED_MOVIES_PATH}...")
        movies_df = pd.read_csv(ENRICHED_MOVIES_PATH)
        # Ensure required columns exist
        if "Genre" not in movies_df.columns:
            movies_df["Genre"] = None
        # Ensure consistent dtypes with initial dataframe creation
        movies_df["Movie_ID"] = movies_df["Movie_ID"].astype("int32")
        movies_df["Year"] = pd.to_numeric(movies_df["Year"], errors="coerce")
    else:
        print(f"Loading raw movie metadata from {MOVIE_TITLES_PATH}...")
        movies: list[list[str | None]] = []
        with open(MOVIE_TITLES_PATH, encoding="latin-1") as f:
            # Skip header line
            f.readline()
            for line in f:
                parts = line.strip().split(",", 2)
                if len(parts) == 3:
                    movies.append([parts[0], parts[1], parts[2]])
                elif len(parts) == 2:
                    movies.append([parts[0], None, parts[1]])

        movies_df = pd.DataFrame(movies, columns=["Movie_ID", "Year", "Title"])
        movies_df["Movie_ID"] = movies_df["Movie_ID"].astype("int32")
        movies_df["Year"] = pd.to_numeric(movies_df["Year"], errors="coerce")
        movies_df["Genre"] = None

    # Identify movies that still need enrichment
    # A movie needs enrichment if:
    # 1. Genre is None/NaN, OR
    # 2. Genre is "Unknown" (failed previous attempt), OR
    # 3. Genre is empty string
    mask = (
        movies_df["Genre"].isna()
        | (movies_df["Genre"] == "Unknown")
        | (movies_df["Genre"] == "")
    )
    missing_idx = movies_df[mask].index.tolist()

    if len(missing_idx) == 0:
        print("All movies already enriched. Skipping API calls.")
        return movies_df

    print(
        f"Found {len(missing_idx)} movies needing enrichment "
        f"out of {len(movies_df)} total."
    )

    def fetch_worker(idx):
        title = movies_df.at[idx, "Title"]
        year = movies_df.at[idx, "Year"]
        return idx, get_movie_genres(title, year, TMDB_API_KEY)

    batch_size = 500
    for start in range(0, len(missing_idx), batch_size):
        batch_indices = missing_idx[start : start + batch_size]
        print(
            f"Processing batch {start} to {start + len(batch_indices)} "
            f"({len(batch_indices)} movies)..."
        )

        with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
            results = list(
                tqdm(
                    executor.map(fetch_worker, batch_indices), total=len(batch_indices)
                )
            )

        # Update results and handle any failures gracefully
        for idx, genre in results:
            movies_df.at[idx, "Genre"] = genre

        # Save incrementally after each batch to prevent losing progress
        movies_df.to_csv(ENRICHED_MOVIES_PATH, index=False)
        print(
            f"  Batch saved. Progress: "
            f"{len(movies_df) - len(missing_idx) + start + len(batch_indices)}"
            f"/{len(movies_df)} movies enriched."
        )

    print(f"Enrichment complete! Saved to {ENRICHED_MOVIES_PATH}")
    return movies_df
