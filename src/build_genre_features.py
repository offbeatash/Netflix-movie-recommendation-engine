import duckdb
import pandas as pd
from sklearn.preprocessing import MultiLabelBinarizer

con = duckdb.connect()

movies = con.sql("""
    SELECT DISTINCT Movie_ID, Title, Genre
    FROM 'data/train.parquet'
    WHERE Genre IS NOT NULL
      AND TRIM(Genre) != 'Unknown'
""").df()

movies["Genre_List"] = movies["Genre"].apply(
    lambda x: [g.strip() for g in x.split(",")]
)

mlb = MultiLabelBinarizer()
genre_matrix = mlb.fit_transform(movies["Genre_List"])

genre_df = pd.DataFrame(
    genre_matrix,
    columns=mlb.classes_
)

genre_df.insert(0, "Movie_ID", movies["Movie_ID"].values)
genre_df.insert(1, "Title", movies["Title"].values)

genre_df.to_parquet(
    "data/movie_genre_features.parquet",
    index=False
)

print(f"Movies: {len(movies)}")
print(f"Genres: {len(mlb.classes_)}")
print("Saved: data/movie_genre_features.parquet")
