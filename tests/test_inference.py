import pytest
import pandas as pd
from src.inference.recommend import generate_genre_recommendations

def test_cold_start_fallback():
    """Ensures unknown users receive the global popularity baseline without crashing."""
    fake_user = "NON_EXISTENT_USER_999"
    msg, df = generate_genre_recommendations(user_id=fake_user, top_n=2)
    
    assert "not found" in msg.lower(), "Cold start message not triggered."
    assert not df.empty, "Recommendation dataframe is empty."
    assert len(df) > 0, "No recommendations returned for cold start."

def test_prediction_sanity_bounds():
    """Ensures all predicted ratings are mathematically clipped within the 1.0 to 5.0 scale."""
    #Testing with cold start user to verify popularity bounds
    _, df_cold = generate_genre_recommendations(user_id="UNKNOWN_123", top_n=10)
    
    assert df_cold["Predicted Rating"].min() >= 1.0, "Rating fell below minimum bound of 1.0"
    assert df_cold["Predicted Rating"].max() <= 5.0, "Rating exceeded maximum bound of 5.0"

def test_data_schema_types():
    """Validates the output DataFrame strictly matches the required data types for the API."""
    _, df = generate_genre_recommendations(user_id="UNKNOWN_123", top_n=1)
    
    expected_columns = ["Genre", "Movie Title", "Predicted Rating"]
    assert list(df.columns) == expected_columns, "Output schema mismatch."
    
    assert pd.api.types.is_string_dtype(df["Genre"]), "Genre must be string type."
    assert pd.api.types.is_string_dtype(df["Movie Title"]), "Title must be string type."
    assert pd.api.types.is_numeric_dtype(df["Predicted Rating"]), "Rating must be numeric."