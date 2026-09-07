import pytest
from src.inference.recommend import generate_genre_recommendations

def test_cold_start_recommendation():
    """Ensures that an unknown user ID triggers the cold start fallback safely without crashing."""
    fake_user_id = "NON_EXISTENT_USER_999999"
    
    status_msg, results_df = generate_genre_recommendations(fake_user_id, top_n=1)
    
    assert "not found" in status_msg.lower() or "cold start" in status_msg.lower()
    assert not results_df.empty
    assert "Genre" in results_df.columns
    assert "Movie Title" in results_df.columns
    assert "Predicted Rating" in results_df.columns