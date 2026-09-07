from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from src.inference.recommend import generate_genre_recommendations

#Initialize API
app = FastAPI(
    title="Netflix Recommendation Engine",
    description="Production API for collaborative filtering and popularity-based movie recommendations.",
    version="1.0.0"
)

#Define request schema
class RecommendationRequest(BaseModel):
    user_id: str
    top_n: int = 5

@app.get("/health")
def health_check():
    """Liveness probe for container orchestration."""
    return {"status": "healthy", "service": "recommendation-api"}

@app.post("/recommend")
def get_recommendations(request: RecommendationRequest):
    """Generates top-N movie recommendations per genre for a given user."""
    try:
        status_msg, results_df = generate_genre_recommendations(
            user_id=request.user_id, 
            top_n=request.top_n
        )
        
        return {
            "status_message": status_msg,
            #Convert pd DF to a JSON-safe dictionary
            "recommendations": results_df.to_dict(orient="records")
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference error: {str(e)}")