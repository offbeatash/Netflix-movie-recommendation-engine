import gradio as gr
import pandas as pd
import sys
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(project_root))

from src.inference.recommend import generate_genre_recommendations

def interface_wrapper(user_id_str):
    try:
        # Handle empty inputs or invalid characters smoothly
        if not user_id_str or not str(user_id_str).strip():
            user_id = "Unknown"
        else:
            try:
                user_id = int(user_id_str)
            except ValueError:
                user_id = str(user_id_str).strip()
                
        status_msg, df = generate_genre_recommendations(user_id)
        return status_msg, df
    except Exception as e:
        error_msg = f"System Error: {str(e)}"
        return error_msg, pd.DataFrame({"Error": [error_msg]})

demo = gr.Interface(
    fn=interface_wrapper,
    inputs=gr.Textbox(label="Enter Customer ID", placeholder="e.g. 51254 (Leave blank for Popularity Baseline)"),
    outputs=[
        gr.Textbox(label="System Status"),
        gr.Dataframe(label="Top Movie Recommendation per Genre", interactive=False)
    ],
    title="Netflix Movie Recommender",
    description="Enter a User ID for personalized picks, or type an unknown ID to see the Cold Start global baseline."
)

if __name__ == "__main__":
    print("Launching Gradio Server...")
    demo.launch(share=True, inline=False)