import gradio as gr
import pandas as pd
import sys
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(project_root))

from src.inference.recommend import generate_genre_recommendations

def interface_wrapper(user_id_str):
    try:
        user_id = int(user_id_str)
        return generate_genre_recommendations(user_id)
    except ValueError as e:
        return pd.DataFrame({"Error": [str(e)]})
    except Exception as e:
        return pd.DataFrame({"Error": [f"System Error: {str(e)}"]})

demo = gr.Interface(
    fn=interface_wrapper,
    inputs=gr.Textbox(label="Enter Customer ID", placeholder="e.g. 51254"),
    outputs=gr.Dataframe(label="Top Movie Recommendation per Genre", interactive=False),
    title="Netflix Movie Recommender",
    description="Enter a User ID to get the best movie recommendation per genre."
)

if __name__ == "__main__":
    print("Launching Gradio Server...")
    demo.launch(share=True, inline=False)