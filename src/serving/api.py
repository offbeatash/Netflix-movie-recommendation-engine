import sys
from pathlib import Path

import gradio as gr
import pandas as pd

project_root = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(project_root))

from src.inference.recommend import generate_genre_recommendations

# Real Customer IDs from the Netflix Prize training dataset
SAMPLE_USERS = [
    822109,
    823519,
    124105,
    1248029,
    1842128,
    2238063,
    1503895,
    2207774,
    2590061,
    2442,
]


def interface_wrapper(user_id_str):
    try:
        # Handle empty input
        if not user_id_str or not str(user_id_str).strip():
            user_id = "Unknown"
        else:
            try:
                user_id = int(user_id_str)
            except (TypeError, ValueError):
                user_id = str(user_id_str).strip()

        # Generate genre-specific recommendations.
        # top_n=5 gives enough candidates to build the Top 10 section.
        status_msg, genre_df = generate_genre_recommendations(
            user_id,
            top_n=5,
        )

        # Build a clean Top 10 unique-movie list.
        # A movie can belong to multiple genres, so remove duplicates here.
        top_10_df = (
            genre_df[["Movie Title", "Predicted Rating"]]
            .drop_duplicates(subset=["Movie Title"])
            .sort_values(
                "Predicted Rating",
                ascending=False,
            )
            .head(10)
            .reset_index(drop=True)
        )

        top_10_df.insert(
            0,
            "Rank",
            range(1, len(top_10_df) + 1),
        )

        return status_msg, top_10_df, genre_df

    except Exception as e:
        error_msg = f"System Error: {str(e)}"

        return (
            error_msg,
            pd.DataFrame({"Error": [error_msg]}),
            pd.DataFrame({"Error": [error_msg]}),
        )


# Main Gradio interface
with gr.Blocks() as demo:

    gr.Markdown("""
        # 🎬 Netflix Movie Recommender

        Enter a valid Netflix Customer ID to generate personalized
        movie recommendations using a collaborative-filtering SVD model.

        **Click one of the sample Customer IDs below, or enter a Customer ID manually.**

        Valid users receive personalized recommendations.
        Leave blank or enter an unknown ID to test the **Cold Start global baseline**.
        """)

    with gr.Row():
        user_id_input = gr.Textbox(
            label="Customer ID",
            placeholder="e.g. 2336536",
            scale=3,
        )

        recommend_button = gr.Button(
            "Get Recommendations",
            variant="primary",
            scale=1,
        )

    gr.Examples(
        examples=[[str(user_id)] for user_id in SAMPLE_USERS],
        inputs=user_id_input,
        label="Try these real Customer IDs",
        examples_per_page=10,
    )

    status_output = gr.Textbox(
        label="System Status",
        interactive=False,
    )

    gr.Markdown("## ⭐ Top 10 Recommendations")

    top_10_output = gr.Dataframe(
        headers=[
            "Rank",
            "Movie Title",
            "Predicted Rating",
        ],
        datatype=[
            "number",
            "str",
            "number",
        ],
        interactive=False,
        wrap=True,
    )

    gr.Markdown("## 🎭 Top Recommendations by Genre")

    genre_output = gr.Dataframe(
        headers=[
            "Genre",
            "Movie Title",
            "Predicted Rating",
        ],
        datatype=[
            "str",
            "str",
            "number",
        ],
        interactive=False,
        wrap=True,
    )

    recommend_button.click(
        fn=interface_wrapper,
        inputs=user_id_input,
        outputs=[
            status_output,
            top_10_output,
            genre_output,
        ],
    )

    # Allow pressing Enter in the Customer ID box.
    user_id_input.submit(
        fn=interface_wrapper,
        inputs=user_id_input,
        outputs=[
            status_output,
            top_10_output,
            genre_output,
        ],
    )


if __name__ == "__main__":
    print("Launching Gradio Server...")
    demo.launch(
        share=True,
        inline=False,
        server_port=7860,
    )
