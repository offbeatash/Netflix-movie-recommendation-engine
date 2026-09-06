# Netflix Movie Recommendation Engine

An end-to-end, high-performance recommendation pipeline trained on 18M+ ratings from the Netflix Prize dataset. This engine features customized data loading configurations, hyperparameter optimization via `GridSearchCV`, an engineered Blended Hybrid Ensemble, and an interactive frontend interface deployed via Gradio.

---

## 🚀 Key Architectural Features & Engineering Highlights

* **Memory-Optimized Out-of-Core Processing:** Designed a lean data pipeline to bypass standard Pandas data duplication overhead, successfully training large matrices inside a restricted cloud container environment without a fallback disk-swap configuration.
* **Linux OS Memory Flush System:** Leveraged lower-level `ctypes` bindings to force immediate `malloc_trim(0)` garbage collection allocations via the Linux kernel, minimizing RAM fragmentation between heavy model training steps.
* **Hybrid Blended Ensemble:** Engineered a linear combination model that strategically unites explicit rating factorization (SVD) and implicit interaction signals (ALS) tuned via a held-out validation split, successfully reducing individual model noise.
* **Gradio User Interface:** Deployed a functional browser UI that allows real-time user-driven prediction execution and returns sorted Top-N personalized recommendations across distinct movie genres.

---

## 📊 Evaluation & Model Showdown

The engine evaluates recommendations using explicit rating tracking metrics: **Root Mean Square Error (RMSE)** and **Mean Absolute Error (MAE)**. 

### Performance Metrics Table
| Model | Test RMSE | Test MAE | Computational Strategy |
|---|---|---|---|
| Naive Mean Baseline | 1.0841 | 0.9183 | Global Average Rating Imputation |
| Model A — Popularity Heuristic | 1.0371 | 0.8379 | Non-personalized Volume Ranking Baseline (Train-isolated) |
| Model B — ALS (Implicit) | 2.8969 | 2.6889 | Alternating Least Squares via `implicit` |
| Model C — Tuned SVD | 0.9945 | 0.7868 | Latent-Factor Explicit Matrix Factorization |
| **Model D — Hybrid Ensemble** | **0.9941** | **0.7888** | **Linear Weight Blend Optimization (ALS + SVD)** |

### 💡 Key Data Science Insight: The Implicit Feedback Trap
In our evaluation showdown, **Model B (ALS)** yields a disproportionately high error score (2.8969). This is a deliberate demonstration of algorithm selection criteria:
* The `implicit` library treats unobserved user-movie cells as explicit **0s** (representing a dislike signal). Because our structural interaction matrix is ~99% sparse, the factor optimization collapses toward 0.
* When forced into a 1-to-5 star metric evaluation via clipping, its absolute output defaults entirely to 1.0, failing rating prediction tasks.
* **Conclusion:** ALS is a **Top-N Ranking algorithm** (designed to uncover *what* item a user will engage with next) whereas SVD is a **Rating Prediction algorithm** (designed to guess an exact star score).

---

## 🛠️ Hyperparameter Fine-Tuning
Using parallel worker processing (GridSearchCV), I performed five stages of targeted cross-validation to progressively refine the SVD hyperparameter search space. The final search achieved the best CV RMSE of 0.9647, with the selected configuration:
* `n_factors`: 50
* `n_epochs`: 20
* `lr_all` (Learning Rate): 0.005
* `reg_all` (Regularization Penalty): 0.04

---

## 💻 Tech Stack & Core Libraries
* **Core Frameworks:** Python, Modular Script Architecture
* **Matrix Operations & Math:** `scipy.sparse`, `numpy`, `pandas`, `pyarrow`
* **Recommendation Backends:** `scikit-surprise` (SVD), `implicit` (ALS)
* **Performance Tracking & UI:** `scikit-learn`, `ctypes`, `gradio`

-------------------------------------------------------------------------------------------------------------

## 🏃‍♂️ How to Setup and Run the Pipeline

## Setup & Usage

> **Recommended:** A machine with at least **4 CPU cores and 16 GB RAM** is recommended. Alternatively, you can run this project directly using **GitHub Codespaces**.

### 1. Clone the Repository

Clone the repository and navigate into the project directory:

```bash
git clone https://github.com/offbeatash/Netflix-movie-recommendation-engine.git
cd Netflix-movie-recommendation-engine
```

### 2. Environment & API Setup

Install the required Python dependencies:

```bash
pip install -r requirements.txt
```

Duplicate the example environment file:

```bash
cp .env.example .env
```

Open the newly created `.env` file in your editor and replace:

```text
your_tmdb_api_key_here
```

with your actual **TMDb API key**.

For instructions on obtaining and authenticating your TMDb API key, see the official documentation:

**TMDb:** [Authentication Documentation](https://developer.themoviedb.org/reference/authentication)

### 3. Download the Raw Dataset

Due to GitHub's file size limitations, the massive raw **Netflix Prize dataset** is hosted externally.

Download the raw dataset archive from:

**[📥 Download Raw Netflix Dataset — Google Drive](https://drive.google.com/drive/folders/1eldCFc5M0ElypZ9fU_jz4OpXD_-AQEUi?usp=sharing)**

After downloading, extract the contents and move or upload them into the project's `data/` directory.

Your project structure should look approximately like:

```text
Netflix-movie-recommendation-engine/
├── data/
│   └── <Netflix dataset files>
├── src/
├── .env
├── .env.example
├── Makefile
├── requirements.txt
└── README.md
```

### 4. Execute the ML Pipeline

Once the dataset is in place and your TMDb API key has been configured, use the `Makefile` to run the end-to-end machine learning pipeline.

#### Run the complete pipeline

This performs data ingestion, enrichment, data splitting, and model training:

```bash
make all
```

#### Evaluate the trained model

Verify the evaluation metrics on the test split:

```bash
make evaluate
```

#### Launch the Interactive Web Server

Launch the interactive **Gradio Web Server**:

```bash
make app
```

The application will start locally and provide a URL that you can open in your browser.
e
Note for VS Code / GitHub Codespaces users: The application automatically handles connection port-forwarding. Once executed via make serve, check your editor’s Ports tab to view your active local hosting address.
