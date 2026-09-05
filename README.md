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
| **Model D — Hybrid Ensemble** | **0.9941** | **0.7888** | **Linear Weight Blend Optimization (ALS + SVD)** |
| Model C — Tuned SVD | 0.9945 | 0.7868 | Latent-Factor Explicit Matrix Factorization |
| Model A — Popularity Heuristic | 1.0371 | 0.8379 | Non-personalized Volume Ranking Baseline (Train-isolated) |
| Naive Mean Baseline | 1.0841 | 0.9183 | Global Average Rating Imputation |
| Model B — ALS (Implicit) | 2.8969 | 2.6889 | Alternating Least Squares via `implicit` |

### 💡 Key Data Science Insight: The Implicit Feedback Trap
In our evaluation showdown, **Model B (ALS)** yields a disproportionately high error score (2.8969). This is a deliberate demonstration of algorithm selection criteria:
* The `implicit` library treats unobserved user-movie cells as explicit **0s** (representing a dislike signal). Because our structural interaction matrix is ~99% sparse, the factor optimization collapses toward 0.
* When forced into a 1-to-5 star metric evaluation via clipping, its absolute output defaults entirely to 1.0, failing rating prediction tasks.
* **Conclusion:** ALS is a **Top-N Ranking algorithm** (designed to uncover *what* item a user will engage with next) whereas SVD is a **Rating Prediction algorithm** (designed to guess an exact star score).

---

## 🛠️ Hyperparameter Fine-Tuning
Using parallel worker processing (GridSearchCV), I performed five stages of targeted cross-validation to progressively refine the SVD hyperparameter search space. The final search achieved the best CV RMSE of 0.9647, with the selected configuration:
* `n_factors`: 10
* `n_epochs`: 30
* `lr_all` (Learning Rate): 0.005
* `reg_all` (Regularization Penalty): 0.08

---

## 💻 Tech Stack & Core Libraries
* **Core Frameworks:** Python, Modular Script Architecture
* **Matrix Operations & Math:** `scipy.sparse`, `numpy`, `pandas`, `pyarrow`
* **Recommendation Backends:** `scikit-surprise` (SVD), `implicit` (ALS)
* **Performance Tracking & UI:** `scikit-learn`, `ctypes`, `gradio`

---

## 🏃‍♂️ How to Setup and Run the Pipeline

```bash
# 1. Clone the Repository
git clone [https://github.com/offbeatash/Netflix-movie-recommendation-engine.git](https://github.com/offbeatash/Netflix-movie-recommendation-engine.git)
cd Netflix-movie-recommendation-engine

# 2. Install Dependencies
pip install -r requirements.txt
# (or manually: pip install pandas numpy scipy scikit-surprise implicit scikit-learn gradio pyarrow)

# 3. Run the Complete Data & Training Pipeline via Makefile
make all

# 4. Run the Evaluation Showdown to verify metrics
make evaluate

# 5. Launch the Interactive Gradio Web Server
make serve