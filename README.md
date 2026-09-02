# Netflix Movie Recommendation Engine

An end-to-end, high-performance recommendation pipeline trained on 18M+ ratings from the Netflix Prize dataset. This engine features customized data loading configurations, hyperparameter optimization via `GridSearchCV`, an engineered Blended Hybrid Ensemble, and an interactive frontend interface deployed via Gradio.

----

## 🚀 Key Architectural Features & Engineering Highlights

* **Memory-Optimized Out-of-Core Processing:** Designed a lean data pipeline to bypass standard Pandas data duplication overhead, successfully training large matrices inside a restricted 16GB RAM cloud container environment without a fallback disk-swap configuration.
* **Linux OS Memory Flush System:** Leveraged lower-level `ctypes` bindings to force immediate `malloc_trim(0)` garbage collection allocations via the Linux kernel, minimizing RAM fragmentation between training cells.
* **Hybrid Blended Ensemble:** Engineered a linear combination model that strategically unites explicit rating factorization (SVD) and implicit interaction signals (ALS), successfully reducing individual model noise and lowering overall test error.
* **Gradio User Interface:** Deployed a functional browser UI that allows real-time user-driven prediction execution and returns sorted Top-N personalized recommendations across distinct movie genres.

---

## 📊 Evaluation & Model Showdown

The engine evaluates recommendations using explicit rating tracking metrics: **Root Mean Square Error (RMSE)** and **Mean Absolute Error (MAE)**. 

### Performance Metrics Table
| Model | Test RMSE | Test MAE | Computational Strategy |
|---|---|---|---|
| **Model D — Hybrid Ensemble** | **0.9821** | **0.7710** | **Linear Weight Blend Optimization (ALS + SVD)** |
| Model C — Tuned SVD | 0.9927 | 0.7814 | Latent-Factor Explicit Matrix Factorization |
| Model A — Popularity Heuristic | 1.0300 | 0.7514 | Non-personalized Volume Ranking Baseline |
| Naive Mean Baseline | 1.0879 | 0.9213 | Global Average Rating Imputation (~3.52 Stars) |
| Model B — ALS (Implicit) | 2.8954 | 2.6856 | Alternating Least Squares via `implicit` |

### 💡 Key Data Science Insight: The Implicit Feedback Trap
In our evaluation showdown, **Model B (ALS)** yields a disproportionately high error score (2.8954). This is a deliberate demonstration of algorithm selection criteria:
* The `implicit` library treats unobserved user-movie cells as explicit **0s** (representing a dislike signal). Because our structural interaction matrix is ~99% sparse, the factor optimization collapses toward 0.
* When forced into a 1-to-5 star metric evaluation via clipping, its absolute output defaults entirely to 1.0, failing rating prediction tasks.
* **Conclusion:** ALS is a **Top-N Ranking algorithm** (designed to uncover *what* item a user will engage with next) whereas SVD is a **Rating Prediction algorithm** (designed to guess an exact star score).

---

## 🛠️ Hyperparameter Fine-Tuning
Using parallel worker processing (`GridSearchCV`), we constrained cross-validation structures to find the mathematically optimal hyperparameter bounds for explicit item convergence:
* `n_factors`: 50
* `n_epochs`: 20
* `lr_all` (Learning Rate): 0.005
* `reg_all` (Regularization Penalty): 0.04

---

## 💻 Tech Stack & Core Libraries
* **Core Frameworks:** Python, Jupyter Notebooks
* **Matrix Operations & Math:** `scipy.sparse`, `numpy`, `pandas`
* **Recommendation Backends:** `scikit-surprise` (SVD), `implicit` (ALS)
* **Performance Tracking & UI:** `scikit-learn`, `psutil`, `gradio`

----

## 🏃‍♂️ How to Setup and Run Local Server

```bash
1. Clone the Repository
git clone [https://github.com/offbeatash/Netflix-movie-recommendation-engine.git](https://github.com/offbeatash/Netflix-movie-recommendation-engine.git)
cd Netflix-movie-recommendation-engine
2. Install Dependencies
Bash
pip install pandas numpy scipy scikit-surprise implicit sklearn gradio psutil
3. Run the UI Application
Open the primary notebook file or extract the interface cell to execute the application server:

Bash
python app.py
Note for VS Code / GitHub Codespaces users: The application automatically handles connection port-forwarding. Once executed, check your editor’s Ports tab to view your active local hosting address or use the shared public live link provided in the log history.
