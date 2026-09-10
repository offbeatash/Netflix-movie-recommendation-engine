# Netflix Movie Recommendation Engine

An end-to-end, production-oriented ML system trained on the Netflix Prize dataset. The project includes data ingestion, TMDb genre enrichment, feature generation, popularity-based recommendations, collaborative filtering, model evaluation, and interactive serving APIs.

## Features

* Parses Netflix Prize rating files into optimized Parquet data.
* Enriches movie metadata with genres from TMDb.
* Builds chronological train, validation, and test splits.
* Prevents temporal leakage by calculating user and movie activity thresholds using only the training period.
* Trains:

  * Popularity baseline
  * Implicit ALS model
  * Explicit SVD model
  * SVD + popularity blended ensemble
* Selects SVD hyperparameters through prior offline hyperparameter experimentation.
* Evaluates models with RMSE and MAE.
* Uses validation-tuned SVD + popularity ensemble weights during serving.
* Uses popularity-based recommendations as the cold-start fallback for unknown users.
* Provides:

  * A Gradio interface for interactive recommendations
  * A FastAPI service with health checks, readiness checks, metrics, and JSON recommendations
* Stores experiment results locally with MLflow.
* Includes Docker support for the FastAPI service.
* Includes automated tests for the API, inference behavior, recommendation logic, and data-pipeline leakage prevention.

## Project Structure

```text
.
├── artifacts/              # Trained model artifacts
├── data/                   # Raw, processed, and enriched datasets
├── mlruns/                 # Local MLflow tracking data
├── notebooks/              # Exploratory analysis and model-development work
├── scripts/                # Pipeline entry points
├── src/
│   ├── data/               # Ingestion, enrichment, and feature building
│   ├── evaluation/         # Evaluation utilities
│   ├── inference/          # Recommendation generation
│   ├── models/             # Popularity, ALS, SVD, and ensemble models
│   └── serving/            # Gradio and FastAPI applications
├── tests/                  # Automated tests
├── Dockerfile
├── Makefile
├── requirements.txt
└── README.md
```

## Requirements

* Python 3.10 through 3.14
* At least 16 GB RAM recommended for the full dataset pipeline
* At least 4 CPU cores recommended for training
* A TMDb API key if genre enrichment must be performed

The project can also be run in GitHub Codespaces or with Docker.

## Installation

Clone the repository:

```bash
git clone https://github.com/offbeatash/Netflix-movie-recommendation-engine.git
cd Netflix-movie-recommendation-engine
```

### Create a virtual environment

#### Windows PowerShell

```powershell
py -3.13 -m venv .venv
.\.venv\Scripts\Activate.ps1
```

#### Linux and macOS

```bash
python3.13 -m venv .venv
source .venv/bin/activate
```

Install dependencies:

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

For tests, notebooks, and development tools, install the additional dependencies
from `requirements-dev.txt` after activating the environment:

```bash
python -m pip install -r requirements-dev.txt
```

## Configuration

Copy the example environment file:

```bash
cp .env.example .env
```

Set your TMDb API key in `.env`:

```text
TMDB_API_KEY=your_tmdb_api_key_here
```

TMDb authentication documentation:

https://developer.themoviedb.org/reference/authentication

A TMDb key is only required when `data/movies_with_genres.csv` does not already exist or needs to be regenerated.

## Dataset and Pretrained Artifacts

The full dataset and trained artifacts are not committed to Git because of their size.

For evaluation or immediate use, prepared `data/` and `artifacts/` directories are available via the link below. However, for true reproducibility and production use, we recommend generating these artifacts yourself by running the complete pipeline.

[Download sample dataset and artifacts](https://drive.google.com/drive/folders/1eldCFc5M0ElypZ9fU_jz4OpXD_-AQEUi?usp=sharing)

If using the downloaded artifacts, place them in the repository as:

```text
data/
├── train.parquet
└── movies_with_genres.csv

artifacts/
├── popularity_model.pkl
├── svd_model.pkl
└── ensemble_weights.json
```

> **Note**: While this repository implements production-quality ML practices (temporal leakage prevention, proper experiment tracking, containerization, etc.), claiming "production-readiness" depends on your deployment environment and operational practices. The Google Drive artifact distribution is provided for convenience and evaluation purposes only. For production use, you should:
> 1. Generate artifacts using the provided Makefile pipeline
> 2. Implement proper artifact storage (model registry, versioned storage)
> 3. Set up monitoring and alerting beyond basic metrics emission
> 4. Establish CI/CD pipelines for automated retraining and deployment

Additional dataset files may be included for reproducing the complete training pipeline.

This prepared runtime allows the FastAPI service and Docker deployment to be evaluated without downloading or processing the full Netflix dataset.

### From-scratch dataset preparation

To reproduce the complete pipeline from raw data, obtain the Netflix Prize rating files and place them in `data/`:

```text
data/
├── combined_data_1.txt
└── movie_titles.csv
```

The raw Netflix dataset is available here:

[Download the Netflix Prize dataset](https://drive.google.com/drive/folders/1eldCFc5M0ElypZ9fU_jz4OpXD_-AQEUi?usp=sharing)

The ingestion step processes whichever `combined_data_*.txt` files are present. At least one raw rating file is required.

## Running the Pipeline

The Makefile provides commands for each pipeline stage.

### Ingest ratings

Converts raw Netflix text files into the processed Parquet dataset:

```bash
make ingest
```

### Enrich movie genres

Uses TMDb to create `data/movies_with_genres.csv`:

```bash
make enrich
```

If the enriched CSV already exists, this step skips the API requests.

### Build features and splits

Creates the train, validation, and test datasets:

```bash
make split
```

The split is chronological.

The pipeline first determines the temporal boundaries and then calculates user and movie activity thresholds using **only the training period**. This prevents future validation/test interactions from influencing which users and movies are considered active.

The resulting datasets follow:

```text
Training:
Date <= q80

Validation:
q80 < Date <= q90

Test:
Date > q90
```

### Train models

Trains or loads the popularity, ALS, SVD, and ensemble artifacts:

```bash
make train
```

Generated artifacts are stored in `artifacts/`.

The SVD configuration used by the current training pipeline is:

```text
n_factors = 50
n_epochs  = 20
lr_all    = 0.005
reg_all   = 0.04
```

These hyperparameters were selected through prior offline hyperparameter search and experimentation. The exhaustive search workflow was removed from the final notebook because repeatedly running the searches was computationally expensive under the project's RAM constraints. The selected configuration is preserved in the project configuration and training pipeline.

### Evaluate models

Evaluates the trained models on the held-out test split and logs selected metrics to MLflow:

```bash
make evaluate
```

The evaluation reports RMSE and MAE for:

* Global mean baseline
* Popularity baseline
* ALS
* SVD
* SVD + popularity ensemble

### Run the complete pipeline

```bash
make all
```

This runs:

```text
ingest -> enrich -> split -> train -> evaluate
```

## Model Evaluation

The current evaluation compares several approaches on the held-out test split. 
Note: ALS is included for demonstration purposes only to illustrate why implicit 
feedback models are inappropriate for explicit rating prediction.

Representative results:

| Model                                 |       RMSE |    MAE |
| ------------------------------------- | ---------: | -----: |
| Naive Mean Baseline                   |     1.0841 | 0.9183 |
| Popularity                            |     1.0371 | 0.8379 |
| ALS (Demo Only - Implicit/Explicit Mismatch) |     2.8969 | 2.6889 |
| SVD                                   |     0.9945 | 0.7868 |
| **SVD + Popularity Ensemble**         | **0.9927** | 0.7882 |

### Interpretation

SVD provides the strongest standalone performance:

* RMSE: **0.9945**
* MAE: **0.7868**

The validation-tuned SVD + popularity ensemble achieves:

* RMSE: **0.9927**
* MAE: **0.7882**

The ensemble therefore reduces RMSE by approximately **0.18%** compared with standalone SVD, while producing a slightly higher MAE.

The ensemble should consequently be interpreted as **broadly comparable to SVD rather than a substantial accuracy improvement**.

The ensemble is nevertheless used by the serving layer so that the production recommendation path is consistent with the evaluated ensemble model.

## Gradio Interface

Start the interactive browser interface:

```bash
make serve
```

The Gradio application accepts a customer ID and returns the highest-scoring recommendations by genre.

### Known users

Known customer IDs receive personalized recommendations using the **SVD + popularity ensemble**.

The inference pipeline:

```text
User
 ↓
Identify unseen movies
 ↓
SVD prediction
 +
Popularity prediction
 ↓
Validation-tuned ensemble weights
 ↓
Final predicted rating
 ↓
Top recommendation per genre
```

### Unknown users

Unknown or blank IDs receive popularity-based **cold-start recommendations**.

This provides a fallback when there is no user-specific rating history available.

The application uses Gradio's share mode and prints the accessible URL when it starts.

## FastAPI Service

The Docker image runs the FastAPI service.

Because the dataset and model artifacts are intentionally excluded from the Docker image, the runtime directories must be mounted when starting the container.

### Recommended Docker usage

After downloading the prepared `data/` and `artifacts/` directories:

```bash
docker build -t netflix-recommender .

docker run --rm \
    -p 8000:8000 \
    -v "$PWD/data:/app/data:ro" \
    -v "$PWD/artifacts:/app/artifacts:ro" \
    netflix-recommender
```

The `:ro` mounts keep the dataset and trained artifacts read-only inside the container.

The service is available at:

```text
http://localhost:8000
```

Interactive API documentation:

```text
http://localhost:8000/docs
```

### Health check

```bash
curl http://localhost:8000/health
```

### Readiness check

```bash
curl http://localhost:8000/ready
```

The readiness endpoint verifies that the required training data, movie metadata, and model artifacts are available.

### Recommendations

```bash
curl -X POST http://localhost:8000/recommend \
    -H "Content-Type: application/json" \
    -d '{"user_id": "51254", "top_n": 5}'
```

Example response shape:

```json
{
    "status_message": "Showing personalized results for user: 51254",
    "recommendations": [
        {
            "Genre": "Drama",
            "Movie Title": "Example Movie",
            "Predicted Rating": 4.52
        }
    ]
}
```

### Metrics Instrumentation

The service emits Prometheus metrics at the `/metrics` endpoint. For a complete monitoring solution, you would need to:
- Set up a Prometheus server to scrape these metrics
- Configure alerting rules for key metrics
- Create dashboards (e.g., in Grafana) to visualize service performance

```bash
curl http://localhost:8000/metrics
```

### Available Metrics

The following metrics are currently exposed:

#### HTTP Metrics
- `http_requests_total`: Counter of all HTTP requests (labels: method, path, status)
- `http_errors_total`: Counter of HTTP 4xx/5xx responses (labels: method, path, status)
- `http_request_duration_seconds`: Histogram of request latency in seconds (labels: method, path)

#### Recommendation Metrics
- `recommendation_requests_total`: Counter of all recommendation requests
- `cold_start_requests_total`: Counter of requests for unknown users (cold-start)
- `personalized_recommendation_requests_total`: Counter of requests for known users (personalized)
- `empty_recommendation_results_total`: Counter of requests returning zero recommendations
- `recommendations_returned_total`: Counter of total recommendation rows returned
- `recommendation_generation_duration_seconds`: Histogram of recommendation generation time (labels: request_type - personalized or cold_start)

#### Error Metrics
- `inference_errors_total`: Counter of inference errors (labels: error_type - inference, model_loading, data_loading)
- `model_loading_errors_total`: Counter of model loading errors (labels: model_type - popularity, svd, ensemble)
- `data_loading_errors_total`: Counter of data loading errors (labels: data_type - train, movies)

#### Health & Service Metrics
- `model_load_status`: Gauge of model loading status (1 = loaded, 0 = failed, -1 = not attempted) (labels: model_type)
- `data_load_status`: Gauge of data loading status (1 = loaded, 0 = failed, -1 = not attempted) (labels: data_type)
- `service_uptime_seconds`: Gauge of seconds since service started
- `model_cache_initialization_seconds`: Gauge of time spent initializing inference cache

## Model Notes

### Popularity Baseline

Ranks movies using average observed ratings and falls back to the global mean for movies without sufficient rating history.

The popularity model also acts as the cold-start recommendation strategy for users who do not exist in the training data.

### SVD

Uses explicit user-rating data to predict estimated ratings for unseen movies.

The selected configuration is:

```text
n_factors = 50
n_epochs  = 20
lr_all    = 0.005
reg_all   = 0.04
```

These values were selected through prior offline hyperparameter experimentation under the project's computational constraints.

The final production training pipeline uses the selected configuration directly rather than performing an expensive hyperparameter search during every training run.

### ALS (Demonstration Only)

Uses a sparse user-item matrix and is primarily useful for implicit interaction or ranking scenarios.
It is **not appropriate** for explicit 1-to-5 rating prediction tasks.

Its predictions are evaluated against the explicit 1-to-5 rating task **only for demonstration purposes** 
to show why implicit feedback models like ALS are unsuitable for explicit rating prediction.
The high error rate (~2.9 RMSE) illustrates the "Implicit vs. Explicit Feedback Trap."

ALS is **not part** of the production recommendation path and is included solely for educational 
purposes to highlight this important distinction in recommendation system design.

### Ensemble

The current ensemble optimizes a linear blend of SVD and popularity predictions on the validation split:

```text
prediction = alpha * SVD + (1 - alpha) * popularity
```

Predictions are clipped to the valid rating range of 1 to 5.

The optimized ensemble weights are stored in:

```text
artifacts/ensemble_weights.json
```

For the current model run, the validation optimization selected approximately:

```text
87% SVD
13% Popularity
```

The inference layer loads these saved weights and applies the same blend used during offline evaluation for known users.

The ensemble weights are loaded at inference startup and are **not re-optimized during API requests**.

## Evaluation Scope

The test split is constructed after applying the train-period activity filters.

Therefore, the reported RMSE and MAE primarily measure **warm-start performance** for users and movies that meet the training-period activity thresholds.

Cold-start recommendations for unknown users use the popularity fallback, but this cold-start behavior is **not included in the reported RMSE/MAE evaluation**.

Consequently, the reported test metrics should not be interpreted as system-wide performance across both warm-start and cold-start users.

## Business & Analytical Findings

### Most Popular and Most Liked Genre

* **Most popular by rating volume:** **Drama**
* **Most liked by average rating:** **War & Politics**
* The most frequently rated genre and highest-rated genre are not necessarily the same.

A volume metric indicates demand, while average rating provides a measure of user satisfaction among sufficiently reviewed genres.

### Best and Worst Rated Genres

Using a minimum threshold of 500 ratings:

* **Best rated:** **War & Politics** — average rating **4.14**
* **Worst rated:** **Reality** — average rating **3.22**

The minimum-rating threshold reduces the influence of genres with very small numbers of observations.

## Retraining Strategy

The current implementation uses the static Netflix Prize dataset, so automated production retraining is not applicable.

The dataset does not continuously receive new Netflix user ratings through this project. Therefore, there is no production data stream that would naturally trigger model retraining.

The training pipeline is nevertheless designed to support retraining when a new dataset snapshot is supplied:

```text
New Dataset Snapshot
        ↓
Data Validation
        ↓
Feature Engineering
        ↓
Model Training
        ↓
Offline Evaluation
        ↓
Compare Against Current Model
        ↓
Promote If Evaluation Criteria Are Met
```

## Testing

Run the automated tests with:

```bash
pytest
```

The test suite covers:

* Health and readiness endpoints
* Recommendation API responses
* Prometheus metrics
* Cold-start recommendations
* Personalized recommendations
* Recommendation output schema
* Rating bounds
* Excluding movies already seen by a user
* Temporal train/validation/test separation
* Training-period-only user activity filtering
* Training-period-only movie activity filtering

The data-pipeline test specifically verifies that:

```text
train dates <= validation dates
```

and that user/movie activity filtering is based only on the training-period data.

## Continuous Integration

The project uses GitHub Actions for continuous integration. The CI workflow runs on pushes and pull requests to the main, master, and 2.0 branches.

The CI pipeline includes:

1. **Validation Job**: Checks imports, code syntax (flake8), formatting (black), and type safety (mypy)
2. **Build and Test Job**: 
   - Tests across Python 3.11, 3.12, 3.13, and 3.14
   - Installs dependencies using requirements-dev.txt
   - Runs the complete test suite
   - Builds and validates Docker image
   - Performs runtime smoke tests (health, readiness, and inference endpoints)

To trigger the CI workflow locally, you can use the same commands that the CI runs:

```bash
# Validation
pip install -r requirements-dev.txt
python -c "import src.serving.fastapi_app; import src.inference.recommend; import src.models.popularity; import src.models.svd_model; import src.models.als_model; print('All imports successful')"
flake8 src/ --count --select=E9,F63,F7,F82 --show-source --statistics
black --check --diff src/
mypy src/ --ignore-missing-imports

# Testing
PYTHONPATH=. pytest tests/ -v

# Docker validation (optional)
docker build -t netflix-recommender:local .
# Then run the container and test endpoints as shown in the CI workflow
```

## MLflow

MLflow is used to track model evaluation runs locally in `mlruns/`.

Each evaluation run records the evaluated model metrics and the model configuration used for the run, providing a reproducible record of evaluation results.

The production training pipeline uses a fixed SVD configuration selected through prior offline experimentation rather than performing hyperparameter search during every training run.

To inspect the tracking data:

```bash
mlflow ui --backend-store-uri ./mlruns

## Resource Considerations

The full Netflix dataset and model-training process can require substantial memory.

The pipeline uses:

* Parquet storage
* Sparse matrices for ALS
* Chunked prediction generation
* Explicit garbage collection between training stages
* Memory cleanup after SVD training
* Single-threaded BLAS settings for ALS stability

The hyperparameter search was performed as an offline model-development activity rather than as part of the production training pipeline because repeated exhaustive searches are computationally expensive for the full dataset.

For development or testing, use the fixtures in `tests/` instead of processing the full dataset.
