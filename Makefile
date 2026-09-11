ifeq ($(OS),Windows_NT)
PYTHON ?= $(if $(wildcard .venv/Scripts/python.exe),.venv/Scripts/python.exe,py -3.13)
else
PYTHON ?= $(if $(wildcard .venv/bin/python),.venv/bin/python,python3.13)
endif

.PHONY: ingest enrich split train evaluate quality-gate version load-test serve-gradio serve-api docker-build docker-run all test lint format-check type-check

ingest:
	$(PYTHON) scripts/run_ingest.py

enrich:
	$(PYTHON) scripts/run_enrich_genres.py

split:
	$(PYTHON) scripts/run_build_features.py

train:
	$(PYTHON) scripts/run_train.py

evaluate:
	$(PYTHON) scripts/run_evaluate.py

quality-gate:
	$(PYTHON) scripts/run_quality_gate.py

version:
	$(PYTHON) scripts/run_version.py

load-test:
	$(PYTHON) scripts/load_test.py

serve-gradio:
	$(PYTHON) src/serving/api.py

serve-api:
	$(PYTHON) -m uvicorn src.serving.fastapi_app:app --host 0.0.0.0 --port 8000

docker-build:
	docker build -t netflix-recommender .

docker-run:
	docker run --rm -p 8000:8000 --env-file .env -v $(PWD)/data:/app/data:ro -v $(PWD)/artifacts:/app/artifacts:ro netflix-recommender

test:
	$(PYTHON) -m pytest tests/ -v

lint:
	$(PYTHON) -m flake8 src scripts tests

format-check:
	$(PYTHON) -m black --check --diff src scripts tests

type-check:
	$(PYTHON) -m mypy src scripts --config-file mypy.ini

all: ingest enrich split train evaluate
