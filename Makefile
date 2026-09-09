ifeq ($(OS),Windows_NT)
PYTHON ?= $(if $(wildcard .venv/Scripts/python.exe),.venv/Scripts/python.exe,py -3.13)
else
PYTHON ?= $(if $(wildcard .venv/bin/python),.venv/bin/python,python3.13)
endif

.PHONY: ingest enrich split train evaluate serve-gradio serve-api docker-build docker-run all

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

serve-gradio:
	$(PYTHON) src/serving/api.py

serve-api:
	$(PYTHON) -m uvicorn src.serving.fastapi_app:app --host 0.0.0.0 --port 8001

docker-build:
	docker build -t netflix-recommender .

docker-run:
	docker run --rm -p 8000:8000 \
		-v $(PWD)/data:/app/data:ro \
		-v $(PWD)/artifacts:/app/artifacts:ro \
		netflix-recommender

all: ingest enrich split train evaluate
