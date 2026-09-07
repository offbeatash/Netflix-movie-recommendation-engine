PYTHON ?= $(if $(wildcard .venv/bin/python),.venv/bin/python,python)

.PHONY: ingest enrich split train evaluate serve all

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

serve:
	$(PYTHON) src/serving/api.py

all: ingest enrich split train evaluate