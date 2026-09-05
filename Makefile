.PHONY: ingest enrich split train evaluate serve all

ingest:
	python scripts/run_ingest.py

enrich:
	python scripts/run_enrich_genres.py

split:
	python scripts/run_build_features.py

train:
	python scripts/run_train.py

evaluate:
	python scripts/run_evaluate.py

serve:
	python src/serving/api.py

all: ingest enrich split train evaluate