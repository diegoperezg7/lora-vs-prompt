# Reproducible pipeline. Every step is seeded; predictions are cached in predictions/.
# Optional: COMPUTE_LOCK=/path/to/dir makes model-loading steps wait for a shared lock.
RUN := uv run

.PHONY: all setup data train eval report test lint clean-predictions

all: data train eval report

setup:
	uv sync

data:
	$(RUN) lvp data

train: data
	$(RUN) lvp train

eval:
	$(RUN) lvp eval

report:
	$(RUN) lvp report

test:
	$(RUN) pytest -q

lint:
	$(RUN) ruff check src tests
	$(RUN) ruff format --check src tests

clean-predictions:
	rm -rf predictions results
