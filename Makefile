.PHONY: install test lint format tools data splits show

install:
	pip install -e ".[dev]"

test:
	pytest -q

lint:
	ruff check .
	ruff format --check .

format:
	ruff check --fix .
	ruff format .

tools:
	python -m distill

data:
	python -m distill.data.generate --n 3000

splits:
	python -m distill.data.build --embeddings

show:
	python -m distill.data.show data/processed/test.jsonl --n 5
