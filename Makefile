.PHONY: install test lint format tools data splits show eval-teacher compare

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

eval-teacher:
	python -m distill.eval.run --name teacher --model openai/gpt-oss-120b --limit 200
	python -m distill.eval.run --name teacher --model openai/gpt-oss-120b --limit 200 --test data/processed/test_noisy.jsonl

compare:
	python -m distill.eval.compare results/*
