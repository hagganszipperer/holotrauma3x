.PHONY: help install dev lint format type test smoke docker clean

help:
	@echo "targets: install dev lint format type test smoke docker clean"

install:
	pip install -r requirements.txt && pip install --no-deps -e .

dev:
	pip install -e .[dev,backbones]

lint:
	ruff check . && isort --check . && black --check .

format:
	isort . && black .

type:
	mypy --strict src/holotrauma3x

test:
	pytest -q

smoke:
	holotrauma3x train --config configs/experiment/_smoke.yaml

docker:
	docker build -t holotrauma3x .

clean:
	rm -rf .pytest_cache .mypy_cache .ruff_cache runs *.egg-info src/*.egg-info
