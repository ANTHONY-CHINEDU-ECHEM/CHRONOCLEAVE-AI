.PHONY: install data train report all test api ui docker clean

install:
	pip install -e ".[dev]"

data:
	python -m chronocleave.cli simulate

train:
	python -m chronocleave.cli train

report:
	python -m chronocleave.cli report

all:
	python -m chronocleave.cli all

test:
	pytest -q

api:
	uvicorn chronocleave.api.main:app --host 0.0.0.0 --port 8000

ui:
	mlflow ui --backend-store-uri sqlite:///mlflow.db --port 5000

docker:
	docker compose up --build

clean:
	rm -rf mlruns mlflow.db data/raw .pytest_cache
