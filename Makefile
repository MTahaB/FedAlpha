.PHONY: test lint download oracle dashboard hardhat-test

test:
	python -m pytest

lint:
	python -m compileall quant federated_learning oracle data dashboard tests

download:
	python data/download.py --preset sp100 --start 2014-01-01 --end 2025-01-01

oracle:
	uvicorn oracle.validation_api:app --reload --port 8000

dashboard:
	streamlit run dashboard/app.py

hardhat-test:
	cd blockchain && npx hardhat test
