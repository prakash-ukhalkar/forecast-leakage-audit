.PHONY: setup data test notebooks all

PY ?= python

setup:
	$(PY) -m venv .venv
	. .venv/bin/activate || . .venv/Scripts/activate; \
	pip install --upgrade pip; \
	pip install -r requirements.txt; \
	pip install -e .

data:
	. .venv/bin/activate || . .venv/Scripts/activate; \
	$(PY) -m sdg_leakage_audit.data.download --help

test:
	. .venv/bin/activate || . .venv/Scripts/activate; \
	pytest -v

notebooks:
	. .venv/bin/activate || . .venv/Scripts/activate; \
	jupyter nbconvert --to notebook --execute --inplace notebooks/00_project_setup_and_environment_check.ipynb

all: setup test
