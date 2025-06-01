#################################################################################
# GLOBALS                                                                       #
#################################################################################

PROJECT_NAME = vertebrae_cls
PYTHON_VERSION = 3.13
PYTHON_INTERPRETER = python

#################################################################################
# COMMANDS                                                                      #
#################################################################################


## Install Python Dependencies
.PHONY: requirements
requirements:
	pip install uv
	uv sync

## Delete all compiled Python files
.PHONY: clean
clean:
	find . -type f -name "*.py[co]" -delete
	find . -type d -name "__pycache__" -delete

## Lint using flake8 and black (use `make format` to do formatting)
.PHONY: lint
lint:
	flake8 src
	isort --check --diff --profile black src
	black --check --config pyproject.toml src

## Format source code with black
.PHONY: format
format:
	black --config pyproject.toml src
	isort --profile black src

## Prepare dataset
.PHONY: prepare_dataset
prepare_dataset:
	@echo "🔍 Extracting XLS..."
	uv run python -c "from src.data_utils.XLSExtractor import extract_xls; extract_xls()"
	@echo "📦 Creating dataset..."
	uv run python -c "from src.data_utils.dataset_creator import create_dataset; create_dataset()"


## Run experiments
.PHONY: run_experiments
run_experiments:
	uv run src/training/sweep_runner.py 


#################################################################################
# Self Documenting Commands                                                     #
#################################################################################

.DEFAULT_GOAL := help

define PRINT_HELP_PYSCRIPT
import re, sys; \
lines = '\n'.join([line for line in sys.stdin]); \
matches = re.findall(r'\n## (.*)\n[\s\S]+?\n([a-zA-Z_-]+):', lines); \
print('Available rules:\n'); \
print('\n'.join(['{:25}{}'.format(*reversed(match)) for match in matches]))
endef
export PRINT_HELP_PYSCRIPT

help:
	@$(PYTHON_INTERPRETER) -c "${PRINT_HELP_PYSCRIPT}" < $(MAKEFILE_LIST)
