# Makefile for Py_PaRSEC (lives inside parsec/python/)

.PHONY: help install install-dev test test-cov lint format clean build docs

help:  ## Show this help message
	@echo "Available targets:"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'

install:  ## Install the package
	pip install -e .

install-dev:  ## Install the package with development dependencies
	pip install -e ".[dev,test]"

test:  ## Run tests
	pytest

test-cov:  ## Run tests with coverage
	pytest --cov=py_parsec --cov-report=html --cov-report=term-missing

test-mpi:  ## Run MPI tests
	pytest -m mpi -v

test-integration:  ## Run integration tests
	pytest -m integration -v

lint:  ## Run linting
	flake8 src/ tests/
	mypy src/

format:  ## Format code
	black src/ tests/
	isort src/ tests/

clean:  ## Clean build artifacts
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info/
	rm -rf .pytest_cache/
	rm -rf .coverage
	rm -rf htmlcov/
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	find . -type f -name "*.so" -delete
	find . -type f -name "*.pyd" -delete

build:  ## Build the package
	python -m build

build-ext:  ## Build Cython extensions
	python setup.py build_ext --inplace

build-parsec:  ## Build PaRSEC from the parent repo
	python build_parsec4python.py --skip-env

docs:  ## Build documentation
	cd docs && make html

check:  ## Run all checks (lint, format, test)
	@echo "Running format check..."
	black --check src/ tests/
	isort --check-only src/ tests/
	@echo "Running linting..."
	flake8 src/ tests/
	mypy src/
	@echo "Running tests..."
	pytest

setup-dev:  ## Set up development environment
	pip install -e ".[dev,test]"
	pre-commit install

dev: install-dev  ## Alias for install-dev
check-all: check  ## Alias for check
