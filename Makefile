.PHONY: help install install-dev test lint format clean

help:
	@echo "Available commands:"
	@echo "  make install      Install package in development mode"
	@echo "  make install-dev  Install package with development dependencies"
	@echo "  make test         Run tests"
	@echo "  make lint         Run linting checks"
	@echo "  make format       Format code with black and isort"
	@echo "  make clean        Clean up temporary files"

install:
	pip install -e .

install-dev: install
	pip install -r requirements-dev.txt

test:
	pytest tests/ -v

lint:
	flake8 gym_nim/ tests/
	mypy gym_nim/

format:
	black gym_nim/ tests/ examples/
	isort gym_nim/ tests/ examples/

clean:
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -delete
	find . -type d -name "*.egg-info" -exec rm -rf {} +
	rm -rf build/ dist/ .coverage htmlcov/ .pytest_cache/