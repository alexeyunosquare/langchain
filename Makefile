.PHONY: install install-dev install-docs test test-fast test-integration test-all clean-cov help test-agentic test-agentic-fast lint lint-check format

# Install main project dependencies
install:
	poetry install --no-root
# Install development dependencies
install-dev:
	poetry install --with dev --no-root
# Install documentation dependencies
install-docs:
	poetry install --with docs
# Run all tests (default)
test:
	poetry run pytest test/ -v --tb=short
# Run only fast unit tests (non-integration)
test-fast:
	poetry run pytest test/ -v --tb=short -m "not integration"
# Run only integration tests
test-integration:
	poetry run pytest test/ -v --tb=short -m "integration"
# Run all tests with coverage
test-all:
	poetry run pytest test/ -v --tb=short --cov=test --cov-report=html
# Run agentic RAG tests
test-agentic:
	poetry run pytest test/agentic_rag/ -v --tb=short
# Run agentic RAG tests (non-integration only)
test-agentic-fast:
	poetry run pytest test/agentic_rag/ -v --tb=short -m "not integration"
# Run tests with verbose output and detailed tracebacks
test-debug:
	poetry run pytest test/ -vvv --tb=long
# Run specific test file
test-file:
	poetry run pytest test/$(file) -v --tb=short
# Run specific test function
test-func:
	poetry run pytest test/$(file) -v --tb=short -k "$(function)"
# Clean pytest cache and coverage files
clean:
	find . -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true
	find . -type f -name ".coverage" -delete 2>/dev/null || true
	rm -rf htmlcov/ 2>/dev/null || true

# Lint with ruff and pyright
lint:
	poetry run ruff check .
	poetry run pyright .
# Lint check (exit with error if issues found)
lint-check:
	poetry run ruff check . --exit-non-zero-on-fix
	poetry run pyright .
# Type check with pyright (semantic checks)
pyright-check:
	poetry run pyright .
# Format code with ruff
format:
	poetry run ruff format .

# Show help message
help:
	@echo "Available targets:"
	@echo "  test          - Run all tests"
	@echo "  test-fast     - Run only fast unit tests"
	@echo "  test-integration - Run integration tests"
	@echo "  test-all      - Run all tests with coverage"
	@echo "  test-debug    - Run tests with verbose output"
	@echo "  test-file     - Run specific test file (use TEST_FILE=filename)"
	@echo "  test-func     - Run specific test function (use TEST_FILE=filename TEST_FUNC=function_name)"
	@echo "  test-docker   - Run Docker-dependent tests"
	@echo "  test-agentic  - Run agentic RAG tests"
	@echo "  test-agentic-fast - Run agentic RAG tests (non-integration)"
	@echo "  clean         - Clean pytest cache and coverage files"
	@echo "  lint          - Run ruff lint and pyright"
	@echo "  lint-check    - Run ruff lint check and pyright"
	@echo "  pyright-check - Run pyright type checker (semantic checks)"
	@echo "  format        - Format code with ruff"
