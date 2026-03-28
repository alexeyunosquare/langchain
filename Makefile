.PHONY: install install-dev install-docs test test-fast test-integration test-all clean-cov help

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
# Run tests with verbose output and detailed tracebacks
test-debug:
	poetry run pytest test/ -vvv --tb=long
# Run specific test file
test-file:
	poetry run pytest test/$(file) -v --tb=short
# Run specific test function
test-func:
	poetry run pytest test/$(file) -v --tb=short -k "$(function)"
# Run tests with Docker services
test-docker:
	docker-compose up -d
	poetry run pytest test/ -v --tb=short -m "integration and requires_docker"
	docker-compose down
# Clean pytest cache and coverage files
clean:
	find . -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true
	find . -type f -name ".coverage" -delete 2>/dev/null || true
	rm -rf htmlcov/ 2>/dev/null || true

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
	@echo "  clean         - Clean pytest cache and coverage files"
