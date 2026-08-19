.PHONY: install ruff_check clean clean_all

install:
	uv sync
	uv run pre-commit install

ruff_check:
	uv run ruff check .
	uv run ruff format .

clean:
	rm -rf .pytest_cache .logs/* .ruff_cache .mypy_cache allure-* allure-results allure-report dist build *.egg-info *:Zone.Identifier
	find . -type d -name ".venv" -prune -o -type d -name "__pycache__" -prune -exec rm -rf {} +
	find . -type f \( -name "*.Identifier" -o -name "*:Zone.Identifier" \) -delete

clean_all:
	rm -rf .pytest_cache .venv .internallogs .logs attachments/* .ruff_cache .mypy_cache allure-* allure-results allure-report dist build *.egg-info migration_*.db
	find . -type d -name ".venv" -prune -o -type d -name "__pycache__" -prune -exec rm -rf {} +
	find . -type f \( -name "*.Identifier" -o -name "*:Zone.Identifier" \) -delete
