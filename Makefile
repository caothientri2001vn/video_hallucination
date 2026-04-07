default_env_dry_run:
	UV_PROJECT_ENVIRONMENT=.venv uv sync --group qwen3vl --group openai --dry-run

default_env:
	UV_PROJECT_ENVIRONMENT=.venv uv sync --group qwen3vl --group openai