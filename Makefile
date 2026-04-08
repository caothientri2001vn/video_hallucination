default_env_dry_run:
	UV_PROJECT_ENVIRONMENT=.venv uv sync --group qwen3vl --group openai --dry-run

default_env:
	UV_PROJECT_ENVIRONMENT=.venv uv sync --group qwen3vl --group openai

openai_env_dry_run:
	UV_PROJECT_ENVIRONMENT=.venv-openai uv sync --group openai --dry-run

openai_env:
	UV_PROJECT_ENVIRONMENT=.venv-openai uv sync --group openai