default_env_dry_run:
	UV_PROJECT_ENVIRONMENT=.venv uv sync --group qwen3vl --group openai --dry-run

default_env:
	UV_PROJECT_ENVIRONMENT=.venv uv sync --group qwen3vl --group openai

openai_env_dry_run:
	UV_PROJECT_ENVIRONMENT=.venv-openai uv sync --group openai --dry-run

openai_env:
	UV_PROJECT_ENVIRONMENT=.venv-openai uv sync --group openai

google_env_dry_run:
	UV_PROJECT_ENVIRONMENT=.venv-google uv sync --group gemini --dry-run

google_env:
	UV_PROJECT_ENVIRONMENT=.venv-google uv sync --group gemini

qwen35vl_env_dry_run:
	UV_PROJECT_ENVIRONMENT=.venv_qwen35vl uv sync --group qwen35vl --dry-run

qwen35vl_env:
	UV_PROJECT_ENVIRONMENT=.venv_qwen35vl uv sync --group qwen35vl