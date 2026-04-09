# Video hallucination detection

## Download model

```shell
hf download Qwen/Qwen3-VL-8B-Instruct --local-dir ./weights/qwen3_vl_8b_inst
```

## Download data

```shell
cd raw_data
curl -O https://link/to/file.mp4
```

## Setup

```shell
sudo apt update
sudo apt install -y libgl1
```

Each model backend has its own dependency group. Install only what you need:

| Use-case                                 | Command                                                                |
| ---------------------------------------- | ---------------------------------------------------------------------- |
| Qwen3-VL-8B (local GPU)                  | `uv sync --group qwen3vl`                                              |
| Qwen3.5-VL (local GPU, dev transformers) | `uv sync --group qwen35vl`                                             |
| Claude API only                          | `uv sync --group claude`                                               |
| Gemini API only                          | `uv sync --group gemini`                                               |
| OpenAI API only                          | `uv sync --group openai`                                               |
| Qwen3-VL + all API backends              | `uv sync --group qwen3vl --group claude --group gemini --group openai` |

Then activate:

```shell
source .venv/bin/activate
```

For API backends, add the relevant key(s) to a `.env` file:

```text
ANTHROPIC_API_KEY=...
GEMINI_API_KEY=...
OPENAI_API_KEY=...
```

## Run

```shell
CUDA_VISIBLE_DEVICES=0 python benchmark.py \
    --model_id weights/qwen3_vl_8b_inst \
    --max_new_tokens 1024
```

**NOTE**: You can read `benchmark.py` to understand the flow of the code before hitting run it. This makes it easier to develop custom pipeline for other baselines to utilize the benchmark.

**NOTE**: The consitency and sub-questions are currently under development. Future work may expand the benchmark structure but backward compatibility is ensure.

## Beta version

### Local models

```shell
# Qwen3-VL-8B
CUDA_VISIBLE_DEVICES=0 python benchmark_sub.py \
    --model_id weights/qwen3_vl_8b_inst \
    --metrics all \
    --questions_dir benchmark_subq

# Qwen3-compatible checkpoint that expects a fixed FPS (for example, 4 FPS)
CUDA_VISIBLE_DEVICES=0 python benchmark_sub.py \
    --model_id /path/to/Cosmos-Reason2 \
    --force_fps 4 \
    --metrics all \
    --questions_dir benchmark_subq

# Qwen3.5-VL (auto-detected by model_id prefix)
CUDA_VISIBLE_DEVICES=3 python benchmark_sub.py \
    --model_id Qwen/Qwen3.5-2B \
    --metrics all \
    --questions_dir benchmark_subq

# Qwen3.5-VL with thinking mode
CUDA_VISIBLE_DEVICES=0 python benchmark_sub.py \
    --model_id Qwen/Qwen3.5-VL-7B-Instruct \
    --prompt_method thinking \
    --metrics all \
    --questions_dir benchmark_subq
```

### API models

No GPU required. Make sure the relevant key is set in `.env` first.

```shell
# Gemini (direct)
python benchmark_sub.py \
    --model_id gemini-2.5-pro \
    --metrics all \
    --questions_dir benchmark_subq

# Claude (direct)
python benchmark_sub.py \
    --model_id claude-3-7-sonnet-20250219 \
    --metrics all \
    --questions_dir benchmark_subq

# OpenAI (direct)
python benchmark_sub.py \
    --model_id gpt-4o \
    --metrics all \
    --questions_dir benchmark_subq
```

### OpenRouter

Prefix `openrouter/` routes any model through OpenRouter (single API key, OpenAI-compatible).
Add `OPENROUTER_API_KEY` to `.env`. No extra dependency group needed beyond `openai`.

```shell
uv sync --group openai
```

```shell
# Gemini 2.5 Pro via OpenRouter
python benchmark_sub.py \
    --model_id openrouter/google/gemini-2.5-pro \
    --metrics all \
    --questions_dir benchmark_subq

# GPT-5 via OpenRouter
python benchmark_sub.py \
    --model_id openrouter/openai/gpt-5 \
    --metrics all \
    --questions_dir benchmark_subq

# Claude Sonnet 4.6 via OpenRouter
python benchmark_sub.py \
    --model_id openrouter/anthropic/claude-sonnet-4-6 \
    --metrics all \
    --questions_dir benchmark_subq

# Qwen3-VL-32B via OpenRouter
python benchmark_sub.py \
    --model_id openrouter/qwen/qwen3-vl-32b-instruct \
    --metrics all \
    --questions_dir benchmark_subq

# InternVL3.5-78B via OpenRouter
python benchmark_sub.py \
    --model_id openrouter/internvl/internvl3.5-78b \
    --metrics all \
    --questions_dir benchmark_subq
```

> The `openrouter/` prefix is stripped automatically before calling the API,
> so `openrouter/google/gemini-2.5-pro` → model slug `google/gemini-2.5-pro`
> (exactly as shown on openrouter.ai).

### vLLM (localhost)

Prefix `vllm/` routes any model through a local vLLM OpenAI-compatible server.
This uses a separate backend file and leaves `src/models/openai_gpt.py` unchanged.

Add these to `.env` if needed:

```text
VLLM_BASE_URL=http://localhost:8000/v1
VLLM_API_KEY=EMPTY
```

`VLLM_API_KEY` is optional for the common localhost setup; the vLLM backend falls
back to `EMPTY` if it is unset.

```shell
uv sync --group openai
```

```shell
# Qwen3-VL-32B-Thinking via local vLLM
python benchmark_sub.py \
    --model_id vllm/Qwen/Qwen3-VL-32B-Thinking \
    --metrics all \
    --questions_dir benchmark_subq
```

> The `vllm/` prefix is stripped automatically before calling the API, so
> `vllm/Qwen/Qwen3-VL-32B-Thinking` → model ID `Qwen/Qwen3-VL-32B-Thinking`.

### Common flags

| Flag               | Default                     | Description                                                                                      |
| ------------------ | --------------------------- | ------------------------------------------------------------------------------------------------ |
| `--model_id`       | `Qwen/Qwen3-VL-8B-Instruct` | Model to evaluate                                                                                |
| `--metrics`        | `all`                       | Metrics: `accuracy`, `sub_accuracy`, `consistency`, `consistency_tc`, `consistency_tw`, or `all` |
| `--questions_dir`  | `benchmark`                 | Folder containing benchmark JSON files                                                           |
| `--prompt_method`  | `vanilla`                   | Prompt variant label (also used as cache namespace suffix)                                       |
| `--force_fps`      | _(none)_                    | Force the local Qwen3-VL backend to use a fixed FPS, e.g. `4` for Cosmos-Reason2                |
| `--max_new_tokens` | `256`                       | Max tokens per answer                                                                            |
| `--output_json`    | _(none)_                    | Write full results to a JSON file                                                                |
| `--cache_dir`      | `cache`                     | Directory for caching model predictions                                                          |
| `--video_dir`      | `raw_data`                  | Directory containing video files                                                                 |

Model routing is automatic based on the `--model_id` prefix:

| Prefix | Backend | Required group | API key env |
|---|---|---|---|
| `vllm/*` | `VLLMOpenAIModel` | `uv sync --group openai` | `VLLM_API_KEY` |
| `openrouter/*` | `OpenAIModel` → OpenRouter | `uv sync --group openai` | `OPENROUTER_API_KEY` |
| `gemini-*` | `GeminiModel` | `uv sync --group gemini` | `GEMINI_API_KEY` |
| `claude-*` | `ClaudeModel` | `uv sync --group claude` | `ANTHROPIC_API_KEY` |
| `gpt-*` / `o1-*` / `o3-*` | `OpenAIModel` | `uv sync --group openai` | `OPENAI_API_KEY` |
| `Qwen/Qwen3.5-*` / `qwen3.5-*` | `Qwen35VLModel` | `uv sync --group qwen35vl` | — |
| anything else | `Qwen3VLModel` | `uv sync --group qwen3vl` | — |

## Prompt for augmentation

1. Create a `.env` file

```text
GEMINI_API_KEY=....
```

```shell
uv run augmnet_question.py
```
