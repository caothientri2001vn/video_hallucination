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

| Use-case | Command | Notes |
|---|---|---|
| Qwen3-VL-8B (local GPU) | `uv sync --group qwen3vl` | |
| Qwen3.5-VL + frame selectors (local GPU) | `uv sync --group qwen35vl` | dev transformers |
| TraveLER (vLLM server) | `uv sync --group openai` | vLLM served separately |
| Claude API | `uv sync --group claude` | |
| Gemini API | `uv sync --group gemini` | |
| OpenAI / OpenRouter | `uv sync --group openai` | |
| Qwen3-VL + all API backends | `uv sync --group qwen3vl --group claude --group gemini --group openai` | |

Then activate:

```shell
source .venv/bin/activate
```

For API backends, add the relevant key(s) to a `.env` file:

```text
ANTHROPIC_API_KEY=...
GEMINI_API_KEY=...
OPENAI_API_KEY=...
OPENROUTER_API_KEY=...

# TraveLER: URL of the running vLLM server (default: http://localhost:8123/v1)
VLLM_BASE_URL=http://localhost:8123/v1
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

> **Default frame sampling**: `Qwen3VLModel` and `Qwen35VLModel` pass the raw
> video file directly to the model and let the processor sample frames
> internally (no explicit frame selector).  Use the `<selector>+<model>` syntax
> below if you want explicit frame selection strategies.

```shell
# Qwen3-VL-8B  (internal video sampling — default)
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

### Frame-selection pipelines

Use `<selector>+<model_id>` to run an explicit frame selection strategy before
inference.  The selector extracts PIL frames and feeds them to the backbone as
images instead of a raw video file.

| Selector | Strategy |
|---|---|
| `uniform` | Uniform temporal sampling (baseline, query-agnostic) |
| `clip` | CLIP cosine similarity — top-K frames most relevant to the question |
| `aks` | Adaptive Keyframe Selection — recursive segment splitting via CLIP scores |
| `efs` | Event-anchored Frame Selection — DINOv2 scene segmentation + CLIP scoring |

```shell
# Uniform  (no CLIP needed — fastest)
CUDA_VISIBLE_DEVICES=0 python benchmark_sub.py \
    --model_id uniform+Qwen/Qwen3.5-2B \
    --metrics all \
    --questions_dir benchmark_subq

# CLIP Retrieval
CUDA_VISIBLE_DEVICES=0 python benchmark_sub.py \
    --model_id clip+Qwen/Qwen3.5-2B \
    --metrics all \
    --questions_dir benchmark_subq

# AKS
CUDA_VISIBLE_DEVICES=0 python benchmark_sub.py \
    --model_id aks+Qwen/Qwen3.5-2B \
    --metrics all \
    --questions_dir benchmark_subq

# EFS  (also loads DINOv2 — slowest selector but most scene-aware)
CUDA_VISIBLE_DEVICES=0 python benchmark_sub.py \
    --model_id efs+Qwen/Qwen3.5-2B \
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

### TraveLER (multi-agent structured pipeline)

TraveLER runs an iterative **Planner → Retriever → Extractor → Summarizer → Evaluator**
loop instead of a single forward pass.  The model is served externally by vLLM
and communicated with via the OpenAI-compatible API — no GPU memory is used
by the benchmark process itself.

**Environment**: only needs the `openai` group (no transformers).

```shell
uv sync --group openai
```

**Step 1 — serve the model** (separate terminal, runs persistently):

```shell
# Install vLLM (one-time, separate from uv env)
pip install vllm --torch-backend=auto --extra-index-url https://wheels.vllm.ai/nightly

# Serve Qwen3.5-2B on GPU 0
CUDA_VISIBLE_DEVICES=0 vllm serve Qwen/Qwen3.5-2B \
    --port 8123 \
    --gpu-memory-utilization 0.25 \
    --max-model-len 8192 \
    --reasoning-parser qwen3 \
    --default-chat-template-kwargs '{"enable_thinking": false}' \
    --enable-prefix-caching \
    --max-cudagraph-capture-size 256
```

**Step 2 — run the benchmark** (in a different terminal, server must be running):

```shell
# Default: connects to http://localhost:8123/v1
python benchmark_sub.py \
    --model_id traveler/Qwen/Qwen3.5-2B \
    --metrics all \
    --questions_dir benchmark_subq

# Custom server URL
VLLM_BASE_URL=http://my-server:8123/v1 python benchmark_sub.py \
    --model_id traveler/Qwen/Qwen3.5-2B \
    --metrics all \
    --questions_dir benchmark_subq
```

> **Cost**: TraveLER runs the full pipeline once **per question** (not per video),
> so it makes many more API calls than a single-pass model.  Use the built-in
> caching (`--cache_dir`) to avoid re-running completed questions.

> **Pipeline hyperparams** can be overridden via `load_model(**kwargs)` in
> Python code: `max_iters` (default 3), `view_range` (2 s), `num_questions` (3),
> `init_frames` (5), `video_fps` (10).

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

For the 32B served model setup:

```shell
OMP_NUM_THREADS=1 \
vllm serve weights/qwen3_vl_32b_inst \
    --served-model-name qwen3_vl_32b_inst \
    --tensor-parallel-size 8 \
    --mm-encoder-tp-mode data \
    --async-scheduling \
    --reasoning-parser qwen3 \
    --limit-mm-per-prompt.image 64 \
    --limit-mm-per-prompt.video 0 \
    --max-model-len 128000
```

Then call the served model name with the `vllm/` prefix:

```shell
python benchmark_sub.py \
    --model_id vllm/qwen3_vl_32b_inst \
    --metrics all \
    --questions_dir benchmark_subq
```

Use `vllm_video/` when the vLLM server is on the same machine and can read the
video path directly. Launch the server with local media access and video inputs
enabled:

```shell
OMP_NUM_THREADS=1 \
vllm serve weights/qwen3_vl_32b_inst \
    --served-model-name qwen3_vl_32b_inst \
    --tensor-parallel-size 8 \
    --mm-encoder-tp-mode data \
    --async-scheduling \
    --reasoning-parser qwen3 \
    --limit-mm-per-prompt.image 0 \
    --limit-mm-per-prompt.video 1 \
    --allowed-local-media-path /path/to/raw_data \
    --max-model-len 128000
```

Then run:

```shell
python benchmark_sub.py \
    --model_id vllm_video/qwen3_vl_32b_inst \
    --metrics all \
    --questions_dir benchmark_subq
```

> The `vllm/` and `vllm_video/` prefixes are stripped automatically before
> calling the API, so `vllm_video/qwen3_vl_32b_inst` -> model ID
> `qwen3_vl_32b_inst`.
> If `--reasoning-parser qwen3` returns `content=None` with text in `reasoning`,
> the vLLM backends read that field as the answer.

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

| Prefix | Backend | Required group | Notes |
|---|---|---|---|
| `traveler/*` | `TraveLERModel` | `uv sync --group openai` | vLLM server must be running; set `VLLM_BASE_URL` |
| `openrouter/*` | `OpenAIModel` → OpenRouter | `uv sync --group openai` | Set `OPENROUTER_API_KEY` |
| `gemini-*` | `GeminiModel` | `uv sync --group gemini` | Set `GOOGLE_API_KEY` |
| `claude-*` | `ClaudeModel` | `uv sync --group claude` | Set `ANTHROPIC_API_KEY` |
| `gpt-*` / `o1-*` / `o3-*` | `OpenAIModel` | `uv sync --group openai` | Set `OPENAI_API_KEY` |
| `<sel>+<model>` | `FramePipelineModel` | `uv sync --group qwen35vl` | sel = uniform / clip / aks / efs |
| `Qwen/Qwen3.5-*` / `qwen3.5-*` | `Qwen35VLModel` | `uv sync --group qwen35vl` | Internal video sampling |
| anything else | `Qwen3VLModel` | `uv sync --group qwen3vl` | Internal video sampling |
| `vllm_video/*` | `VLLMVideoModel` | `uv sync --group openai` | Native `video_url`; server needs `--allowed-local-media-path` |
| `vllm/*` | `VLLMOpenAIModel` | `uv sync --group openai` | Frame-based image prompts; `VLLM_API_KEY` |
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
