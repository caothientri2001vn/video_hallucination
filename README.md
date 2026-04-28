# Video Hallucination

A three-stage pipeline for long-form video question answering, designed to localise and mitigate hallucination in monolithic vision-language models.

## Pipeline

The pipeline decomposes video QA into three stages, each of which can be filled by a different model:

1. **Stage A1 — Extractor (VLM).** Splits the video into 15 s chunks (60 frames each) and runs a VLM with a structured-extraction prompt. Output: per-chunk JSON "states" with fields `event_type`, `description`, `sub_events`, `outcome`, `start_time`, `end_time`. All chunk states for a video are concatenated into one text document.
2. **Stage B — Filter + identity linking (text LLM).** Takes the question and the concatenated states, and produces (a) a filtered list of relevant events whose description / sub_events match the question, and (b) a cross-chunk identity table that links recurring entities across chunks.
3. **Stage C — Answerer (VLM).** Takes the question + the filtered events text + 64 video frames, and emits a short reasoning trace formatted as `Evidence: … / Answer: …`.

Each slot is model-agnostic. Supported backends include Gemini 3 Flash (native API), Qwen3-VL-32B-Thinking served through local vLLM, Qwen3-235B-A22B (text-only, OpenRouter), and others reachable through OpenAI-compatible APIs.

## Setup

```shell
sudo apt install -y libgl1
make env          # uv sync --group openai --group gemini  →  .venv/
source .venv/bin/activate
```

Add API keys to `.env` at the repo root:

```text
GOOGLE_API_KEY=...        # Gemini native API
OPENROUTER_API_KEY=...    # OpenRouter text calls (Qwen3 text, identity linking)
VLLM_API_KEY=EMPTY        # Local vLLM usually accepts any non-empty value
VLLM_BASE_URL=http://localhost:8700/v1
```

Serve the Qwen3-VL answerer locally before running experiments that use `q3vl` in Stage C:

```bash
vllm serve Qwen/Qwen3-VL-32B-Thinking \
  --host 0.0.0.0 --port 8700 \
  --served-model-name Qwen/Qwen3-VL-32B-Thinking
```

Experiments that use `q3vl` in Stage A1 can either point `--state_extractor_vllm_base_url` at the same server or run a second local vLLM server on the default extractor port, `http://localhost:8200/v1`.

## Data

The benchmark lives under `benchmark/` and `benchmark_*/`, organised by difficulty:

| Directory               | Use                                                |
| ----------------------- | -------------------------------------------------- |
| `benchmark/`            | Standard set (~88 samples)                         |

Each sample is a JSON file describing a video + a list of questions. Videos themselves are referenced from a sibling directory and exposed via the `normalized_videos`, `raw_data`, and `videos` symlinks at the repo root.

## Repo layout

```
.
├── benchmark_sub_with_states.py    # main pipeline runner (Stages A1 → B → C)
├── analyze_failures.py             # failure-mode analysis
├── llm_judge_accuracy.py           # LLM-judge accuracy scoring
├── judge_vanilla.py                # vanilla (no-pipeline) baseline judge
├── inspect_outputs.py              # quick-look at cached answers
├── inspect_accuracy.py             # accuracy summary helper
├── stages/                         # core stage implementations
│   ├── eval_tier2_flash_aggregator.py          # Stage A1 + Stage B + helpers
│   ├── eval_tier2_flash_aggregator_planner.py  # extractor planner
│   └── stage_a_planner.py                      # biased-timeline builder
├── src/
│   ├── eval_module.py              # eval orchestration + scoring
│   ├── load_data.py                # benchmark loader
│   ├── answer_processing.py        # yes/no extraction, answer matching
│   ├── cache/                      # AnswerCache
│   ├── metrics/                    # accuracy, consistency
│   └── models/
│       ├── base.py                 # BaseVideoQAModel ABC
│       └── vllm_openai.py          # OpenAI-compatible / vLLM client
├── benchmark/                      # standard difficulty samples (~88)
├── benchmark_hard/, benchmark_super_hard/, benchmark_small/
├── tests/                          # tests for the live core
├── normalized_videos/, raw_data/, videos/  →  symlinks to video files
├── cache/                          # stage outputs (gitignored)
├── pyproject.toml, uv.lock, Makefile
└── README.md
```

## Running experiments

### Model tags

| Tag    | Slug / model                            | Role                                          |
| ------ | --------------------------------------- | --------------------------------------------- |
| `gfl`  | `gemini-3-flash-preview`                | proprietary VLM/LLM (any slot)                |
| `q3vl` | `Qwen/Qwen3-VL-32B-Thinking`            | open-source VLM served by local vLLM (Stage A1 extractor / Stage C answerer) |
| `q3t`  | `qwen/qwen3-235b-a22b`                  | open-source text MoE (Stage B filter + identity linking) |
| `q35`  | `qwen/qwen3.5-27b`                      | open-source dense VLM (any slot)              |

### Experiment matrix

| ID           | A1 Extractor | B Filter + IdLink | C Answerer | Cache dir                  |
| ------------ | ------------ | ----------------- | ---------- | -------------------------- |
| **Baseline** | `gfl`        | `gfl`             | `gfl`      | `cache/pipeline_baseline/` |
| **A1**       | `gfl`        | `q3t`             | `q3vl`     | `cache/pipeline_a1/`       |
| **B1**       | `q3vl`       | `gfl`             | `q3vl`     | `cache/pipeline_b1_d1/`    |
| **C1**       | `q3vl`       | `q3t`             | `gfl`      | `cache/pipeline_c1/`       |
| **D1**       | `q3vl`       | `q3t`             | `q3vl`     | `cache/pipeline_b1_d1/`    |
| **D1-Cans**  | `q3vl`       | `q3t`             | `q35`      | `cache/pipeline_b1_d1/`    |

Baseline, A1, and C1 each have their own dedicated state cache directory. **B1, D1, and D1-Cans all share `cache/pipeline_b1_d1/`** — all three have a `q3vl` extractor, and use different `--prompt_method` tags so their Stage-C answers land in distinct `answers_<prompt_method>/` subdirectories within the shared cache. D1-Cans additionally reuses D1's `q3t` filter cache and only re-runs Stage C with the `q35` answerer.

### Baseline (all-Gemini)

```bash
python benchmark_sub_with_states.py \
  --state_strategy filter --aggregator_backend concat \
  --stage_b_backend gemini --stage_b_model gemini-3-flash-preview \
  --state_extractor_backend gemini --state_extractor_model gemini-3-flash-preview \
  --states_cache_dir cache/pipeline_baseline \
  --chunk_prompt_version v6 --answerer_prompt_version v3 \
  --enable_identity_link --aggregation_routing \
  --answerer_backend gemini --model_id gemini-3-flash-preview \
  --gemini_answerer_thinking_budget 0 --gemini_answerer_max_concurrency 4 \
  --vllm_n_frames 64 --frames_per_chunk 60 \
  --prompt_method filter_v3_gemini \
  --mode all --metrics accuracy \
  --max_new_tokens 4096 \
  --questions_dir benchmark
```

### A1 — Gemini extractor + Qwen filter + Qwen answerer

```bash
python benchmark_sub_with_states.py \
  --state_strategy filter --aggregator_backend concat \
  --stage_b_backend openrouter --stage_b_model qwen/qwen3-235b-a22b \
  --state_extractor_backend gemini --state_extractor_model gemini-3-flash-preview \
  --states_cache_dir cache/pipeline_a1 \
  --chunk_prompt_version v6 --answerer_prompt_version v3 \
  --enable_identity_link --aggregation_routing \
  --answerer_backend vllm --model_id vllm/Qwen/Qwen3-VL-32B-Thinking \
  --vllm_base_url http://localhost:8700/v1 \
  --vllm_api_key_env VLLM_API_KEY \
  --vllm_n_frames 64 --vllm_max_concurrency 4 \
  --frames_per_chunk 60 \
  --prompt_method filter_q3t_v3_q3vl \
  --mode all --metrics accuracy \
  --max_new_tokens 16384 \
  --questions_dir benchmark
```

### B1 — Qwen extractor + Gemini filter + Qwen answerer

```bash
python benchmark_sub_with_states.py \
  --state_strategy filter --aggregator_backend concat \
  --stage_b_backend gemini --stage_b_model gemini-3-flash-preview \
  --state_extractor_backend vllm --state_extractor_model Qwen/Qwen3-VL-32B-Thinking \
  --state_extractor_vllm_base_url http://localhost:8200/v1 \
  --state_extractor_vllm_api_key_env VLLM_API_KEY \
  --states_cache_dir cache/pipeline_b1_d1 \
  --chunk_prompt_version v6 --answerer_prompt_version v3 \
  --enable_identity_link --aggregation_routing \
  --answerer_backend vllm --model_id vllm/Qwen/Qwen3-VL-32B-Thinking \
  --vllm_base_url http://localhost:8700/v1 \
  --vllm_api_key_env VLLM_API_KEY \
  --vllm_n_frames 64 --vllm_max_concurrency 4 \
  --frames_per_chunk 60 \
  --prompt_method filter_gfl_v3_q3vl \
  --mode all --metrics accuracy \
  --max_new_tokens 16384 \
  --questions_dir benchmark
```

### C1 — Qwen extractor + Qwen filter + Gemini answerer

```bash
python benchmark_sub_with_states.py \
  --state_strategy filter --aggregator_backend concat \
  --stage_b_backend openrouter --stage_b_model qwen/qwen3-235b-a22b \
  --state_extractor_backend vllm --state_extractor_model Qwen/Qwen3-VL-32B-Thinking \
  --state_extractor_vllm_base_url http://localhost:8200/v1 \
  --state_extractor_vllm_api_key_env VLLM_API_KEY \
  --states_cache_dir cache/pipeline_c1 \
  --chunk_prompt_version v6 --answerer_prompt_version v3 \
  --enable_identity_link --aggregation_routing \
  --answerer_backend gemini --model_id gemini-3-flash-preview \
  --gemini_answerer_thinking_budget 0 --gemini_answerer_max_concurrency 4 \
  --vllm_n_frames 64 --frames_per_chunk 60 \
  --prompt_method filter_q3t_v3_gfl \
  --mode all --metrics accuracy \
  --max_new_tokens 4096 \
  --questions_dir benchmark
```

### D1 — All open-source (no Gemini)

```bash
python benchmark_sub_with_states.py \
  --state_strategy filter --aggregator_backend concat \
  --stage_b_backend openrouter --stage_b_model qwen/qwen3-235b-a22b \
  --state_extractor_backend vllm --state_extractor_model Qwen/Qwen3-VL-32B-Thinking \
  --state_extractor_vllm_base_url http://localhost:8200/v1 \
  --state_extractor_vllm_api_key_env VLLM_API_KEY \
  --states_cache_dir cache/pipeline_b1_d1 \
  --chunk_prompt_version v6 --answerer_prompt_version v3 \
  --enable_identity_link --aggregation_routing \
  --answerer_backend vllm --model_id vllm/Qwen/Qwen3-VL-32B-Thinking \
  --vllm_base_url http://localhost:8700/v1 \
  --vllm_api_key_env VLLM_API_KEY \
  --vllm_n_frames 64 --vllm_max_concurrency 4 \
  --frames_per_chunk 60 \
  --prompt_method filter_q3t_v3_q3vl \
  --mode all --metrics accuracy \
  --max_new_tokens 16384 \
  --questions_dir benchmark
```

### D1-Cans — Qwen3.5-27B replaces the Stage C answerer

Reuses D1's chunks AND q3t filter outputs. Only Stage C runs fresh. The `--vllm_fallback_*` flags are optional but recommended: they make the answerer transparently retry against a local vLLM (here at `http://localhost:8900/v1`) when OpenRouter returns a malformed or empty response, which is otherwise frequent enough on q35 to lose ~30/88 videos to fallback-less skips. To use the fallback, start a vLLM instance with the matching `--served-model-name` and bump `--max-model-len` high enough for 64 frames + thinking (we recommend 131072).

```bash
python benchmark_sub_with_states.py \
  --state_strategy filter --aggregator_backend concat \
  --stage_b_backend openrouter --stage_b_model qwen/qwen3-235b-a22b \
  --state_extractor_backend openrouter --state_extractor_model qwen/qwen3-vl-235b-a22b-thinking \
  --states_cache_dir cache/pipeline_b1_d1 \
  --chunk_prompt_version v6 --answerer_prompt_version v3 \
  --enable_identity_link --aggregation_routing \
  --answerer_backend vllm --model_id vllm/qwen/qwen3.5-27b \
  --vllm_base_url https://openrouter.ai/api/v1 \
  --vllm_api_key_env OPENROUTER_API_KEY \
  --vllm_n_frames 64 --vllm_max_concurrency 4 \
  --frames_per_chunk 60 \
  --vllm_fallback_base_url http://localhost:8900/v1 \
  --vllm_fallback_model_id Qwen3.5-27B \
  --prompt_method filter_q3t_v3_q35 \
  --mode all --metrics accuracy \
  --max_new_tokens 32768 \
  --questions_dir benchmark
```

## Cache layout

Stage outputs are cached under `--states_cache_dir`. Layout per video:

```
<cache_dir>/<video_stem>/
├── chunks.json                     # Stage A1 per-chunk states
├── stage_a_concat.txt              # concatenated states text
├── plan.json                       # extractor plan
├── aliases.txt, aliases.json       # cross-chunk identity table (one per video)
├── filter[<model_tag>]/<qid>.json  # Stage B filter outputs (suffixed by model)
└── answers_<prompt_method>/<qid>.json  # Stage C final answers
```

**Aliases caveat.** `aliases.txt` / `aliases.json` are written once per video by the first run that reaches the identity-link step, and reused unconditionally by every subsequent run regardless of which Stage-B backend is configured. If you need strict per-config alias semantics, delete `aliases.txt` and `aliases.json` from each `<video_stem>/` directory before the run.

## Notes

- `--max_new_tokens 16384` is required when the answerer is `q3vl` (a thinking model — reasoning tokens consume budget before the final `Evidence:` / `Answer:` lines). Use `4096` for non-thinking answerers.
- `VLLM_API_KEY` / `VLLM_BASE_URL` are used for local Qwen3-VL vLLM calls. `OPENROUTER_API_KEY` is still required whenever Stage B, planner, aggregator, or identity linking uses an OpenRouter-backed text model such as `q3t`.
- B1 and D1 share `cache/pipeline_b1_d1/` so the `q3vl` Stage-A1 chunks are extracted once and reused. Within that directory, B1 and D1 use different filters and different `--prompt_method` tags, which means Stage-B outputs (`filter/` for B1's gemini filter vs. `filter_qwen3-235b-a22b/` for D1's q3t filter) and Stage-C outputs (`answers_filter_gfl_v3_q3vl/` vs. `answers_filter_q3t_v3_q3vl/`) live in distinct subdirectories and never collide.
- C1 runs Stage A1 fresh against its own `cache/pipeline_c1/` directory. To avoid paying for re-extraction, symlink `chunks.json`, `stage_a_concat.txt`, and `plan.json` from `cache/pipeline_b1_d1/<video>/` into `cache/pipeline_c1/<video>/` before running C1.


python benchmark_sub_with_states.py --state_strategy filter --aggregator_backend concat --stage_b_backend openrouter --stage_b_model qwen/qwen3-235b-a22b --state_extractor_backend openrouter --state_extractor_model qwen/qwen3-vl-235b-a22b-thinking --states_cache_dir cache/pipeline_b1_d1 --chunk_prompt_version v6 --answerer_prompt_version v3 --enable_identity_link --aggregation_routing --answerer_backend vllm --model_id vllm/qwen/qwen3.5-27b --vllm_base_url https://openrouter.ai/api/v1 --vllm_api_key_env OPENROUTER_API_KEY --vllm_n_frames 64 --vllm_max_concurrency 4 --frames_per_chunk 60 --vllm_fallback_base_url http://localhost:8900/v1 --vllm_fallback_model_id Qwen3.5-27B --prompt_method filter_q3t_v3_q35 --mode all --metrics accuracy --max_new_tokens 16384 --questions_dir benchmark