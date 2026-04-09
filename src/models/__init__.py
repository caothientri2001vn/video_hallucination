"""
src/models/__init__.py
-----------------------
Model registry.

``load_model`` is the single entry point used by benchmark scripts.
It maps a model_id string to the correct backend class without any
if/elif chains in the calling code.

Routing rules (checked in order)
---------------------------------
1. Starts with ``"vllm/"``                     → VLLMOpenAIModel (local vLLM)
2. Starts with ``"openrouter/"``               → OpenAIModel     (OpenRouter API)
3. Starts with ``"claude"``                    → ClaudeModel     (Anthropic API)
4. Starts with ``"gemini"``                    → GeminiModel     (Google API)
5. Starts with ``"gpt"`` or ``"o1"``/``"o3"`` → OpenAIModel     (OpenAI API)
6. Anything else                               → Qwen3VLModel    (local HuggingFace)

Lazy imports
------------
Each backend is imported only when it is actually selected by ``load_model``.
This means you do NOT need to install the dependencies for backends you are
not using.  For example, running Gemini does not require ``transformers`` to
be installed, and vice versa.

Adding a new backend
---------------------
1. Create ``src/models/your_model.py`` inheriting from ``BaseVideoQAModel``.
2. Add a lambda (or function) to ``_LAZY_PREFIX_MAP`` below that imports and
   returns the class, then add the routing prefix string.
"""

from typing import Any, Callable, Optional, Type

from .base import BaseVideoQAModel

__all__ = [
    "BaseVideoQAModel",
    "load_model",
]

_OPENAI_FAMILY_DEFAULT_N_FRAMES = 32
_OPENROUTER_ANTHROPIC_DEFAULT_N_FRAMES = 128
_OPENROUTER_QWEN_DEFAULT_N_FRAMES = 64
_OPENROUTER_INTERNVL_DEFAULT_N_FRAMES = 64
_OPENROUTER_GEMINI_DEFAULT_N_FRAMES = 128


# ---------------------------------------------------------------------------
# Lazy loader helpers — each value is a zero-arg callable that imports and
# returns the backend *class* (not an instance).  The import happens only
# when load_model actually routes to that backend.
# ---------------------------------------------------------------------------

def _load_claude() -> Type[BaseVideoQAModel]:
    from .claude import ClaudeModel  # requires: anthropic
    return ClaudeModel


def _load_gemini() -> Type[BaseVideoQAModel]:
    from .gemini import GeminiModel  # requires: google-genai
    return GeminiModel


def _load_pseudo_gemini() -> Type[BaseVideoQAModel]:
    from .pseudo_gemini import PseudoGeminiModel  # requires: google-genai
    return PseudoGeminiModel


def _load_openai() -> Type[BaseVideoQAModel]:
    from .openai_gpt import OpenAIModel  # requires: openai
    return OpenAIModel


def _load_vllm_openai() -> Type[BaseVideoQAModel]:
    from .vllm_openai import VLLMOpenAIModel  # requires: openai
    return VLLMOpenAIModel


def _load_openrouter_gemini() -> Type[BaseVideoQAModel]:
    from .openrouter_gemini import OpenRouterGeminiModel  # requires: openai
    return OpenRouterGeminiModel


def _load_qwen3vl() -> Type[BaseVideoQAModel]:
    from .qwen3_vl import Qwen3VLModel  # requires: transformers==4.57.1, qwen-vl-utils
    return Qwen3VLModel


def _load_qwen35vl() -> Type[BaseVideoQAModel]:
    from .qwen35_vl import Qwen35VLModel  # requires: transformers@HEAD, qwen-vl-utils
    return Qwen35VLModel


# ---------------------------------------------------------------------------
# Prefix map for regular backends (prefix → lazy class loader).
# NOTE: longer prefixes must come first so they match before shorter ones
# (e.g. "qwen3.5" before a hypothetical "qwen3" catch-all, "qwen/qwen3.5"
# before "qwen3.5").
# ---------------------------------------------------------------------------
_LAZY_PREFIX_MAP: dict[str, Callable[[], Type[BaseVideoQAModel]]] = {
    "claude": _load_claude,
    "gemini": _load_gemini,
    "gpt": _load_openai,
    "o1": _load_openai,
    "o3": _load_openai,
    "qwen/qwen3.5": _load_qwen35vl,
    "qwen3.5": _load_qwen35vl,
}

# ---------------------------------------------------------------------------
# OpenRouter: model_id convention is "openrouter/<provider>/<model-slug>".
# Routed separately because we need to pass extra kwargs (base_url, api_key_env)
# to the OpenAIModel constructor rather than just swapping the class.
# ---------------------------------------------------------------------------
_VLLM_PREFIX = "vllm/"
_OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
_OPENROUTER_PREFIX = "openrouter/"
_OPENROUTER_ANTHROPIC_PREFIX = "openrouter/anthropic/"
_OPENROUTER_GOOGLE_GEMINI_PREFIX = "openrouter/google/gemini"
_OPENROUTER_QWEN_PREFIX = "openrouter/qwen/"
_OPENROUTER_INTERNVL_PREFIX = "openrouter/internvl/"
_PSEUDO_GEMINI_PREFIXES = ("pseudo_gemini/", "pseudo-gemini/")


def _with_openai_compatible_defaults(model_id: str, kwargs: dict[str, Any]) -> dict[str, Any]:
    resolved = dict(kwargs)
    if "n_frames" not in resolved:
        lower = model_id.lower()
        if lower.startswith(_OPENROUTER_ANTHROPIC_PREFIX):
            resolved["n_frames"] = _OPENROUTER_ANTHROPIC_DEFAULT_N_FRAMES
        elif lower.startswith(_OPENROUTER_QWEN_PREFIX):
            resolved["n_frames"] = _OPENROUTER_QWEN_DEFAULT_N_FRAMES
        elif lower.startswith(_OPENROUTER_INTERNVL_PREFIX):
            resolved["n_frames"] = _OPENROUTER_INTERNVL_DEFAULT_N_FRAMES
        else:
            resolved["n_frames"] = _OPENAI_FAMILY_DEFAULT_N_FRAMES
    return resolved


def load_model(
    model_id: str,
    prompt_method: str = "vanilla",
    debug_with_n_frames: Optional[int] = None,
    force_fps: Optional[float] = None,
    **kwargs,
) -> BaseVideoQAModel:
    """
    Instantiate the right model backend for *model_id*.

    Each backend is imported lazily so that missing optional dependencies
    (e.g. ``anthropic``, ``google-genai``, ``transformers``) only cause an
    error when you actually try to use that backend, not at import time.

    Parameters
    ----------
    model_id : str
        Model identifier.  Routing rules (checked in order):

        1. Starts with ``"vllm/"``                 → VLLMOpenAIModel via local vLLM
        2. Starts with ``"openrouter/"``           → OpenAIModel via OpenRouter API
        3. Matches a prefix in ``_LAZY_PREFIX_MAP`` → corresponding API backend
        4. Anything else                            → Qwen3VLModel (local HuggingFace)

        OpenRouter model IDs use the convention ``"openrouter/<provider>/<slug>"``,
        e.g. ``"openrouter/google/gemini-2.5-pro"``.  The ``"openrouter/"`` prefix
        is stripped before passing to the model so the provider sees the real slug.
    prompt_method : str
        Short label for the prompt template.  Injected into every backend so
        it flows through to the cache namespace.  Default: ``"vanilla"``.
    debug_with_n_frames : int | None
        Only meaningful for local (Qwen3VL) models.  Ignored by API backends.
    force_fps : float | None
        Only meaningful for the local Qwen3-VL backend. When set, injects a
        fixed FPS into video preprocessing, e.g. ``4`` for Cosmos-Reason2.
    **kwargs
        Forwarded to the backend constructor.  Useful for overriding
        per-model defaults like ``n_frames``, ``video_upload``, etc.

    Returns
    -------
    BaseVideoQAModel
        Ready-to-use model instance.
    """
    lower = model_id.lower()

    # 1. Local vLLM
    if lower.startswith(_VLLM_PREFIX):
        real_model_id = model_id[len(_VLLM_PREFIX):]
        cls = _load_vllm_openai()
        resolved_kwargs = _with_openai_compatible_defaults(model_id, kwargs)
        return cls(
            model_id=real_model_id,
            prompt_method=prompt_method,
            **resolved_kwargs,
        )

    # 2. OpenRouter
    if lower.startswith(_OPENROUTER_PREFIX):
        real_model_id = model_id[len(_OPENROUTER_PREFIX):]
        if lower.startswith(_OPENROUTER_GOOGLE_GEMINI_PREFIX):
            cls = _load_openrouter_gemini()
            resolved_kwargs = dict(kwargs)
            resolved_kwargs.setdefault("n_frames", _OPENROUTER_GEMINI_DEFAULT_N_FRAMES)
            resolved_kwargs.setdefault("prefer_video", False)
            return cls(
                model_id=real_model_id,
                prompt_method=prompt_method,
                api_key_env="OPENROUTER_API_KEY",
                base_url=_OPENROUTER_BASE_URL,
                **resolved_kwargs,
            )

        from .openai_gpt import OpenAIModel

        resolved_kwargs = _with_openai_compatible_defaults(model_id, kwargs)
        return OpenAIModel(
            model_id=real_model_id,
            prompt_method=prompt_method,
            api_key_env="OPENROUTER_API_KEY",
            base_url=_OPENROUTER_BASE_URL,
            **resolved_kwargs,
        )

    # 3. Pseudo Gemini estimator
    for prefix in _PSEUDO_GEMINI_PREFIXES:
        if lower.startswith(prefix):
            real_model_id = model_id[len(prefix):]
            cls = _load_pseudo_gemini()
            resolved_kwargs = dict(kwargs)
            resolved_kwargs.setdefault("n_frames", _OPENROUTER_GEMINI_DEFAULT_N_FRAMES)
            return cls(
                model_id=model_id,
                api_model_id=real_model_id,
                prompt_method=prompt_method,
                **resolved_kwargs,
            )

    # 4. Named API backends
    for prefix, loader in _LAZY_PREFIX_MAP.items():
        if lower.startswith(prefix):
            cls = loader()
            resolved_kwargs = kwargs
            if prefix in {"gpt", "o1", "o3"}:
                resolved_kwargs = _with_openai_compatible_defaults(model_id, kwargs)
            return cls(model_id=model_id, prompt_method=prompt_method, **resolved_kwargs)

    # 5. Default: local HuggingFace model
    cls = _load_qwen3vl()
    return cls(
        model_id=model_id,
        prompt_method=prompt_method,
        debug_with_n_frames=debug_with_n_frames,
        force_fps=force_fps,
        **kwargs,
    )
