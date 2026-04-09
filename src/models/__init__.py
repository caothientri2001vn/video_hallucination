"""
src/models/__init__.py
-----------------------
Model registry.

``load_model`` is the single entry point used by benchmark scripts.
It maps a model_id string to the correct backend class without any
if/elif chains in the calling code.

Routing rules (checked in order)
---------------------------------
1. Starts with ``"claude"``                    → ClaudeModel   (Anthropic API)
2. Starts with ``"gemini"``                    → GeminiModel   (Google API)
3. Starts with ``"gpt"`` or ``"o1"``/``"o3"`` → OpenAIModel   (OpenAI API)
4. Anything else                               → Qwen3VLModel  (local HuggingFace)

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

from typing import Callable, Optional, Type

from .base import BaseVideoQAModel

__all__ = [
    "BaseVideoQAModel",
    "load_model",
]

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


def _load_openai() -> Type[BaseVideoQAModel]:
    from .openai_gpt import OpenAIModel  # requires: openai
    return OpenAIModel




def _load_videorft() -> Type[BaseVideoQAModel]:
    from .video_rft import VideoRFTModel  # requires: transformers>=4.51, qwen-vl-utils
    return VideoRFTModel


def _load_videor1() -> Type[BaseVideoQAModel]:
    from .video_r1 import VideoR1Model  # requires: transformers>=4.51, qwen-vl-utils
    return VideoR1Model


def _load_qwen3vl() -> Type[BaseVideoQAModel]:
    from .qwen3_vl import Qwen3VLModel  # requires: transformers==4.57.1, qwen-vl-utils
    return Qwen3VLModel


def _load_qwen35vl() -> Type[BaseVideoQAModel]:
    from .qwen35_vl import Qwen35VLModel  # requires: transformers@HEAD, qwen-vl-utils
    return Qwen35VLModel


def _load_traveler(model_id: str, prompt_method: str, **kwargs) -> BaseVideoQAModel:
    """Instantiate TraveLERModel. model_id format: ``"traveler/<vllm_model_name>"``."""
    from .traveler import TraveLERModel  # requires: openai
    return TraveLERModel(model_id=model_id, prompt_method=prompt_method, **kwargs)


def _load_frame_pipeline(
    model_id: str,
    prompt_method: str,
    **kwargs,
) -> BaseVideoQAModel:
    """
    Parse ``"<selector>+<backbone>"`` and return a ``FramePipelineModel`` instance.
    Called directly from ``load_model`` — returns an instance, not a class.
    """
    from .frame_pipeline import FramePipelineModel
    selector_name, backbone_model_id = model_id.split("+", 1)
    return FramePipelineModel(
        selector_name=selector_name.lower(),
        backbone_model_id=backbone_model_id,
        prompt_method=prompt_method,
        **kwargs,
    )


_FRAME_SELECTOR_NAMES = {"uniform", "clip", "aks", "efs"}

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
    "qiwang98/": _load_videorft,
    "video-r1/": _load_videor1,
}

# ---------------------------------------------------------------------------
# OpenRouter: model_id convention is "openrouter/<provider>/<model-slug>".
# Routed separately because we need to pass extra kwargs (base_url, api_key_env)
# to the OpenAIModel constructor rather than just swapping the class.
# ---------------------------------------------------------------------------
_OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
_OPENROUTER_PREFIX = "openrouter/"


def load_model(
    model_id: str,
    prompt_method: str = "vanilla",
    debug_with_n_frames: Optional[int] = None,
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

        1. Starts with ``"openrouter/"``           → OpenAIModel via OpenRouter API
        2. Matches a prefix in ``_LAZY_PREFIX_MAP`` → corresponding API backend
        3. Anything else                            → Qwen3VLModel (local HuggingFace)

        OpenRouter model IDs use the convention ``"openrouter/<provider>/<slug>"``,
        e.g. ``"openrouter/google/gemini-2.5-pro"``.  The ``"openrouter/"`` prefix
        is stripped before passing to the model so the provider sees the real slug.
    prompt_method : str
        Short label for the prompt template.  Injected into every backend so
        it flows through to the cache namespace.  Default: ``"vanilla"``.
    debug_with_n_frames : int | None
        Only meaningful for local (Qwen3VL) models.  Ignored by API backends.
    **kwargs
        Forwarded to the backend constructor.  Useful for overriding
        per-model defaults like ``n_frames``, ``video_upload``, etc.

    Returns
    -------
    BaseVideoQAModel
        Ready-to-use model instance.
    """
    lower = model_id.lower()

    # 1. TraveLER multi-agent pipeline: "traveler/<vllm_model_name>"
    if lower.startswith("traveler/"):
        return _load_traveler(model_id=model_id, prompt_method=prompt_method, **kwargs)

    # 2. Frame-selection pipeline: "<selector>+<backbone_model_id>"
    if "+" in model_id:
        prefix = model_id.split("+", 1)[0].lower()
        if prefix in _FRAME_SELECTOR_NAMES:
            return _load_frame_pipeline(
                model_id=model_id,
                prompt_method=prompt_method,
                **kwargs,
            )

    # 2. OpenRouter
    if lower.startswith(_OPENROUTER_PREFIX):
        from .openai_gpt import OpenAIModel
        # Strip the "openrouter/" prefix — the real slug goes to the API
        real_model_id = model_id[len(_OPENROUTER_PREFIX):]
        return OpenAIModel(
            model_id=real_model_id,
            prompt_method=prompt_method,
            api_key_env="OPENROUTER_API_KEY",
            base_url=_OPENROUTER_BASE_URL,
            **kwargs,
        )

    # 2. Named API backends
    for prefix, loader in _LAZY_PREFIX_MAP.items():
        if lower.startswith(prefix):
            cls = loader()
            return cls(model_id=model_id, prompt_method=prompt_method, **kwargs)

    # 3. Default: local HuggingFace model
    cls = _load_qwen3vl()
    return cls(
        model_id=model_id,
        prompt_method=prompt_method,
        debug_with_n_frames=debug_with_n_frames,
        **kwargs,
    )