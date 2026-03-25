"""
src/models/__init__.py
-----------------------
Model registry.

``load_model`` is the single entry point used by benchmark scripts.
It maps a model_id string to the correct backend class without any
if/elif chains in the calling code.

Routing rules (checked in order)
---------------------------------
1. Starts with ``"claude"``        → ClaudeModel   (Anthropic API)
2. Starts with ``"gemini"``        → GeminiModel   (Google API)
3. Starts with ``"gpt"`` or ``"o1"``/``"o3"``  → OpenAIModel   (OpenAI API)
4. Anything else                   → Qwen3VLModel  (local HuggingFace)

Adding a new backend
---------------------
1. Create ``src/models/your_model.py`` inheriting from ``BaseVideoQAModel``.
2. Import it here and add a routing rule in ``load_model``.
"""

from typing import Optional

from .base import BaseVideoQAModel
from .claude import ClaudeModel
from .gemini import GeminiModel
from .openai_gpt import OpenAIModel
from .qwen3_vl import Qwen3VLModel

__all__ = [
    "BaseVideoQAModel",
    "ClaudeModel",
    "GeminiModel",
    "OpenAIModel",
    "Qwen3VLModel",
    "load_model",
]

# Prefixes that identify API-based backends.
# Extend this dict when adding new providers.
_PREFIX_MAP = {
    "claude": ClaudeModel,
    "gemini": GeminiModel,
    "gpt": OpenAIModel,
    "o1": OpenAIModel,
    "o3": OpenAIModel,
}


def load_model(
    model_id: str,
    prompt_method: str = "vanilla",
    debug_with_n_frames: Optional[int] = None,
    **kwargs,
) -> BaseVideoQAModel:
    """
    Instantiate the right model backend for *model_id*.

    Parameters
    ----------
    model_id : str
        Model identifier.  API models are identified by prefix (see module
        docstring).  Everything else is treated as a HuggingFace repo ID
        and loaded locally with Qwen3VLModel.
    prompt_method : str
        Short label for the prompt template.  Injected into every backend so
        it flows through to the cache namespace.  Default: ``"vanilla"``.
    debug_with_n_frames : int | None
        Only meaningful for local (Qwen3VL) models.  Ignored by API stubs.
    **kwargs
        Forwarded to the backend constructor.  Useful for overriding
        per-model defaults like ``n_frames``, ``video_upload``, etc.

    Returns
    -------
    BaseVideoQAModel
        Ready-to-use model instance.
    """
    lower = model_id.lower()
    for prefix, cls in _PREFIX_MAP.items():
        if lower.startswith(prefix):
            return cls(model_id=model_id, prompt_method=prompt_method, **kwargs)

    # Default: local HuggingFace model
    return Qwen3VLModel(
        model_id=model_id,
        prompt_method=prompt_method,
        debug_with_n_frames=debug_with_n_frames,
        **kwargs,
    )