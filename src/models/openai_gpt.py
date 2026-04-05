"""
src/models/openai_gpt.py
-------------------------
OpenAI GPT-4o / GPT-4-vision API backend.

STATUS: Stub — not yet implemented.

Implementation notes (for when API access is available)
--------------------------------------------------------
* Use the `openai` SDK: ``pip install openai``
* OpenAI does not accept raw video.  Extract N evenly-spaced frames, encode
  each as base64 JPEG, and pass them as ``image_url`` content blocks with
  ``url: "data:image/jpeg;base64,<data>"``.
* The API key is read from the env var defined in configs/models.yaml
  (default: ``OPENAI_API_KEY``).

Example skeleton (do not delete — fill in when ready)
------------------------------------------------------
    import openai, base64, cv2

    client = openai.OpenAI(api_key=os.environ[api_key_env])
    frames_b64 = _extract_frames_b64(video_path, n_frames=self.n_frames)
    content = [
        {
            "type": "image_url",
            "image_url": {"url": f"data:image/jpeg;base64,{f}", "detail": "low"},
        }
        for f in frames_b64
    ]
    content.append({"type": "text", "text": prompt})
    response = client.chat.completions.create(
        model=self.model_id,
        messages=[{"role": "user", "content": content}],
        max_tokens=max_new_tokens,
    )
    answer = response.choices[0].message.content
"""

import os
from typing import List, Optional

from .base import BaseVideoQAModel


class OpenAIModel(BaseVideoQAModel):
    """
    OpenAI-compatible API backend (GPT-4o, or any OpenRouter model).

    Parameters
    ----------
    model_id : str
        Model name, e.g. ``"gpt-4o"`` or an OpenRouter slug like
        ``"openrouter/google/gemini-2.5-pro"``.
    prompt_method : str
        Prompt template label (affects cache namespace).  Default: "vanilla".
    n_frames : int
        Number of evenly-spaced frames to extract from the video.  Default: 16.
    api_key_env : str
        Env-var name for the API key.  Default: ``"OPENAI_API_KEY"``.
        For OpenRouter set to ``"OPENROUTER_API_KEY"``.
    base_url : str | None
        Override the API base URL.  Pass ``"https://openrouter.ai/api/v1"``
        to route through OpenRouter.  Default: None (uses the openai SDK
        default, i.e. ``"https://api.openai.com/v1"``).
    """

    def __init__(
        self,
        model_id: str,
        prompt_method: str = "vanilla",
        n_frames: int = 16,
        api_key_env: str = "OPENAI_API_KEY",
        base_url: Optional[str] = None,
    ) -> None:
        super().__init__(model_id, prompt_method)
        self.n_frames = n_frames
        self.api_key_env = api_key_env
        self.base_url = base_url

    def answer_questions(
        self,
        video_path: str,
        questions: List[str],
        max_new_tokens: int = 256,
    ) -> List[str]:
        raise NotImplementedError(
            "OpenAIModel is not yet implemented.  "
            "See the docstring in src/models/openai_gpt.py for the implementation plan."
        )