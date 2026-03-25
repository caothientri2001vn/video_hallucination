"""
src/models/gemini.py
---------------------
Google Gemini API backend.

STATUS: Stub — not yet implemented.

Implementation notes (for when API access is available)
--------------------------------------------------------
* Use the `google-genai` SDK: ``pip install google-genai``
* Gemini supports direct video upload via the Files API, which is preferred
  for videos longer than ~30 s.  Set ``video_upload=True`` in configs/models.yaml
  to enable this path; otherwise frames are extracted and base64-encoded.
* The API key is read from the env var defined in configs/models.yaml
  (default: ``GOOGLE_API_KEY``).

Example skeleton — Files API path (do not delete — fill in when ready)
-----------------------------------------------------------------------
    import google.genai as genai

    client = genai.Client(api_key=os.environ[api_key_env])
    video_file = client.files.upload(file=video_path)
    # Poll until processing is complete
    while video_file.state.name == "PROCESSING":
        time.sleep(2)
        video_file = client.files.get(name=video_file.name)
    response = client.models.generate_content(
        model=self.model_id,
        contents=[video_file, prompt],
    )
    answer = response.text

Example skeleton — base64 frames path
--------------------------------------
    parts = [
        genai.types.Part.from_bytes(data=frame_bytes, mime_type="image/jpeg")
        for frame_bytes in _extract_frames_bytes(video_path, self.n_frames)
    ]
    parts.append(genai.types.Part.from_text(prompt))
    response = client.models.generate_content(model=self.model_id, contents=parts)
    answer = response.text
"""

import os
from typing import List

from .base import BaseVideoQAModel


class GeminiModel(BaseVideoQAModel):
    """
    Google Gemini API backend (stub).

    Parameters
    ----------
    model_id : str
        Gemini model name, e.g. ``"gemini-2.5-pro"``.
    prompt_method : str
        Prompt template label (affects cache namespace).  Default: "vanilla".
    n_frames : int
        Frames to extract when NOT using the Files API.  Default: 16.
    video_upload : bool
        If True, upload the whole video via the Files API.  Default: False.
    api_key_env : str
        Env-var name for the Google API key.  Default: ``"GOOGLE_API_KEY"``.
    """

    def __init__(
        self,
        model_id: str,
        prompt_method: str = "vanilla",
        n_frames: int = 16,
        video_upload: bool = False,
        api_key_env: str = "GOOGLE_API_KEY",
    ) -> None:
        super().__init__(model_id, prompt_method)
        self.n_frames = n_frames
        self.video_upload = video_upload
        self.api_key_env = api_key_env

    def answer_questions(
        self,
        video_path: str,
        questions: List[str],
        max_new_tokens: int = 256,
    ) -> List[str]:
        raise NotImplementedError(
            "GeminiModel is not yet implemented.  "
            "See the docstring in src/models/gemini.py for the implementation plan."
        )