"""
src/models/openai_gpt.py
-------------------------
OpenAI GPT-4o / GPT-4-vision API backend.
"""

import base64
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict, List, Optional

import cv2

try:
    from openai import OpenAI
except Exception as exc:
    raise RuntimeError(
        "Failed to import the OpenAI SDK. Install with: uv sync --group openai\n"
        f"Error: {exc}"
    ) from exc

from .base import BaseVideoQAModel


_DEFAULT_N_FRAMES = 32
_DEFAULT_MAX_CONCURRENCY = 4


def _encode_frame_b64(frame: Any) -> str:
    ok, buf = cv2.imencode(".jpg", frame)
    if not ok:
        raise ValueError("Failed to encode sampled video frame as JPEG")
    return base64.b64encode(buf.tobytes()).decode("ascii")


def _extract_frames_b64(video_path: str, n_frames: int) -> List[str]:
    """
    Sample exactly ``n_frames`` uniformly-spaced frame slots and return them as
    base64-encoded JPEG strings.

    If the source video has fewer than ``n_frames`` decoded frames, some slots
    will map to the same underlying frame. This preserves a fixed-size visual
    prompt for the API backend.
    """
    if n_frames <= 0:
        raise ValueError(f"n_frames must be positive, got {n_frames}")
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video file not found: {video_path}")

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Failed to open video: {video_path}")

    try:
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        if total_frames <= 0:
            ok, frame = cap.read()
            if not ok or frame is None:
                raise ValueError(f"Could not read any frames from video: {video_path}")
            encoded = _encode_frame_b64(frame)
            return [encoded] * n_frames

        if n_frames == 1:
            indices = [total_frames // 2]
        else:
            indices = [
                round(i * (total_frames - 1) / (n_frames - 1))
                for i in range(n_frames)
            ]

        frames_b64: List[str] = []
        cache: Dict[int, str] = {}
        for idx in indices:
            if idx not in cache:
                cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
                ok, frame = cap.read()
                if not ok or frame is None:
                    continue
                cache[idx] = _encode_frame_b64(frame)
            frames_b64.append(cache[idx])

        if not frames_b64:
            raise ValueError(f"Failed to extract frames from video: {video_path}")

        if len(frames_b64) < n_frames:
            frames_b64.extend([frames_b64[-1]] * (n_frames - len(frames_b64)))

        return frames_b64
    finally:
        cap.release()


class OpenAIModel(BaseVideoQAModel):
    """
    OpenAI-compatible API backend (GPT-4o, GPT-5, or any OpenRouter model).

    Parameters
    ----------
    model_id : str
        Model name, e.g. ``"gpt-4o"``, ``"gpt-5"`` or an OpenRouter slug like
        ``"openrouter/google/gemini-2.5-pro"``.
    prompt_method : str
        Prompt template label (affects cache namespace).  Default: "vanilla".
    n_frames : int
        Number of uniformly-spaced frame slots to send for every video.
        Default: 32.
    api_key_env : str
        Env-var name for the API key.  Default: ``"OPENAI_API_KEY"``.
        For OpenRouter set to ``"OPENROUTER_API_KEY"``.
    base_url : str | None
        Override the API base URL.  Pass ``"https://openrouter.ai/api/v1"``
        to route through OpenRouter.  Default: None (uses the openai SDK
        default, i.e. ``"https://api.openai.com/v1"``).
    max_concurrency : int
        Maximum number of per-question API calls to run in parallel.
        Default: 4.
    """

    def __init__(
        self,
        model_id: str,
        prompt_method: str = "vanilla",
        n_frames: int = _DEFAULT_N_FRAMES,
        api_key_env: str = "OPENAI_API_KEY",
        base_url: Optional[str] = None,
        max_concurrency: int = _DEFAULT_MAX_CONCURRENCY,
    ) -> None:
        super().__init__(model_id, prompt_method)
        if max_concurrency <= 0:
            raise ValueError(f"max_concurrency must be positive, got {max_concurrency}")
        self.n_frames = n_frames
        self.api_key_env = api_key_env
        self.base_url = base_url
        self.max_concurrency = max_concurrency
        self._client: Optional[OpenAI] = None

    def answer_questions(
        self,
        video_path: str,
        questions: List[str],
        max_new_tokens: int = 256,
    ) -> List[str]:
        if not questions:
            return []

        normalized_questions: List[str] = []
        for question in questions:
            question = question.strip()
            if not question:
                raise ValueError(f"Empty question passed to {self!r}")
            normalized_questions.append(question)

        frames_b64 = _extract_frames_b64(video_path, self.n_frames)
        results = [""] * len(normalized_questions)
        worker_count = min(self.max_concurrency, len(normalized_questions))

        if worker_count == 1:
            for idx, question in enumerate(normalized_questions):
                results[idx] = self._answer_one(question, frames_b64, max_new_tokens)
            return results

        with ThreadPoolExecutor(max_workers=worker_count) as executor:
            future_to_idx = {
                executor.submit(self._answer_one, question, frames_b64, max_new_tokens): idx
                for idx, question in enumerate(normalized_questions)
            }
            for future in as_completed(future_to_idx):
                idx = future_to_idx[future]
                results[idx] = future.result()

        return results

    def _get_client(self) -> OpenAI:
        if self._client is not None:
            return self._client

        api_key = os.environ.get(self.api_key_env)
        if not api_key:
            raise RuntimeError(
                f"Environment variable {self.api_key_env!r} is not set; "
                f"required for model {self.model_id!r}."
            )

        kwargs: Dict[str, Any] = {"api_key": api_key}
        if self.base_url:
            kwargs["base_url"] = self.base_url

        self._client = OpenAI(**kwargs)
        return self._client

    def _answer_one(
        self,
        question: str,
        frames_b64: List[str],
        max_new_tokens: int,
    ) -> str:
        client = self._get_client()
        response = client.chat.completions.create(
            model=self.model_id,
            messages=[{"role": "user", "content": self._build_content(question, frames_b64)}],
            max_tokens=int(max_new_tokens),
        )
        answer = response.choices[0].message.content
        if isinstance(answer, str):
            return answer.strip()
        if answer is None:
            return ""
        return str(answer).strip()

    def _build_content(self, question: str, frames_b64: List[str]) -> List[Dict[str, Any]]:
        grounding = (
            "IMPORTANT: only use information you can directly verify from the "
            "video frames. Answer the question directly. When possible, cite rough timestamps."
        )
        prompt = f"{grounding}\n\nQuestion: {question}"

        content: List[Dict[str, Any]] = [
            {
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/jpeg;base64,{frame_b64}",
                    "detail": "low",
                },
            }
            for frame_b64 in frames_b64
        ]
        content.append({"type": "text", "text": prompt})
        return content
