"""
src/models/openai_gpt.py
-------------------------
OpenAI GPT-4o / GPT-4-vision API backend.
"""

import base64
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from collections import Counter
from typing import Any, Dict, List, Optional

try:
    import av
except Exception as exc:
    raise RuntimeError(
        "Failed to import PyAV. Install with: uv sync --group openai\n"
        f"Error: {exc}"
    ) from exc

import cv2

try:
    from openai import BadRequestError, OpenAI
except Exception as exc:
    raise RuntimeError(
        "Failed to import the OpenAI SDK. Install with: uv sync --group openai\n"
        f"Error: {exc}"
    ) from exc

from .base import BaseVideoQAModel

_FALLBACK_N_FRAMES = 32
_DEFAULT_MAX_CONCURRENCY = 4
_MIN_RETRY_FRAMES = 4


def _resize_longest_side_bgr(bgr_frame: Any, max_longest_side: Optional[int]) -> Any:
    if max_longest_side is None:
        return bgr_frame
    if max_longest_side <= 0:
        raise ValueError(f"max_longest_side must be positive, got {max_longest_side}")

    h, w = bgr_frame.shape[:2]
    if h <= 0 or w <= 0:
        raise ValueError(f"Invalid frame shape: {bgr_frame.shape}")

    longest_side = max(h, w)
    if longest_side <= max_longest_side:
        return bgr_frame

    scale = max_longest_side / longest_side
    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))
    return cv2.resize(bgr_frame, (new_w, new_h), interpolation=cv2.INTER_AREA)


def _encode_frame_b64(
    frame: av.VideoFrame,
    max_longest_side: Optional[int] = None,
) -> str:
    bgr_frame = frame.to_ndarray(format="bgr24")
    bgr_frame = _resize_longest_side_bgr(bgr_frame, max_longest_side)
    ok, buf = cv2.imencode(".jpg", bgr_frame)
    if not ok:
        raise ValueError("Failed to encode sampled video frame as JPEG")
    return base64.b64encode(buf.tobytes()).decode("ascii")


def _sample_evenly(items: List[str], n: int) -> List[str]:
    if not items:
        raise ValueError("Cannot sample from an empty frame list")
    if n <= 1:
        return [items[len(items) // 2]]
    if len(items) == 1:
        return items * n
    indices = [round(i * (len(items) - 1) / (n - 1)) for i in range(n)]
    return [items[idx] for idx in indices]


def _get_video_stream(container: av.container.input.InputContainer) -> av.video.stream.VideoStream:
    if not container.streams.video:
        raise ValueError("Video file does not contain any video streams")
    return container.streams.video[0]


def _extract_frames_b64_sequential(
    video_path: str,
    n_frames: int,
    max_longest_side: Optional[int] = None,
) -> List[str]:
    with av.open(video_path) as container:
        stream = _get_video_stream(container)
        decoded: List[str] = []
        for frame in container.decode(stream):
            decoded.append(_encode_frame_b64(frame, max_longest_side=max_longest_side))

        if not decoded:
            raise ValueError(f"Failed to decode any frames sequentially from video: {video_path}")

        return _sample_evenly(decoded, n_frames)


def _extract_frames_b64(
    video_path: str,
    n_frames: int,
    max_longest_side: Optional[int] = None,
) -> List[str]:
    """
    Sample exactly ``n_frames`` uniformly-spaced frame slots and return them as
    base64-encoded JPEG strings.

    If the source video has fewer than ``n_frames`` decoded frames, some slots
    will map to the same underlying frame. This preserves a fixed-size visual
    prompt for the API backend. When ``max_longest_side`` is set, larger sampled
    frames are downscaled before JPEG encoding.
    """
    if n_frames <= 0:
        raise ValueError(f"n_frames must be positive, got {n_frames}")
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video file not found: {video_path}")

    with av.open(video_path) as container:
        stream = _get_video_stream(container)
        total_frames = int(stream.frames or 0)
        if total_frames <= 0:
            return _extract_frames_b64_sequential(
                video_path,
                n_frames,
                max_longest_side=max_longest_side,
            )

        if n_frames == 1:
            indices = [total_frames // 2]
        else:
            indices = [
                round(i * (total_frames - 1) / (n_frames - 1))
                for i in range(n_frames)
            ]

        wanted = Counter(indices)
        cache: Dict[int, str] = {}
        last_needed = max(wanted)

        for frame_idx, frame in enumerate(container.decode(stream)):
            if frame_idx > last_needed:
                break
            if frame_idx in wanted and frame_idx not in cache:
                cache[frame_idx] = _encode_frame_b64(
                    frame,
                    max_longest_side=max_longest_side,
                )
                if len(cache) == len(wanted):
                    break

        if not cache:
            return _extract_frames_b64_sequential(
                video_path,
                n_frames,
                max_longest_side=max_longest_side,
            )

        if any(idx not in cache for idx in wanted):
            return _extract_frames_b64_sequential(
                video_path,
                n_frames,
                max_longest_side=max_longest_side,
            )

        return [cache[idx] for idx in indices]


def _extract_response_text(response: Any) -> str:
    choices = getattr(response, "choices", None)
    if not choices:
        payload = response.model_dump() if hasattr(response, "model_dump") else repr(response)
        raise RuntimeError(f"API response did not include any choices: {payload}")

    message = getattr(choices[0], "message", None)
    if message is None:
        payload = response.model_dump() if hasattr(response, "model_dump") else repr(response)
        raise RuntimeError(f"API response choice did not include a message: {payload}")

    content = getattr(message, "content", None)
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        parts: List[str] = []
        for item in content:
            if isinstance(item, dict):
                text_part = item.get("text")
                if isinstance(text_part, str) and text_part.strip():
                    parts.append(text_part.strip())
                continue
            text_part = getattr(item, "text", None)
            if isinstance(text_part, str) and text_part.strip():
                parts.append(text_part.strip())
        if parts:
            return "\n".join(parts)

    refusal = getattr(message, "refusal", None)
    if isinstance(refusal, str) and refusal.strip():
        return refusal.strip()

    payload = response.model_dump() if hasattr(response, "model_dump") else repr(response)
    raise RuntimeError(f"API response did not contain text content: {payload}")


def _is_input_too_long_error(exc: Exception) -> bool:
    message = str(exc).lower()
    patterns = (
        "range of input length should be",
        "maximum context length",
        "context length",
        "input length",
        "request too large",
        "payload too large",
        "invalid_parameter_error",
    )
    return any(pattern in message for pattern in patterns)


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
        The preferred default should be injected by ``load_model`` based on the
        model family.  This constructor keeps a fallback default of 32 for
        direct instantiation.
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
    max_frame_longest_side : int | None
        Optional cap for the longest side of sampled JPEG frames. When set,
        larger frames are resized before being encoded and sent to the API.
    """

    def __init__(
        self,
        model_id: str,
        prompt_method: str = "vanilla",
        n_frames: int = _FALLBACK_N_FRAMES,
        api_key_env: str = "OPENAI_API_KEY",
        base_url: Optional[str] = None,
        max_concurrency: int = _DEFAULT_MAX_CONCURRENCY,
        max_frame_longest_side: Optional[int] = None,
    ) -> None:
        super().__init__(model_id, prompt_method)
        if max_concurrency <= 0:
            raise ValueError(f"max_concurrency must be positive, got {max_concurrency}")
        if max_frame_longest_side is not None and max_frame_longest_side <= 0:
            raise ValueError(
                f"max_frame_longest_side must be positive, got {max_frame_longest_side}"
            )
        self.n_frames = n_frames
        self.api_key_env = api_key_env
        self.base_url = base_url
        self.max_concurrency = max_concurrency
        self.max_frame_longest_side = max_frame_longest_side
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

        frames_b64 = _extract_frames_b64(
            video_path,
            self.n_frames,
            max_longest_side=self.max_frame_longest_side,
        )
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
        current_frames = list(frames_b64)

        while True:
            try:
                response = client.chat.completions.create(
                    model=self.model_id,
                    messages=[{"role": "user", "content": self._build_content(question, current_frames)}],
                    max_tokens=int(max_new_tokens),
                )
                return _extract_response_text(response)
            except BadRequestError as exc:
                if not _is_input_too_long_error(exc) or len(current_frames) <= _MIN_RETRY_FRAMES:
                    raise

                next_n_frames = max(_MIN_RETRY_FRAMES, len(current_frames) // 2)
                if next_n_frames >= len(current_frames):
                    raise
                current_frames = current_frames[:next_n_frames]

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
