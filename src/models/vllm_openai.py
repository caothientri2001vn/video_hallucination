"""
src/models/vllm_openai.py
-------------------------
OpenAI-compatible backend for locally served vLLM models.

This file intentionally mirrors the OpenAI/OpenRouter backend in a separate
module so vLLM-specific defaults can evolve without touching
``src/models/openai_gpt.py``.
"""

import base64
import os
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict, List, Optional

import cv2

try:
    import av
except Exception as exc:
    raise RuntimeError(
        "Failed to import PyAV. Install with: uv sync --group openai\n"
        f"Error: {exc}"
    ) from exc

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
_DEFAULT_VLLM_API_KEY = "EMPTY"
_DEFAULT_VLLM_BASE_URL = "http://localhost:8000/v1"


def _encode_frame_b64(frame: av.VideoFrame) -> str:
    bgr_frame = frame.to_ndarray(format="bgr24")
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


def _extract_frames_b64_sequential(video_path: str, n_frames: int) -> List[str]:
    with av.open(video_path) as container:
        stream = _get_video_stream(container)
        decoded: List[str] = []
        for frame in container.decode(stream):
            decoded.append(_encode_frame_b64(frame))

        if not decoded:
            raise ValueError(f"Failed to decode any frames sequentially from video: {video_path}")

        return _sample_evenly(decoded, n_frames)


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

    with av.open(video_path) as container:
        stream = _get_video_stream(container)
        total_frames = int(stream.frames or 0)
        if total_frames <= 0:
            return _extract_frames_b64_sequential(video_path, n_frames)

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
                cache[frame_idx] = _encode_frame_b64(frame)
                if len(cache) == len(wanted):
                    break

        if not cache:
            return _extract_frames_b64_sequential(video_path, n_frames)

        if any(idx not in cache for idx in wanted):
            return _extract_frames_b64_sequential(video_path, n_frames)

        return [cache[idx] for idx in indices]


def _get_message_field(message: Any, name: str) -> Any:
    if isinstance(message, dict):
        return message.get(name)

    value = getattr(message, name, None)
    if value is not None:
        return value

    model_extra = getattr(message, "model_extra", None)
    if isinstance(model_extra, dict) and name in model_extra:
        return model_extra[name]

    if hasattr(message, "model_dump"):
        payload = message.model_dump()
        if isinstance(payload, dict):
            return payload.get(name)

    return None


def _extract_response_text(response: Any) -> str:
    choices = getattr(response, "choices", None)
    if not choices:
        payload = response.model_dump() if hasattr(response, "model_dump") else repr(response)
        raise RuntimeError(f"API response did not include any choices: {payload}")

    message = getattr(choices[0], "message", None)
    if message is None:
        payload = response.model_dump() if hasattr(response, "model_dump") else repr(response)
        raise RuntimeError(f"API response choice did not include a message: {payload}")

    content = _get_message_field(message, "content")
    if isinstance(content, str) and content.strip():
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

    # Some vLLM/Qwen reasoning-parser responses put the assistant text here
    # with ``content=None``.
    for attr_name in ("reasoning", "reasoning_content"):
        reasoning = _get_message_field(message, attr_name)
        if isinstance(reasoning, str) and reasoning.strip():
            return reasoning.strip()

    refusal = _get_message_field(message, "refusal")
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


class VLLMOpenAIModel(BaseVideoQAModel):
    """
    OpenAI-compatible API backend for vLLM-served models.

    Parameters
    ----------
    model_id : str
        Model name exposed by the vLLM server, e.g.
        ``"Qwen/Qwen3-VL-32B-Thinking"``.
    prompt_method : str
        Prompt template label (affects cache namespace). Default: "vanilla".
    n_frames : int
        Number of uniformly-spaced frame slots to send for every video.
        The preferred default should be injected by ``load_model`` based on the
        model family. This constructor keeps a fallback default of 32 for
        direct instantiation.
    api_key_env : str
        Env-var name for the API key. Default: ``"VLLM_API_KEY"``.
        If unset, the client falls back to ``"EMPTY"`` which matches the
        standard vLLM localhost examples.
    base_url : str | None
        Override the API base URL. If unset, falls back to the env var
        ``VLLM_BASE_URL`` and finally ``"http://localhost:8000/v1"``.
    max_concurrency : int
        Maximum number of per-question API calls to run in parallel.
        Default: 4.
    extra_body : dict[str, Any] | None
        Optional vLLM-specific extra request parameters, e.g.
        ``{"chat_template_kwargs": {"enable_thinking": False}}``.
    """

    def __init__(
        self,
        model_id: str,
        prompt_method: str = "vanilla",
        n_frames: int = _FALLBACK_N_FRAMES,
        api_key_env: str = "VLLM_API_KEY",
        base_url: Optional[str] = None,
        max_concurrency: int = _DEFAULT_MAX_CONCURRENCY,
        extra_body: Optional[Dict[str, Any]] = None,
    ) -> None:
        super().__init__(model_id, prompt_method)
        if max_concurrency <= 0:
            raise ValueError(f"max_concurrency must be positive, got {max_concurrency}")
        self.n_frames = n_frames
        self.api_key_env = api_key_env
        self.base_url = base_url
        self.max_concurrency = max_concurrency
        self.extra_body = dict(extra_body) if extra_body else None
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

        api_key = os.environ.get(self.api_key_env, _DEFAULT_VLLM_API_KEY)
        base_url = self.base_url or os.environ.get("VLLM_BASE_URL", _DEFAULT_VLLM_BASE_URL)

        self._client = OpenAI(api_key=api_key, base_url=base_url)
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
                request_kwargs: Dict[str, Any] = {
                    "model": self.model_id,
                    "messages": [{"role": "user", "content": self._build_content(question, current_frames)}],
                    "max_tokens": int(max_new_tokens),
                }
                if self.extra_body:
                    request_kwargs["extra_body"] = self.extra_body

                response = client.chat.completions.create(**request_kwargs)
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
                },
            }
            for frame_b64 in frames_b64
        ]
        content.append({"type": "text", "text": prompt})
        return content
