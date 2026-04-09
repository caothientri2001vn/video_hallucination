"""
src/models/video_rft.py
------------------------
Local HuggingFace backend for VideoRFT (QiWang98/VideoRFT).

VideoRFT is fine-tuned from Qwen/Qwen2.5-VL-7B-Instruct using Reinforced
Fine-Tuning for video reasoning (NeurIPS 2025).

Paper  : https://arxiv.org/abs/2505.12434
Model  : https://huggingface.co/QiWang98/VideoRFT

Supported model IDs
-------------------
    QiWang98/VideoRFT          (7B, RFT)
    QiWang98/VideoRFT-SFT      (7B, SFT only)
    QiWang98/VideoRFT-3B       (3B, RFT)
    QiWang98/VideoRFT-SFT-3B   (3B, SFT only)

Dependencies
------------
Requires the same deps as the ``qwen3vl`` group (stable transformers ≥ 4.51,
which ships Qwen2_5_VLForConditionalGeneration):

    uv sync --group videorft

or reuse:

    uv sync --group qwen3vl

Inference settings (from paper)
--------------------------------
    temperature = 0.01
    top_p       = 0.001   (nearly deterministic)
    max_frames  = 32
    max_pixels  = 256 × 28 × 28 = 200 704
"""

import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch

from .base import BaseVideoQAModel

try:
    from transformers import AutoProcessor
    from transformers import Qwen2_5_VLForConditionalGeneration
except Exception as exc:
    raise RuntimeError(
        "Failed to import Qwen2_5_VLForConditionalGeneration.\n"
        "Install with: uv sync --group videorft  (or --group qwen3vl)\n"
        f"Error: {exc}"
    ) from exc

try:
    from qwen_vl_utils import process_vision_info
except Exception as exc:
    raise RuntimeError(
        "Failed to import qwen_vl_utils. Install with: pip install qwen-vl-utils\n"
        f"Error: {exc}"
    ) from exc

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")


def _to_file_uri(p: str) -> str:
    return Path(p).expanduser().resolve().as_uri()


@dataclass
class _LoadedWeights:
    model_id: str
    model: Any
    processor: Any


class VideoRFTModel(BaseVideoQAModel):
    """
    Runs inference locally using a VideoRFT checkpoint (Qwen2.5-VL based).

    Parameters
    ----------
    model_id : str
        HuggingFace repo ID, e.g. ``"QiWang98/VideoRFT"``.
    prompt_method : str
        Prompt template label (affects cache namespace). Default: ``"vanilla"``.
    max_frames : int
        Number of frames sampled from the video. Paper recommends 32.
        Default: 32.
    max_pixels : int
        Max pixels per frame. Paper recommends 256*28*28. Default: 200704.
    debug_with_n_frames : int | None
        When set, overrides ``max_frames``.
    """

    # Paper-recommended inference settings
    _TEMPERATURE: float = 0.01
    _TOP_P: float = 0.001

    def __init__(
        self,
        model_id: str,
        prompt_method: str = "vanilla",
        max_frames: int = 32,
        max_pixels: int = 256 * 28 * 28,
        debug_with_n_frames: Optional[int] = None,
    ) -> None:
        super().__init__(model_id, prompt_method)
        self.max_frames = debug_with_n_frames if debug_with_n_frames is not None else max_frames
        self.max_pixels = max_pixels
        self._loaded: Optional[_LoadedWeights] = None

    # ------------------------------------------------------------------
    # BaseVideoQAModel interface
    # ------------------------------------------------------------------

    def answer_questions(
        self,
        video_path: str,
        questions: List[str],
        max_new_tokens: int = 256,
    ) -> List[str]:
        results = []
        for question in questions:
            question = question.strip()
            if not question:
                raise ValueError(f"Empty question passed to {self!r}")
            messages = self._build_messages(video_path, question)
            answer, _ = self._generate_one(messages, max_new_tokens)
            results.append(answer)
        return results

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _load_weights(self) -> _LoadedWeights:
        if self._loaded is not None and self._loaded.model_id == self.model_id:
            return self._loaded

        processor = AutoProcessor.from_pretrained(self.model_id)
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            self.model_id,
            torch_dtype=torch.bfloat16,
            device_map="auto",
        )
        model.eval()
        self._loaded = _LoadedWeights(self.model_id, model, processor)
        return self._loaded

    def _build_messages(
        self, video_path: str, question: str
    ) -> List[Dict[str, Any]]:
        grounding = (
            "IMPORTANT: only use information you can directly verify from the "
            "video. If you are unsure, say 'Not sure'. When possible, cite "
            "rough timestamps."
        )
        prompt = f"{grounding}\n\nQuestion: {question}"
        content: List[Dict[str, Any]] = [
            {
                "type": "video",
                "video": _to_file_uri(video_path),
                "max_pixels": self.max_pixels,
                "nframes": self.max_frames,
            },
            {"type": "text", "text": prompt},
        ]
        return [{"role": "user", "content": content}]

    def _generate_one(
        self,
        messages: List[Dict[str, Any]],
        max_new_tokens: int,
    ) -> Tuple[str, Dict[str, Any]]:
        loaded = self._load_weights()
        model, processor = loaded.model, loaded.processor

        text = processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        image_inputs, video_inputs, video_kwargs = process_vision_info(
            messages, return_video_kwargs=True
        )

        # process_vision_info may return fps as a list; processor needs a scalar
        if "fps" in video_kwargs and isinstance(video_kwargs["fps"], list):
            video_kwargs["fps"] = video_kwargs["fps"][0]

        inputs = processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
            **video_kwargs,
        ).to(next(model.parameters()).device)

        gen_kwargs: Dict[str, Any] = dict(
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=self._TEMPERATURE,
            top_p=self._TOP_P,
        )

        t0 = time.time()
        with torch.inference_mode():
            generated_ids = model.generate(**inputs, **gen_kwargs)
        dt = time.time() - t0

        trimmed = [
            out[len(inp):]
            for inp, out in zip(inputs.input_ids, generated_ids)
        ]
        answer = processor.batch_decode(
            trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0].strip()

        debug = {
            "model_id": self.model_id,
            "max_new_tokens": int(max_new_tokens),
            "temperature": self._TEMPERATURE,
            "top_p": self._TOP_P,
            "latency_sec": round(dt, 3),
        }
        return answer, debug
