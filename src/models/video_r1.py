"""
src/models/video_r1.py
-----------------------
Local HuggingFace backend for Video-R1 (Video-R1/Video-R1-7B).

Video-R1 is fine-tuned from Qwen/Qwen2.5-VL-7B-Instruct using reinforcement
learning to incentivise chain-of-thought video reasoning.

Paper  : https://arxiv.org/abs/2503.21776
Model  : https://huggingface.co/Video-R1/Video-R1-7B
Code   : https://github.com/tulerfeng/Video-R1

Supported model IDs
-------------------
    Video-R1/Video-R1-7B

Key differences from VideoRFTModel
------------------------------------
* Uses the Video-R1 prompt template that elicits a <think>…</think> chain of
  thought followed by <answer>…</answer> — the text inside <answer> is
  returned as the final answer string.
* Inference settings from the paper: temperature=0.1, top_p=0.001.

Dependencies
------------
    uv sync --group videorft   (or --group qwen3vl)

Both install transformers ≥ 4.51 which ships Qwen2_5_VLForConditionalGeneration.
"""

import os
import re
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

# ---------------------------------------------------------------------------
# Prompt template (from the official Video-R1 inference example)
# ---------------------------------------------------------------------------
_QUESTION_TEMPLATE = (
    "{question}\n"
    "Please think about this question as if you were a human pondering deeply. "
    "Engage in an internal dialogue using expressions such as 'let me think', "
    "'wait', 'Hmm', 'oh, I see', 'let's break it down', etc, or other natural "
    "language thought expressions. "
    "It's encouraged to include self-reflection or verification in the reasoning "
    "process. "
    "Provide your detailed reasoning between the <think> and </think> tags, and "
    "then give your final answer between the <answer> and </answer> tags."
    " Please provide your text answer within the <answer> </answer> tags."
)


def _to_file_uri(p: str) -> str:
    return Path(p).expanduser().resolve().as_uri()


@dataclass
class _LoadedWeights:
    model_id: str
    model: Any
    processor: Any


class VideoR1Model(BaseVideoQAModel):
    """
    Runs inference locally using a Video-R1 checkpoint (Qwen2.5-VL based).

    The model is prompted to produce a chain-of-thought inside <think>…</think>
    and a final answer inside <answer>…</answer>.  Only the <answer> content is
    returned to the evaluation pipeline.

    Parameters
    ----------
    model_id : str
        HuggingFace repo ID, e.g. ``"Video-R1/Video-R1-7B"``.
    prompt_method : str
        Prompt template label (affects cache namespace). Default: ``"vanilla"``.
    max_frames : int
        Number of frames sampled from the video. Default: 32 (same as paper).
    max_pixels : int
        Max pixels per frame. Default: 200704 (256×28×28, same as paper).
    debug_with_n_frames : int | None
        When set, overrides ``max_frames``.
    """

    # Inference settings from the official Video-R1 inference example
    _TEMPERATURE: float = 0.1
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
        max_new_tokens: int = 1024,
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
        prompt = _QUESTION_TEMPLATE.format(question=question)
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
        raw = processor.batch_decode(
            trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0]

        answer = self._parse_answer(raw)

        debug = {
            "model_id": self.model_id,
            "max_new_tokens": int(max_new_tokens),
            "temperature": self._TEMPERATURE,
            "top_p": self._TOP_P,
            "raw_output": raw,
            "latency_sec": round(dt, 3),
        }
        return answer, debug

    @staticmethod
    def _parse_answer(raw: str) -> str:
        """
        Extract the content inside <answer>…</answer>.
        Falls back to the full output (stripped) if no tags are found.
        """
        match = re.search(r"<answer>(.*?)</answer>", raw, flags=re.DOTALL | re.IGNORECASE)
        if match:
            return match.group(1).strip()
        # Fallback: no <answer> tags — return the whole output
        return raw.strip()
