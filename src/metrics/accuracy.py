"""
src/metrics/accuracy.py
------------------------
Per-sample (per-video) accuracy over the target questions answered by
the pipeline. Sub-questions are never sent to the answerer, so they are
not part of the score.

accuracy_v = (correct target answers in video v) / (target questions in video v)

The cross-video aggregation is a macro-average — each video contributes
equally regardless of its target-question count.
"""

from typing import List

from .base import BaseMetric, QuestionGroup


class TargetAccuracy(BaseMetric):
    """Per-video target-question accuracy. Macro-averaged across videos."""

    @property
    def name(self) -> str:
        return "accuracy"

    def compute(self, groups: List[QuestionGroup]) -> float:
        if not groups:
            return 0.0
        correct = sum(1 for g in groups if g.is_target_correct())
        return correct / len(groups)


class SubQuestionAccuracy(BaseMetric):
    """
    Flat accuracy over all sub-questions (target questions excluded).

    Useful as a standalone diagnostic to see how well the model handles
    atomic, decomposed questions.
    """

    @property
    def name(self) -> str:
        return "sub_accuracy"

    def compute(self, groups: List[QuestionGroup]) -> float:
        correct = total = 0
        for g in groups:
            # Skip the first element (target) — only sub-questions
            cv = g.correctness_vector()[1:]
            correct += sum(cv)
            total += len(cv)
        if total == 0:
            return 0.0
        return correct / total