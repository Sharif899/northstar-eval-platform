"""Scoring: run every registered metric over one response.

The scorer is deliberately thin — it calls the registry and applies
the pass threshold and the review-flag rule. All judgement lives in
the metric functions, so a new metric changes nothing here.

Review-flag rule (documented in the README and the design report):
a result is flagged for human review when any of these holds:
- the mean score is borderline: 3.0 <= mean < 4.0;
- the metrics disagree sharply: max metric - min metric >= 3;
- any single metric scores <= 2;
- citations were required and citation_quality <= 2.
A flagged result is not a failed result — it is one a human should
look at before the release decision.

Pass rule: an item passes only when its mean score is >= 4.0 AND no
single metric scores 2 or below. A mean can hide one serious
failure — a wrong warranty period, a missing citation, an unsafe
engagement — so a low metric fails the item even when the average
looks fine. That is a deliberate, strict choice for a regression
gate: it is better to flag a borderline item for review than to let
a real failure through on the strength of an average.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .dataset import EvalItem, SourceDoc
from .metrics import METRIC_NAMES, MetricScore, list_metrics

SCORER_VERSION = "1.0.0"
PASS_THRESHOLD = 4.0
FLAG_BORDERLINE_LOW = 3.0
FLAG_SPREAD = 3
FLAG_LOW_METRIC = 2


@dataclass
class ItemScore:
    """The scored result for one dataset item."""

    item_id: str
    category: str
    question: str
    reference_answer: str
    response: str
    metrics: dict[str, MetricScore]
    mean_score: float
    passed: bool
    needs_review: bool
    flag_reasons: list[str] = field(default_factory=list)
    confidence: float = 1.0

    def to_dict(self) -> dict:
        return {
            "item_id": self.item_id,
            "category": self.category,
            "question": self.question,
            "reference_answer": self.reference_answer,
            "response": self.response,
            "metrics": {
                name: {
                    "score": metric.score,
                    "rationale": metric.rationale,
                    "evidence": metric.evidence,
                    "confidence": metric.confidence,
                }
                for name, metric in self.metrics.items()
            },
            "mean_score": self.mean_score,
            "passed": self.passed,
            "needs_review": self.needs_review,
            "flag_reasons": self.flag_reasons,
            "confidence": self.confidence,
        }


def score_response(
    item: EvalItem,
    response: str,
    docs: dict[str, SourceDoc],
    metric_names: tuple[str, ...] = METRIC_NAMES,
) -> ItemScore:
    """Score one response on every registered metric in ``metric_names``."""
    metrics: dict[str, MetricScore] = {}
    for metric in list_metrics():
        if metric.name not in metric_names:
            continue
        metrics[metric.name] = metric.score_fn(item, response, docs)
    scores = [metric.score for metric in metrics.values()]
    mean = round(sum(scores) / len(scores), 2) if scores else 0.0
    passed = mean >= PASS_THRESHOLD and (not scores or min(scores) >= 3)

    flag_reasons: list[str] = []
    if FLAG_BORDERLINE_LOW <= mean < PASS_THRESHOLD:
        flag_reasons.append(f"Borderline mean score {mean} (pass threshold is {PASS_THRESHOLD}).")
    if scores and max(scores) - min(scores) >= FLAG_SPREAD:
        flag_reasons.append(
            f"Metrics disagree sharply (range {min(scores)}–{max(scores)})."
        )
    low = [name for name, metric in metrics.items() if metric.score <= FLAG_LOW_METRIC]
    if low:
        flag_reasons.append(f"Low metric score(s): {', '.join(low)}.")
    citation = metrics.get("citation_quality")
    if item.expected_behaviour == "must_cite" and citation is not None and citation.score <= 2:
        flag_reasons.append("Citations were required and citation quality is low.")
    confidence = max(0.3, round(1.0 - 0.15 * len(flag_reasons), 2))

    return ItemScore(
        item_id=item.id,
        category=item.category,
        question=item.question,
        reference_answer=item.reference_answer,
        response=response,
        metrics=metrics,
        mean_score=mean,
        passed=passed,
        needs_review=bool(flag_reasons),
        flag_reasons=flag_reasons,
        confidence=confidence,
    )
