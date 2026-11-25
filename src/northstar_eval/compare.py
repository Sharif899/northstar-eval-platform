"""Comparing two runs: where did the new version get better or worse?

A comparison only makes sense on the same dataset, so runs with
different dataset hashes are refused instead of compared. An item
counts as improved when its mean score rises by at least 0.5, and
regressed when it falls by at least 0.5 — half a rubric point is
large enough to matter and small enough to catch real movement.
Every changed item carries both responses and both sets of metric
scores, so a reviewer can see exactly what changed and decide
whether it is a genuine improvement or a scoring artefact.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .errors import NorthstarError
from .store import ResultRecord, RunRecord, Store

IMPROVED_THRESHOLD = 0.5
REGRESSED_THRESHOLD = -0.5


@dataclass
class ItemDiff:
    """One item's outcome under both runs."""

    item_id: str
    category: str
    question: str
    mean_a: float
    mean_b: float
    delta: float
    verdict: str  # "improved", "regressed", "unchanged"
    response_a: str
    response_b: str
    metrics_a: dict
    metrics_b: dict


@dataclass
class Comparison:
    """The full comparison between two runs."""

    run_a: RunRecord
    run_b: RunRecord
    overall_delta: float
    mean_a: float
    mean_b: float
    pass_rate_a: float
    pass_rate_b: float
    by_category: dict
    improved: list[ItemDiff] = field(default_factory=list)
    regressed: list[ItemDiff] = field(default_factory=list)
    unchanged: list[ItemDiff] = field(default_factory=list)

    def to_dict(self) -> dict:
        def diff_dict(diff: ItemDiff) -> dict:
            return {
                "item_id": diff.item_id,
                "category": diff.category,
                "question": diff.question,
                "mean_a": diff.mean_a,
                "mean_b": diff.mean_b,
                "delta": diff.delta,
                "verdict": diff.verdict,
                "response_a": diff.response_a,
                "response_b": diff.response_b,
                "metrics_a": diff.metrics_a,
                "metrics_b": diff.metrics_b,
            }

        return {
            "run_a": {"id": self.run_a.id, "name": self.run_a.name, "provider": self.run_a.provider,
                      "model": self.run_a.model, "prompt_version": self.run_a.prompt_version},
            "run_b": {"id": self.run_b.id, "name": self.run_b.name, "provider": self.run_b.provider,
                      "model": self.run_b.model, "prompt_version": self.run_b.prompt_version},
            "overall_delta": self.overall_delta,
            "mean_a": self.mean_a,
            "mean_b": self.mean_b,
            "pass_rate_a": self.pass_rate_a,
            "pass_rate_b": self.pass_rate_b,
            "by_category": self.by_category,
            "improved": [diff_dict(d) for d in self.improved],
            "regressed": [diff_dict(d) for d in self.regressed],
            "unchanged": [diff_dict(d) for d in self.unchanged],
            "counts": {
                "improved": len(self.improved),
                "regressed": len(self.regressed),
                "unchanged": len(self.unchanged),
            },
        }


def compare_runs(store: Store, run_a_id: str, run_b_id: str) -> Comparison:
    """Compare two runs from the same dataset. Raises on mismatch."""
    run_a = store.get_run(run_a_id)
    run_b = store.get_run(run_b_id)
    if run_a.dataset_hash != run_b.dataset_hash:
        raise NorthstarError(
            f"Cannot compare runs on different datasets: run {run_a_id[:8]} uses "
            f"dataset {run_a.dataset_hash[:8]}, run {run_b_id[:8]} uses "
            f"{run_b.dataset_hash[:8]}. Compare runs from the same dataset version."
        )
    results_a = {r.item_id: r for r in store.get_results(run_a_id)}
    results_b = {r.item_id: r for r in store.get_results(run_b_id)}
    if set(results_a) != set(results_b):
        raise NorthstarError(
            "Cannot compare runs with different item sets. Both runs must cover the same items."
        )
    diffs: list[ItemDiff] = []
    for item_id in sorted(results_a):
        a: ResultRecord = results_a[item_id]
        b: ResultRecord = results_b[item_id]
        delta = round(b.mean_score - a.mean_score, 2)
        if delta >= IMPROVED_THRESHOLD:
            verdict = "improved"
        elif delta <= REGRESSED_THRESHOLD:
            verdict = "regressed"
        else:
            verdict = "unchanged"
        diffs.append(ItemDiff(
            item_id=item_id,
            category=a.category,
            question=a.question,
            mean_a=a.mean_score,
            mean_b=b.mean_score,
            delta=delta,
            verdict=verdict,
            response_a=a.response,
            response_b=b.response,
            metrics_a={k: v["score"] for k, v in a.metrics.items()},
            metrics_b={k: v["score"] for k, v in b.metrics.items()},
        ))
    by_category: dict[str, dict] = {}
    for category in sorted({d.category for d in diffs}):
        group = [d for d in diffs if d.category == category]
        by_category[category] = {
            "n": len(group),
            "mean_a": round(sum(d.mean_a for d in group) / len(group), 2),
            "mean_b": round(sum(d.mean_b for d in group) / len(group), 2),
            "delta": round(sum(d.delta for d in group) / len(group), 2),
        }
    return Comparison(
        run_a=run_a,
        run_b=run_b,
        overall_delta=round(run_b.mean_score - run_a.mean_score, 2),
        mean_a=run_a.mean_score,
        mean_b=run_b.mean_score,
        pass_rate_a=run_a.pass_rate,
        pass_rate_b=run_b.pass_rate,
        by_category=by_category,
        improved=[d for d in diffs if d.verdict == "improved"],
        regressed=[d for d in diffs if d.verdict == "regressed"],
        unchanged=[d for d in diffs if d.verdict == "unchanged"],
    )
