"""Reports: Markdown for humans, JSON for machines.

A report leads with the headline numbers, then makes failures easy
to find: the worst items are shown with their responses, scores and
rationales, not buried in an appendix. A report that hides failures
is worse than no report, because it lets a bad release through.
"""

from __future__ import annotations

from .compare import Comparison
from .runner import manifest
from .store import ResultRecord, RunRecord

N_FAILURE_EXAMPLES = 5


def worst_items(results: list[ResultRecord], n: int = N_FAILURE_EXAMPLES) -> list[ResultRecord]:
    """The lowest-scoring results, worst first (ties: lowest min metric)."""

    def sort_key(result: ResultRecord):
        min_metric = min(v["score"] for v in result.metrics.values())
        return (result.mean_score, min_metric, result.item_id)

    return sorted(results, key=sort_key)[:n]


def build_json_report(
    run: RunRecord,
    results: list[ResultRecord],
    comparison: Comparison | None = None,
    review_stats: dict | None = None,
) -> dict:
    """The full report as structured data."""
    by_category: dict[str, dict] = {}
    for category in sorted({r.category for r in results}):
        group = [r for r in results if r.category == category]
        by_category[category] = {
            "n": len(group),
            "mean": round(sum(r.mean_score for r in group) / len(group), 2),
            "pass_rate": round(sum(1 for r in group if r.passed) / len(group), 3),
        }
    by_metric: dict[str, dict] = {}
    metric_names = sorted({name for r in results for name in r.metrics})
    for name in metric_names:
        values = [r.metrics[name]["score"] for r in results]
        by_metric[name] = {
            "mean": round(sum(values) / len(values), 2),
            "min": min(values),
            "max": max(values),
            "at_or_below_2": sum(1 for v in values if v <= 2),
        }
    return {
        "manifest": manifest(run),
        "overall": {
            "n_items": run.n_items,
            "mean_score": run.mean_score,
            "pass_rate": run.pass_rate,
            "flagged_for_review": sum(1 for r in results if r.needs_review),
        },
        "by_category": by_category,
        "by_metric": by_metric,
        "failure_examples": [
            {
                "item_id": r.item_id,
                "category": r.category,
                "question": r.question,
                "response": r.response,
                "mean_score": r.mean_score,
                "metrics": {
                    name: {"score": m["score"], "rationale": m["rationale"]}
                    for name, m in r.metrics.items()
                },
                "flag_reasons": r.flag_reasons,
            }
            for r in worst_items(results)
        ],
        "review": review_stats or {},
        "comparison": comparison.to_dict() if comparison else None,
        "results": [
            {
                "item_id": r.item_id,
                "category": r.category,
                "question": r.question,
                "response": r.response,
                "mean_score": r.mean_score,
                "passed": r.passed,
                "needs_review": r.needs_review,
                "metrics": r.metrics,
                "review": r.review,
            }
            for r in results
        ],
    }


def build_markdown_report(
    run: RunRecord,
    results: list[ResultRecord],
    comparison: Comparison | None = None,
    review_stats: dict | None = None,
) -> str:
    """The full report as Markdown."""
    data = build_json_report(run, results, comparison, review_stats)
    lines: list[str] = []
    lines.append(f"# Evaluation report — {run.name}")
    lines.append("")
    lines.append(f"Run `{run.id}` · {run.created_at} · provider `{run.provider}` "
                 f"(model `{run.model}`, prompt `{run.prompt_version}`) · "
                 f"scorer v{run.scorer_version}")
    lines.append(f"Dataset `{run.dataset_name}` · SHA-256 `{run.dataset_hash[:12]}…`")
    lines.append("")
    lines.append("## Overall")
    lines.append("")
    lines.append(f"- Items: **{data['overall']['n_items']}**")
    lines.append(f"- Mean score: **{data['overall']['mean_score']:.2f} / 5**")
    lines.append(f"- Pass rate (mean ≥ 4.0 and no metric ≤ 2): **{data['overall']['pass_rate'] * 100:.1f}%**")
    lines.append(f"- Flagged for human review: **{data['overall']['flagged_for_review']}**")
    lines.append("")
    lines.append("## By category")
    lines.append("")
    lines.append("| Category | Items | Mean | Pass rate |")
    lines.append("|---|---:|---:|---:|")
    for category, stats in data["by_category"].items():
        lines.append(
            f"| {category} | {stats['n']} | {stats['mean']:.2f} | {stats['pass_rate'] * 100:.1f}% |"
        )
    lines.append("")
    lines.append("## By metric")
    lines.append("")
    lines.append("| Metric | Mean | Min | Max | Scores ≤ 2 |")
    lines.append("|---|---:|---:|---:|---:|")
    for name, stats in data["by_metric"].items():
        lines.append(
            f"| {name} | {stats['mean']:.2f} | {stats['min']} | {stats['max']} | {stats['at_or_below_2']} |"
        )
    lines.append("")
    lines.append("## Failure examples (worst items first)")
    lines.append("")
    if not data["failure_examples"]:
        lines.append("No failures: every item passed.")
    for example in data["failure_examples"]:
        lines.append(f"### {example['item_id']} — {example['category']} (mean {example['mean_score']:.2f})")
        lines.append("")
        lines.append(f"**Question:** {example['question']}")
        lines.append("")
        lines.append(f"**Response:** {example['response']}")
        lines.append("")
        for name, metric in example["metrics"].items():
            lines.append(f"- {name}: **{metric['score']}** — {metric['rationale']}")
        if example["flag_reasons"]:
            lines.append(f"- Flagged for review: {'; '.join(example['flag_reasons'])}")
        lines.append("")
    if data["review"]:
        stats = data["review"]
        lines.append("## Human review")
        lines.append("")
        lines.append(f"- Flagged: {stats['flagged']}")
        lines.append(f"- Reviewed: {stats['flagged_reviewed']} of flagged "
                     f"(coverage {stats['coverage'] * 100:.1f}%)")
        lines.append(f"- Overturned: {stats['overturned']} "
                     f"(overturn rate {stats['overturn_rate'] * 100:.1f}%)")
        lines.append("")
    if comparison is not None:
        comp = data["comparison"]
        lines.append("## Comparison")
        lines.append("")
        lines.append(f"{comp['run_a']['name']} → {comp['run_b']['name']}: "
                     f"mean {comp['mean_a']:.2f} → {comp['mean_b']:.2f} "
                     f"(Δ {comp['overall_delta']:+.2f}), pass rate "
                     f"{comp['pass_rate_a'] * 100:.1f}% → {comp['pass_rate_b'] * 100:.1f}%")
        lines.append("")
        lines.append(f"Improved: **{comp['counts']['improved']}** · "
                     f"Regressed: **{comp['counts']['regressed']}** · "
                     f"Unchanged: **{comp['counts']['unchanged']}**")
        lines.append("")
        lines.append("| Category | Mean A | Mean B | Δ |")
        lines.append("|---|---:|---:|---:|")
        for category, stats in comp["by_category"].items():
            lines.append(
                f"| {category} | {stats['mean_a']:.2f} | {stats['mean_b']:.2f} | {stats['delta']:+.2f} |"
            )
        lines.append("")
        for title, group in (("Regressions", comp["regressed"]), ("Improvements", comp["improved"])):
            if not group:
                continue
            lines.append(f"### {title}")
            lines.append("")
            for item in group:
                lines.append(
                    f"- **{item['item_id']}** ({item['category']}, {item['mean_a']:.2f} → "
                    f"{item['mean_b']:.2f}): {item['question']}"
                )
                lines.append(f"  - Before: {item['response_a']}")
                lines.append(f"  - After: {item['response_b']}")
            lines.append("")
    lines.append("---")
    lines.append("")
    lines.append(
        "Scores come from deterministic rubric checks, not a judge model. "
        "Treat them as a triage signal: read the failure examples and the "
        "human-review queue before approving a release."
    )
    lines.append("")
    return "\n".join(lines)
