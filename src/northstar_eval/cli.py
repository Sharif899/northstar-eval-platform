"""Command-line interface: ``nev``.

Commands:
    nev load DATASET             Validate a dataset and show its summary.
    nev run DATASET              Run an evaluation and write the reports.
    nev compare RUN_A RUN_B      Diff two runs from the same dataset.
    nev review list|show|judge   Work the human-review queue.
    nev report RUN_ID            Export a run's report (Markdown or JSON).
    nev serve                    Start the web UI.

Every command accepts --help. API keys are read from the
environment (NORTHSTAR_LLM_API_KEY) and never appear in output.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path

from . import __version__
from .compare import compare_runs
from .dataset import dataset_summary, load_dataset, load_knowledge_base
from .errors import DatasetError, NorthstarError, ProviderError, ReviewError
from .providers import ALL_PROVIDERS, ProviderConfig
from .report import build_json_report, build_markdown_report
from .review import get_review_queue, record_judgement
from .runner import RunConfig, run_evaluation
from .store import Store

log = logging.getLogger("northstar_eval")


def _add_common(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--db", default=os.environ.get("NORTHSTAR_DB", "northstar.db"),
                        help="SQLite database file (default: $NORTHSTAR_DB or northstar.db).")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose logging.")


def _setup_logging(verbose: bool) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(levelname)s %(name)s: %(message)s",
    )


def _kb_path_for(dataset: Path, explicit: str | None) -> Path:
    if explicit:
        return Path(explicit)
    candidate = dataset.parent / "knowledge_base.json"
    return candidate


def _print_summary_table(title: str, rows: list[tuple[str, str]]) -> None:
    print(f"\n{title}")
    width = max(len(label) for label, _ in rows)
    for label, value in rows:
        print(f"  {label:<{width}}  {value}")


def cmd_load(args: argparse.Namespace) -> int:
    dataset_path = Path(args.dataset)
    items = load_dataset(dataset_path)
    kb_path = _kb_path_for(dataset_path, args.kb)
    docs = load_knowledge_base(kb_path)
    summary = dataset_summary(items)
    missing = sorted({d for item in items for d in item.source_doc_ids if d not in docs})
    _print_summary_table(f"Dataset: {dataset_path}", [
        ("Examples", str(summary["total"])),
        ("SHA-256", summary["hash"][:16] + "…"),
        ("Categories", ", ".join(f"{k}={v}" for k, v in summary["by_category"].items())),
        ("Expected behaviours", ", ".join(f"{k}={v}" for k, v in summary["by_behaviour"].items())),
        ("Knowledge base", f"{kb_path} ({len(docs)} documents)"),
    ])
    if missing:
        print(f"\nWARNING: dataset references missing documents: {', '.join(missing)}")
        return 1
    print("\nDataset is valid.")
    return 0


def cmd_run(args: argparse.Namespace) -> int:
    dataset_path = Path(args.dataset)
    config = RunConfig(
        dataset_path=dataset_path,
        kb_path=_kb_path_for(dataset_path, args.kb),
        provider_name=args.provider,
        model=args.model or args.provider,
        prompt_version=args.prompt_version,
        db_path=args.db,
        run_name=args.name or "",
        base_url=os.environ.get("NORTHSTAR_LLM_BASE_URL", ""),
        api_key=os.environ.get("NORTHSTAR_LLM_API_KEY", ""),
    )
    summary = run_evaluation(config)
    _print_summary_table(f"Run complete: {summary.run_id}", [
        ("Name", summary.name),
        ("Provider / model", f"{summary.provider} / {summary.model}"),
        ("Prompt version", summary.prompt_version),
        ("Dataset", f"{summary.dataset_name} ({summary.n_items} items)"),
        ("Mean score", f"{summary.mean_score:.2f} / 5"),
        ("Pass rate", f"{summary.pass_rate * 100:.1f}%"),
        ("Flagged for review", str(summary.flagged)),
    ])
    print("\nMean score by category:")
    for category, stats in summary.by_category.items():
        print(f"  {category:<16} {stats['mean']:.2f}  (pass {stats['pass_rate'] * 100:.0f}%, n={stats['n']})")
    out_dir = Path(args.out) if args.out else None
    if out_dir:
        out_dir.mkdir(parents=True, exist_ok=True)
        with Store(args.db) as store:
            run = store.get_run(summary.run_id)
            results = store.get_results(summary.run_id)
            stats = store.review_stats(summary.run_id)
        (out_dir / f"{summary.run_id}.md").write_text(
            build_markdown_report(run, results, None, stats), encoding="utf-8"
        )
        (out_dir / f"{summary.run_id}.json").write_text(
            json.dumps(build_json_report(run, results, None, stats), indent=2),
            encoding="utf-8",
        )
        print(f"\nReports written to {out_dir}/")
    print(f"\nReview the flagged items with: nev review list --run {summary.run_id} --db {args.db}")
    return 0


def cmd_compare(args: argparse.Namespace) -> int:
    with Store(args.db) as store:
        comparison = compare_runs(store, args.run_a, args.run_b)
    print(f"\nComparison: {comparison.run_a.name} → {comparison.run_b.name}")
    print(f"  Mean score: {comparison.mean_a:.2f} → {comparison.mean_b:.2f} "
          f"(Δ {comparison.overall_delta:+.2f})")
    print(f"  Pass rate:  {comparison.pass_rate_a * 100:.1f}% → {comparison.pass_rate_b * 100:.1f}%")
    print(f"  Improved: {len(comparison.improved)}  Regressed: {len(comparison.regressed)}  "
          f"Unchanged: {len(comparison.unchanged)}")
    print("\n  Mean score by category:")
    for category, stats in comparison.by_category.items():
        print(f"    {category:<16} {stats['mean_a']:.2f} → {stats['mean_b']:.2f} "
              f"(Δ {stats['delta']:+.2f})")
    if comparison.regressed:
        print("\n  Regressions (read these before releasing):")
        for diff in comparison.regressed:
            print(f"    - {diff.item_id} ({diff.category}): {diff.mean_a:.2f} → {diff.mean_b:.2f} — {diff.question}")
    if comparison.improved:
        print("\n  Improvements:")
        for diff in comparison.improved:
            print(f"    - {diff.item_id} ({diff.category}): {diff.mean_a:.2f} → {diff.mean_b:.2f} — {diff.question}")
    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with Store(args.db) as store:
            run_b = store.get_run(args.run_b)
            results_b = store.get_results(args.run_b)
        if out_path.suffix == ".json":
            out_path.write_text(json.dumps(comparison.to_dict(), indent=2), encoding="utf-8")
        else:
            out_path.write_text(
                build_markdown_report(run_b, results_b, comparison),
                encoding="utf-8",
            )
        print(f"\nComparison report written to {out_path}")
    return 0


def cmd_review_list(args: argparse.Namespace) -> int:
    with Store(args.db) as store:
        queue = get_review_queue(store, args.run)
        stats = store.review_stats(args.run)
    print(f"\nReview queue for run {args.run}: {len(queue)} item(s) awaiting review")
    print(f"  Flagged: {stats['flagged']} · Reviewed: {stats['flagged_reviewed']} "
          f"(coverage {stats['coverage'] * 100:.1f}%) · Overturned: {stats['overturned']}")
    for result in queue:
        print(f"\n  [{result.id}] {result.item_id} ({result.category}) — mean {result.mean_score:.2f}")
        print(f"    Q: {result.question}")
        print(f"    Flags: {'; '.join(result.flag_reasons)}")
    if not queue:
        print("  Queue is clear.")
    return 0


def cmd_review_show(args: argparse.Namespace) -> int:
    with Store(args.db) as store:
        result = store.get_result(args.result_id)
    print(f"\n{result.item_id} ({result.category}) — run {result.run_id}")
    print(f"Q: {result.question}")
    print(f"Reference: {result.reference_answer}")
    print(f"Response: {result.response}")
    print(f"Mean score: {result.mean_score:.2f} · Passed: {result.passed} · "
          f"Confidence: {result.confidence}")
    for name, metric in result.metrics.items():
        print(f"  {name}: {metric['score']} — {metric['rationale']}")
    if result.flag_reasons:
        print(f"Flags: {'; '.join(result.flag_reasons)}")
    if result.review:
        review = result.review
        print(f"Review: {review['judgement']} by {review['reviewer']}"
              + (f" (corrected score {review['corrected_score']})" if review['corrected_score'] else "")
              + (f" — {review['note']}" if review['note'] else ""))
    return 0


def cmd_review_judge(args: argparse.Namespace) -> int:
    with Store(args.db) as store:
        record = record_judgement(
            store,
            result_id=args.result_id,
            reviewer=args.reviewer,
            judgement=args.judgement,
            corrected_score=args.corrected_score,
            note=args.note or "",
        )
    print(f"Review recorded: {record['judgement']} on {args.result_id} by {args.reviewer}")
    if record["corrected_score"] is not None:
        print(f"  Corrected score: {record['corrected_score']}")
    return 0


def cmd_report(args: argparse.Namespace) -> int:
    with Store(args.db) as store:
        run = store.get_run(args.run)
        results = store.get_results(args.run)
        stats = store.review_stats(args.run)
    if args.format == "json":
        text = json.dumps(build_json_report(run, results, None, stats), indent=2)
    else:
        text = build_markdown_report(run, results, None, stats)
    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(text, encoding="utf-8")
        print(f"Report written to {out_path}")
    else:
        print(text)
    return 0


def cmd_serve(args: argparse.Namespace) -> int:
    import uvicorn

    from .web.app import create_app

    app = create_app(args.db)
    print(f"Northstar Eval Platform — web UI at http://{args.host}:{args.port}")
    print(f"Database: {args.db}")
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="nev",
        description="Northstar Eval Platform — deterministic LLM evaluation, "
                    "regression comparison and human review.",
    )
    parser.add_argument("--version", action="version", version=f"nev {__version__}")
    sub = parser.add_subparsers(dest="command", required=True)

    p_load = sub.add_parser("load", help="Validate a dataset and show its summary.")
    p_load.add_argument("dataset", help="Dataset file (.jsonl or .csv).")
    p_load.add_argument("--kb", help="Knowledge-base JSON file (default: knowledge_base.json next to the dataset).")
    p_load.set_defaults(func=cmd_load)

    p_run = sub.add_parser("run", help="Run an evaluation over a dataset.")
    p_run.add_argument("dataset", help="Dataset file (.jsonl or .csv).")
    p_run.add_argument("--kb", help="Knowledge-base JSON file.")
    p_run.add_argument("--provider", choices=ALL_PROVIDERS, default="mock-a",
                       help="LLM provider (default: mock-a).")
    p_run.add_argument("--model", default="", help="Model name (default: the provider name).")
    p_run.add_argument("--prompt-version", choices=("v1", "v2"), default="v2",
                       help="Prompt version recorded in the run manifest (default: v2).")
    p_run.add_argument("--name", default="", help="Human-readable run name.")
    p_run.add_argument("--out", help="Directory to write the Markdown and JSON reports into.")
    _add_common(p_run)
    p_run.set_defaults(func=cmd_run)

    p_cmp = sub.add_parser("compare", help="Compare two runs from the same dataset.")
    p_cmp.add_argument("run_a", help="Run id of the baseline (A).")
    p_cmp.add_argument("run_b", help="Run id of the new version (B).")
    p_cmp.add_argument("--out", help="File to write the comparison report to (.md or .json).")
    _add_common(p_cmp)
    p_cmp.set_defaults(func=cmd_compare)

    p_rev = sub.add_parser("review", help="Work the human-review queue.")
    rev_sub = p_rev.add_subparsers(dest="review_command", required=True)
    p_list = rev_sub.add_parser("list", help="List flagged items awaiting review.")
    p_list.add_argument("--run", required=True, help="Run id.")
    _add_common(p_list)
    p_list.set_defaults(func=cmd_review_list)
    p_show = rev_sub.add_parser("show", help="Show one result in full, with its review if any.")
    p_show.add_argument("result_id", help="Result id from the review queue.")
    _add_common(p_show)
    p_show.set_defaults(func=cmd_review_show)
    p_judge = rev_sub.add_parser("judge", help="Record a judgement on one result.")
    p_judge.add_argument("result_id", help="Result id from the review queue.")
    p_judge.add_argument("--reviewer", required=True, help="Name of the reviewer.")
    p_judge.add_argument("--judgement", choices=("agree", "disagree"), required=True)
    p_judge.add_argument("--corrected-score", type=float,
                         help="Required with --judgement disagree: the corrected 1–5 score.")
    p_judge.add_argument("--note", default="", help="Note explaining the judgement.")
    _add_common(p_judge)
    p_judge.set_defaults(func=cmd_review_judge)

    p_rep = sub.add_parser("report", help="Export a run's report.")
    p_rep.add_argument("run", help="Run id.")
    p_rep.add_argument("--format", choices=("md", "json"), default="md")
    p_rep.add_argument("--out", help="File to write the report to (default: print to stdout).")
    _add_common(p_rep)
    p_rep.set_defaults(func=cmd_report)

    p_serve = sub.add_parser("serve", help="Start the web UI.")
    p_serve.add_argument("--host", default="127.0.0.1")
    p_serve.add_argument("--port", type=int, default=8000)
    _add_common(p_serve)
    p_serve.set_defaults(func=cmd_serve)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    _setup_logging(getattr(args, "verbose", False))
    try:
        return args.func(args)
    except (DatasetError, ProviderError, ReviewError, NorthstarError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    except FileNotFoundError as exc:
        print(f"Error: file not found: {exc.filename or exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
