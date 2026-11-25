"""Running an evaluation: dataset + provider + scorer, reproducibly.

A run records everything needed to reproduce it in its manifest:
the dataset's SHA-256 hash, the provider and model, the prompt
version, the scorer version and the timestamp. The same dataset and
config always produce the same scores, because the scorer and the
mock providers are deterministic and the real provider is called
with temperature 0. Reproducibility is a feature, not an accident:
it is what makes a regression comparison meaningful.
"""

from __future__ import annotations

import logging
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from .dataset import dataset_hash, dataset_summary, load_dataset, load_knowledge_base
from .errors import ProviderError
from .metrics import METRIC_NAMES
from .providers import REAL_PROVIDERS, ProviderConfig, build_provider
from .scorer import PASS_THRESHOLD, SCORER_VERSION, ItemScore, score_response
from .store import RunRecord, Store

log = logging.getLogger(__name__)

# Recorded in every manifest. The scorer and the mock providers are
# fully deterministic; the seed is reserved for any future metric
# that needs randomness, so a run's provenance stays complete.
DETERMINISM_SEED = 42


@dataclass
class RunConfig:
    """Everything a run needs, in one place."""

    dataset_path: str | Path
    kb_path: str | Path | None = None
    provider_name: str = "mock-a"
    model: str = ""
    prompt_version: str = "v2"
    db_path: str | Path = "northstar.db"
    run_name: str = ""
    base_url: str = ""
    api_key: str = ""
    # Full request URL for providers that take one (Gemini). When
    # set, it is used verbatim; see providers.GeminiProvider.
    endpoint_url: str = ""


@dataclass
class RunSummary:
    """The outcome of a run, for display and reporting."""

    run_id: str
    name: str
    dataset_name: str
    dataset_hash: str
    provider: str
    model: str
    prompt_version: str
    n_items: int
    mean_score: float
    pass_rate: float
    flagged: int
    by_category: dict
    by_metric: dict


def _default_kb_path(dataset_path: Path) -> Path:
    return dataset_path.parent / "knowledge_base.json"


def _provider_config(config: RunConfig) -> ProviderConfig:
    """Resolve the run's provider config.

    Real providers read their credentials from the environment when
    the run config does not supply them, so the CLI, the web UI and
    direct callers all behave the same way. A value supplied in the
    run config wins over the environment.
    """
    if config.provider_name in REAL_PROVIDERS:
        cfg = ProviderConfig.from_env(
            config.provider_name,
            model=config.model,
            prompt_version=config.prompt_version,
        )
        if config.api_key:
            cfg.api_key = config.api_key
        if config.base_url:
            cfg.base_url = config.base_url
        if config.endpoint_url:
            cfg.endpoint_url = config.endpoint_url
        return cfg
    return ProviderConfig(
        name=config.provider_name,
        model=config.model or config.provider_name,
        prompt_version=config.prompt_version,
        base_url=config.base_url,
        api_key=config.api_key,
    )


def run_evaluation(config: RunConfig) -> RunSummary:
    """Load the dataset, run it through the provider, score, persist."""
    dataset_path = Path(config.dataset_path)
    items = load_dataset(dataset_path)
    kb_path = Path(config.kb_path) if config.kb_path else _default_kb_path(dataset_path)
    docs = load_knowledge_base(kb_path)
    missing_docs = sorted({d for item in items for d in item.source_doc_ids if d not in docs})
    if missing_docs:
        raise ProviderError(
            f"Dataset references knowledge-base documents that do not exist: "
            f"{', '.join(missing_docs)}. Check {kb_path}."
        )
    provider_config = _provider_config(config)
    provider = build_provider(config.provider_name, provider_config)
    log.info(
        "Starting run: dataset=%s (%d items), provider=%s, model=%s, prompt=%s",
        dataset_path.name, len(items), config.provider_name, provider_config.model,
        config.prompt_version,
    )

    results: list[ItemScore] = []
    for item in items:
        try:
            response = provider.generate(item, docs)
        except ProviderError:
            raise
        except Exception as exc:  # A provider bug must fail the run, not fake a result.
            raise ProviderError(f"Provider {config.provider_name} failed on item {item.id}: {exc}") from exc
        results.append(score_response(item, response, docs))

    run_id = str(uuid.uuid4())
    name = config.run_name or f"{config.provider_name}-{provider_config.model}-{config.prompt_version}"
    created_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
    mean = round(sum(r.mean_score for r in results) / len(results), 2)
    passed = sum(1 for r in results if r.passed)
    record = RunRecord(
        id=run_id,
        name=name,
        dataset_name=dataset_path.name,
        dataset_hash=dataset_hash(items),
        provider=config.provider_name,
        model=provider_config.model,
        prompt_version=config.prompt_version,
        scorer_version=SCORER_VERSION,
        created_at=created_at,
        n_items=len(items),
        mean_score=mean,
        pass_rate=round(passed / len(items), 3),
    )
    with Store(config.db_path) as store:
        store.save_run(record, results)
    by_category: dict[str, dict] = {}
    for category in sorted({r.category for r in results}):
        group = [r for r in results if r.category == category]
        by_category[category] = {
            "n": len(group),
            "mean": round(sum(r.mean_score for r in group) / len(group), 2),
            "pass_rate": round(sum(1 for r in group if r.passed) / len(group), 3),
        }
    by_metric: dict[str, dict] = {}
    for metric_name in METRIC_NAMES:
        values = [r.metrics[metric_name].score for r in results]
        by_metric[metric_name] = {
            "mean": round(sum(values) / len(values), 2),
            "min": min(values),
            "max": max(values),
        }
    summary = RunSummary(
        run_id=run_id,
        name=name,
        dataset_name=dataset_path.name,
        dataset_hash=dataset_hash(items),
        provider=config.provider_name,
        model=provider_config.model,
        prompt_version=config.prompt_version,
        n_items=len(items),
        mean_score=mean,
        pass_rate=round(passed / len(items), 3),
        flagged=sum(1 for r in results if r.needs_review),
        by_category=by_category,
        by_metric=by_metric,
    )
    log.info(
        "Run %s finished: mean %.2f, pass rate %.1f%%, %d flagged for review",
        run_id, mean, summary.pass_rate * 100, summary.flagged,
    )
    return summary


def manifest(run: RunRecord) -> dict:
    """The reproducibility manifest for a stored run."""
    return {
        "run_id": run.id,
        "name": run.name,
        "dataset_name": run.dataset_name,
        "dataset_hash": run.dataset_hash,
        "provider": run.provider,
        "model": run.model,
        "prompt_version": run.prompt_version,
        "scorer_version": run.scorer_version,
        "determinism_seed": DETERMINISM_SEED,
        "pass_threshold": PASS_THRESHOLD,
        "created_at": run.created_at,
    }
