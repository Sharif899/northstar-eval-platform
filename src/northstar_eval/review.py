"""Human review: the queue, judgements and coverage statistics.

Automated scores are a triage signal, not a verdict. Anything the
scorer flags goes to a human reviewer, who records whether they
agree with the automated score and, when they disagree, the
corrected score and a note. The review history is kept in full —
every judgement is a row, never an overwrite — so overturn rates
and calibration can be measured honestly later.
"""

from __future__ import annotations

from .errors import ReviewError
from .store import ResultRecord, Store

JUDGEMENTS = ("agree", "disagree")


def get_review_queue(store: Store, run_id: str) -> list[ResultRecord]:
    """Flagged results in a run that have no review yet, worst first."""
    results = store.get_results(run_id)
    queue = [r for r in results if r.needs_review and r.review is None]
    return sorted(queue, key=lambda r: (r.mean_score, r.item_id))


def record_judgement(
    store: Store,
    result_id: str,
    reviewer: str,
    judgement: str,
    corrected_score: float | None = None,
    note: str = "",
) -> dict:
    """Record a reviewer's judgement on one result.

    ``agree`` means the automated score stands. ``disagree`` means
    the reviewer supplies a corrected score (1–5) and a note saying
    why. A disagreement without a corrected score is refused —
    otherwise the review would say "wrong" without saying what is
    right.
    """
    if judgement not in JUDGEMENTS:
        raise ReviewError(
            f"Unknown judgement {judgement!r}; use 'agree' or 'disagree'."
        )
    if not reviewer.strip():
        raise ReviewError("A reviewer name is required so judgements are attributable.")
    if judgement == "disagree":
        if corrected_score is None:
            raise ReviewError(
                "A 'disagree' judgement needs a corrected score (1–5)."
            )
        if not 1.0 <= corrected_score <= 5.0:
            raise ReviewError(
                f"Corrected score {corrected_score} is outside the 1–5 rubric scale."
            )
        if not note.strip():
            raise ReviewError("A 'disagree' judgement needs a note explaining why.")
    else:
        corrected_score = None
    return store.save_review(
        result_id=result_id,
        reviewer=reviewer.strip(),
        judgement=judgement,
        corrected_score=corrected_score,
        note=note.strip(),
    )
