"""``repowise risk`` — just-in-time change risk, for a change or for paths.

Two questions, one command, because they are the same question asked of
different subjects.

A REVSPEC scores a *change* from its diff shape — size, diffusion, author
familiarity — and prints a benchmarked review priority plus a supporting 0-10
diff-shape score with an attributable breakdown. Runs
in-process: pure git + learned constants, no LLM, no network. A natural
pre-merge / PR gate, complementary to ``repowise health`` (which scores files).

``--target <path>`` instead asks what history says about touching those files:
bug-fix pressure, churn trend, dependents, co-change partners, ownership. That
is the ``get_risk`` MCP tool, and this command is a thin adapter over it — the
same seam ``ask`` / ``context`` / ``symbol`` / ``why`` use. It reads the index
rather than git, so the repo must be indexed.

``--fail-above-percentile P`` makes the REVSPEC path a CI gate on the
repo-relative percentile (never the 0-10 score), with the shared CI exit codes
(:mod:`repowise.cli.ci`): 1 above the gate, 2 when the change cannot be read
or ranked.

Examples:
    repowise risk                       # score uncommitted work, else HEAD
    repowise risk HEAD                  # score the last commit
    repowise risk abc123                # score a single commit
    repowise risk main..HEAD            # score a branch / PR range as one change
    repowise risk --fail-above-percentile 95 --format github  # gate the CI change
    repowise risk --target a.py -t b.py # what history says about those files
"""

from __future__ import annotations

import json
import subprocess
from dataclasses import dataclass
from pathlib import Path

import click
from rich.table import Table

from repowise.cli.ci import (
    CI_FORMATS,
    EXIT_GATE_FAILED,
    CannotEvaluateError,
    append_step_summary,
    cannot_evaluate,
    ci_notices,
    ci_revspec,
    repo_root,
)
from repowise.cli.commands import _tool_adapters as _ta
from repowise.cli.helpers import console, repo_index_session, run_async
from repowise.cli.output import emit_json, format_option, full_option
from repowise.core.analysis.change_risk import (
    ChangeRiskResult,
    change_risk_payload,
    review_priority_classification,
    score_live_change,
)
from repowise.core.analysis.change_risk.render import ordinal as _ordinal
from repowise.core.analysis.change_risk.render import percentile_text

_PRIORITY_LEAD = {
    "low": "Smaller or more focused than a typical commit in this repo",
    "moderate": "Typical diff size and spread for this repo",
    "high": "Larger or more dispersed than most commits in this repo",
}


#: Per-target keys the trim drops, and nothing else.
#:
#: A denylist, not an allowlist, and deliberately so: the failure mode of a
#: denylist is a payload slightly larger than intended, while the failure mode
#: of an allowlist is a silently discarded answer — which is the mistake the
#: adapter commands' first projections made eight times over. ``get_risk``'s
#: card is a page of small scalars rather than a block of source, so there is
#: little to win by trimming harder and a lot to lose.
#:
#: ``_base_dep_count`` is bookkeeping the tool uses to rebuild ``risk_summary``
#: after enrichment. ``impact_surface`` is the transitive dependent set, the one
#: genuinely unbounded block, and its own summary counts survive in
#: ``risk_summary`` and ``dependents_count``. It is opt-in on the tool now and
#: this command does not ask for it; the entry stays so that a caller passing a
#: hand-built payload gets the same projection.
_DROPPED_TARGET_KEYS = ("_base_dep_count", "impact_surface")


def _project_target(card: dict) -> dict:
    """One ``get_risk`` target card, minus the blocks in ``_DROPPED_TARGET_KEYS``."""
    return {k: v for k, v in card.items() if k not in _DROPPED_TARGET_KEYS}


def project_risk(payload: dict) -> dict:
    """``get_risk``'s dict, trimmed to what a CLI caller reads.

    Kept whole: ``directive`` (in PR mode this *is* the answer, and the tool's
    own docstring tells a reader to lead with it), ``pr_blast_radius``,
    ``global_hotspots``, and the freshness half of ``_meta``. Dropped: the
    timing half of ``_meta``, and the per-target keys named above.

    ``pr_blast_radius`` survives although ``directive`` summarises much of it,
    because the tool has *already* capped its four noisy lists (15/10/10/5)
    before it gets here, so keeping it costs almost nothing.
    """
    out: dict = {}
    if payload.get("directive"):
        out["directive"] = payload["directive"]
    out["targets"] = {
        name: _project_target(card) for name, card in (payload.get("targets") or {}).items()
    }
    for key in ("risk_scales", "pr_blast_radius", "global_hotspots", "omission_marker"):
        if payload.get(key):
            out[key] = payload[key]
    note = _ta.index_note(payload)
    if note:
        out["index_note"] = note
    return out


@dataclass(frozen=True)
class _Subject:
    repo_path: Path
    revspec: str | None
    fmt: str
    fail_above: float | None


def _score(
    subject: _Subject, extensions: tuple[str, ...], exclude: tuple[str, ...], baseline: int
) -> ChangeRiskResult:
    """Score the change, or exit 2 when it cannot be read or, gated, ranked."""
    status = ci_notices(subject.fmt)
    # Verify a single commit ref exists before announcing baseline sampling.
    # Skip ranges (contain ..) and empty revspecs; score_live_change handles those.
    revspec = subject.revspec
    if revspec and ".." not in revspec:
        try:
            subprocess.run(
                ["git", "rev-parse", "--verify", f"{revspec}^{{commit}}"],
                cwd=subject.repo_path,
                check=True,
                capture_output=True,
                text=True,
            )
        except subprocess.CalledProcessError as exc:
            detail = (exc.stderr or "").strip().split("\n")[0] or str(exc)
            cannot_evaluate(
                subject.fmt,
                "change_unreadable",
                f"Could not read change {revspec!r} in {subject.repo_path}: {detail}",
            )
    if baseline and subject.fmt == "table":
        status.print(f"[dim]Sampling up to {baseline} recent commits…[/dim]")
    try:
        result = score_live_change(
            subject.repo_path,
            subject.revspec,
            extensions=extensions,
            exclude_patterns=exclude,
            baseline=baseline,
        )
    except subprocess.CalledProcessError as exc:
        detail = (exc.stderr or "").strip().split("\n")[0] or str(exc)
        cannot_evaluate(
            subject.fmt,
            "change_unreadable",
            f"Could not read change {subject.revspec or 'HEAD'!r} in {subject.repo_path}: {detail}",
        )
    except Exception as exc:
        # Other errors (not a repo, missing history, etc.) cannot be scored.
        cannot_evaluate(
            subject.fmt,
            "change_unreadable",
            f"Could not read change {subject.revspec or 'HEAD'!r} in {subject.repo_path}: {exc}",
        )
    if subject.fail_above is not None and result.percentile is None:
        from repowise.core.analysis.change_risk.render import unranked_reason

        cannot_evaluate(subject.fmt, "no_baseline", unranked_reason(result, baseline))
    if result.features.nf == 0:
        status.print(
            f"[yellow]No counted file changes in {result.features.ref!r} "
            f"(check the revspec, --ext, or exclusion filters).[/yellow]"
        )
    return result
