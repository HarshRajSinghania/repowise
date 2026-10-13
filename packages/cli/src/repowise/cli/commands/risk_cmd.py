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
        out["index"] = note
    return out


def _target_risk(
    repo_path: str,
    targets: tuple[str, ...],
    changed_files: tuple[str, ...],
    fmt: str,
    full: bool,
) -> None:
    """``--target``: what history says about touching these files."""
    fmt = _ta.resolve_format_for(fmt, full)
    repo = _ta.resolve_indexed_repo(
        path=repo_path,
        repo_alias=None,
        no_workspace=False,
        fmt=fmt,
        command="risk",
    )

    def _factory():
        from repowise.server.mcp_server.tool_risk.get_risk import get_risk

        return get_risk(
            targets=list(targets),
            changed_files=list(changed_files) or None,
            # _render_target_risk prints risk_type, change_magnitude, owner share,
            # bus factor, contributor count, and recent owner; --format json keeps
            # the PR blast radius the tool serves only on request.
            include=["churn", "owners", "blast"],
        )

    payload = _ta.run(repo, _factory, "get_risk")

    if full:
        _ta.emit_full(payload)
        return
    _ta.emit_error(payload, fmt, extra={"targets": list(targets)})
    projected = project_risk(payload)
    if fmt == "json":
        emit_json(projected)
        return
    _render_target_risk(projected, targets)
    _ta.print_index_note(payload, fmt)


def _render_target_risk(projected: dict, requested: tuple[str, ...]) -> None:
    """The human path for ``--target``: the directive first, then a card each."""
    from rich.markup import escape

    directive = projected.get("directive")
    if directive:
        # PR mode leads with the directive because that is the tool's contract:
        # what may break, what is missing, what to run.
        console.print(f"\n[bold]Directive[/bold] {escape(str(directive.get('summary', '')))}")
        for label, key in (
            ("May break", "may_break"),
            ("Missing co-changes", "missing_cochanges"),
            ("Files without tests", "missing_tests"),
            ("Tests to run", "tests_to_run"),
        ):
            _print_list(label, directive.get(key) or [])
        for label, key in _DIRECTIVE_RECORD_BLOCKS:
            _print_records(
                label,
                directive.get(key) or [],
                key,
                truncated=directive.get(f"{key}_truncated") or 0,
            )
        calls = directive.get("next_calls") or []
        _print_list("Then run", [c["cli"] for c in calls if c.get("cli")])

    targets = projected.get("targets") or {}
    for name in requested:
        card = targets.get(name)
        if card is None:
            # An excluded path is filtered out of the request before the tool
            # sees it, so a requested target with no card is an answer ("this
            # is not indexed here"), not a gap to render as an empty section.
            console.print(f"\n[yellow]{escape(name)} — not indexed (or excluded).[/yellow]")
            continue
        _render_card(name, card)
    # Anything the tool returned that was not asked for by name (it normalises
    # paths), so a caller never loses a card to a spelling difference.
    for name, card in targets.items():
        if name not in requested:
            _render_card(name, card)

    hotspots = projected.get("global_hotspots") or []
    if hotspots:
        table = Table(title="Elsewhere in this repo")
        table.add_column("File", style="cyan")
        table.add_column("Hotspot", justify="right")
        table.add_column("Owner")
        table.add_column("Fixes", justify="right")
        for h in hotspots:
            score = h.get("hotspot_score")
            fixes = h.get("fix_count")
            table.add_row(
                escape(str(h.get("file_path", ""))),
                "?" if score is None else f"{float(score):.0%}",
                escape(str(h.get("primary_owner") or "unknown")),
                "" if fixes is None else str(fixes),
            )
        console.print(table)
        console.print(
            "  [dim]Prior fixes are recency-weighted against this change's own date — "
            "one from a year earlier counts a half.[/dim]"
        )


def _print_diff_shape(result: ChangeRiskResult) -> None:
    """The supporting 0-10 diff-size score and its drivers."""
    risk = result.risk
    console.print(
        f"  [dim]Diff-size score: {risk.score:.1f}/10 — how big and spread out the change is, "
        f"not where it lands. Corpus-anchored to a single commit.[/dim]"
    )

    table = Table(title="Diff shape (each driver vs. the model's baseline commit)")
    table.add_column("Driver")
    table.add_column("Value", justify="right")
    table.add_column("Push", justify="right")
    for d in risk.top_drivers:
        push_color = "red" if d.contribution > 0 else "green"
        sign = "+" if d.contribution >= 0 else ""
        table.add_row(
            d.label,
            "-" if d.value is None else f"{d.value:g}",
            f"[{push_color}]{sign}{d.contribution:.2f}[/{push_color}]",
        )
    console.print(table)


def _emit_machine(subject: _Subject, result: ChangeRiskResult) -> None:
    """The json, markdown and github renderings of a scored change."""
    from repowise.core.analysis.change_risk import render

    fmt, fail_above = subject.fmt, subject.fail_above
    if fmt == "json":
        payload = change_risk_payload(result, scales=True)
        groups = _independent_changes_for(subject.repo_path, result, subject.revspec)
        if groups:
            payload["independent_changes"] = groups
        if fail_above is not None:
            payload["gate"] = {
                "fail_above_percentile": fail_above,
                # Unrounded, as the gate compares it: ``risk_percentile`` is
                # rounded to one decimal and can read as equal to the gate.
                "percentile": result.percentile,
                "status": render.percentile_gate(result, fail_above),
            }
        click.echo(json.dumps(payload, indent=2))
    elif fmt == "markdown":
        click.echo(render.render_markdown(result, fail_above), nl=False)
    else:
        for line in render.github_annotations(result, fail_above):
            click.echo(line)
        append_step_summary(render.render_markdown(result, fail_above))
        console.print(render.headline(result, fail_above, markdown=False), markup=False)
