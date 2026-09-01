"""Orchestration for the opt-in AI-assisted trajectory analysis mode."""

from __future__ import annotations

import json
import os
import re
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Callable

from llm import LLMBackend
from prompts.ai_assisted_prompt import ISSUE_CLUE_PROMPT, TRAJECTORY_ANALYSIS_PROMPT

from .checks import grep_issue_keywords, run_deterministic_checks
from .input import discover_cases
from .report import save_batch_index, save_case_report
from .validator import (
    PHASES,
    locate_quote,
    validate_clue_analysis,
    validate_trajectory_analysis,
)


def _log(message: str) -> None:
    """Print an immediately-flushed, timestamped progress line."""
    timestamp = time.strftime("%H:%M:%S")
    print(f"[{timestamp}] {message}", flush=True)


def _json_for_prompt(value: Any) -> str:
    return json.dumps(value, indent=2, ensure_ascii=False)


def _truncate_text(value: Any, limit: int) -> str:
    text = value if isinstance(value, str) else "" if value is None else str(value)
    if len(text) <= limit:
        return text
    keep = max(0, limit - 18)
    return text[:keep] + "\n...[truncated]"


def _summarize_spt_mutations(spt: dict[str, Any]) -> dict[str, Any]:
    """Summarize raw SPT entries into file/transformation context for prompting."""
    entries = spt.get("data") if isinstance(spt.get("data"), list) else []
    if not entries:
        return {
            "total_entries": 0,
            "files_touched": [],
            "transformations": [],
            "sample_mutations": [],
        }

    file_counts: Counter[str] = Counter()
    transformation_counts: Counter[str] = Counter()
    sample = []

    for item in entries:
        if not isinstance(item, dict):
            continue
        file_path = item.get("file")
        transformation = item.get("transformation")
        positions = item.get("positions") if isinstance(item.get("positions"), list) else []

        if isinstance(file_path, str) and file_path:
            file_counts[file_path] += 1
        if isinstance(transformation, str) and transformation:
            transformation_counts[transformation] += 1

        if len(sample) < 40:
            sample.append(
                {
                    "order": item.get("order"),
                    "file": file_path,
                    "transformation": transformation,
                    "position_count": len(positions),
                    "positions": positions[:6],
                }
            )

    return {
        "total_entries": len(entries),
        "files_touched": [
            {"file": file_path, "count": count}
            for file_path, count in file_counts.most_common(40)
        ],
        "transformations": [
            {"name": name, "count": count}
            for name, count in transformation_counts.most_common(20)
        ],
        "sample_mutations": sample,
    }


def _compact_enriched_context(case: dict[str, Any], deterministic: dict[str, Any]) -> dict[str, Any]:
    """Build a compact metadata context block for optional prompt enrichment."""
    result = case.get("result", {}) if isinstance(case.get("result"), dict) else {}
    spt = case.get("spt", {}) if isinstance(case.get("spt"), dict) else {}
    metadata = case.get("metadata", {}) if isinstance(case.get("metadata"), dict) else {}
    tests = deterministic.get("tests", {}) if isinstance(deterministic.get("tests"), dict) else {}
    patches = deterministic.get("patches", {}) if isinstance(deterministic.get("patches"), dict) else {}
    iterations = (
        deterministic.get("iterations", {})
        if isinstance(deterministic.get("iterations"), dict)
        else {}
    )
    errors = deterministic.get("errors", {}) if isinstance(deterministic.get("errors"), dict) else {}

    return {
        "case": {
            "case_id": case.get("case_id"),
            "case_name": case.get("case_name"),
            "format": case.get("format"),
            "issue_description_excerpt": _truncate_text(case.get("issue_description", ""), 1200),
        },
        "result": {
            "status": result.get("status"),
            "resolved": result.get("resolved"),
            "patch_successfully_applied": result.get("patch_successfully_applied"),
            "patch_exists": result.get("patch_exists"),
            "tests_status": result.get("tests_status"),
        },
        "spt": {
            key: spt.get(key)
            for key in (
                "metadata_available",
                "entry_count",
                "applied",
                "note",
                "error",
            )
        },
        "spt_mutations_summary": _summarize_spt_mutations(spt),
        "run_metadata": {
            key: metadata.get(key)
            for key in ("model", "provider", "cost", "api_calls", "tokens", "exit_status")
        },
        "deterministic_summary": {
            "event_count": deterministic.get("event_count"),
            "test_run_count": tests.get("run_count"),
            "test_failure_count": tests.get("failure_count"),
            "test_rerun_count": tests.get("rerun_count"),
            "patch_iteration_count": patches.get("iteration_count"),
            "backtracking_signal_count": iterations.get("backtracking_signal_count"),
            "git_reversion_event_ids": (iterations.get("git_reversion_event_ids") or [])[:20],
            "error_event_count": errors.get("count"),
            "result_reconciliation": deterministic.get("result_reconciliation", {}),
        },
    }


def _prompt_events(events: list[dict[str, Any]], character_budget: int = 200_000) -> list[dict[str, Any]]:
    if not events:
        return []
    if character_budget < 0:
        raise ValueError("character_budget must be non-negative")
    per_event_budget = min(12_000, character_budget // len(events))
    prompt_events = []
    for event in events:
        content = event.get("content", "")
        if not isinstance(content, str):
            content = str(content)
        item = {
            "id": event["id"],
            "role": event.get("role"),
            "kind": event.get("kind"),
            "timestamp": event.get("timestamp"),
            "duration_seconds": event.get("duration_seconds"),
            "duration_is_estimated": event.get("duration_is_estimated", False),
            "tool_name": event.get("tool_name"),
            "tool_status": event.get("tool_status"),
        }
        if per_event_budget == 0:
            item["content_omitted"] = True
            item["content_length"] = len(content)
        elif len(content) <= per_event_budget:
            item["content"] = content
        else:
            half = per_event_budget // 2
            item["content_start"] = content[:half]
            item["content_end"] = content[-(per_event_budget - half):]
            item["content_length"] = len(content)
        prompt_events.append(item)
    return prompt_events


def _generate_validated(
    backend: Any,
    prompt: str,
    validator: Callable[[dict[str, Any]], dict[str, Any]],
    attempts: int = 3,
    label: str = "model call",
) -> tuple[dict[str, Any], list[str]]:
    errors = []
    current_prompt = prompt
    for attempt in range(attempts):
        started = time.monotonic()
        _log(f"    {label}: requesting model (attempt {attempt + 1}/{attempts})…")
        try:
            response = backend.generate_json(current_prompt)
        except Exception as exc:
            _log(f"    {label}: request failed after {time.monotonic() - started:.1f}s — {exc}")
            raise ValueError(f"Model request failed: {exc}") from exc
        try:
            result = validator(response)
            _log(f"    {label}: valid response in {time.monotonic() - started:.1f}s")
            return result, errors
        except Exception as exc:
            errors.append(str(exc))
            _log(
                f"    {label}: response invalid ({time.monotonic() - started:.1f}s) — {exc}"
            )
            if attempt + 1 < attempts:
                current_prompt = (
                    prompt
                    + "\n\nYour previous response was invalid:\n"
                    + _json_for_prompt(response)
                    + "\n\nValidation error:\n"
                    + str(exc)
                    + "\nReturn a corrected complete JSON object only. Preserve all valid "
                    + "content and correct every occurrence of this problem."
                )
    raise ValueError(f"Model output remained invalid after {attempts} attempts: {' | '.join(errors)}")


def _recover_exact_quote(quote: Any, source: str) -> tuple[Any, bool | None]:
    """Reconcile a model-provided quote against its source text.

    Returns (quote, exact_match):
    - exact_match=True  → verbatim substring (quote returned unchanged).
    - exact_match=False → recovered via normalized matching; the quote is
      replaced with the exact source span so it stays faithful.
    - exact_match=None  → no match found at all; the original quote is kept and
      must be flagged for the reader.
    """
    if not isinstance(quote, str) or not isinstance(source, str):
        return quote, None
    located = locate_quote(quote, source)
    if located is None:
        return quote, None
    start, end, exact = located
    return source[start:end], exact


def _repair_evidence_quotes(
    evidence: Any, event_map: dict[str, dict[str, Any]]
) -> None:
    if not isinstance(evidence, list):
        return
    for item in evidence:
        if not isinstance(item, dict):
            continue
        event = event_map.get(item.get("event_id"))
        if not event:
            continue
        recovered, exact = _recover_exact_quote(item.get("quote"), event.get("content", ""))
        item["quote"] = recovered
        if exact is True:
            item["exact_match"] = True
            item.pop("match_warning", None)
        elif exact is False:
            item["exact_match"] = False
            item["match_warning"] = (
                "Recovered by normalized matching; may differ slightly from the model's quote."
            )
        else:
            item["exact_match"] = False
            item["match_warning"] = (
                "Quote could not be located in the source event; shown as provided by the model."
            )


def _repair_trajectory_response(
    response: dict[str, Any], events: list[dict[str, Any]]
) -> dict[str, Any]:
    """Apply only evidence-preserving repairs before strict schema validation."""
    if not isinstance(response, dict):
        return response
    event_map = {event["id"]: event for event in events}
    warnings = []
    repaired_nodes = []
    nodes = response.get("nodes")
    if isinstance(nodes, list):
        for node in nodes:
            if not isinstance(node, dict):
                repaired_nodes.append(node)
                continue
            phase = node.get("phase")
            if phase not in PHASES and isinstance(phase, str):
                candidates = [candidate.strip() for candidate in phase.split("|")]
                valid_candidates = [candidate for candidate in candidates if candidate in PHASES]
                if valid_candidates:
                    node["phase"] = valid_candidates[0]
                    if isinstance(node.get("confidence"), (int, float)):
                        node["confidence"] = min(float(node["confidence"]), 0.75)
                    warnings.append(
                        f"Node {node.get('id')} phase '{phase}' was normalized to "
                        f"'{valid_candidates[0]}'."
                    )
            if not isinstance(node.get("resources"), list):
                node["resources"] = []
            _repair_evidence_quotes(node.get("evidence"), event_map)
            if not node.get("event_ids") and isinstance(node.get("evidence"), list):
                inferred_ids = []
                for item in node["evidence"]:
                    event_id = item.get("event_id") if isinstance(item, dict) else None
                    if event_id in event_map and event_id not in inferred_ids:
                        inferred_ids.append(event_id)
                if inferred_ids:
                    node["event_ids"] = inferred_ids
                    warnings.append(
                        f"Node {node.get('id')} event_ids were recovered from its evidence."
                    )
            repaired_nodes.append(node)
    response["nodes"] = _resolve_node_ordering(repaired_nodes, events, warnings)

    classified = {
        event_id
        for node in response["nodes"]
        if isinstance(node, dict) and isinstance(node.get("event_ids"), list)
        for event_id in node["event_ids"]
        if event_id in event_map
    }
    response["unclassified_event_ids"] = [
        event["id"] for event in events if event["id"] not in classified
    ]
    response["normalization_warnings"] = warnings
    return response


def _resolve_node_ordering(
    nodes: list[Any], events: list[dict[str, Any]], warnings: list[str]
) -> list[Any]:
    """Make node event ranges strictly chronological and non-overlapping.

    Real trajectories (especially OpenCode, whose messages interleave reasoning,
    tool, and empty step-marker parts) often lead the model to emit nodes whose
    event ranges are slightly out of order or overlapping. Rather than failing
    the whole case, reorder nodes by their first event and trim any event that an
    earlier node already claimed. Trimmed evidence is removed with it; a node that
    is left with no events (or no evidence) is dropped and its events fall back to
    unclassified. Every change is recorded as a normalization warning.
    """
    event_order = {event["id"]: index for index, event in enumerate(events)}

    def first_position(node: Any) -> int:
        if not isinstance(node, dict) or not isinstance(node.get("event_ids"), list):
            return 10**12
        positions = [event_order[e] for e in node["event_ids"] if e in event_order]
        return min(positions, default=10**12)

    ordered = sorted(nodes, key=first_position)
    resolved: list[Any] = []
    previous_last = -1
    for node in ordered:
        if not isinstance(node, dict) or not isinstance(node.get("event_ids"), list):
            resolved.append(node)
            continue

        # Unique, valid, chronological event IDs for this node.
        seen: set[str] = set()
        positioned = []
        for event_id in node["event_ids"]:
            if event_id in event_order and event_id not in seen:
                seen.add(event_id)
                positioned.append(event_id)
        positioned.sort(key=lambda e: event_order[e])

        # Drop events already claimed by (or inside the range of) earlier nodes.
        kept = [e for e in positioned if event_order[e] > previous_last]
        dropped = [e for e in positioned if e not in kept]

        if not kept:
            warnings.append(
                f"Node {node.get('id')} was dropped because its events overlap earlier nodes."
            )
            continue

        if dropped or kept != node["event_ids"]:
            node["event_ids"] = kept
            if dropped and isinstance(node.get("evidence"), list):
                node["evidence"] = [
                    item
                    for item in node["evidence"]
                    if not (isinstance(item, dict) and item.get("event_id") in dropped)
                ]
            warnings.append(
                f"Node {node.get('id')} event range was trimmed to keep the graph "
                f"chronological and non-overlapping."
            )

        if not node.get("evidence"):
            warnings.append(
                f"Node {node.get('id')} was dropped after trimming left it without evidence."
            )
            continue

        previous_last = event_order[kept[-1]]
        resolved.append(node)
    return resolved


def _phase_statistics(
    nodes: list[dict[str, Any]],
    events: list[dict[str, Any]],
    unclassified_event_ids: list[str],
) -> dict[str, Any]:
    event_map = {event["id"]: event for event in events}
    phase_events: dict[str, set[str]] = defaultdict(set)
    phase_durations: dict[str, float] = defaultdict(float)
    node_counts = Counter()
    segments = []

    for node in nodes:
        phase = node["phase"]
        node_counts[phase] += 1
        for event_id in node["event_ids"]:
            if event_id in phase_events[phase]:
                continue
            phase_events[phase].add(event_id)
            duration = event_map[event_id].get("duration_seconds")
            if isinstance(duration, (int, float)):
                phase_durations[phase] += float(duration)

        if segments and segments[-1]["phase"] == phase:
            segments[-1]["node_ids"].append(node["id"])
            segments[-1]["event_ids"].extend(
                event_id for event_id in node["event_ids"] if event_id not in segments[-1]["event_ids"]
            )
        else:
            segments.append(
                {"phase": phase, "node_ids": [node["id"]], "event_ids": list(node["event_ids"])}
            )

    phase_names = sorted(node_counts)
    classified_event_count = len(set().union(*phase_events.values())) if phase_events else 0
    total_event_count = len(events)
    by_phase = {
        phase: {
            "node_count": node_counts[phase],
            "event_count": len(phase_events[phase]),
            "classified_event_percentage": round(
                100 * len(phase_events[phase]) / classified_event_count, 2
            ) if classified_event_count else 0.0,
            "total_event_percentage": round(
                100 * len(phase_events[phase]) / total_event_count, 2
            ) if total_event_count else 0.0,
            "duration_seconds": round(phase_durations[phase], 3),
        }
        for phase in phase_names
    }
    largest_by_events = max(phase_names, key=lambda phase: len(phase_events[phase])) if phase_names else None
    phases_with_time = [phase for phase in phase_names if phase_durations[phase] > 0]
    largest_by_time = (
        max(phases_with_time, key=lambda phase: phase_durations[phase]) if phases_with_time else None
    )
    phase_order = {
        "Localization": 0,
        "Debugging": 1,
        "Planning": 2,
        "Patching": 3,
        "Validation": 4,
    }
    regressions = []
    for previous, current in zip(segments, segments[1:]):
        previous_phase = previous["phase"]
        current_phase = current["phase"]
        if previous_phase in phase_order and current_phase in phase_order:
            if phase_order[current_phase] < phase_order[previous_phase]:
                regressions.append(
                    {
                        "from_phase": previous_phase,
                        "to_phase": current_phase,
                        "at_node_id": current["node_ids"][0],
                    }
                )
    return {
        "by_phase": by_phase,
        "segments": segments,
        "phase_pattern": " → ".join(segment["phase"] for segment in segments),
        "largest_phase_by_events": largest_by_events,
        "largest_phase_by_time": largest_by_time,
        "classified_event_count": classified_event_count,
        "unclassified_event_ids": unclassified_event_ids,
        "unclassified_event_count": len(unclassified_event_ids),
        "classification_coverage": round(
            classified_event_count / total_event_count, 4
        ) if total_event_count else 0.0,
        "phase_regressions": regressions,
        "phase_regression_count": len(regressions),
    }


def analyze_case(
    case: dict[str, Any],
    backend: Any,
    enriched_prompt_context: bool = False,
    run_issue_prompt: bool = True,
    run_trajectory_prompt: bool = True,
) -> dict[str, Any]:
    if not run_issue_prompt and not run_trajectory_prompt:
        raise ValueError("At least one prompt stage must be enabled")

    _log(f"  {len(case['events'])} events · running deterministic checks…")
    deterministic = run_deterministic_checks(case)
    _log(
        f"  deterministic checks done: {deterministic['git']['command_count']} git command(s), "
        f"{deterministic['tests']['run_count']} test run(s)"
    )

    if run_issue_prompt and case["issue_description"]:
        _log(f"  extracting issue keywords ({len(case['issue_description'])} chars)…")
        clue_prompt = ISSUE_CLUE_PROMPT.format(ISSUE_DESCRIPTION=case["issue_description"])
        clue_analysis, clue_errors = _generate_validated(
            backend,
            clue_prompt,
            lambda response: validate_clue_analysis(response, case["issue_description"]),
            label="issue keywords",
        )
        _log(f"  extracted {len(clue_analysis.get('clues', []))} keyword(s)")
    elif run_issue_prompt:
        _log("  no issue description found; skipping keyword extraction")
        clue_analysis = {
            "clues": [],
            "summary": {"summary": "No issue description was found."},
        }
        clue_errors = []
    else:
        _log("  issue keyword stage disabled by flag")
        clue_analysis = {
            "clues": [],
            "summary": {"summary": "Issue keyword stage disabled by flag."},
        }
        clue_errors = []

    if run_trajectory_prompt:
        prompt_events = _prompt_events(case["events"])
        prompt_chars = sum(
            len(str(item.get("content", "")))
            + len(str(item.get("content_start", "")))
            + len(str(item.get("content_end", "")))
            for item in prompt_events
        )
        _log(
            f"  building trajectory graph from {len(prompt_events)} event(s) "
            f"(~{prompt_chars // 1000}k chars sent)…"
        )
        enriched_context = (
            _compact_enriched_context(case, deterministic) if enriched_prompt_context else {}
        )
        if enriched_prompt_context:
            _log("  enriched prompt context enabled for trajectory graph generation")

        if enriched_prompt_context:
            spt_hypothesis_schema_suffix = (
                ',\n  "spt_impact_hypothesis": {\n'
                '    "likely_impacted": "yes | no | uncertain",\n'
                '    "confidence": 0.0,\n'
                '    "signals": ["short, concrete reasons supporting likely impact"],\n'
                '    "counter_signals": ["short reasons against likely impact"],\n'
                '    "evidence": [\n'
                '      {\n'
                '        "event_id": "E2",\n'
                '        "quote": "exact substring copied from that event content",\n'
                '        "why": "why this supports the impact hypothesis"\n'
                '      }\n'
                '    ]\n'
                '  }'
            )
            spt_hypothesis_rules = (
                "- If OPTIONAL ENRICHED CONTEXT contains meaningful SPT metadata, include "
                "spt_impact_hypothesis. If SPT metadata is absent/empty, omit it.\n"
                "- When OPTIONAL ENRICHED CONTEXT includes SPT mutation details "
                "(files, transformations, and positions), use them to reason about "
                "whether observed trajectory behavior appears influenced by those perturbations."
            )
        else:
            spt_hypothesis_schema_suffix = ""
            spt_hypothesis_rules = ""

        trajectory_prompt = TRAJECTORY_ANALYSIS_PROMPT.format(
            ISSUE_CLUES_JSON=_json_for_prompt(clue_analysis),
            EVENTS_JSON=_json_for_prompt(prompt_events),
            ENRICHED_CONTEXT_JSON=_json_for_prompt(enriched_context),
            SPT_HYPOTHESIS_SCHEMA_SUFFIX=spt_hypothesis_schema_suffix,
            SPT_HYPOTHESIS_RULES=spt_hypothesis_rules,
        )
        trajectory_analysis, trajectory_errors = _generate_validated(
            backend,
            trajectory_prompt,
            lambda response: validate_trajectory_analysis(
                _repair_trajectory_response(response, case["events"]), case["events"]
            ),
            label="trajectory graph",
        )
        _log(f"  graph built: {len(trajectory_analysis.get('nodes', []))} node(s)")
    else:
        _log("  trajectory graph stage disabled by flag")
        trajectory_analysis = {
            "nodes": [],
            "unclassified_event_ids": [event["id"] for event in case["events"]],
            "normalization_warnings": [
                "Trajectory graph stage disabled by flag; all events are unclassified."
            ],
        }
        trajectory_errors = []

    phase_stats = _phase_statistics(
        trajectory_analysis["nodes"],
        case["events"],
        trajectory_analysis["unclassified_event_ids"],
    )
    keyword_grep = grep_issue_keywords(
        case["events"], clue_analysis.get("clues", []), case.get("issue_event_id")
    )
    rule_based = {
        "git_commands": deterministic["git"]["commands"],
        "keyword_grep": keyword_grep,
        "phase_counts": {
            phase: values["node_count"]
            for phase, values in phase_stats["by_phase"].items()
        },
    }
    spt_hypothesis = trajectory_analysis.get("spt_impact_hypothesis") if enriched_prompt_context else None
    return {
        "schema_version": "ai-trajectory-analysis-v1",
        "case": {
            key: case.get(key)
            for key in (
                "case_name",
                "case_id",
                "case_root",
                "trajectory_path",
                "format",
                "issue_description",
                "issue_event_id",
                "metadata",
                "result",
                "spt",
                "submission",
            )
        },
        "events": case["events"],
        "nodes": trajectory_analysis["nodes"],
        "unclassified_event_ids": trajectory_analysis["unclassified_event_ids"],
        "phase_statistics": phase_stats,
        "deterministic_checks": deterministic,
        "spt_impact_hypothesis": spt_hypothesis,
        "issue_clues": clue_analysis,
        "rule_based_results": rule_based,
        "data_quality": {
            "result_match_status": case["result"].get("status"),
            "result_match_errors": case["result"].get("errors", []),
            "enriched_prompt_context_enabled": enriched_prompt_context,
            "issue_prompt_enabled": run_issue_prompt,
            "trajectory_prompt_enabled": run_trajectory_prompt,
            "clue_generation_retry_errors": clue_errors,
            "trajectory_generation_retry_errors": trajectory_errors,
            "spt_hypothesis_present": isinstance(spt_hypothesis, dict),
            "trajectory_normalization_warnings": trajectory_analysis.get(
                "normalization_warnings", []
            ),
            "classified_event_count": phase_stats["classified_event_count"],
            "unclassified_event_count": phase_stats["unclassified_event_count"],
            "classification_coverage": phase_stats["classification_coverage"],
        },
    }


def _safe_name(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "_", value).strip("._")
    return cleaned or "case"


def run_ai_assisted_pipeline(
    input_dir: str,
    output_dir: str,
    config_path: str,
    error_log: str | None = None,
    backend: Any | None = None,
    resume: bool = False,
    enriched_prompt_context: bool = False,
    run_issue_prompt: bool = True,
    run_trajectory_prompt: bool = True,
) -> dict[str, Any]:
    if not run_issue_prompt and not run_trajectory_prompt:
        raise ValueError("At least one prompt stage must be enabled")

    cases, discovery_errors = discover_cases(input_dir)
    if not cases:
        details = f" Discovery errors: {discovery_errors}" if discovery_errors else ""
        raise ValueError(f"No supported *.traj.json files found under {input_dir}.{details}")

    _log(f"Discovered {len(cases)} case(s) under {input_dir}")
    if discovery_errors:
        _log(f"{len(discovery_errors)} case(s) could not be loaded during discovery")

    output_root = Path(output_dir)
    output_root.mkdir(parents=True, exist_ok=True)
    if resume:
        _log("Resume mode: skipping cases that already have a completed report")

    # Create the backend lazily so a full resume with nothing to do does not
    # require model credentials.
    _backend = backend

    def get_backend() -> Any:
        nonlocal _backend
        if _backend is None:
            _backend = LLMBackend(config_path)
        return _backend

    summary_cases = []
    used_names = Counter()
    batch_started = time.monotonic()

    total = len(cases)
    for index, case in enumerate(cases, start=1):
        base_name = _safe_name(case["case_name"])
        used_names[base_name] += 1
        output_name = base_name if used_names[base_name] == 1 else f"{base_name}_{used_names[base_name]}"
        case_output = output_root / output_name
        case_output.mkdir(parents=True, exist_ok=True)

        if resume and (case_output / "analysis.json").is_file():
            _log(f"[{index}/{total}] Skipping {case['case_name']} (already completed)")
            summary_cases.append(
                {
                    "case_name": case["case_name"],
                    "case_id": case["case_id"],
                    "format": case["format"],
                    "resolved": case["result"].get("resolved"),
                    "status": "completed",
                    "output_directory": output_name,
                    "skipped": True,
                }
            )
            continue

        _log(f"[{index}/{total}] Analyzing {case['case_name']}")
        case_started = time.monotonic()
        try:
            artifact = analyze_case(
                case,
                get_backend(),
                enriched_prompt_context=enriched_prompt_context,
                run_issue_prompt=run_issue_prompt,
                run_trajectory_prompt=run_trajectory_prompt,
            )
            save_case_report(artifact, case_output)
            elapsed = time.monotonic() - case_started
            _log(
                f"[{index}/{total}] Completed {case['case_name']} in {elapsed:.1f}s "
                f"→ {output_name}/report.html"
            )
            summary_cases.append(
                {
                    "case_name": case["case_name"],
                    "case_id": case["case_id"],
                    "format": case["format"],
                    "resolved": case["result"].get("resolved"),
                    "status": "completed",
                    "output_directory": output_name,
                    "largest_phase": artifact["phase_statistics"].get("largest_phase_by_events"),
                    "event_count": len(case["events"]),
                }
            )
        except Exception as exc:
            elapsed = time.monotonic() - case_started
            message = f"Error processing {case['trajectory_path']}: {exc}"
            _log(f"[{index}/{total}] FAILED {case['case_name']} after {elapsed:.1f}s — {exc}")
            summary_cases.append(
                {
                    "case_name": case["case_name"],
                    "case_id": case["case_id"],
                    "format": case["format"],
                    "resolved": case["result"].get("resolved"),
                    "status": "failed",
                    "error": str(exc),
                    "output_directory": output_name,
                }
            )
            if error_log:
                with open(error_log, "a", encoding="utf-8") as handle:
                    handle.write(message + "\n\n")

    summary = {
        "schema_version": "ai-trajectory-analysis-batch-v1",
        "input_directory": str(Path(input_dir).resolve()),
        "cases": summary_cases,
        "discovery_errors": discovery_errors,
        "completed": sum(case["status"] == "completed" for case in summary_cases),
        "failed": sum(case["status"] == "failed" for case in summary_cases),
        "skipped": sum(case.get("skipped", False) for case in summary_cases),
    }
    with (output_root / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, ensure_ascii=False)
    save_batch_index(summary, output_root)
    _log(
        f"Batch done in {time.monotonic() - batch_started:.1f}s: "
        f"{summary['completed']} completed ({summary['skipped']} skipped), "
        f"{summary['failed']} failed → {output_root}/index.html"
    )
    return summary