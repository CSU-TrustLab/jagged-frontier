"""LLM-assisted trajectory triage for fast filtering before manual evaluation."""

from __future__ import annotations

import csv
import json
import re
import time
from collections import Counter
from pathlib import Path
from typing import Any

from llm import LLMBackend
from prompts.ai_assisted_prompt import TRAJECTORY_FILTER_PROMPT

from .checks import run_deterministic_checks
from .input import discover_cases
from .validator import locate_quote


MAX_PROMPT_CHARS = 55_000
DEFAULT_EVENTS_CHAR_BUDGET = 18_000


def _log(message: str) -> None:
    timestamp = time.strftime("%H:%M:%S")
    print(f"[{timestamp}] {message}", flush=True)


def _json_for_prompt(value: Any) -> str:
    return json.dumps(value, indent=2, ensure_ascii=False)


def _safe_name(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "_", value).strip("._")
    return cleaned or "case"


def _truncate_text(value: Any, limit: int) -> str:
    text = value if isinstance(value, str) else "" if value is None else str(value)
    if limit < 0:
        raise ValueError("limit must be non-negative")
    if len(text) <= limit:
        return text
    keep = max(0, limit - 18)
    return text[:keep] + "\n...[truncated]"


def _compact_events(events: list[dict[str, Any]], character_budget: int = 28_000) -> list[dict[str, Any]]:
    """Build a token-conscious event list for triage prompting."""
    compact = []
    if not events:
        return compact

    # Skip system and issue-description restatement content to preserve budget.
    candidate_events = [
        event for event in events
        if event.get("role") != "system" and not event.get("is_issue_description")
    ]
    if not candidate_events:
        candidate_events = list(events)

    per_event_budget = max(180, min(850, character_budget // max(1, len(candidate_events))))
    for event in candidate_events:
        content = event.get("content", "")
        if not isinstance(content, str):
            content = str(content)

        item = {
            "id": event.get("id"),
            "role": event.get("role"),
            "kind": event.get("kind"),
            "tool_name": event.get("tool_name"),
            "tool_status": event.get("tool_status"),
        }
        if len(content) <= per_event_budget:
            item["content"] = content
        else:
            head = int(per_event_budget * 0.65)
            item["content_start"] = content[:head]
            item["content_end"] = content[-(per_event_budget - head):]
            item["content_length"] = len(content)
        compact.append(item)
    return compact


def _count_eval_failures(result: dict[str, Any]) -> int:
    tests_status = result.get("tests_status") if isinstance(result.get("tests_status"), dict) else {}
    total = 0
    for group in tests_status.values():
        if isinstance(group, dict) and isinstance(group.get("failure"), list):
            total += len(group["failure"])
    return total


def _compact_case_metadata(case: dict[str, Any]) -> dict[str, Any]:
    result = case.get("result", {}) if isinstance(case.get("result"), dict) else {}
    spt = case.get("spt", {}) if isinstance(case.get("spt"), dict) else {}
    metadata = case.get("metadata", {}) if isinstance(case.get("metadata"), dict) else {}
    return {
        "case_id": case.get("case_id"),
        "case_name": case.get("case_name"),
        "format": case.get("format"),
        "trajectory_path": case.get("trajectory_path"),
        "issue_description": _truncate_text(case.get("issue_description", ""), 4000),
        "result": {
            "status": result.get("status"),
            "source_path": result.get("source_path"),
            "instance_id": result.get("instance_id"),
            "resolved": result.get("resolved"),
            "patch_successfully_applied": result.get("patch_successfully_applied"),
            "patch_exists": result.get("patch_exists"),
            "tests_status": result.get("tests_status"),
            "errors": result.get("errors", []),
        },
        "spt": {
            key: spt.get(key)
            for key in (
                "metadata_available",
                "source_path",
                "entry_count",
                "applied",
                "note",
                "error",
            )
        },
        "metadata": {
            key: metadata.get(key)
            for key in ("model", "provider", "cost", "api_calls", "tokens", "exit_status")
        },
    }


def _compact_deterministic_for_prompt(
    case: dict[str, Any], deterministic: dict[str, Any]
) -> dict[str, Any]:
    tests = deterministic.get("tests", {}) if isinstance(deterministic.get("tests"), dict) else {}
    patches = deterministic.get("patches", {}) if isinstance(deterministic.get("patches"), dict) else {}
    iterations = (
        deterministic.get("iterations", {})
        if isinstance(deterministic.get("iterations"), dict)
        else {}
    )
    errors = deterministic.get("errors", {}) if isinstance(deterministic.get("errors"), dict) else {}

    compact_test_events = []
    for item in tests.get("events", [])[:8]:
        if not isinstance(item, dict):
            continue
        compact_test_events.append(
            {
                "event_id": item.get("event_id"),
                "command": _truncate_text(item.get("command", ""), 140),
                "outcome": item.get("outcome"),
            }
        )

    compact_patch_events = []
    for item in patches.get("events", [])[:8]:
        if not isinstance(item, dict):
            continue
        compact_patch_events.append(
            {
                "event_id": item.get("event_id"),
                "files_modified": item.get("files_modified", [])[:20],
                "additions": item.get("additions"),
                "deletions": item.get("deletions"),
                "hunks": item.get("hunks"),
            }
        )

    compact_error_events = []
    for item in errors.get("events", [])[:8]:
        if not isinstance(item, dict):
            continue
        compact_error_events.append(
            {
                "event_id": item.get("event_id"),
                "matched": _truncate_text(item.get("matched", ""), 120),
                "snippet": _truncate_text(item.get("snippet", ""), 260),
                "tool_status": item.get("tool_status"),
            }
        )

    return {
        "features": _feature_payload(case, deterministic),
        "result_reconciliation": deterministic.get("result_reconciliation", {}),
        "tests": {
            "run_count": tests.get("run_count"),
            "failure_count": tests.get("failure_count"),
            "rerun_count": tests.get("rerun_count"),
            "evaluation_counts": tests.get("evaluation_counts"),
            "events": compact_test_events,
        },
        "patches": {
            "iteration_count": patches.get("iteration_count"),
            "events": compact_patch_events,
            "repeatedly_edited_files": patches.get("repeatedly_edited_files", {}),
            "submitted_patch": patches.get("submitted_patch"),
        },
        "iterations": {
            "test_edit_cycle_count": iterations.get("test_edit_cycle_count"),
            "test_edit_cycles": (iterations.get("test_edit_cycles") or [])[:10],
            "git_reversion_event_ids": (iterations.get("git_reversion_event_ids") or [])[:20],
            "repeated_file_edits": iterations.get("repeated_file_edits", {}),
            "backtracking_signal_count": iterations.get("backtracking_signal_count"),
        },
        "errors": {
            "count": errors.get("count"),
            "events": compact_error_events,
        },
    }


def _feature_payload(case: dict[str, Any], deterministic: dict[str, Any]) -> dict[str, Any]:
    result = case.get("result", {})
    patches = deterministic.get("patches", {})
    tests = deterministic.get("tests", {})
    iterations = deterministic.get("iterations", {})
    errors = deterministic.get("errors", {})
    resources = deterministic.get("resources", {})
    spt = deterministic.get("spt", {})

    patch_event_files = sorted(
        {
            file_path
            for event in patches.get("events", [])
            if isinstance(event, dict)
            for file_path in event.get("files_modified", [])
            if isinstance(file_path, str)
        }
    )

    return {
        "case_id": case.get("case_id"),
        "case_name": case.get("case_name"),
        "trajectory_format": case.get("format"),
        "event_count": deterministic.get("event_count"),
        "resolved": result.get("resolved"),
        "result_match_status": result.get("status"),
        "patch_successfully_applied": result.get("patch_successfully_applied"),
        "patch_exists": result.get("patch_exists"),
        "structured_eval_failure_count": _count_eval_failures(result),
        "test_runs": tests.get("run_count"),
        "test_failures_seen_in_trajectory": tests.get("failure_count"),
        "test_reruns": tests.get("rerun_count"),
        "patch_iterations": patches.get("iteration_count"),
        "patched_files_from_events": patch_event_files,
        "repeated_file_edits": iterations.get("repeated_file_edits", {}),
        "backtracking_signal_count": iterations.get("backtracking_signal_count"),
        "git_reversion_count": deterministic.get("git", {}).get("reversion_count"),
        "error_event_count": errors.get("count"),
        "unique_resource_count": resources.get("unique_resource_count"),
        "spt_metadata_available": spt.get("metadata_available"),
        "spt_entry_count": spt.get("entry_count"),
        "spt_applied": spt.get("applied"),
    }


def _validate_triage_response(response: dict[str, Any], compact_events: list[dict[str, Any]]) -> dict[str, Any]:
    if not isinstance(response, dict):
        raise ValueError("Triage response must be a JSON object")

    include = response.get("include")
    if not isinstance(include, bool):
        raise ValueError("include must be a boolean")

    priority = response.get("priority")
    if priority not in {"high", "medium", "low"}:
        raise ValueError("priority must be one of: high, medium, low")
    if include and priority == "low":
        raise ValueError("priority cannot be low when include is true")

    score = response.get("score")
    if not isinstance(score, int) or isinstance(score, bool) or not 0 <= score <= 100:
        raise ValueError("score must be an integer from 0 to 100")

    confidence = response.get("confidence")
    if not isinstance(confidence, (int, float)) or isinstance(confidence, bool) or not 0 <= confidence <= 1:
        raise ValueError("confidence must be a number from 0 to 1")

    tags = response.get("tags")
    if not isinstance(tags, list) or any(not isinstance(tag, str) or not tag.strip() for tag in tags):
        raise ValueError("tags must be a list of non-empty strings")

    rationale = response.get("rationale")
    if not isinstance(rationale, str) or not rationale.strip():
        raise ValueError("rationale must be a non-empty string")

    metadata_evidence = response.get("metadata_evidence")
    if not isinstance(metadata_evidence, list) or any(
        not isinstance(item, str) or not item.strip() for item in metadata_evidence
    ):
        raise ValueError("metadata_evidence must be a list of non-empty strings")

    event_map = {event.get("id"): event for event in compact_events}
    trajectory_evidence = response.get("trajectory_evidence")
    if not isinstance(trajectory_evidence, list) or not trajectory_evidence:
        raise ValueError("trajectory_evidence must be a non-empty list")

    for item in trajectory_evidence:
        if not isinstance(item, dict):
            raise ValueError("trajectory_evidence entries must be JSON objects")
        event_id = item.get("event_id")
        if event_id not in event_map:
            raise ValueError(f"trajectory_evidence references unknown event: {event_id}")
        quote = item.get("quote")
        if not isinstance(quote, str) or not quote:
            raise ValueError("trajectory_evidence quote must be non-empty")
        source = event_map[event_id].get("content")
        source_start = event_map[event_id].get("content_start")
        source_end = event_map[event_id].get("content_end")
        if isinstance(source, str) and quote in source:
            pass
        elif isinstance(source_start, str) and quote in source_start:
            pass
        elif isinstance(source_end, str) and quote in source_end:
            pass
        else:
            located = locate_quote(quote, source or "") if isinstance(source, str) else None
            if located is None:
                raise ValueError("trajectory_evidence quote must match an available event content span")
            start, end, _ = located
            item["quote"] = source[start:end]
        why = item.get("why")
        if not isinstance(why, str) or not why.strip():
            raise ValueError("trajectory_evidence why must be a non-empty string")

    return response


def _generate_validated_triage(
    backend: Any,
    prompt: str,
    compact_events: list[dict[str, Any]],
    attempts: int = 3,
) -> tuple[dict[str, Any], list[str]]:
    errors = []
    current_prompt = prompt
    for attempt in range(1, attempts + 1):
        _log(f"    filter model call: requesting triage (attempt {attempt}/{attempts})...")
        started = time.monotonic()
        response = backend.generate_json(current_prompt)
        try:
            validated = _validate_triage_response(response, compact_events)
            _log(f"    filter model call: valid response in {time.monotonic() - started:.1f}s")
            return validated, errors
        except Exception as exc:
            errors.append(str(exc))
            _log(
                f"    filter model call: invalid response ({time.monotonic() - started:.1f}s) - {exc}"
            )
            if attempt < attempts:
                current_prompt = (
                    prompt
                    + "\n\nYour previous response was invalid:\n"
                    + _json_for_prompt(response)
                    + "\n\nValidation error:\n"
                    + str(exc)
                    + "\nReturn a corrected complete JSON object only."
                )
    raise ValueError("Model output remained invalid after retries: " + " | ".join(errors))


def _build_prompt(
    case: dict[str, Any],
    deterministic: dict[str, Any],
    compact_events: list[dict[str, Any]],
) -> str:
    case_metadata = _compact_case_metadata(case)
    compact_signals = _compact_deterministic_for_prompt(case, deterministic)

    return (
        TRAJECTORY_FILTER_PROMPT.replace("{CASE_METADATA_JSON}", _json_for_prompt(case_metadata))
        .replace("{DETERMINISTIC_SIGNALS_JSON}", _json_for_prompt(compact_signals))
        .replace("{EVENTS_JSON}", _json_for_prompt(compact_events))
    )


def _build_prompt_with_budget(
    case: dict[str, Any], deterministic: dict[str, Any]
) -> tuple[str, list[dict[str, Any]], int]:
    """Build prompt with adaptive shrinking to stay under prompt character cap."""
    events = case.get("events", [])
    budgets = [
        DEFAULT_EVENTS_CHAR_BUDGET,
        12_000,
        9_000,
        6_000,
        4_500,
        3_000,
        2_000,
        1_200,
    ]
    selected_prompt = ""
    selected_events: list[dict[str, Any]] = []
    selected_budget = budgets[-1]
    for budget in budgets:
        compact_events = _compact_events(events, character_budget=budget)
        prompt = _build_prompt(case, deterministic, compact_events)
        selected_prompt = prompt
        selected_events = compact_events
        selected_budget = budget
        if len(prompt) <= MAX_PROMPT_CHARS:
            break
    return selected_prompt, selected_events, selected_budget


def _sort_key(case_summary: dict[str, Any]) -> tuple[int, int, float, int]:
    triage = case_summary.get("triage", {})
    include_boost = 1 if triage.get("include") else 0
    priority_order = {"high": 2, "medium": 1, "low": 0}
    priority_score = priority_order.get(triage.get("priority"), 0)
    confidence = float(triage.get("confidence", 0.0) or 0.0)
    llm_score = int(triage.get("score", 0) or 0)
    return (include_boost, priority_score, confidence, llm_score)


def _save_shortlist_csv(cases: list[dict[str, Any]], path: Path) -> None:
    fieldnames = [
        "rank",
        "case_name",
        "case_id",
        "status",
        "include",
        "priority",
        "score",
        "confidence",
        "tags",
        "rationale",
        "resolved",
        "event_count",
        "spt_entry_count",
        "test_runs",
        "test_failures_seen_in_trajectory",
        "patch_iterations",
        "backtracking_signal_count",
        "error_event_count",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for rank, case in enumerate(cases, start=1):
            triage = case.get("triage", {})
            features = case.get("features", {})
            writer.writerow(
                {
                    "rank": rank,
                    "case_name": case.get("case_name"),
                    "case_id": case.get("case_id"),
                    "status": case.get("status"),
                    "include": triage.get("include"),
                    "priority": triage.get("priority"),
                    "score": triage.get("score"),
                    "confidence": triage.get("confidence"),
                    "tags": " | ".join(triage.get("tags", [])),
                    "rationale": triage.get("rationale"),
                    "resolved": features.get("resolved"),
                    "event_count": features.get("event_count"),
                    "spt_entry_count": features.get("spt_entry_count"),
                    "test_runs": features.get("test_runs"),
                    "test_failures_seen_in_trajectory": features.get("test_failures_seen_in_trajectory"),
                    "patch_iterations": features.get("patch_iterations"),
                    "backtracking_signal_count": features.get("backtracking_signal_count"),
                    "error_event_count": features.get("error_event_count"),
                }
            )


def run_ai_filter_pipeline(
    input_dir: str,
    output_dir: str,
    config_path: str,
    error_log: str | None = None,
    backend: Any | None = None,
    resume: bool = False,
) -> dict[str, Any]:
    """Run lightweight LLM triage and emit a ranked shortlist for evaluation."""
    cases, discovery_errors = discover_cases(input_dir)
    if not cases:
        details = f" Discovery errors: {discovery_errors}" if discovery_errors else ""
        raise ValueError(f"No supported *.traj.json files found under {input_dir}.{details}")

    _log(f"Discovered {len(cases)} case(s) under {input_dir} for AI filtering")
    if discovery_errors:
        _log(f"{len(discovery_errors)} case(s) could not be loaded during discovery")

    output_root = Path(output_dir)
    output_root.mkdir(parents=True, exist_ok=True)
    case_root = output_root / "filter_cases"
    case_root.mkdir(parents=True, exist_ok=True)

    _backend = backend

    def get_backend() -> Any:
        nonlocal _backend
        if _backend is None:
            _backend = LLMBackend(config_path)
        return _backend

    used_names = Counter()
    summary_cases = []
    total = len(cases)
    batch_started = time.monotonic()

    for index, case in enumerate(cases, start=1):
        base_name = _safe_name(case["case_name"])
        used_names[base_name] += 1
        output_name = base_name if used_names[base_name] == 1 else f"{base_name}_{used_names[base_name]}"
        output_dir_path = case_root / output_name
        output_dir_path.mkdir(parents=True, exist_ok=True)
        triage_path = output_dir_path / "triage.json"

        if resume and triage_path.is_file():
            _log(f"[{index}/{total}] Skipping {case['case_name']} (triage.json already exists)")
            existing = json.loads(triage_path.read_text(encoding="utf-8"))
            summary_cases.append(existing)
            summary_cases[-1]["skipped"] = True
            continue

        _log(f"[{index}/{total}] Filtering {case['case_name']}")
        started = time.monotonic()
        try:
            deterministic = run_deterministic_checks(case)
            prompt, compact_events, used_budget = _build_prompt_with_budget(case, deterministic)
            _log(
                f"    prompt size: {len(prompt)} chars (event budget {used_budget}, "
                f"{len(compact_events)} compact event(s))"
            )
            triage, retry_errors = _generate_validated_triage(get_backend(), prompt, compact_events)

            case_summary = {
                "case_name": case.get("case_name"),
                "case_id": case.get("case_id"),
                "format": case.get("format"),
                "status": "completed",
                "output_directory": output_name,
                "triage": triage,
                "features": _feature_payload(case, deterministic),
                "data_quality": {
                    "result_match_status": case.get("result", {}).get("status"),
                    "result_match_errors": case.get("result", {}).get("errors", []),
                    "triage_retry_errors": retry_errors,
                },
            }
            triage_path.write_text(json.dumps(case_summary, indent=2, ensure_ascii=False), encoding="utf-8")
            summary_cases.append(case_summary)
            _log(f"[{index}/{total}] Completed {case['case_name']} in {time.monotonic() - started:.1f}s")
        except Exception as exc:
            message = f"Error processing {case['trajectory_path']}: {exc}"
            _log(f"[{index}/{total}] FAILED {case['case_name']} after {time.monotonic() - started:.1f}s - {exc}")
            failure = {
                "case_name": case.get("case_name"),
                "case_id": case.get("case_id"),
                "format": case.get("format"),
                "status": "failed",
                "error": str(exc),
                "output_directory": output_name,
            }
            (output_dir_path / "triage.json").write_text(
                json.dumps(failure, indent=2, ensure_ascii=False), encoding="utf-8"
            )
            summary_cases.append(failure)
            if error_log:
                with open(error_log, "a", encoding="utf-8") as handle:
                    handle.write(message + "\n\n")

    completed_cases = [case for case in summary_cases if case.get("status") == "completed"]
    ranked = sorted(completed_cases, key=_sort_key, reverse=True)
    shortlist = [case for case in ranked if case.get("triage", {}).get("include")]

    summary = {
        "schema_version": "ai-trajectory-filter-v1",
        "input_directory": str(Path(input_dir).resolve()),
        "cases": ranked + [case for case in summary_cases if case.get("status") != "completed"],
        "discovery_errors": discovery_errors,
        "completed": len(completed_cases),
        "failed": sum(case.get("status") == "failed" for case in summary_cases),
        "skipped": sum(bool(case.get("skipped")) for case in summary_cases),
        "selected": len(shortlist),
        "selected_case_ids": [case.get("case_id") for case in shortlist],
        "run_seconds": round(time.monotonic() - batch_started, 2),
    }

    summary_path = output_root / "filter_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    _save_shortlist_csv(ranked, output_root / "filter_shortlist.csv")

    _log(
        f"Filter batch done in {summary['run_seconds']:.1f}s: "
        f"{summary['completed']} completed, {summary['failed']} failed, {summary['selected']} selected"
    )
    return summary
