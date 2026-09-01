"""Deterministic checks used by the AI-assisted analysis mode."""

from __future__ import annotations

import re
import shlex
from collections import Counter, defaultdict
from typing import Any

from utils.diff import summarize_diff


TEST_COMMAND_RE = re.compile(
    r"(?:^|\s)(?:python\s+-m\s+)?pytest\b|"
    r"(?:^|\s)python\s+-m\s+unittest\b|"
    r"(?:^|\s)(?:npm|pnpm|yarn)\s+(?:run\s+)?test\b|"
    r"(?:^|\s)go\s+test\b|(?:^|\s)cargo\s+test\b|(?:^|\s)tox\b",
    re.IGNORECASE,
)
FILE_TOKEN_RE = re.compile(
    r"(?<![\w:/.-])(?:[A-Za-z0-9_.@+-]+/)+[A-Za-z0-9_.@+-]+|"
    r"(?<![\w.-])[A-Za-z0-9_@+-]+\.(?:py|js|jsx|ts|tsx|go|rs|java|rb|php|cpp|c|h|hpp|md|yaml|yml|toml|json)(?![\w.-])"
)
ERROR_RE = re.compile(
    r"Traceback \(most recent call last\)|\b[A-Za-z_][A-Za-z0-9_]*(?:Error|Exception):|"
    r"\bcommand failed\b|\btool error\b|\bERRORS?\b|\bFAILURES?\b",
    re.IGNORECASE,
)


def _event_index(event_id: str) -> int:
    try:
        return int(event_id[1:])
    except (TypeError, ValueError):
        return -1


def _commands_from_event(event: dict[str, Any]) -> list[str]:
    if event.get("role") not in {"assistant", "tool"} and event.get("kind") != "tool":
        return []
    commands = []
    tool_input = event.get("tool_input")
    if isinstance(tool_input, dict):
        command = tool_input.get("command")
        if isinstance(command, str) and command.strip():
            commands.append(command.strip())
    elif isinstance(tool_input, str) and tool_input.strip():
        commands.append(tool_input.strip())

    content = event.get("content", "")
    if isinstance(content, str):
        for match in re.finditer(r"```(?:bash|sh|shell)?\s*\n(.*?)```", content, re.DOTALL | re.IGNORECASE):
            command = match.group(1).strip()
            if command:
                commands.append(command)

    unique = []
    seen = set()
    for command in commands:
        if command not in seen:
            seen.add(command)
            unique.append(command)
    return unique


def _command_fragments(command: str) -> list[str]:
    return [
        fragment.strip()
        for fragment in re.split(r"(?:&&|\|\||;|\n)", command)
        if fragment.strip()
    ]


def _git_invocations(command: str) -> list[str]:
    invocations = []
    for fragment in _command_fragments(command):
        match = re.search(r"\bgit\s+.+", fragment)
        if match:
            invocations.append(match.group(0).strip())
    return invocations


def _classify_git_command(command: str) -> tuple[str, bool, str | None]:
    try:
        parts = shlex.split(command)
    except ValueError:
        parts = command.split()
    subcommand = "unknown"
    index = 1
    options_with_values = {"-C", "-c", "--git-dir", "--work-tree", "--namespace"}
    while index < len(parts):
        part = parts[index]
        if part in options_with_values:
            index += 2
            continue
        if part.startswith("--git-dir=") or part.startswith("--work-tree="):
            index += 1
            continue
        if part.startswith("-"):
            index += 1
            continue
        subcommand = part
        break
    lowered = command.lower()

    if subcommand in {"status", "diff", "log", "show", "branch", "blame", "rev-parse"}:
        return "inspection", False, subcommand
    if subcommand in {"add", "commit"}:
        return "recording", False, subcommand
    if subcommand == "revert":
        return "reversion", True, subcommand
    if subcommand == "restore":
        is_reversion = "--staged" not in lowered or "--worktree" in lowered
        return "reversion" if is_reversion else "index_change", is_reversion, subcommand
    if subcommand == "reset":
        is_reversion = "--hard" in lowered or bool(
            re.search(r"\b(?:head(?:[~^]\d*)?|[0-9a-f]{4,64})\b", lowered)
        )
        return "reversion" if is_reversion else "index_change", is_reversion, subcommand
    if subcommand == "checkout":
        is_reversion = " -- " in f" {lowered} " or bool(
            re.search(r"\b(?:head(?:[~^]\d*)?|[0-9a-f]{4,64})\b", lowered)
        )
        return "reversion" if is_reversion else "state_change", is_reversion, subcommand
    if subcommand == "clean":
        return "destructive_cleanup", False, subcommand
    return "state_change", False, subcommand


def analyze_git(events: list[dict[str, Any]]) -> dict[str, Any]:
    git_events = []
    for event in events:
        for command in _commands_from_event(event):
            for invocation in _git_invocations(command):
                category, is_reversion, subcommand = _classify_git_command(invocation)
                git_events.append(
                    {
                        "event_id": event["id"],
                        "command": invocation,
                        "subcommand": subcommand,
                        "category": category,
                        "is_reversion": is_reversion,
                    }
                )
    reversions = [event for event in git_events if event["is_reversion"]]
    return {
        "commands": git_events,
        "command_count": len(git_events),
        "reversions": reversions,
        "reversion_count": len(reversions),
    }


def grep_issue_keywords(
    events: list[dict[str, Any]],
    clues: list[dict[str, Any]],
    issue_event_id: str | None,
) -> list[dict[str, Any]]:
    """Grep each extracted issue keyword across trajectory events."""
    results = []
    for clue in clues:
        keyword = clue.get("quote")
        if not isinstance(keyword, str) or len(keyword.strip()) < 3:
            continue
        keyword = keyword.strip()
        matches = []
        for event in events:
            if event["id"] == issue_event_id or event.get("role") == "system":
                continue
            content = event.get("content", "")
            if isinstance(content, str) and keyword.lower() in content.lower():
                matches.append(event["id"])
        results.append(
            {
                "keyword": keyword,
                "category": clue.get("category"),
                "match_count": len(matches),
                "event_ids": matches,
            }
        )
    return results


def _clean_resource(resource: str) -> str | None:
    resource = resource.strip("'\"`()[]{}:,;<> ")
    resource = re.sub(r"^(?:a|b)/", "", resource)
    if not resource or resource.startswith(("http://", "https://")):
        return None
    if resource in {"dev/null", "usr/bin/env"}:
        return None
    return resource


def _resources_from_text(text: str) -> list[str]:
    resources = []
    for match in FILE_TOKEN_RE.finditer(text or ""):
        resource = _clean_resource(match.group(0))
        if resource and resource not in resources:
            resources.append(resource)
    return resources


def _command_kind(command: str) -> str:
    lowered = command.strip().lower()
    if re.search(r"\b(?:apply_patch|patch)\b|\bsed\s+-i\b|\b(?:tee|cat)\b[^\n]*>", lowered):
        return "edit"
    if re.search(r"(?:^|\s)(?:rg|grep|find|fd)\b", lowered):
        return "search"
    if re.search(r"(?:^|\s)(?:cat|sed|head|tail|less|more|bat)\b", lowered):
        return "read"
    if TEST_COMMAND_RE.search(command):
        return "test"
    return "other"


def analyze_resources(
    events: list[dict[str, Any]], target_resources: list[str] | None = None
) -> dict[str, Any]:
    accesses = []
    by_kind: dict[str, set[str]] = defaultdict(set)
    first_seen: dict[str, str] = {}
    seen_accesses = set()

    def add_access(event_id: str, resource: str, kind: str) -> None:
        key = (event_id, resource, kind)
        if key in seen_accesses:
            return
        seen_accesses.add(key)
        accesses.append({"event_id": event_id, "resource": resource, "kind": kind})
        by_kind[kind].add(resource)
        first_seen.setdefault(resource, event_id)

    for event in events:
        commands = _commands_from_event(event)
        for command in commands:
            kind = _command_kind(command)
            for resource in _resources_from_text(command):
                add_access(event["id"], resource, kind)

        content = event.get("content", "")
        if "diff --git" in content or "*** Update File:" in content:
            for resource in _resources_from_text(content):
                add_access(event["id"], resource, "edit")

    all_resources = sorted(first_seen, key=lambda resource: _event_index(first_seen[resource]))
    counts = Counter(access["resource"] for access in accesses)
    cumulative = []
    cumulative_seen = set()
    for event in events:
        new_resources = []
        for access in accesses:
            if access["event_id"] != event["id"] or access["resource"] in cumulative_seen:
                continue
            cumulative_seen.add(access["resource"])
            new_resources.append(access["resource"])
        if new_resources:
            cumulative.append(
                {
                    "event_id": event["id"],
                    "new_resources": new_resources,
                    "cumulative_unique_count": len(cumulative_seen),
                }
            )

    normalized_targets = {
        cleaned
        for target in target_resources or []
        if (cleaned := _clean_resource(target)) is not None
    }
    target_hits = [
        access
        for access in accesses
        if access["resource"] in normalized_targets
        or any(access["resource"].endswith(f"/{target}") for target in normalized_targets)
    ]
    first_target_event = target_hits[0]["event_id"] if target_hits else None
    return {
        "accesses": accesses,
        "unique_resources": all_resources,
        "unique_resource_count": len(all_resources),
        "resources_by_kind": {kind: sorted(resources) for kind, resources in sorted(by_kind.items())},
        "revisited_resources": sorted(resource for resource, count in counts.items() if count > 1),
        "cumulative_unique_resources": cumulative,
        "target_resources": sorted(normalized_targets),
        "target_hits": target_hits,
        "first_target_event": first_target_event,
        "events_before_first_target": _event_index(first_target_event) if first_target_event else None,
        "coverage_note": "Counts observed trajectory resources; repository-wide coverage requires a repository inventory.",
    }


def analyze_patches(events: list[dict[str, Any]], submission: str | None) -> dict[str, Any]:
    patch_events = []
    files_to_events: dict[str, list[str]] = defaultdict(list)
    for event in events:
        content = event.get("content", "")
        if not isinstance(content, str):
            continue
        if "diff --git" not in content and "apply_patch" not in content and "*** Update File:" not in content:
            continue
        summary = summarize_diff(content)
        if not summary["files_modified"]:
            summary["files_modified"] = [
                match.group(1).strip()
                for match in re.finditer(r"\*\*\* (?:Update|Add|Delete) File:\s*(.+)", content)
            ]
        patch_event = {"event_id": event["id"], **summary}
        patch_events.append(patch_event)
        for path in summary["files_modified"]:
            files_to_events[path].append(event["id"])

    submitted_summary = summarize_diff(submission) if submission else None
    return {
        "events": patch_events,
        "iteration_count": len(patch_events),
        "repeatedly_edited_files": {
            path: event_ids for path, event_ids in files_to_events.items() if len(event_ids) > 1
        },
        "submitted_patch": submitted_summary,
    }


def _test_outcome(text: str) -> str:
    if re.search(r"\b(?:FAILED|FAILURES?)\b|\b[1-9]\d* failed\b|AssertionError", text, re.IGNORECASE):
        return "failed"
    if re.search(r"\b(?:[1-9]\d*\s+)?passed\b|\bPASS(?:ED)?\b|\btests?\s+ok\b", text, re.IGNORECASE):
        return "passed"
    if re.search(r"\bERRORS?\b|Traceback", text, re.IGNORECASE):
        return "error"
    return "unknown"


def analyze_tests(events: list[dict[str, Any]], result: dict[str, Any]) -> dict[str, Any]:
    test_events = []
    for index, event in enumerate(events):
        for command in _commands_from_event(event):
            if not TEST_COMMAND_RE.search(command):
                continue
            evidence_text = event.get("content", "")
            if index + 1 < len(events):
                evidence_text += "\n" + str(events[index + 1].get("content", ""))
            test_events.append(
                {
                    "event_id": event["id"],
                    "command": command,
                    "outcome": _test_outcome(evidence_text),
                }
            )

    tests_status = result.get("tests_status") if isinstance(result.get("tests_status"), dict) else {}
    result_counts = {}
    for group_name, group in tests_status.items():
        if isinstance(group, dict):
            result_counts[group_name] = {
                "success": len(group.get("success", [])) if isinstance(group.get("success"), list) else 0,
                "failure": len(group.get("failure", [])) if isinstance(group.get("failure"), list) else 0,
            }

    return {
        "events": test_events,
        "run_count": len(test_events),
        "failure_count": sum(event["outcome"] in {"failed", "error"} for event in test_events),
        "rerun_count": max(0, len(test_events) - len({event["command"] for event in test_events})),
        "evaluation_counts": result_counts,
    }


def analyze_iterations(
    events: list[dict[str, Any]],
    git: dict[str, Any],
    resources: dict[str, Any],
    patches: dict[str, Any],
    tests: dict[str, Any],
) -> dict[str, Any]:
    edit_event_ids = {
        access["event_id"]
        for access in resources.get("accesses", [])
        if access.get("kind") == "edit"
    }
    edit_event_ids.update(event["event_id"] for event in patches.get("events", []))
    ordered_edit_ids = sorted(edit_event_ids, key=_event_index)
    ordered_tests = sorted(tests.get("events", []), key=lambda item: _event_index(item["event_id"]))

    test_edit_cycles = []
    for test_index, test_event in enumerate(ordered_tests):
        if test_event.get("outcome") not in {"failed", "error"}:
            continue
        failed_index = _event_index(test_event["event_id"])
        next_test = next(
            (candidate for candidate in ordered_tests[test_index + 1:] if _event_index(candidate["event_id"]) > failed_index),
            None,
        )
        next_test_index = _event_index(next_test["event_id"]) if next_test else float("inf")
        edits = [
            event_id
            for event_id in ordered_edit_ids
            if failed_index < _event_index(event_id) < next_test_index
        ]
        test_edit_cycles.append(
            {
                "failed_test_event_id": test_event["event_id"],
                "edit_event_ids": edits,
                "retest_event_id": next_test.get("event_id") if next_test else None,
                "retest_outcome": next_test.get("outcome") if next_test else None,
            }
        )

    edit_counts = Counter(
        access["resource"]
        for access in resources.get("accesses", [])
        if access.get("kind") == "edit"
    )
    repeated_edits = {
        resource: count for resource, count in sorted(edit_counts.items()) if count > 1
    }
    return {
        "edit_event_ids": ordered_edit_ids,
        "test_edit_cycles": test_edit_cycles,
        "test_edit_cycle_count": len(test_edit_cycles),
        "git_reversion_event_ids": [
            event["event_id"] for event in git.get("reversions", [])
        ],
        "repeated_file_edits": repeated_edits,
        "backtracking_signal_count": (
            len(git.get("reversions", []))
            + len(test_edit_cycles)
            + len(repeated_edits)
        ),
        "note": "Signals identify observable backtracking; the reason for a revision remains an evidence-grounded semantic judgment.",
    }


def analyze_errors(events: list[dict[str, Any]], issue_event_id: str | None) -> dict[str, Any]:
    errors = []
    for event in events:
        if event["id"] == issue_event_id or event.get("role") == "system":
            continue
        content = event.get("content", "")
        match = ERROR_RE.search(content) if isinstance(content, str) else None
        if match:
            start = max(0, match.start() - 160)
            end = min(len(content), match.end() + 300)
            errors.append(
                {
                    "event_id": event["id"],
                    "matched": match.group(0),
                    "snippet": content[start:end],
                    "tool_status": event.get("tool_status"),
                }
            )
        elif event.get("tool_status") in {"error", "failed"}:
            errors.append(
                {
                    "event_id": event["id"],
                    "matched": f"tool status: {event.get('tool_status')}",
                    "snippet": content[:460],
                    "tool_status": event.get("tool_status"),
                }
            )
    return {"events": errors, "count": len(errors)}


def reconcile_result(case: dict[str, Any], tests: dict[str, Any]) -> dict[str, Any]:
    result = case.get("result", {})
    resolved = result.get("resolved")
    exit_status = (case.get("metadata") or {}).get("exit_status")
    submitted = bool(case.get("submission"))
    evaluation_failures = sum(
        group.get("failure", 0)
        for group in tests.get("evaluation_counts", {}).values()
        if isinstance(group, dict)
    )
    contradictions = []
    if resolved is True and result.get("patch_successfully_applied") is False:
        contradictions.append("Evaluator reports resolved although patch application is false.")
    if resolved is True and evaluation_failures:
        contradictions.append("Evaluator reports resolved although structured test failures remain.")
    if resolved is True and isinstance(exit_status, str) and exit_status.lower() in {"failed", "error"}:
        contradictions.append("Trajectory exit status reports failure although evaluator reports resolved.")
    if resolved is False and not submitted:
        likely_outcome_context = "No submitted patch was found."
    elif resolved is False:
        likely_outcome_context = "A patch was submitted but the evaluator did not resolve the case."
    elif resolved is True:
        likely_outcome_context = "The evaluator reports the case as resolved."
    else:
        likely_outcome_context = "No definitive evaluator outcome was matched."
    return {
        "outcome": "resolved" if resolved is True else "unresolved" if resolved is False else "unknown",
        "exit_status": exit_status,
        "submitted_patch_available": submitted,
        "patch_successfully_applied": result.get("patch_successfully_applied"),
        "structured_evaluation_failure_count": evaluation_failures,
        "summary": likely_outcome_context,
        "contradictions": contradictions,
        "consistent": not contradictions,
    }


def run_deterministic_checks(case: dict[str, Any]) -> dict[str, Any]:
    events = case["events"]
    timestamps = [event["timestamp"] for event in events if event.get("timestamp") is not None]
    elapsed = max(timestamps) - min(timestamps) if len(timestamps) >= 2 else None
    patches = analyze_patches(events, case.get("submission"))
    submitted_patch = patches.get("submitted_patch") or {}
    resources = analyze_resources(events, submitted_patch.get("files_modified", []))
    git = analyze_git(events)
    tests = analyze_tests(events, case.get("result", {}))
    checks = {
        "event_count": len(events),
        "elapsed_seconds": elapsed,
        "git": git,
        "resources": resources,
        "patches": patches,
        "tests": tests,
        "iterations": analyze_iterations(events, git, resources, patches, tests),
        "errors": analyze_errors(events, case.get("issue_event_id")),
        "result_reconciliation": reconcile_result(case, tests),
        "result": {
            key: case.get("result", {}).get(key)
            for key in (
                "status",
                "source_path",
                "instance_id",
                "resolved",
                "tests_status",
                "patch_successfully_applied",
                "patch_exists",
                "errors",
            )
            if key in case.get("result", {})
        },
        "spt": {
            key: case.get("spt", {}).get(key)
            for key in (
                "metadata_available", "source_path", "entry_count", "applied", "note", "error"
            )
            if key in case.get("spt", {})
        },
    }
    return checks