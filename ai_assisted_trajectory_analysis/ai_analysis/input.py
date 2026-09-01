"""Input discovery and normalization for supported trajectory formats."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any


RESULT_BASENAMES = {
    "result.json",
    "results.json",
    "report.json",
    "eval_results.json",
    "eval_results_detailed.json",
}


def _read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _is_result_candidate(path: Path) -> bool:
    name = path.name.lower()
    return (
        path.is_file()
        and not name.endswith(".traj.json")
        and (name in RESULT_BASENAMES or name.endswith("_output.json"))
    )


def _result_candidates(root: Path) -> list[Path]:
    return sorted(path for path in root.rglob("*.json") if _is_result_candidate(path))


def _find_case_root(trajectory_path: Path, input_root: Path) -> tuple[Path, list[Path]]:
    """Find the closest ancestor that contains a recognizable result artifact."""
    current = trajectory_path.parent
    last_single_case_root = current
    while True:
        # Reaching a dataset root from a nested case must not expose result files
        # from sibling cases. A root may still be a case when the trajectory is
        # stored directly in it.
        if current == input_root and trajectory_path.parent != input_root:
            return last_single_case_root, []
        trajectories = list(current.rglob("*.traj.json"))
        if len(trajectories) > 1:
            return last_single_case_root, []
        if trajectories:
            last_single_case_root = current
        candidates = _result_candidates(current)
        if candidates:
            return current, candidates
        if current == input_root:
            return last_single_case_root, []
        if input_root not in current.parents:
            return last_single_case_root, []
        current = current.parent


def detect_format(data: dict[str, Any]) -> str:
    if str(data.get("trajectory_format", "")).startswith("mini-swe-agent"):
        return "mini-swe-agent"

    messages = data.get("messages", [])
    if any(isinstance(message, dict) and "parts" in message for message in messages):
        return "opencode"
    if any(isinstance(message, dict) and "content" in message for message in messages):
        return "mini-swe-agent"
    raise ValueError("Unsupported trajectory format: expected mini-swe-agent or OpenCode messages")


def _normalize_timestamp(value: Any) -> float | None:
    if not isinstance(value, (int, float)):
        return None
    timestamp = float(value)
    if timestamp > 100_000_000_000:
        timestamp /= 1000.0
    return timestamp


def _json_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    return json.dumps(value, ensure_ascii=False, sort_keys=True)


def _tool_content(tool_name: str, tool_input: Any, tool_output: Any) -> str:
    sections = [f"TOOL: {tool_name or 'unknown'}"]
    if tool_input not in (None, "", {}):
        sections.append(f"INPUT:\n{_json_text(tool_input)}")
    if tool_output not in (None, ""):
        sections.append(f"OUTPUT:\n{_json_text(tool_output)}")
    return "\n".join(sections)


def _normalize_mini_events(messages: list[Any]) -> list[dict[str, Any]]:
    events = []
    for message_index, message in enumerate(messages):
        if not isinstance(message, dict):
            continue
        content = message.get("content", "")
        if not isinstance(content, str):
            content = _json_text(content)
        events.append(
            {
                "id": f"E{len(events)}",
                "message_index": message_index,
                "part_index": None,
                "role": message.get("role", "unknown"),
                "kind": "message",
                "timestamp": _normalize_timestamp(message.get("timestamp")),
                "duration_seconds": None,
                "tool_name": None,
                "tool_status": None,
                "tool_input": None,
                "tool_output": None,
                "content": content,
            }
        )
    for index, event in enumerate(events[:-1]):
        current = event.get("timestamp")
        following = events[index + 1].get("timestamp")
        if current is not None and following is not None and following >= current:
            event["duration_seconds"] = following - current
    return events


def _normalize_opencode_events(messages: list[Any]) -> list[dict[str, Any]]:
    events = []
    for message_index, message in enumerate(messages):
        if not isinstance(message, dict):
            continue
        info = message.get("info") if isinstance(message.get("info"), dict) else {}
        time_info = info.get("time") if isinstance(info.get("time"), dict) else {}
        created = _normalize_timestamp(time_info.get("created"))
        completed = _normalize_timestamp(time_info.get("completed"))
        duration = None
        if created is not None and completed is not None and completed >= created:
            duration = completed - created

        parts = message.get("parts") if isinstance(message.get("parts"), list) else []
        valid_part_count = sum(isinstance(part, dict) for part in parts) or 1
        for part_index, part in enumerate(parts):
            if not isinstance(part, dict):
                continue
            part_type = part.get("type", "unknown")
            tool_name = None
            tool_status = None
            tool_input = None
            tool_output = None

            if part_type == "tool":
                state = part.get("state") if isinstance(part.get("state"), dict) else {}
                tool_name = str(part.get("tool", "unknown"))
                tool_status = state.get("status")
                tool_input = state.get("input")
                tool_output = state.get("output")
                content = _tool_content(tool_name, tool_input, tool_output)
            else:
                content = part.get("text", "")
                if not isinstance(content, str):
                    content = _json_text(content)

            events.append(
                {
                    "id": f"E{len(events)}",
                    "message_index": message_index,
                    "part_index": part_index,
                    "role": info.get("role", "unknown"),
                    "kind": part_type,
                    "timestamp": created,
                    "duration_seconds": duration / valid_part_count if duration is not None else None,
                    "duration_is_estimated": duration is not None and valid_part_count > 1,
                    "tool_name": tool_name,
                    "tool_status": tool_status,
                    "tool_input": tool_input,
                    "tool_output": tool_output,
                    "content": content,
                }
            )
    return events


def _extract_issue_description(events: list[dict[str, Any]]) -> tuple[str, str | None]:
    for event in events:
        if event.get("role") != "user" or not event.get("content"):
            continue
        content = event["content"]
        tagged = re.search(
            r"<(?:issue|pr_description)>\s*(.*?)\s*</(?:issue|pr_description)>",
            content,
            flags=re.IGNORECASE | re.DOTALL,
        )
        event["is_issue_description"] = True
        return (tagged.group(1).strip() if tagged else content.strip(), event["id"])
    return "", None


def _clean_issue_description(raw: Any) -> str:
    """Decode an ``info.issue_description`` value into plain text.

    Some exports store the issue as a loosely JSON-escaped string (leading quote,
    ``\\n`` newlines, ``\\"`` quotes). Prefer a clean JSON decode; fall back to a
    conservative unescape when the value is not valid JSON.
    """
    if not isinstance(raw, str):
        return ""
    text = raw.strip()
    if not text:
        return ""
    try:
        decoded = json.loads(text)
        if isinstance(decoded, str):
            return decoded.strip()
    except (json.JSONDecodeError, ValueError):
        pass
    if len(text) >= 2 and text[0] == '"' and text[-1] == '"':
        text = text[1:-1]
    text = (
        text.replace("\\r\\n", "\n")
        .replace("\\n", "\n")
        .replace("\\t", "\t")
        .replace('\\"', '"')
        .replace("\\\\", "\\")
    )
    return text.strip()


def _normalize_identifier(value: Any) -> str:
    identifier = str(value or "").strip().lower()
    identifier = re.sub(r"^instance_", "", identifier)
    identifier = re.sub(r"_(?:miniswe|opencode)(?:-sample)?$", "", identifier)
    return identifier


def _coerce_resolved(value: Any) -> bool | None:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)) and value in (0, 1):
        return bool(value)
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"resolved", "passed", "pass", "success", "true"}:
            return True
        if lowered in {"unresolved", "failed", "fail", "failure", "false"}:
            return False
    return None


def _records_from_result(data: Any) -> list[tuple[str | None, dict[str, Any]]]:
    if not isinstance(data, dict):
        return []

    if any(key in data for key in ("resolved", "tests_status", "patch_is_None")):
        return [(data.get("instance_id"), data)]

    records = []
    for key, value in data.items():
        if isinstance(value, dict):
            record = dict(value)
            record.setdefault("instance_id", key)
        else:
            record = {"instance_id": key, "resolved": _coerce_resolved(value)}
        records.append((key, record))
    return records


def _record_quality(record: dict[str, Any]) -> int:
    score = 0
    if _coerce_resolved(record.get("resolved")) is not None:
        score += 10
    if isinstance(record.get("tests_status"), dict):
        score += 30
    if "patch_successfully_applied" in record:
        score += 5
    return score


def _select_result(
    candidates: list[Path], trajectory_data: dict[str, Any], case_root: Path
) -> dict[str, Any]:
    trajectory_id = trajectory_data.get("instance_id")
    trajectory_id_normalized = _normalize_identifier(trajectory_id)
    path_hint = _normalize_identifier(case_root.name)
    options = []
    errors = []

    for path in candidates:
        try:
            records = _records_from_result(_read_json(path))
        except (OSError, json.JSONDecodeError) as exc:
            errors.append(f"{path}: {exc}")
            continue

        for record_id, record in records:
            normalized_record_id = _normalize_identifier(record_id or record.get("instance_id"))
            score = _record_quality(record)
            if trajectory_id_normalized and normalized_record_id == trajectory_id_normalized:
                score += 100
            elif normalized_record_id and normalized_record_id in path_hint:
                score += 60
            elif len(records) == 1:
                score += 20
            if path.name.lower() in {"result.json", "results.json"}:
                score += 2
            options.append((score, str(path), path, record_id, record))

    if not options:
        return {
            "status": "missing",
            "source_path": None,
            "candidate_paths": [str(path) for path in candidates],
            "instance_id": trajectory_id,
            "resolved": None,
            "tests_status": {},
            "errors": errors,
        }

    options.sort(key=lambda option: (-option[0], option[1]))
    best = options[0]
    if len(options) > 1 and options[1][0] == best[0] and options[1][3] != best[3]:
        return {
            "status": "ambiguous",
            "source_path": None,
            "candidate_paths": sorted({option[1] for option in options}),
            "instance_id": trajectory_id,
            "resolved": None,
            "tests_status": {},
            "errors": [f"Equally ranked result records: {best[1]} and {options[1][1]}"],
        }

    _, _, path, record_id, record = best
    return {
        "status": "matched",
        "source_path": str(path),
        "candidate_paths": sorted({option[1] for option in options}),
        "instance_id": record.get("instance_id") or record_id or trajectory_id,
        "resolved": _coerce_resolved(record.get("resolved")),
        "tests_status": record.get("tests_status", {}),
        "patch_successfully_applied": record.get("patch_successfully_applied"),
        "patch_exists": record.get("patch_exists"),
        "raw": record,
        "errors": errors,
    }


def _find_artifact(case_root: Path, name: str) -> Path | None:
    matches = sorted(path for path in case_root.rglob(name) if path.is_file())
    return matches[0] if matches else None


def _load_spt(case_root: Path) -> dict[str, Any]:
    path = _find_artifact(case_root, "spt_log.json")
    if path is None:
        return {
            "metadata_available": False,
            "source_path": None,
            "entry_count": None,
            "applied": None,
            "data": None,
        }
    try:
        data = _read_json(path)
        if isinstance(data, (list, dict)):
            entry_count = len(data)
            applied = True if entry_count > 0 else None
        else:
            entry_count = None
            applied = None
        return {
            "metadata_available": True,
            "source_path": str(path),
            "entry_count": entry_count,
            "applied": applied,
            "data": data,
            "note": (
                "SPT entries were recorded."
                if applied is True
                else "The SPT log is empty; absence of an applied perturbation is not proven."
            ),
        }
    except (OSError, json.JSONDecodeError) as exc:
        return {
            "metadata_available": False,
            "source_path": str(path),
            "entry_count": None,
            "applied": None,
            "data": None,
            "error": str(exc),
        }


def _extract_model_metadata(data: dict[str, Any], trajectory_format: str) -> dict[str, Any]:
    info = data.get("info") if isinstance(data.get("info"), dict) else {}
    if trajectory_format == "mini-swe-agent":
        config = info.get("config") if isinstance(info.get("config"), dict) else {}
        model_config = config.get("model") if isinstance(config.get("model"), dict) else {}
        stats = info.get("model_stats") if isinstance(info.get("model_stats"), dict) else {}
        return {
            "model": model_config.get("model_name"),
            "cost": stats.get("instance_cost"),
            "api_calls": stats.get("api_calls"),
            "tokens": None,
            "exit_status": info.get("exit_status"),
        }

    model = info.get("model") if isinstance(info.get("model"), dict) else {}
    return {
        "model": model.get("id") or model.get("modelID"),
        "provider": model.get("providerID"),
        "cost": info.get("cost"),
        "api_calls": None,
        "tokens": info.get("tokens"),
        "exit_status": None,
    }


def load_case(trajectory_path: Path, input_root: Path) -> dict[str, Any]:
    data = _read_json(trajectory_path)
    if not isinstance(data, dict):
        raise ValueError(f"Trajectory must be a JSON object: {trajectory_path}")

    trajectory_format = detect_format(data)
    messages = data.get("messages")
    if not isinstance(messages, list):
        raise ValueError(f"Trajectory messages must be a list: {trajectory_path}")

    if trajectory_format == "opencode":
        events = _normalize_opencode_events(messages)
    else:
        events = _normalize_mini_events(messages)

    info = data.get("info") if isinstance(data.get("info"), dict) else {}
    embedded_issue = _clean_issue_description(info.get("issue_description"))
    if embedded_issue:
        issue_description, issue_event_id = embedded_issue, None
    else:
        issue_description, issue_event_id = _extract_issue_description(events)

    case_root, result_candidates = _find_case_root(trajectory_path, input_root)
    result = _select_result(result_candidates, data, case_root)
    case_id = result.get("instance_id") or data.get("instance_id") or case_root.name

    submission = info.get("submission") if isinstance(info.get("submission"), str) else None
    patch_path = _find_artifact(case_root, "patch.diff")
    if submission is None and patch_path is not None:
        try:
            submission = patch_path.read_text(encoding="utf-8")
        except OSError:
            submission = None

    return {
        "case_name": case_root.name,
        "case_id": str(case_id),
        "case_root": str(case_root),
        "trajectory_path": str(trajectory_path),
        "format": trajectory_format,
        "issue_description": issue_description,
        "issue_event_id": issue_event_id,
        "events": events,
        "result": result,
        "spt": _load_spt(case_root),
        "submission": submission,
        "metadata": _extract_model_metadata(data, trajectory_format),
    }


def discover_cases(input_dir: str) -> tuple[list[dict[str, Any]], list[dict[str, str]]]:
    input_root = Path(input_dir).resolve()
    if not input_root.is_dir():
        raise ValueError(f"AI-assisted input must be a directory: {input_dir}")

    trajectories = sorted(input_root.rglob("*.traj.json"))
    cases = []
    errors = []
    for trajectory_path in trajectories:
        try:
            cases.append(load_case(trajectory_path, input_root))
        except Exception as exc:
            errors.append({"trajectory_path": str(trajectory_path), "error": str(exc)})
    return cases, errors