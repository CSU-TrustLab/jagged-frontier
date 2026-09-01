"""Schema and evidence validation for AI-assisted model outputs."""

from __future__ import annotations

import re
from typing import Any


PHASES = {
    "Localization",
    "Debugging",
    "Planning",
    "Patching",
    "Validation",
    "Recovery",
    "General",
}
ASSESSMENT_STATUSES = {"answered", "partial", "not_assessable"}
CLUE_CODES = {
    "A1", "A2", "A3", "A4",
    "B1", "B2", "B3", "B4", "B5", "B6", "B7",
    "C1", "C2", "C3", "C4",
    "D1", "D2", "D3",
    "E1", "E2", "E3", "E4",
    "F1", "F2", "F3", "F4",
    "G1", "G2", "G3",
    "H1", "H2",
    "I1",
}
CLUE_ROLES = {
    "localization", "reproduction", "diagnosis", "solution", "constraint",
    "environment", "structure", "metadata",
}
SIGNAL_STRENGTHS = {"low", "medium", "high"}


def _validate_confidence(value: Any, context: str) -> None:
    if not isinstance(value, (int, float)) or isinstance(value, bool) or not 0 <= value <= 1:
        raise ValueError(f"{context} confidence must be a number from 0 to 1")


def _normalize_for_match(text: str) -> tuple[str, list[int]]:
    """Lower-case, fold smart punctuation, and collapse whitespace.

    Returns the normalized text plus an index map from each normalized
    character back to its original offset (with a final sentinel offset).
    """
    smart = {
        "\u2018": "'", "\u2019": "'", "\u201b": "'", "\u2032": "'",
        "\u201c": '"', "\u201d": '"', "\u201f": '"', "\u2033": '"',
        "\u2013": "-", "\u2014": "-", "\u00a0": " ",
    }
    norm_chars: list[str] = []
    index_map: list[int] = []
    prev_space = False
    for i, ch in enumerate(text):
        c = smart.get(ch, ch)
        if c.isspace():
            if prev_space:
                continue
            norm_chars.append(" ")
            index_map.append(i)
            prev_space = True
        else:
            norm_chars.append(c.lower())
            index_map.append(i)
            prev_space = False
    index_map.append(len(text))
    return "".join(norm_chars), index_map


def locate_quote(quote: Any, source: str) -> tuple[int, int, bool] | None:
    """Locate a model-provided quote inside the source text as exhaustively as possible.

    Tries, in order: exact substring, ellipsis/whitespace-trimmed substring, and a
    normalized match that ignores case, smart punctuation, and whitespace
    differences. Returns (start, end, is_exact) over the ORIGINAL source, or None
    if no match can be found. `is_exact` is True only when the model's quote is a
    verbatim substring of the source.
    """
    if not isinstance(quote, str) or not quote or not isinstance(source, str):
        return None

    idx = source.find(quote)
    if idx != -1:
        return idx, idx + len(quote), True

    trimmed = re.sub(r"^\.{3}|\.{3}$", "", quote.strip().strip("…")).strip()
    if trimmed and trimmed in source:
        idx = source.find(trimmed)
        return idx, idx + len(trimmed), False

    norm_source, index_map = _normalize_for_match(source)
    norm_query, _ = _normalize_for_match(quote)
    norm_query = norm_query.strip()
    if not norm_query:
        return None
    pos = norm_source.find(norm_query)
    if pos == -1:
        return None
    start = index_map[pos]
    end = index_map[pos + len(norm_query)]
    return start, end, source[start:end] == quote


def validate_clue_analysis(analysis: dict[str, Any], issue_description: str) -> dict[str, Any]:
    if not isinstance(analysis, dict) or not isinstance(analysis.get("clues"), list):
        raise ValueError("Clue analysis must contain a clues list")

    seen_ids = set()
    seen_spans = set()
    for clue in analysis["clues"]:
        if not isinstance(clue, dict):
            raise ValueError("Every clue must be an object")
        clue_id = clue.get("id")
        if not isinstance(clue_id, str) or not clue_id or clue_id in seen_ids:
            raise ValueError(f"Invalid or duplicate clue id: {clue_id}")
        seen_ids.add(clue_id)

        category = clue.get("category")
        if category not in CLUE_CODES:
            raise ValueError(f"Invalid clue category: {category}")
        if not isinstance(clue.get("type"), str) or not clue["type"].strip():
            raise ValueError(f"Clue {clue_id} type must be a non-empty string")
        if clue.get("role") not in CLUE_ROLES:
            raise ValueError(f"Clue {clue_id} has invalid role: {clue.get('role')}")
        if clue.get("signal_strength") not in SIGNAL_STRENGTHS:
            raise ValueError(
                f"Clue {clue_id} has invalid signal strength: {clue.get('signal_strength')}"
            )
        if not isinstance(clue.get("value"), str):
            raise ValueError(f"Clue {clue_id} value must be a string")
        quote = clue.get("quote")
        if not isinstance(quote, str) or not quote:
            raise ValueError(f"Clue {clue_id} quote must be a non-empty string")

        located = locate_quote(quote, issue_description)
        if located is None:
            # Option 2: keep the clue rather than failing the whole case.
            clue["start"] = None
            clue["end"] = None
            clue["exact_match"] = False
            clue["match_warning"] = (
                "Quote not found in the issue description; shown as provided by the model."
            )
        else:
            start, end, exact = located
            # Use the exact source span so highlighting is always precise.
            clue["quote"] = issue_description[start:end]
            clue["start"] = start
            clue["end"] = end
            clue["exact_match"] = exact
            if not exact:
                clue["match_warning"] = (
                    "Recovered by normalized matching; may differ slightly from the model's quote."
                )
            span = (start, end, category)
            if span in seen_spans:
                raise ValueError(f"Duplicate clue span: {clue_id}")
            seen_spans.add(span)
        _validate_confidence(clue.get("confidence"), f"Clue {clue_id}")

    summary = analysis.get("summary")
    if not isinstance(summary, dict):
        raise ValueError("Clue analysis must contain a summary object")
    primary_categories = summary.get("primary_categories")
    if not isinstance(primary_categories, list) or any(
        category not in set("ABCDEFGHI") for category in primary_categories
    ):
        raise ValueError("Clue summary primary_categories must contain category letters A-I")
    if summary.get("solution_leakage") not in {"none", "low", "medium", "high"}:
        raise ValueError("Clue summary has invalid solution_leakage")
    if not isinstance(summary.get("summary"), str):
        raise ValueError("Clue summary text must be a string")
    return analysis


def _validate_evidence(
    evidence: Any, event_map: dict[str, dict[str, Any]], context: str
) -> None:
    if not isinstance(evidence, list):
        raise ValueError(f"{context} evidence must be a list")
    for item in evidence:
        if not isinstance(item, dict):
            raise ValueError(f"{context} evidence entries must be objects")
        event_id = item.get("event_id")
        if event_id not in event_map:
            raise ValueError(f"{context} references unknown event: {event_id}")
        quote = item.get("quote")
        if not isinstance(quote, str) or not quote:
            raise ValueError(f"{context} evidence quote must be non-empty")
        if quote not in event_map[event_id].get("content", ""):
            # A quote that the repair step already reconciled is kept and flagged
            # for the reader rather than failing the whole case.
            if item.get("exact_match") is False:
                item.setdefault(
                    "match_warning",
                    "Quote could not be matched exactly to the source event.",
                )
            else:
                raise ValueError(f"{context} quote is not an exact substring of {event_id}")
        if "why" in item and not isinstance(item["why"], str):
            raise ValueError(f"{context} evidence why must be a string")


def validate_trajectory_analysis(
    analysis: dict[str, Any], events: list[dict[str, Any]]
) -> dict[str, Any]:
    if not isinstance(analysis, dict) or not isinstance(analysis.get("nodes"), list):
        raise ValueError("Trajectory analysis must contain a nodes list")
    if not analysis["nodes"]:
        raise ValueError("Trajectory analysis must contain at least one node")

    event_map = {event["id"]: event for event in events}
    event_order = {event["id"]: index for index, event in enumerate(events)}
    seen_node_ids = set()
    classified_event_ids = set()
    previous_last_event = -1
    for node in analysis["nodes"]:
        if not isinstance(node, dict):
            raise ValueError("Every node must be an object")
        node_id = node.get("id")
        if not isinstance(node_id, str) or not node_id or node_id in seen_node_ids:
            raise ValueError(f"Invalid or duplicate node id: {node_id}")
        seen_node_ids.add(node_id)
        if node.get("phase") not in PHASES:
            raise ValueError(f"Node {node_id} has invalid phase: {node.get('phase')}")
        for field in ("title", "summary", "intent", "outcome"):
            if not isinstance(node.get(field), str) or not node[field].strip():
                raise ValueError(f"Node {node_id} {field} must be a non-empty string")
        resources = node.get("resources")
        if not isinstance(resources, list) or any(not isinstance(item, str) for item in resources):
            raise ValueError(f"Node {node_id} resources must be a list of strings")

        event_ids = node.get("event_ids")
        if not isinstance(event_ids, list) or not event_ids:
            raise ValueError(f"Node {node_id} must reference at least one event")
        if any(event_id not in event_map for event_id in event_ids):
            raise ValueError(f"Node {node_id} references an unknown event")
        positions = [event_order[event_id] for event_id in event_ids]
        if positions != sorted(set(positions)):
            raise ValueError(f"Node {node_id} event IDs must be unique and chronological")
        overlaps = classified_event_ids.intersection(event_ids)
        if overlaps:
            raise ValueError(f"Node {node_id} reuses classified events: {sorted(overlaps)}")
        if positions[0] <= previous_last_event:
            raise ValueError("Nodes must be chronological and non-overlapping")
        previous_last_event = positions[-1]
        classified_event_ids.update(event_ids)
        evidence = node.get("evidence")
        _validate_evidence(evidence, event_map, f"Node {node_id}")
        if not evidence:
            raise ValueError(f"Node {node_id} must contain direct evidence")
        evidence_outside_node = {
            item["event_id"] for item in evidence if item["event_id"] not in event_ids
        }
        if evidence_outside_node:
            raise ValueError(
                f"Node {node_id} evidence must come from its assigned events: "
                f"{sorted(evidence_outside_node)}"
            )
        _validate_confidence(node.get("confidence"), f"Node {node_id}")

    unclassified = analysis.get("unclassified_event_ids")
    if not isinstance(unclassified, list):
        raise ValueError("Trajectory analysis must contain unclassified_event_ids")
    if len(unclassified) != len(set(unclassified)):
        raise ValueError("unclassified_event_ids must be unique")
    unknown_unclassified = set(unclassified) - set(event_map)
    if unknown_unclassified:
        raise ValueError(f"Unknown unclassified events: {sorted(unknown_unclassified)}")
    overlap = classified_event_ids.intersection(unclassified)
    if overlap:
        raise ValueError(f"Events cannot be both classified and unclassified: {sorted(overlap)}")
    unaccounted = set(event_map) - classified_event_ids - set(unclassified)
    if unaccounted:
        raise ValueError(f"Trajectory events are not accounted for: {sorted(unaccounted)}")

    return analysis