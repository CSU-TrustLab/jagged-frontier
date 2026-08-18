"""
Compute Metadata: unified analysis of adversarial mutation experiments.

Combines the per-sample trajectory/patch analysis of ``eval_insights.py`` with
the stability-score degradation of ``degradation.py`` into a single pass over a
baseline run and a mutated run.

Unlike ``eval_insights.py`` (which assumed a single baseline run per instance),
this script treats *both* baseline and mutated as having many runs per instance
(~20 each), and reports mean / standard-deviation for steps, cost and file
coverage on *both* sides.

For each instance it also computes a "stability score" (the resolve rate, i.e.
% of runs that were resolved) for baseline and mutated, and their difference
(the degradation) -- the same quantity produced by ``degradation.py``.

Inputs may be supplied either as ``.tar.gz`` archives (default) or as already
extracted directories (``--mode dir``).

Outputs:
  * ``<output>.csv``            -- instance-wise aggregate metadata + stability.
  * ``<output>_detailed.csv``   -- per-sample metadata (no aggregates).
  * ``<output>_spt_correlation.csv``  -- global per-transformation correlation.
  * ``<output>_spt_combinations.csv`` -- global per-combination correlation.

Usage:
    python3 compute_metadata.py \
        --baseline opus_45_swepro_baseline_20x.tar.gz \
        --mutated opus_swepro_traces.tar.gz

    python3 compute_metadata.py --mode dir \
        --baseline opus_45_swepro_baseline_20x \
        --mutated opus_45_swepro_traces \
        --output insights
"""

import argparse
import csv
import json
import random
import re
import statistics
import sys
import tarfile
import tempfile
from collections import Counter, defaultdict
from pathlib import Path

TASK_SPECIFIC_SPTS = {
    "string_literal_splitter",
    "dead_string_assignment",
    "dead_method_injection",
}

# Qwen is served from a local endpoint that does not report a price, so its
# trajectories carry cost 0. When a trajectory is recognised as Qwen and its
# cost is missing, it is reconstructed from token counts at these rates
# (USD per million tokens), mirroring ``fill_qwen_traj_costs.py``.
QWEN_INPUT_COST_PER_MTOK = 0.285
QWEN_OUTPUT_COST_PER_MTOK = 2.4

ERROR_MARKERS = (
    "Too many tokens per day",
    "RateLimitError",
    "429 Too Many Requests",
    "BedrockException",
    "ThrottlingException",
    "ServiceUnavailable",
    "InternalServerError",
)


# ---------------------------------------------------------------------------
# ID helpers
# ---------------------------------------------------------------------------
def extract_instance_id(dir_name: str) -> str:
    parts = dir_name.rsplit("_", 1)
    return parts[0] if len(parts) > 1 else dir_name


def extract_sample_id(dir_name: str) -> str:
    parts = dir_name.rsplit("_", 1)
    return parts[1] if len(parts) > 1 else ""


# ---------------------------------------------------------------------------
# Trajectory / file-coverage analysis (ported from eval_insights.py)
# ---------------------------------------------------------------------------
def find_trajectory(sample_dir: Path, instance_id: str | None = None) -> Path | None:
    """Return the trajectory JSON for this sample.

    A sample directory occasionally contains a *stray* ``*.traj.json`` copied
    from a different instance (e.g. django-13410 run holding a spurious
    django-13417 trajectory). Picking the wrong one corrupts that sample's
    step/cost/file-coverage numbers, so when ``instance_id`` is known we prefer
    the trajectory whose file name or parent directory matches it
    (mini-swe-agent names them ``<instance_id>/<instance_id>.traj.json``).
    opencode names them ``ses_*.traj.json`` (no instance in the name); there is
    only one, so the fallback to the first file is correct there.
    """
    traj_files = list(sample_dir.rglob("*.traj.json"))
    if not traj_files:
        return None
    if instance_id:
        matches = [
            p
            for p in traj_files
            if p.name == f"{instance_id}.traj.json" or p.parent.name == instance_id
        ]
        if matches:
            return matches[0]
    return traj_files[0]


def count_patch_files(sample_dir: Path) -> int:
    """Count distinct files modified in the agent's ``patch.diff``.

    Used as the file-coverage signal for scaffolds (e.g. opencode) whose
    trajectories do not carry shell-command file exploration, or whose
    trajectories are truncated. Robust and scaffold-independent.
    """
    patch = sample_dir / "patch.diff"
    if not patch.exists():
        return 0
    try:
        text = patch.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return 0
    files: set[str] = set()
    for line in text.splitlines():
        if line.startswith("diff --git "):
            parts = line.split()
            if len(parts) >= 4:
                a = parts[2]
                files.add(a[2:] if a.startswith("a/") else a)
        elif line.startswith("+++ "):
            f = line[4:].strip()
            if f.startswith("b/"):
                f = f[2:]
            if f and f != "/dev/null":
                files.add(f)
    return len(files)


def _extract_leading_info(raw: str) -> dict | None:
    """Best-effort recovery of the leading ``"info": {...}`` object from a
    (possibly truncated) trajectory JSON. opencode stores ``info`` first, so its
    cost/token metadata survives even when the trailing ``messages`` array is
    cut off by a 64KB truncation.
    """
    m = re.search(r'"info"\s*:\s*\{', raw)
    if not m:
        return None
    start = m.end() - 1  # index of the opening '{'
    depth = 0
    in_str = False
    esc = False
    for i in range(start, len(raw)):
        c = raw[i]
        if in_str:
            if esc:
                esc = False
            elif c == "\\":
                esc = True
            elif c == '"':
                in_str = False
        else:
            if c == '"':
                in_str = True
            elif c == "{":
                depth += 1
            elif c == "}":
                depth -= 1
                if depth == 0:
                    try:
                        return json.loads(raw[start : i + 1])
                    except json.JSONDecodeError:
                        return None
    return None


def _find_top_level_key(raw: str, key: str) -> int:
    """Index of a top-level (depth-1) ``"key"`` inside a JSON document.

    Scans while tracking brace/bracket depth and string state so keys nested
    inside sub-objects are ignored. Ported from ``repair_opencode_traj_json.py``
    to locate the ``messages`` array in a truncated trajectory.
    """
    target = f'"{key}"'
    depth = 0
    in_string = False
    escaped = False
    index = 0
    while index < len(raw):
        char = raw[index]
        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
            index += 1
            continue
        if char == '"':
            if depth == 1 and raw.startswith(target, index):
                end = index + len(target)
                while end < len(raw) and raw[end] in " \t\r\n":
                    end += 1
                if end < len(raw) and raw[end] == ":":
                    return index
            in_string = True
            index += 1
            continue
        if char in "{[":
            depth += 1
        elif char in "}]":
            depth -= 1
        index += 1
    return -1


def _recover_truncated_messages(raw: str) -> list:
    """Recover the complete elements of a truncated ``messages`` array.

    A 64KB-truncated opencode trajectory cuts off mid-array; this decodes every
    fully-written message object and stops at the first incomplete one, so step
    counts survive truncation. Ported from ``repair_opencode_traj_json.py``.
    """
    key_pos = _find_top_level_key(raw, "messages")
    if key_pos < 0:
        return []
    start = raw.find("[", key_pos)
    if start < 0:
        return []

    decoder = json.JSONDecoder()
    messages: list = []
    index = start + 1
    length = len(raw)
    while index < length:
        while index < length and raw[index] in " \t\r\n,":
            index += 1
        if index >= length or raw[index] == "]":
            break
        try:
            message, index = decoder.raw_decode(raw, index)
        except json.JSONDecodeError:
            break
        messages.append(message)
    return messages


def _traj_is_qwen(info: dict) -> bool:
    """True when a trajectory's ``info`` header identifies a Qwen model.

    Detection is content-based (not path-based) so it works inside the
    streaming pipeline, where sample directories are anonymised scratch names.
    Mirrors the model checks in ``fill_qwen_traj_costs.py``.
    """
    if not isinstance(info, dict):
        return False
    model = info.get("model")
    if isinstance(model, dict):
        model_id = model.get("id")
        if isinstance(model_id, str) and "qwen" in model_id.lower():
            return True
    elif isinstance(model, str) and "qwen" in model.lower():
        return True
    config = info.get("config")
    if isinstance(config, dict):
        model_config = config.get("model")
        if isinstance(model_config, dict):
            for key in ("model_name", "model_name_or_path"):
                value = model_config.get(key)
                if isinstance(value, str) and "qwen" in value.lower():
                    return True
    return False


def _qwen_token_counts(info: dict, messages: list, opencode: bool) -> tuple[float, float]:
    """Return ``(input_tokens, output_tokens)`` for a Qwen trajectory.

    opencode stores an aggregate ``info.tokens``; mini-swe-agent stores per-turn
    usage on each message (``extra.response.usage``), with a ``model_stats``
    fallback. Mirrors ``fill_qwen_traj_costs.get_token_counts``.
    """
    if opencode:
        tokens = info.get("tokens") if isinstance(info.get("tokens"), dict) else {}
        input_tokens = (
            tokens.get("input")
            or tokens.get("input_tokens")
            or tokens.get("prompt_tokens")
            or 0
        )
        output_tokens = (
            tokens.get("output")
            or tokens.get("output_tokens")
            or tokens.get("completion_tokens")
            or 0
        )
        return float(input_tokens), float(output_tokens)

    prompt_total = 0
    completion_total = 0
    for message in messages:
        if not isinstance(message, dict):
            continue
        usage = message.get("usage")
        if not isinstance(usage, dict) or not usage:
            extra = message.get("extra", {})
            if isinstance(extra, dict):
                response = extra.get("response", {})
                if isinstance(response, dict):
                    usage = response.get("usage", {})
        if not isinstance(usage, dict):
            continue
        prompt_tokens = usage.get("prompt_tokens")
        if isinstance(prompt_tokens, int):
            prompt_total += prompt_tokens
        completion_tokens = usage.get("completion_tokens")
        if isinstance(completion_tokens, int):
            completion_total += completion_tokens

    if prompt_total == 0 or completion_total == 0:
        model_stats = info.get("model_stats", {})
        if isinstance(model_stats, dict):
            if prompt_total == 0 and model_stats.get("prompt_tokens") is not None:
                prompt_total = model_stats["prompt_tokens"]
            if completion_total == 0 and model_stats.get("completion_tokens") is not None:
                completion_total = model_stats["completion_tokens"]

    return float(prompt_total), float(completion_total)


def _maybe_fill_qwen_cost(
    result: dict, info: dict, messages: list, opencode: bool
) -> dict:
    """Reconstruct a missing Qwen cost from token counts, in place.

    A no-op unless the trajectory is Qwen and its reported cost is zero/absent.
    """
    if result.get("cost"):
        return result
    if not _traj_is_qwen(info):
        return result
    input_tokens, output_tokens = _qwen_token_counts(info, messages, opencode)
    if input_tokens <= 0 and output_tokens <= 0:
        return result
    result["cost"] = (
        input_tokens * QWEN_INPUT_COST_PER_MTOK
        + output_tokens * QWEN_OUTPUT_COST_PER_MTOK
    ) / 1_000_000
    return result


def _looks_opencode(info: dict, messages: list) -> bool:
    """True when a trajectory is opencode rather than mini-swe-agent.

    Detection is by structure, not by the mere presence of ``model_stats`` in
    ``info`` (both formats now carry it). Message shape is the strongest signal
    (opencode nests role under ``info`` and has a ``parts`` array; mini-swe-agent
    uses a top-level ``role``). Many opencode exports condense ``messages`` to an
    empty array while keeping a full ``info`` header, so we also key off
    opencode-specific header fields: ``model_stats.step_count`` and session keys
    like ``projectID``/``slug``. mini-swe-agent instead stores
    ``model_stats.instance_cost``.
    """
    for m in messages:
        if not isinstance(m, dict):
            continue
        if "parts" in m or (isinstance(m.get("info"), dict) and "role" in m["info"]):
            return True
        if "role" in m:
            return False
    model_stats = info.get("model_stats")
    if isinstance(model_stats, dict) and "step_count" in model_stats:
        return True
    if any(k in info for k in ("projectID", "sessionID", "slug", "directory")):
        return True
    return False


def _opencode_step_count(info: dict, messages: list) -> int:
    """Authoritative opencode step count.

    opencode records the true agent step count in ``info.model_stats.step_count``
    (the packaged ``messages`` array is condensed, so counting turns/parts
    under-reports). Fall back to ``step-start`` parts, then assistant turns, when
    the field is absent (e.g. older opencode exports).
    """
    model_stats = info.get("model_stats")
    if isinstance(model_stats, dict) and isinstance(model_stats.get("step_count"), int):
        return model_stats["step_count"]
    step_starts = sum(
        1
        for m in messages
        if isinstance(m, dict)
        for p in (m.get("parts") or [])
        if isinstance(p, dict) and p.get("type") == "step-start"
    )
    if step_starts:
        return step_starts
    return sum(
        1
        for m in messages
        if isinstance(m, dict) and (m.get("info") or {}).get("role") == "assistant"
    )


# ---------------------------------------------------------------------------
# Token extraction
# ---------------------------------------------------------------------------
_EMPTY_TOKENS = {
    "input_tokens": 0,
    "output_tokens": 0,
    "cache_read_tokens": 0,
    "cache_write_tokens": 0,
    "reasoning_tokens": 0,
}


def _extract_tokens_opencode(info: dict, messages: list) -> dict:
    """Extract token metrics from an opencode trajectory.

    opencode stores session-level totals in ``info.tokens``:
        {input, output, reasoning, cache: {read, write}}
    """
    out = dict(_EMPTY_TOKENS)
    sess = info.get("tokens") if isinstance(info.get("tokens"), dict) else {}
    if not sess:
        return out
    cache = sess.get("cache") or {}
    out["input_tokens"] = int(sess.get("input", 0) or 0)
    out["output_tokens"] = int(sess.get("output", 0) or 0)
    out["reasoning_tokens"] = int(sess.get("reasoning", 0) or 0)
    out["cache_read_tokens"] = int(cache.get("read", 0) or 0)
    out["cache_write_tokens"] = int(cache.get("write", 0) or 0)
    return out


def _extract_tokens_miniswe(info: dict, messages: list) -> dict:
    """Extract token metrics from a mini-swe-agent trajectory.

    mini-swe-agent stores per-turn usage under
    ``message.extra.response.usage`` with OpenAI-style fields.
    """
    out = dict(_EMPTY_TOKENS)
    for m in messages:
        if not isinstance(m, dict):
            continue
        u = ((m.get("extra") or {}).get("response") or {}).get("usage") or {}
        if not u:
            continue
        det = u.get("prompt_tokens_details") or {}
        out["input_tokens"] += int(u.get("prompt_tokens", 0) or 0)
        out["output_tokens"] += int(u.get("completion_tokens", 0) or 0)
        out["cache_read_tokens"] += int(det.get("cached_tokens", 0) or 0)
        out["cache_write_tokens"] += int(u.get("cache_creation_input_tokens", 0) or 0)
    return out


def load_trajectory_info(sample_dir: Path, instance_id: str | None = None) -> dict:
    empty = {
        "steps": 0,
        "cost": 0.0,
        "exit_status": "unknown",
        "submission_length": 0,
        "files_touched": 0,
        **_EMPTY_TOKENS,
    }
    traj_path = find_trajectory(sample_dir, instance_id)
    if not traj_path:
        return empty

    try:
        raw = traj_path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return empty

    truncated = False
    try:
        traj = json.loads(raw)
    except json.JSONDecodeError:
        # Truncated trajectory (typically opencode cut off at 64KB). Repair it
        # in memory: recover the leading ``info`` header plus every complete
        # message so cost and step count survive. Mirrors
        # ``repair_opencode_traj_json.py`` but without rewriting the file.
        truncated = True
        traj = {
            "info": _extract_leading_info(raw) or {},
            "messages": _recover_truncated_messages(raw),
        }

    info = traj.get("info", {}) or {}
    messages = traj.get("messages", []) or []
    is_opencode = _looks_opencode(info, messages)

    # --- mini-swe-agent format ---
    if "model_stats" in info and not is_opencode:
        model_stats = info.get("model_stats", {})
        # Use api_calls from model_stats for consistency with the researcher's
        # pipeline (generate_pipeline_data.py).  Fall back to counting assistant
        # messages when api_calls is not recorded.
        api_calls = model_stats.get("api_calls")
        if isinstance(api_calls, (int, float)) and api_calls > 0:
            steps = int(api_calls)
        else:
            steps = sum(
                1 for m in messages if isinstance(m, dict) and m.get("role") == "assistant"
            )
        cost = model_stats.get("instance_cost", 0.0) or 0.0
        exit_status = "truncated" if truncated else info.get("exit_status", "unknown")
        submission = info.get("submission", "") or ""
        tokens = _extract_tokens_miniswe(info, messages)
        result = {
            "steps": steps,
            "cost": cost,
            "exit_status": exit_status,
            "submission_length": len(submission),
            "files_touched": (
                count_patch_files(sample_dir)
                if truncated
                else extract_file_coverage(traj)
            ),
            **tokens,
        }
        return _maybe_fill_qwen_cost(result, info, messages, opencode=False)

    # --- opencode format ---
    if is_opencode or "cost" in info or "tokens" in info:
        tokens = _extract_tokens_opencode(info, messages)
        result = {
            "steps": _opencode_step_count(info, messages),
            "cost": float(info.get("cost") or 0.0),
            "exit_status": "truncated" if truncated else "ok",
            "submission_length": 0,
            "files_touched": count_patch_files(sample_dir),
            **tokens,
        }
        return _maybe_fill_qwen_cost(result, info, messages, opencode=True)

    # --- unrecognised format: keep whatever generic signal we can ---
    return {
        **empty,
        "exit_status": "truncated" if truncated else "unknown",
        "files_touched": count_patch_files(sample_dir),
    }


def extract_file_coverage(traj: dict) -> int:
    messages = traj.get("messages", [])
    files_touched = set()

    for msg in messages:
        if msg.get("role") != "assistant":
            continue
        content = msg.get("content", "")
        blocks = re.findall(r"```bash\n(.*?)\n```", content, re.DOTALL)
        for cmd in blocks:
            paths = re.findall(
                r"(?:(?:\./|/)[^\s|>;<\"'\\]+\.(?:py|txt|cfg|ini|toml|rst|md|c|h|js|ts|yaml|yml))",
                cmd,
            )
            files_touched.update(paths)
            rel_paths = re.findall(
                r"(?:cat|head|tail|nl|sed|grep|find|less|more|wc|diff)\s+[^|>;<]*?([a-zA-Z][\w/.-]*\.(?:py|txt|cfg|toml|rst|md|c|h))",
                cmd,
            )
            files_touched.update(rel_paths)
            redirect = re.findall(r">\s*([a-zA-Z][\w/.-]*\.py)", cmd)
            files_touched.update(redirect)
            sed_i = re.findall(r"sed\s+-i\s+[^|;]*?([a-zA-Z][\w/.-]*\.py)", cmd)
            files_touched.update(sed_i)

    normalized = set()
    for f in files_touched:
        f = f.lstrip("./")
        basename = f.split("/")[-1]
        if re.match(r"^(test_|reproduce_|fix_|check_|debug_|verify_)", basename):
            continue
        normalized.add(f)

    return len(normalized)


def load_spt_info(sample_dir: Path) -> dict:
    spt_path = sample_dir / "spt_log.json"
    if not spt_path.exists():
        return {"spt_types": "", "spt_entry_count": 0, "spt_breakdown": {}}

    try:
        entries = json.loads(spt_path.read_text())
    except (json.JSONDecodeError, OSError):
        return {"spt_types": "", "spt_entry_count": 0, "spt_breakdown": {}}

    if not isinstance(entries, list):
        return {"spt_types": "", "spt_entry_count": 0, "spt_breakdown": {}}

    breakdown = {}
    for e in entries:
        if not isinstance(e, dict) or "transformation" not in e:
            continue  # skip metadata entries (e.g. task_specific_keywords)
        t = e.get("transformation", "unknown")
        breakdown[t] = breakdown.get(t, 0) + 1

    spt_types = ",".join(sorted(breakdown.keys()))
    return {
        "spt_types": spt_types,
        "spt_entry_count": len(entries),
        "spt_breakdown": breakdown,
    }


# ---------------------------------------------------------------------------
# Status classification (ported from eval_insights.py)
# ---------------------------------------------------------------------------
def classify_patch(sample_dir: Path) -> str:
    """Inspect preds.json to distinguish empty patches from error patches."""
    preds_file = sample_dir / "preds.json"
    if not preds_file.exists():
        return "empty"
    try:
        data = json.loads(preds_file.read_text(encoding="utf-8", errors="replace"))
    except Exception:
        return "error"

    rec = data[0] if isinstance(data, list) and data else data
    patch = ""
    if isinstance(rec, dict):
        patch = rec.get("patch") or rec.get("model_patch") or ""

    if any(m in patch for m in ERROR_MARKERS):
        return "error"
    if not patch.strip():
        return "empty"
    return "ok"


def get_sample_status(sample_dir: Path) -> str:
    report_dir = sample_dir / "report"
    if not report_dir.exists():
        return "unknown"

    # SWE-bench Pro schema: report/eval_results.json = {"<instance_id>": bool}
    eval_results = report_dir / "eval_results.json"
    if eval_results.exists():
        try:
            data = json.loads(eval_results.read_text())
            if isinstance(data, dict) and data:
                if any(bool(v) for v in data.values()):
                    return "resolved"
                kind = classify_patch(sample_dir)
                return "unresolved" if kind == "ok" else kind
        except (json.JSONDecodeError, OSError):
            pass

    # Legacy SWE-bench schema: report/<name>.json with *_instances counters
    for json_file in report_dir.glob("*.json"):
        if json_file.name in ("report.json", "eval_results.json"):
            continue
        try:
            data = json.loads(json_file.read_text())
            if data.get("resolved_instances", 0) > 0:
                return "resolved"
            elif data.get("error_instances", 0) > 0:
                return "error"
            elif data.get("empty_patch_instances", 0) > 0:
                return "empty"
            else:
                return "unresolved"
        except (json.JSONDecodeError, OSError):
            continue

    return "unknown"


# ---------------------------------------------------------------------------
# Directory scanning
# ---------------------------------------------------------------------------
def analyze_directory(traces_dir: Path, run_type: str) -> list[dict]:
    """Scan a traces directory and return one record per sample run."""
    samples = []
    for sample_dir in sorted(traces_dir.iterdir()):
        if not sample_dir.is_dir():
            continue
        samples.append(_analyze_sample_dir(sample_dir, sample_dir.name, run_type))

    return samples


def _analyze_sample_dir(sample_dir: Path, dir_name: str, run_type: str) -> dict:
    """Build one per-sample record from an (extracted) sample directory."""
    instance_id = extract_instance_id(dir_name)
    sample_id = extract_sample_id(dir_name)
    status = get_sample_status(sample_dir)
    traj_info = load_trajectory_info(sample_dir, instance_id)
    spt_info = load_spt_info(sample_dir)

    return {
        "instance_id": instance_id,
        "sample_id": sample_id,
        "status": status,
        "run_type": run_type,
        "source_dir": dir_name,
        # Uniform, scaffold-independent file-coverage signal derived from the
        # agent's final patch. Computed for *every* scaffold so cross-scaffold
        # comparisons are apples-to-apples. ``files_touched`` (from traj_info)
        # remains the richer trajectory-exploration count where available.
        "patch_files_touched": count_patch_files(sample_dir),
        **traj_info,
        **spt_info,
    }


def resolve_traces_root(path: Path) -> Path:
    """Given a directory, return the directory that actually contains the
    per-sample ``instance_*`` run folders.

    Archives commonly extract to a single top-level wrapper directory
    (e.g. ``opus_45_swepro_traces/``); descend into it when present.
    """
    entries = [p for p in path.iterdir() if p.is_dir()]
    has_samples = any((p / "report").exists() or (p / "spt_log.json").exists() for p in entries)
    if has_samples:
        return path
    if len(entries) == 1:
        return resolve_traces_root(entries[0])
    return path


def _is_within(base: Path, target: Path) -> bool:
    try:
        target.resolve().relative_to(base.resolve())
        return True
    except ValueError:
        return False


def safe_extract_tar(archive: Path, dest: Path) -> None:
    """Safely extract a tar.gz, rejecting path-traversal members."""
    with tarfile.open(archive, "r:gz") as tar:
        # Python 3.12+ ships a hardened data filter; use it when available.
        if hasattr(tarfile, "data_filter"):
            tar.extractall(dest, filter="data")
            return
        for member in tar.getmembers():
            member_path = dest / member.name
            if not _is_within(dest, member_path):
                raise RuntimeError(f"Unsafe path in archive: {member.name}")
        tar.extractall(dest)


def _find_sample_index(parts: list[str]) -> int:
    """Index of the path component that names a sample directory, or ``-1``.

    Sample directories are named ``<instance_id>_<sample_hash>`` and every
    SWE-bench / SWE-bench-Pro ``instance_id`` contains ``"__"`` (e.g.
    ``astropy__astropy-12907``). The leftmost component containing ``"__"`` is
    therefore the sample directory, regardless of how many wrapper directories
    precede it (``claude/``, ``claude/claude/``, ``unperturbed_traces/`` ...).

    Detecting sample dirs by this marker -- rather than by a fixed wrapper
    depth -- makes streaming robust to archives that mix sample dirs with extra
    nested wrapper folders, which would otherwise be collapsed into a single
    bogus sample (e.g. an instance literally named ``claude`` or
    ``unperturbed``).
    """
    for i, part in enumerate(parts):
        if "__" in part:
            return i
    return -1


def analyze_tar_streaming(archive: Path, run_type: str, tmp_root: Path) -> list[dict]:
    """Analyze a tar.gz one sample at a time without extracting it whole.

    Members are streamed sequentially; each sample directory's files are
    written to a small scratch dir, analyzed with the same per-sample logic as
    the extracted path, then deleted. Peak disk use is a single sample rather
    than the full (potentially many-GB) archive.
    """
    import shutil

    scratch = tmp_root / f"{run_type}_stream"
    scratch.mkdir(parents=True, exist_ok=True)

    samples: list[dict] = []
    seq = 0
    cur_key: str | None = None
    cur_dir: Path | None = None
    cur_name: str | None = None

    def flush() -> None:
        if cur_dir is not None and cur_name is not None:
            samples.append(_analyze_sample_dir(cur_dir, cur_name, run_type))
            shutil.rmtree(cur_dir, ignore_errors=True)

    with tarfile.open(archive, "r|gz") as tar:  # streaming mode
        for member in tar:
            if not member.isfile():
                continue
            parts = member.name.split("/")
            if ".." in parts:
                continue
            idx = _find_sample_index(parts)
            if idx < 0 or idx >= len(parts) - 1:
                continue  # loose file, not inside a sample dir
            sample_name = parts[idx]
            rel = "/".join(parts[idx + 1 :])
            key = "/".join(parts[: idx + 1])

            if key != cur_key:
                flush()
                cur_key = key
                cur_name = sample_name
                cur_dir = scratch / f"s{seq}"
                seq += 1
                cur_dir.mkdir(parents=True, exist_ok=True)

            dest = cur_dir / rel
            if not _is_within(cur_dir, dest):
                continue
            dest.parent.mkdir(parents=True, exist_ok=True)
            src = tar.extractfile(member)
            if src is None:
                continue
            with src, open(dest, "wb") as out:
                shutil.copyfileobj(src, out)

    flush()
    shutil.rmtree(scratch, ignore_errors=True)
    return samples


def analyze_zip_streaming(
    archive: Path, run_type: str, tmp_root: Path, subpath: str = ""
) -> list[dict]:
    """Analyze one experiment subtree inside a (possibly multi-experiment) zip,
    one sample at a time, without extracting the whole archive.

    ``subpath`` selects the experiment subtree (e.g.
    ``minisweagent/swebench/qwen/vanilla``); wrapper directories beneath it are
    resolved automatically. Peak disk use is a single sample directory.
    """
    import shutil
    import zipfile

    scratch = tmp_root / f"{run_type}_zipstream"
    scratch.mkdir(parents=True, exist_ok=True)

    samples: list[dict] = []
    seq = 0
    cur_key: str | None = None
    cur_dir: Path | None = None
    cur_name: str | None = None

    def flush() -> None:
        if cur_dir is not None and cur_name is not None:
            samples.append(_analyze_sample_dir(cur_dir, cur_name, run_type))
            shutil.rmtree(cur_dir, ignore_errors=True)

    prefix = (subpath.rstrip("/") + "/") if subpath else ""
    with zipfile.ZipFile(archive) as z:
        members = sorted(
            n for n in z.namelist() if n.startswith(prefix) and not n.endswith("/")
        )
        for m in members:
            parts = m.split("/")
            if ".." in parts:
                continue
            idx = _find_sample_index(parts)
            if idx < 0 or idx >= len(parts) - 1:
                continue  # loose file, not inside a sample dir
            sample_name = parts[idx]
            rel = "/".join(parts[idx + 1 :])
            key = "/".join(parts[: idx + 1])

            if key != cur_key:
                flush()
                cur_key = key
                cur_name = sample_name
                cur_dir = scratch / f"s{seq}"
                seq += 1
                cur_dir.mkdir(parents=True, exist_ok=True)

            dest = cur_dir / rel
            if not _is_within(cur_dir, dest):
                continue
            dest.parent.mkdir(parents=True, exist_ok=True)
            with z.open(m) as src, open(dest, "wb") as out:
                shutil.copyfileobj(src, out)
        flush()

    shutil.rmtree(scratch, ignore_errors=True)
    return samples


def load_run(
    path_str: str, mode: str, run_type: str, tmp_root: Path, subpath: str = ""
) -> list[dict]:
    """Load a baseline/mutated run from a tar.gz, a zip subtree, or a directory."""
    path = Path(path_str)
    if not path.exists():
        print(f"Error: {run_type} path '{path_str}' not found.")
        sys.exit(1)

    if path.suffix == ".zip":
        loc = f"{path}::{subpath}" if subpath else str(path)
        print(f"Streaming {run_type} zip subtree: {loc}")
        samples = analyze_zip_streaming(path, run_type, tmp_root, subpath)
        print(f"  Found {len(samples)} {run_type} samples.")
        return samples

    if mode == "tar" or (mode == "auto" and path.is_file()):
        print(f"Streaming {run_type} archive: {path}")
        samples = analyze_tar_streaming(path, run_type, tmp_root)
        print(f"  Found {len(samples)} {run_type} samples.")
        return samples

    traces_root = resolve_traces_root(path)
    print(f"Analyzing {run_type} traces: {traces_root}")
    samples = analyze_directory(traces_root, run_type)
    print(f"  Found {len(samples)} {run_type} samples.")
    return samples


def dedup_and_cap(
    samples: list[dict], run_type: str, max_per_instance: int | None
) -> list[dict]:
    """Drop duplicate sample dirs and cap the number kept per instance.

    * Deduplication: a sample is keyed by ``(instance_id, sample_id)`` -- its
      directory name. Archives sometimes store the same run under two paths
      (e.g. both ``claude/<dir>`` and ``claude/claude/<dir>``); those collapse
      to one record here so a run is never double-counted.
    * Capping: at most ``max_per_instance`` samples are kept per instance,
      chosen deterministically (ordered by ``sample_id``) so results are
      reproducible. ``None``/``0`` disables the cap.

    First-seen order is preserved for kept samples so the earliest archive copy
    wins on duplicates.
    """
    seen: set[tuple[str, str]] = set()
    by_instance: dict[str, list[dict]] = defaultdict(list)
    for s in samples:
        key = (s["instance_id"], s["sample_id"])
        if key in seen:
            continue
        seen.add(key)
        by_instance[s["instance_id"]].append(s)

    result: list[dict] = []
    capped = 0
    for inst in sorted(by_instance):
        group = by_instance[inst]
        if max_per_instance and max_per_instance > 0 and len(group) > max_per_instance:
            group = sorted(group, key=lambda x: x["sample_id"])[:max_per_instance]
            capped += 1
        result.extend(group)

    dropped = len(samples) - len(seen)
    if dropped or capped:
        print(
            f"  {run_type}: removed {dropped} duplicate sample(s); "
            f"capped {capped} instance(s) to {max_per_instance}; "
            f"{len(result)} samples kept."
        )
    return result


def stratified_sample(
    samples: list[dict], run_type: str, n_per_instance: int
) -> list[dict]:
    """Down-sample to ``n_per_instance`` samples per instance, stratified by
    resolve outcome so the resolved/unresolved proportion is preserved.

    Some runs collect more baseline repetitions than others (e.g. minimax's
    50x baseline vs the usual 20x). Truncating naively would bias the resolve
    rate; instead we keep the same fraction of resolved and unresolved samples
    as the full set, allocating the ``n_per_instance`` slots across strata with
    largest-remainder rounding. Selection within a stratum is deterministic:
    the RNG is seeded per instance so re-runs are reproducible.
    """
    if not n_per_instance or n_per_instance <= 0:
        return samples

    by_instance: dict[str, list[dict]] = defaultdict(list)
    for s in samples:
        by_instance[s["instance_id"]].append(s)

    result: list[dict] = []
    reduced = 0
    for inst in sorted(by_instance):
        group = by_instance[inst]
        if len(group) <= n_per_instance:
            result.extend(group)
            continue

        # Partition into strata by resolve status.
        strata: dict[str, list[dict]] = defaultdict(list)
        for s in group:
            stratum = "resolved" if s["status"] == "resolved" else "unresolved"
            strata[stratum].append(s)

        # Proportional allocation with largest-remainder rounding.
        total = len(group)
        exact = {k: len(v) / total * n_per_instance for k, v in strata.items()}
        alloc = {k: int(v) for k, v in exact.items()}
        remaining = n_per_instance - sum(alloc.values())
        for k in sorted(exact, key=lambda x: (-(exact[x] - alloc[x]), x)):
            if remaining <= 0:
                break
            if alloc[k] < len(strata[k]):
                alloc[k] += 1
                remaining -= 1

        rng = random.Random(f"{run_type}:{inst}")
        for k in sorted(strata):
            pool = sorted(strata[k], key=lambda x: x["sample_id"])
            take = min(alloc.get(k, 0), len(pool))
            result.extend(rng.sample(pool, take) if take < len(pool) else pool)
        reduced += 1

    if reduced:
        print(
            f"  {run_type}: stratified-sampled {reduced} instance(s) to "
            f"{n_per_instance} sample(s) each (resolve-rate preserved); "
            f"{len(result)} samples kept."
        )
    return result


def partition_noop_mutants(
    mutant_samples: list[dict], drop: bool = True
) -> tuple[list[dict], list[dict]]:
    """Split mutant samples into (transformed, no_op).

    A mutant sample whose ``spt_types`` is empty had no semantic-preserving
    transformation actually applied (the spt_log was missing/empty), so it is
    effectively a baseline run mislabelled as mutated. Including such samples
    in the mutant pool biases the resolve rate upward and produces spurious
    negative degradation, so they are dropped from the pool and reported in the
    errors log for manual verification. ``drop`` only affects the log wording;
    the caller decides whether to actually drop them.
    """
    transformed, no_op = [], []
    for s in mutant_samples:
        (no_op if not str(s.get("spt_types", "")).strip() else transformed).append(s)
    if no_op:
        verb = "excluded" if drop else "flagged (kept in pool)"
        print(
            f"  mutated: {verb} {len(no_op)} sample(s) with no transformation "
            f"applied (empty spt_types); see errors log."
        )
    return transformed, no_op


def write_errors_log(
    path: str,
    noop_mutants: list[dict],
    low_baseline: list[tuple[str, int, int]],
    baseline_only: list[tuple[str, int]],
    min_baseline_samples: int,
) -> None:
    """Write a human-readable log of samples/instances excluded from the
    degradation computation, for manual verification."""
    lines: list[str] = []
    lines.append("# compute_metadata errors / exclusions log")
    lines.append(
        "# Samples and instances below were excluded from the baseline-vs-mutant "
        "degradation computation."
    )
    lines.append("")

    lines.append(
        f"## Mutant samples with no transformation applied (empty spt_types): "
        f"{len(noop_mutants)}"
    )
    lines.append(
        "# These were dropped from the mutant pool. Verify the source run to "
        "confirm no SPT was applied (missing/empty spt_log.json)."
    )
    if noop_mutants:
        lines.append(
            f"{'instance_id':<70}  {'sample_id':<12}  {'status':<12}  source_dir"
        )
        for s in sorted(
            noop_mutants, key=lambda x: (x["instance_id"], str(x["sample_id"]))
        ):
            lines.append(
                f"{s['instance_id']:<70}  {str(s['sample_id']):<12}  "
                f"{s['status']:<12}  {s.get('source_dir', '')}"
            )
    else:
        lines.append("(none)")
    lines.append("")

    lines.append(
        f"## Instances excluded from degradation for too few baseline samples "
        f"(< {min_baseline_samples}): {len(low_baseline)}"
    )
    if min_baseline_samples <= 0:
        lines.append("# Guard disabled (--min-baseline-samples 0).")
    elif low_baseline:
        lines.append(f"{'instance_id':<70}  {'baseline_n':<10}  mutant_n")
        for inst, bn, mn in sorted(low_baseline):
            lines.append(f"{inst:<70}  {bn:<10}  {mn}")
    else:
        lines.append("(none)")
    lines.append("")

    lines.append(
        f"## Baseline-only instances (no mutant generated), already excluded "
        f"from degradation: {len(baseline_only)}"
    )
    if baseline_only:
        lines.append(f"{'instance_id':<70}  baseline_n")
        for inst, bn in sorted(baseline_only):
            lines.append(f"{inst:<70}  {bn}")
    else:
        lines.append("(none)")
    lines.append("")

    Path(path).write_text("\n".join(lines))
    print(f"Wrote {path}")


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------
def mean(values: list[float]) -> float:
    return statistics.fmean(values) if values else 0.0


def sd(values: list[float]) -> float:
    """Sample standard deviation; 0.0 when fewer than two data points."""
    return statistics.stdev(values) if len(values) > 1 else 0.0


def resolve_rate(samples: list[dict]) -> float:
    if not samples:
        return 0.0
    resolved = sum(1 for s in samples if s["status"] == "resolved")
    return resolved / len(samples) * 100


def _group_by_instance(samples: list[dict]) -> dict[str, list[dict]]:
    grouped = defaultdict(list)
    for s in samples:
        grouped[s["instance_id"]].append(s)
    return grouped


def generate_per_instance_report(
    baseline_samples: list[dict],
    mutant_samples: list[dict],
    min_baseline_samples: int = 0,
) -> list[dict]:
    baseline_by_instance = _group_by_instance(baseline_samples)
    mutant_by_instance = _group_by_instance(mutant_samples)

    all_ids = sorted(set(baseline_by_instance) | set(mutant_by_instance))
    rows = []

    for instance_id in all_ids:
        bl = baseline_by_instance.get(instance_id, [])
        mt = mutant_by_instance.get(instance_id, [])

        # --- Stability scores (resolve rate %) ---
        baseline_stability = resolve_rate(bl)
        mutant_stability = resolve_rate(mt)
        # Degradation is only meaningful when both pools exist and the baseline
        # has enough samples to give a stable estimate.
        enough_baseline = min_baseline_samples <= 0 or len(bl) >= min_baseline_samples
        has_both = bool(bl) and bool(mt) and enough_baseline
        degradation = (
            round(baseline_stability - mutant_stability, 2) if has_both else ""
        )

        # --- Baseline aggregate stats ---
        bl_steps = [s["steps"] for s in bl]
        bl_cost = [s["cost"] for s in bl]
        bl_files = [s["files_touched"] for s in bl]
        bl_pfiles = [s["patch_files_touched"] for s in bl]

        # --- Mutant aggregate stats ---
        mt_steps = [s["steps"] for s in mt]
        mt_cost = [s["cost"] for s in mt]
        mt_files = [s["files_touched"] for s in mt]
        mt_pfiles = [s["patch_files_touched"] for s in mt]

        # --- Mutant status breakdown ---
        mt_total = len(mt)
        mt_resolved = sum(1 for s in mt if s["status"] == "resolved")
        mt_unresolved = sum(1 for s in mt if s["status"] == "unresolved")
        mt_empty = sum(1 for s in mt if s["status"] == "empty")
        mt_error = sum(1 for s in mt if s["status"] == "error")

        bl_total = len(bl)
        bl_resolved = sum(1 for s in bl if s["status"] == "resolved")

        bl_steps_mean = mean(bl_steps)
        mt_steps_mean = mean(mt_steps)
        bl_files_mean = mean(bl_files)
        mt_files_mean = mean(mt_files)
        bl_pfiles_mean = mean(bl_pfiles)
        mt_pfiles_mean = mean(mt_pfiles)
        bl_cost_mean = mean(bl_cost)
        mt_cost_mean = mean(mt_cost)

        step_increase = (
            (mt_steps_mean - bl_steps_mean) / bl_steps_mean * 100
            if bl_steps_mean > 0
            else 0.0
        )
        file_increase = (
            (mt_files_mean - bl_files_mean) / bl_files_mean * 100
            if bl_files_mean > 0
            else 0.0
        )
        patch_file_increase = (
            (mt_pfiles_mean - bl_pfiles_mean) / bl_pfiles_mean * 100
            if bl_pfiles_mean > 0
            else 0.0
        )
        cost_increase = (
            (mt_cost_mean - bl_cost_mean) / bl_cost_mean * 100
            if bl_cost_mean > 0
            else 0.0
        )

        # --- Instance-level SPT usage (from mutant samples) ---
        spt_type_samples = defaultdict(int)
        total_spt_entries = 0
        for s in mt:
            for t in s.get("spt_breakdown", {}):
                spt_type_samples[t] += 1
            total_spt_entries += s.get("spt_entry_count", 0)
        top_spts = sorted(spt_type_samples.items(), key=lambda x: -x[1])
        spt_types_applied = ",".join(f"{t}({c})" for t, c in top_spts)
        avg_spt_entries = total_spt_entries / mt_total if mt_total > 0 else 0.0

        # --- Token metrics ---
        bl_out_tok = [s.get("output_tokens", 0) for s in bl]
        mt_out_tok = [s.get("output_tokens", 0) for s in mt]
        bl_in_tok = [s.get("input_tokens", 0) for s in bl]
        mt_in_tok = [s.get("input_tokens", 0) for s in mt]
        bl_out_mean = mean(bl_out_tok)
        mt_out_mean = mean(mt_out_tok)
        bl_in_mean = mean(bl_in_tok)
        mt_in_mean = mean(mt_in_tok)

        # Per-step intensities
        bl_cost_per_step = bl_cost_mean / bl_steps_mean if bl_steps_mean > 0 else 0.0
        mt_cost_per_step = mt_cost_mean / mt_steps_mean if mt_steps_mean > 0 else 0.0
        bl_out_per_step = bl_out_mean / bl_steps_mean if bl_steps_mean > 0 else 0.0
        mt_out_per_step = mt_out_mean / mt_steps_mean if mt_steps_mean > 0 else 0.0
        bl_in_per_step = bl_in_mean / bl_steps_mean if bl_steps_mean > 0 else 0.0
        mt_in_per_step = mt_in_mean / mt_steps_mean if mt_steps_mean > 0 else 0.0

        cost_per_step_increase = (
            (mt_cost_per_step - bl_cost_per_step) / bl_cost_per_step * 100
            if bl_cost_per_step > 0 else 0.0
        )

        # --- Resolved-only / Failed-only conditional metrics ---
        bl_resolved_samples = [s for s in bl if s["status"] == "resolved"]
        mt_resolved_samples = [s for s in mt if s["status"] == "resolved"]
        bl_failed_samples = [s for s in bl if s["status"] != "resolved"]
        mt_failed_samples = [s for s in mt if s["status"] != "resolved"]

        bl_steps_resolved_mean = mean([s["steps"] for s in bl_resolved_samples])
        mt_steps_resolved_mean = mean([s["steps"] for s in mt_resolved_samples])
        bl_cost_resolved_mean = mean([s["cost"] for s in bl_resolved_samples])
        mt_cost_resolved_mean = mean([s["cost"] for s in mt_resolved_samples])
        bl_steps_failed_mean = mean([s["steps"] for s in bl_failed_samples])
        mt_steps_failed_mean = mean([s["steps"] for s in mt_failed_samples])
        bl_cost_failed_mean = mean([s["cost"] for s in bl_failed_samples])
        mt_cost_failed_mean = mean([s["cost"] for s in mt_failed_samples])

        step_increase_resolved = (
            (mt_steps_resolved_mean - bl_steps_resolved_mean) / bl_steps_resolved_mean * 100
            if bl_steps_resolved_mean > 0 else 0.0
        )
        cost_increase_resolved = (
            (mt_cost_resolved_mean - bl_cost_resolved_mean) / bl_cost_resolved_mean * 100
            if bl_cost_resolved_mean > 0 else 0.0
        )
        step_increase_failed = (
            (mt_steps_failed_mean - bl_steps_failed_mean) / bl_steps_failed_mean * 100
            if bl_steps_failed_mean > 0 else 0.0
        )
        cost_increase_failed = (
            (mt_cost_failed_mean - bl_cost_failed_mean) / bl_cost_failed_mean * 100
            if bl_cost_failed_mean > 0 else 0.0
        )

        rows.append(
            {
                "instance_id": instance_id,
                # stability
                "baseline_stability": round(baseline_stability, 2),
                "mutant_stability": round(mutant_stability, 2),
                "stability_degradation": degradation,
                # baseline aggregates
                "baseline_samples": bl_total,
                "baseline_resolved": bl_resolved,
                "baseline_steps_mean": round(bl_steps_mean, 2),
                "baseline_steps_sd": round(sd(bl_steps), 2),
                "baseline_cost_mean": round(bl_cost_mean, 4),
                "baseline_cost_sd": round(sd(bl_cost), 4),
                "baseline_files_mean": round(bl_files_mean, 2),
                "baseline_files_sd": round(sd(bl_files), 2),
                "baseline_patch_files_mean": round(bl_pfiles_mean, 2),
                "baseline_patch_files_sd": round(sd(bl_pfiles), 2),
                # mutant aggregates
                "mutant_samples": mt_total,
                "mutant_resolved": mt_resolved,
                "mutant_unresolved": mt_unresolved,
                "mutant_empty": mt_empty,
                "mutant_error": mt_error,
                "mutant_steps_mean": round(mt_steps_mean, 2),
                "mutant_steps_sd": round(sd(mt_steps), 2),
                "mutant_cost_mean": round(mt_cost_mean, 4),
                "mutant_cost_sd": round(sd(mt_cost), 4),
                "mutant_files_mean": round(mt_files_mean, 2),
                "mutant_files_sd": round(sd(mt_files), 2),
                "mutant_patch_files_mean": round(mt_pfiles_mean, 2),
                "mutant_patch_files_sd": round(sd(mt_pfiles), 2),
                # deltas
                "step_increase_pct": round(step_increase, 2),
                "file_coverage_increase_pct": round(file_increase, 2),
                "patch_file_coverage_increase_pct": round(patch_file_increase, 2),
                "cost_increase_pct": round(cost_increase, 2),
                # per-step intensity
                "baseline_cost_per_step": round(bl_cost_per_step, 5),
                "mutant_cost_per_step": round(mt_cost_per_step, 5),
                "cost_per_step_increase_pct": round(cost_per_step_increase, 2),
                # token metrics
                "baseline_output_tokens_mean": round(bl_out_mean, 2),
                "mutant_output_tokens_mean": round(mt_out_mean, 2),
                "baseline_input_tokens_mean": round(bl_in_mean, 2),
                "mutant_input_tokens_mean": round(mt_in_mean, 2),
                "baseline_output_tokens_per_step": round(bl_out_per_step, 2),
                "mutant_output_tokens_per_step": round(mt_out_per_step, 2),
                "baseline_input_tokens_per_step": round(bl_in_per_step, 2),
                "mutant_input_tokens_per_step": round(mt_in_per_step, 2),
                # conditional metrics (resolved-only)
                "baseline_steps_resolved_mean": round(bl_steps_resolved_mean, 2),
                "mutant_steps_resolved_mean": round(mt_steps_resolved_mean, 2),
                "step_increase_resolved_pct": round(step_increase_resolved, 2),
                "baseline_cost_resolved_mean": round(bl_cost_resolved_mean, 4),
                "mutant_cost_resolved_mean": round(mt_cost_resolved_mean, 4),
                "cost_increase_resolved_pct": round(cost_increase_resolved, 2),
                # conditional metrics (failed-only)
                "baseline_steps_failed_mean": round(bl_steps_failed_mean, 2),
                "mutant_steps_failed_mean": round(mt_steps_failed_mean, 2),
                "step_increase_failed_pct": round(step_increase_failed, 2),
                "baseline_cost_failed_mean": round(bl_cost_failed_mean, 4),
                "mutant_cost_failed_mean": round(mt_cost_failed_mean, 4),
                "cost_increase_failed_pct": round(cost_increase_failed, 2),
                # instance-level SPT usage
                "spt_types_applied": spt_types_applied,
                "avg_spt_entries": round(avg_spt_entries, 2),
            }
        )

    return rows


def generate_per_sample_report(
    baseline_samples: list[dict], mutant_samples: list[dict]
) -> list[dict]:
    rows = []
    for s in [*baseline_samples, *mutant_samples]:
        rows.append(
            {
                "instance_id": s["instance_id"],
                "sample_id": s["sample_id"],
                "run_type": s["run_type"],
                "status": s["status"],
                "exit_status": s["exit_status"],
                "steps": s["steps"],
                "cost": round(s["cost"], 4),
                "files_touched": s["files_touched"],
                "patch_files_touched": s["patch_files_touched"],
                "submission_length": s["submission_length"],
                "input_tokens": s.get("input_tokens", 0),
                "output_tokens": s.get("output_tokens", 0),
                "cache_read_tokens": s.get("cache_read_tokens", 0),
                "cache_write_tokens": s.get("cache_write_tokens", 0),
                "reasoning_tokens": s.get("reasoning_tokens", 0),
                "spt_types": s["spt_types"],
                "spt_entry_count": s["spt_entry_count"],
            }
        )
    return rows


def generate_per_spt_report(mutant_samples: list[dict]) -> list[dict]:
    """Global per-transformation adversarial effectiveness (cross-instance)."""
    spt_stats = defaultdict(
        lambda: {
            "samples_with": 0,
            "resolved_with": 0,
            "failed_with": 0,
            "entries_resolved": 0,
            "entries_failed": 0,
            "steps_resolved": [],
            "steps_failed": [],
            "cost_resolved": [],
            "cost_failed": [],
        }
    )

    for s in mutant_samples:
        breakdown = s.get("spt_breakdown", {})
        is_resolved = s["status"] == "resolved"
        for spt_name, entry_count in breakdown.items():
            stats = spt_stats[spt_name]
            stats["samples_with"] += 1
            if is_resolved:
                stats["resolved_with"] += 1
                stats["entries_resolved"] += entry_count
                stats["steps_resolved"].append(s["steps"])
                stats["cost_resolved"].append(s["cost"])
            else:
                stats["failed_with"] += 1
                stats["entries_failed"] += entry_count
                stats["steps_failed"].append(s["steps"])
                stats["cost_failed"].append(s["cost"])

    rows = []
    for spt_name, stats in sorted(spt_stats.items()):
        samples_with = stats["samples_with"]
        failure_rate = (
            stats["failed_with"] / samples_with * 100 if samples_with > 0 else 0.0
        )

        samples_without = [
            s for s in mutant_samples if spt_name not in s.get("spt_breakdown", {})
        ]
        failed_without = sum(1 for s in samples_without if s["status"] != "resolved")
        failure_rate_without = (
            failed_without / len(samples_without) * 100 if samples_without else 0.0
        )

        avg_entries_resolved = (
            stats["entries_resolved"] / stats["resolved_with"]
            if stats["resolved_with"] > 0
            else 0.0
        )
        avg_entries_failed = (
            stats["entries_failed"] / stats["failed_with"]
            if stats["failed_with"] > 0
            else 0.0
        )

        rows.append(
            {
                "spt_name": spt_name,
                "is_task_specific": spt_name in TASK_SPECIFIC_SPTS,
                "samples_with": samples_with,
                "resolved_with": stats["resolved_with"],
                "failed_with": stats["failed_with"],
                "failure_rate_with": round(failure_rate, 2),
                "failure_rate_without": round(failure_rate_without, 2),
                "adversarial_lift": round(failure_rate - failure_rate_without, 2),
                "avg_entries_when_resolved": round(avg_entries_resolved, 2),
                "avg_entries_when_failed": round(avg_entries_failed, 2),
                "avg_steps_resolved": round(mean(stats["steps_resolved"]), 2),
                "avg_steps_failed": round(mean(stats["steps_failed"]), 2),
                "avg_cost_resolved": round(mean(stats["cost_resolved"]), 4),
                "avg_cost_failed": round(mean(stats["cost_failed"]), 4),
            }
        )

    return rows


def generate_spt_combination_report(mutant_samples: list[dict]) -> list[dict]:
    """Global per-combination adversarial effectiveness (cross-instance)."""
    combo_stats = defaultdict(
        lambda: {"total": 0, "resolved": 0, "failed": 0, "has_task_specific": False}
    )

    for s in mutant_samples:
        breakdown = s.get("spt_breakdown", {})
        if not breakdown:
            continue
        combo_key = "+".join(sorted(breakdown.keys()))
        has_task_specific = bool(set(breakdown.keys()) & TASK_SPECIFIC_SPTS)
        combo_stats[combo_key]["total"] += 1
        if s["status"] == "resolved":
            combo_stats[combo_key]["resolved"] += 1
        else:
            combo_stats[combo_key]["failed"] += 1
        combo_stats[combo_key]["has_task_specific"] = has_task_specific

    rows = []
    for combo, stats in sorted(combo_stats.items(), key=lambda x: -x[1]["failed"]):
        failure_rate = (
            stats["failed"] / stats["total"] * 100 if stats["total"] > 0 else 0.0
        )
        rows.append(
            {
                "spt_combination": combo,
                "has_task_specific": stats["has_task_specific"],
                "total_samples": stats["total"],
                "resolved": stats["resolved"],
                "failed": stats["failed"],
                "failure_rate": round(failure_rate, 2),
            }
        )

    return rows


# ---------------------------------------------------------------------------
# Statistical computations
# ---------------------------------------------------------------------------
def _newcombe_ci(count1: int, nobs1: int, count2: int, nobs2: int,
                 alpha: float = 0.05) -> tuple[float, float]:
    """Newcombe confidence interval for the difference of two proportions.

    Returns (lower, upper) bounds. Requires statsmodels.
    """
    from statsmodels.stats.proportion import confint_proportions_2indep
    lower, upper = confint_proportions_2indep(
        count1=count1, nobs1=nobs1,
        count2=count2, nobs2=nobs2,
        method='newcombe', compare='diff', alpha=alpha)
    return float(lower), float(upper)


def _bootstrap_mean_ci(values, n_boot: int = 20000, ci: float = 0.95,
                       seed: int = 0) -> tuple[float, float, float]:
    """Percentile bootstrap CI for the mean.

    Returns (mean, ci_lower, ci_upper).
    """
    import numpy as np
    values = np.asarray(values, dtype=float)
    if len(values) == 0:
        return 0.0, 0.0, 0.0
    theta_hat = float(values.mean())
    if len(values) < 2 or np.ptp(values) == 0:
        return theta_hat, theta_hat, theta_hat
    rng = np.random.default_rng(seed)
    boot_means = np.array([
        values[rng.integers(0, len(values), size=len(values))].mean()
        for _ in range(n_boot)
    ])
    alpha = 1 - ci
    lo = float(np.quantile(boot_means, alpha / 2))
    hi = float(np.quantile(boot_means, 1 - alpha / 2))
    return theta_hat, lo, hi


def _fixed_instance_bootstrap_relative_change(
    x_baseline: list, x_mutant: list,
    n_boot: int = 20000, alpha: float = 0.05, seed: int = 0
) -> tuple[float | None, float | None, float | None]:
    """Fixed-instance run-level bootstrap CI for the mean relative % change.

    For each instance, resample runs within that instance, compute
    (mutant_mean - baseline_mean) / baseline_mean * 100, then average across
    instances. Returns (mean_pct_change, ci_lower, ci_upper).
    """
    import numpy as np
    m = len(x_baseline)
    if m == 0:
        return None, None, None
    rng = np.random.default_rng(seed)

    # Point estimate: mean of per-instance relative changes
    d = np.array([
        (np.mean(p) - np.mean(u)) / np.mean(u) * 100.0
        for u, p in zip(x_baseline, x_mutant)
    ])
    dbar = float(d.mean())

    # Bootstrap
    boot = np.zeros(n_boot)
    for u, p in zip(x_baseline, x_mutant):
        u = np.asarray(u, dtype=float)
        p = np.asarray(p, dtype=float)
        u_star = u[rng.integers(0, len(u), size=(n_boot, len(u)))].mean(axis=1)
        p_star = p[rng.integers(0, len(p), size=(n_boot, len(p)))].mean(axis=1)
        boot += (p_star - u_star) / u_star * 100.0
    boot /= m

    lo = float(np.quantile(boot, alpha / 2))
    hi = float(np.quantile(boot, 1 - alpha / 2))
    return dbar, lo, hi


def _binomial_redraw_bootstrap(
    p_baseline, p_mutant, n_baseline: int, n_mutant: int,
    n_boot: int = 20000, alpha: float = 0.05, seed: int = 0
) -> tuple[float, float, float]:
    """Binomial-redraw bootstrap for mean degradation (fixed issue set).

    Treats observed per-instance resolve rates as the true probabilities,
    redraws binomial outcomes, and constructs a percentile CI on the mean
    degradation. Returns (mean_pp, ci_lower_pp, ci_upper_pp) in percentage
    points.
    """
    import numpy as np
    p_b = np.asarray(p_baseline, dtype=float)
    p_m = np.asarray(p_mutant, dtype=float)
    rng = np.random.default_rng(seed)

    degr = p_b - p_m
    theta_hat = float(degr.mean())

    u = rng.binomial(n_baseline, p_b, size=(n_boot, p_b.size)) / n_baseline
    p = rng.binomial(n_mutant, p_m, size=(n_boot, p_m.size)) / n_mutant
    boot_means = (u - p).mean(axis=1)

    lo = float(np.quantile(boot_means, alpha / 2))
    hi = float(np.quantile(boot_means, 1 - alpha / 2))
    # Convert to percentage points
    return theta_hat * 100, lo * 100, hi * 100


def _sign_flip_test(diffs, n_resamples: int = 100000, seed: int = 0
                    ) -> tuple[float, float]:
    """Two-sided sign-flip permutation test that mean(diffs) == 0.

    Returns (statistic, p_value). Zero diffs are dropped first.
    """
    import numpy as np
    from scipy.stats import permutation_test
    diffs = np.asarray(diffs, dtype=float)
    nonzero = diffs[diffs != 0.0]
    if len(nonzero) == 0:
        return 0.0, 1.0

    def stat(x, axis=-1):
        return np.mean(x, axis=axis)

    exact = len(nonzero) <= 24
    res = permutation_test(
        (nonzero,), stat,
        permutation_type="samples",
        vectorized=True,
        alternative="two-sided",
        n_resamples=(2**30 if exact else n_resamples),
        rng=seed,
    )
    return float(res.statistic), float(res.pvalue)


def generate_statistics_report(
    baseline_samples: list[dict],
    mutant_samples: list[dict],
    instance_rows: list[dict],
    n_boot: int = 20000,
    seed: int = 0,
) -> dict:
    """Generate statistical summaries: CIs on degradation, step/cost deltas.

    Returns a dict with keys:
      - "degradation_ci": {mean_pp, ci_lower_pp, ci_upper_pp, ...}
      - "degradation_binomial_ci": {...}  (fixed-issue binomial-redraw)
      - "sign_flip_pvalue": float
      - "step_delta_ci": {mean_pct, ci_lower, ci_upper}
      - "cost_delta_ci": {mean_pct, ci_lower, ci_upper}
      - "step_delta_resolved_ci": {...}
      - "cost_delta_resolved_ci": {...}
      - "per_instance_newcombe": [{instance_id, ci_lower, ci_upper}, ...]
    """
    import numpy as np
    from collections import defaultdict

    bl_by_inst = defaultdict(list)
    mt_by_inst = defaultdict(list)
    for s in baseline_samples:
        bl_by_inst[s["instance_id"]].append(s)
    for s in mutant_samples:
        mt_by_inst[s["instance_id"]].append(s)

    common_ids = sorted(set(bl_by_inst) & set(mt_by_inst))

    result = {}

    # --- Per-instance Newcombe CIs ---
    newcombe_rows = []
    degradations = []
    p_baselines = []
    p_mutants = []
    for inst in common_ids:
        bl = bl_by_inst[inst]
        mt = mt_by_inst[inst]
        n_bl = len(bl)
        n_mt = len(mt)
        bl_resolved = sum(1 for s in bl if s["status"] == "resolved")
        mt_resolved = sum(1 for s in mt if s["status"] == "resolved")
        bl_rate = bl_resolved / n_bl
        mt_rate = mt_resolved / n_mt
        deg = bl_rate - mt_rate
        degradations.append(deg)
        p_baselines.append(bl_rate)
        p_mutants.append(mt_rate)
        try:
            ci_lo, ci_hi = _newcombe_ci(bl_resolved, n_bl, mt_resolved, n_mt)
        except Exception:
            ci_lo, ci_hi = None, None
        newcombe_rows.append({
            "instance_id": inst,
            "degradation": round(deg, 4),
            "newcombe_ci_lower": round(ci_lo, 4) if ci_lo is not None else None,
            "newcombe_ci_upper": round(ci_hi, 4) if ci_hi is not None else None,
            "n_baseline": n_bl,
            "n_mutant": n_mt,
        })
    result["per_instance_newcombe"] = newcombe_rows

    # --- Bootstrap CI on mean degradation (issue-resample) ---
    if degradations:
        deg_arr = np.array(degradations) * 100  # to percentage points
        mean_pp, lo_pp, hi_pp = _bootstrap_mean_ci(deg_arr, n_boot=n_boot, seed=seed)
        result["degradation_ci"] = {
            "mean_pp": round(mean_pp, 4),
            "ci_lower_pp": round(lo_pp, 4),
            "ci_upper_pp": round(hi_pp, 4),
            "n_instances": len(degradations),
            "confidence": 0.95,
            "n_boot": n_boot,
        }
    else:
        result["degradation_ci"] = None

    # --- Binomial-redraw bootstrap CI (fixed issue set) ---
    if p_baselines:
        # Determine n per instance (use most common sample count)
        bl_counts = [len(bl_by_inst[i]) for i in common_ids]
        mt_counts = [len(mt_by_inst[i]) for i in common_ids]
        n_bl_mode = max(set(bl_counts), key=bl_counts.count)
        n_mt_mode = max(set(mt_counts), key=mt_counts.count)
        mean_pp, lo_pp, hi_pp = _binomial_redraw_bootstrap(
            p_baselines, p_mutants, n_bl_mode, n_mt_mode,
            n_boot=n_boot, seed=seed)
        result["degradation_binomial_ci"] = {
            "mean_pp": round(mean_pp, 4),
            "ci_lower_pp": round(lo_pp, 4),
            "ci_upper_pp": round(hi_pp, 4),
            "n_baseline_per_instance": n_bl_mode,
            "n_mutant_per_instance": n_mt_mode,
            "n_instances": len(p_baselines),
            "confidence": 0.95,
            "n_boot": n_boot,
        }
    else:
        result["degradation_binomial_ci"] = None

    # --- Sign-flip p-value ---
    if degradations:
        _, pval = _sign_flip_test(np.array(degradations), seed=seed)
        result["sign_flip_pvalue"] = round(pval, 6)
    else:
        result["sign_flip_pvalue"] = None

    # --- Bootstrap CIs on step/cost deltas ---
    def _build_ci(field: str, resolved_only: bool):
        x_bl, x_mt = [], []
        for inst in common_ids:
            bl = bl_by_inst[inst]
            mt = mt_by_inst[inst]
            if resolved_only:
                bl = [s for s in bl if s["status"] == "resolved"]
                mt = [s for s in mt if s["status"] == "resolved"]
            u = [s[field] for s in bl]
            p = [s[field] for s in mt]
            if not u or not p or sum(u) == 0:
                continue
            x_bl.append(u)
            x_mt.append(p)
        if not x_bl:
            return None
        mean_pct, lo, hi = _fixed_instance_bootstrap_relative_change(
            x_bl, x_mt, n_boot=n_boot, seed=seed)
        return {
            "mean_pct": round(mean_pct, 2) if mean_pct is not None else None,
            "ci_lower": round(lo, 2) if lo is not None else None,
            "ci_upper": round(hi, 2) if hi is not None else None,
            "n_instances": len(x_bl),
        }

    result["step_delta_ci"] = _build_ci("steps", False)
    result["cost_delta_ci"] = _build_ci("cost", False)
    result["step_delta_resolved_ci"] = _build_ci("steps", True)
    result["cost_delta_resolved_ci"] = _build_ci("cost", True)

    return result


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------
def write_csv(rows: list[dict], output_path: str) -> None:
    if not rows:
        print(f"Skipping {output_path}: no rows.")
        return
    fieldnames = list(rows[0].keys())
    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Written: {output_path} ({len(rows)} rows)")


def print_summary(
    instance_rows: list[dict],
    baseline_samples: list[dict],
    mutant_samples: list[dict],
    spt_rows: list[dict],
) -> None:
    print("\n" + "=" * 100)
    print("COMPUTE METADATA SUMMARY")
    print("=" * 100)

    bl_rate = resolve_rate(baseline_samples)
    mt_rate = resolve_rate(mutant_samples)
    print(
        f"\nBaseline resolve rate: "
        f"{sum(1 for s in baseline_samples if s['status'] == 'resolved')}/"
        f"{len(baseline_samples)} ({bl_rate:.1f}%)"
    )
    print(
        f"Mutant resolve rate:   "
        f"{sum(1 for s in mutant_samples if s['status'] == 'resolved')}/"
        f"{len(mutant_samples)} ({mt_rate:.1f}%)"
    )
    # Instances with no mutants are out of scope for degradation: exclude their
    # baseline samples so the pooled comparison is baseline-vs-mutant on the
    # same instance set.
    mutated_ids = {s["instance_id"] for s in mutant_samples}
    baseline_scoped = [s for s in baseline_samples if s["instance_id"] in mutated_ids]
    excluded = len({s["instance_id"] for s in baseline_samples} - mutated_ids)
    bl_rate_scoped = resolve_rate(baseline_scoped)
    degradations = [
        r["stability_degradation"]
        for r in instance_rows
        if isinstance(r["stability_degradation"], (int, float))
    ]
    if excluded:
        print(
            f"  (excluding {excluded} instance(s) with no mutants from degradation; "
            f"scoped baseline resolve rate: "
            f"{sum(1 for s in baseline_scoped if s['status'] == 'resolved')}/"
            f"{len(baseline_scoped)} ({bl_rate_scoped:.1f}%))"
        )
    print(
        f"Overall stability degradation (baseline - mutant): "
        f"{bl_rate_scoped - mt_rate:+.1f}%"
    )
    print(f"Mean per-instance stability degradation: {mean(degradations):+.2f}%")

    print("\n--- SPT Adversarial Effectiveness (sorted by adversarial lift) ---")
    print(
        f"{'SPT Name':<30} | {'Task-Spec':<9} | {'Samples':<8} | "
        f"{'Fail Rate':<10} | {'W/O Rate':<10} | {'Lift':<8}"
    )
    print("-" * 90)
    for r in sorted(spt_rows, key=lambda x: -x["adversarial_lift"]):
        task_spec = "YES" if r["is_task_specific"] else ""
        print(
            f"{r['spt_name']:<30} | {task_spec:<9} | {r['samples_with']:<8} | "
            f"{r['failure_rate_with']:<9.1f}% | {r['failure_rate_without']:<9.1f}% | "
            f"{r['adversarial_lift']:+7.2f}%"
        )
    print()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--baseline",
        required=True,
        help="Baseline run: .tar.gz archive (default mode) or directory (--mode dir).",
    )
    parser.add_argument(
        "--mutated",
        required=True,
        help="Mutated run: .tar.gz archive (default mode) or directory (--mode dir).",
    )
    parser.add_argument(
        "--baseline-subpath",
        default="",
        help="When --baseline is a .zip, the subtree inside it to analyze "
        "(e.g. 'minisweagent/swebench/qwen/vanilla').",
    )
    parser.add_argument(
        "--mutated-subpath",
        default="",
        help="When --mutated is a .zip, the subtree inside it to analyze "
        "(e.g. 'minisweagent/swebench/qwen/perturbed').",
    )
    parser.add_argument(
        "--mode",
        choices=["tar", "dir", "auto"],
        default="auto",
        help="Input type: 'tar' (.tar.gz archives), 'dir' (extracted directories), "
        "or 'auto' (detect per path; default).",
    )
    parser.add_argument(
        "--max-samples-per-instance",
        type=int,
        default=50,
        help="Cap on mutant samples kept per instance after de-duplication "
        "(0 disables; default 50). Baselines are de-duplicated but not capped.",
    )
    parser.add_argument(
        "--baseline-stratified-sample",
        type=int,
        default=0,
        help="Down-sample baseline runs to this many samples per instance, "
        "stratified by resolve outcome to preserve the resolve rate "
        "(0 disables; use e.g. 20 when a baseline has more repetitions than "
        "the others, such as minimax's 50x baseline).",
    )
    parser.add_argument(
        "--min-baseline-samples",
        type=int,
        default=0,
        help="Exclude an instance from the degradation computation when it has "
        "fewer than this many baseline samples (its estimate is too noisy). "
        "0 disables the guard (default).",
    )
    parser.add_argument(
        "--keep-noop-mutants",
        action="store_true",
        help="Keep mutant samples whose spt_log records no transformation "
        "(empty spt_types) in the mutant pool instead of dropping them. Use "
        "when the perturbed set is expected to contain exactly N runs per "
        "instance regardless of whether an SPT actually landed. They are still "
        "listed in the errors log for visibility.",
    )
    parser.add_argument(
        "--output",
        default="metadata",
        help="Output prefix (produces <prefix>.csv, <prefix>_detailed.csv, "
        "<prefix>_spt_correlation.csv, <prefix>_spt_combinations.csv).",
    )
    parser.add_argument(
        "--n-boot",
        type=int,
        default=20000,
        help="Number of bootstrap resamples for CIs (default: 20000).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed for bootstrap and permutation tests (default: 0).",
    )
    args = parser.parse_args()

    with tempfile.TemporaryDirectory(prefix="compute_metadata_") as tmp:
        tmp_root = Path(tmp)
        baseline_samples = load_run(
            args.baseline, args.mode, "baseline", tmp_root, args.baseline_subpath
        )
        mutant_samples = load_run(
            args.mutated, args.mode, "mutated", tmp_root, args.mutated_subpath
        )

        baseline_samples = dedup_and_cap(
            baseline_samples, "baseline", None
        )
        baseline_samples = stratified_sample(
            baseline_samples, "baseline", args.baseline_stratified_sample
        )
        mutant_samples = dedup_and_cap(
            mutant_samples, "mutated", args.max_samples_per_instance
        )
        # Mutant samples where no transformation was actually applied (empty
        # spt_types) are effectively un-mutated runs. By default they are
        # dropped from the mutant pool (they would bias degradation), but with
        # --keep-noop-mutants they are retained so every instance keeps its full
        # run count. Either way they are reported in the errors log.
        if args.keep_noop_mutants:
            _, noop_mutants = partition_noop_mutants(mutant_samples, drop=False)
        else:
            mutant_samples, noop_mutants = partition_noop_mutants(mutant_samples)

        instance_rows = generate_per_instance_report(
            baseline_samples, mutant_samples, args.min_baseline_samples
        )
        sample_rows = generate_per_sample_report(baseline_samples, mutant_samples)
        spt_rows = generate_per_spt_report(mutant_samples)
        combo_rows = generate_spt_combination_report(mutant_samples)

        # Statistical computations
        stats_report = generate_statistics_report(
            baseline_samples, mutant_samples, instance_rows,
            n_boot=args.n_boot, seed=args.seed,
        )

        write_csv(instance_rows, f"{args.output}.csv")
        write_csv(sample_rows, f"{args.output}_detailed.csv")
        write_csv(spt_rows, f"{args.output}_spt_correlation.csv")
        write_csv(combo_rows, f"{args.output}_spt_combinations.csv")

        # Write statistics JSON
        stats_path = f"{args.output}_statistics.json"
        Path(stats_path).write_text(
            json.dumps(stats_report, indent=2, default=str), encoding="utf-8"
        )
        print(f"Written: {stats_path}")

        # Write per-instance Newcombe CIs as CSV
        if stats_report.get("per_instance_newcombe"):
            write_csv(
                stats_report["per_instance_newcombe"],
                f"{args.output}_degradation_ci.csv",
            )

        # Collect instances excluded from degradation for the errors log.
        baseline_counts = Counter(s["instance_id"] for s in baseline_samples)
        mutant_counts = Counter(s["instance_id"] for s in mutant_samples)
        baseline_only = [
            (inst, baseline_counts[inst])
            for inst in baseline_counts
            if mutant_counts.get(inst, 0) == 0
        ]
        low_baseline = []
        if args.min_baseline_samples > 0:
            low_baseline = [
                (inst, baseline_counts[inst], mutant_counts[inst])
                for inst in mutant_counts
                if 0 < baseline_counts.get(inst, 0) < args.min_baseline_samples
            ]
        write_errors_log(
            f"{args.output}_errors.log",
            noop_mutants,
            low_baseline,
            baseline_only,
            args.min_baseline_samples,
        )

        print_summary(instance_rows, baseline_samples, mutant_samples, spt_rows)


if __name__ == "__main__":
    main()
