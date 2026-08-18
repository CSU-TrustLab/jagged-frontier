from __future__ import annotations

import re
from functools import lru_cache
from pathlib import Path
from datasets import load_dataset


DIFF_FILE_PATTERN = re.compile(r"^diff --git a/(.+?) b/.+$", re.MULTILINE)


@lru_cache(maxsize=8)
def _load_dataset_cached(dataset_name: str, split: str):
    return load_dataset(dataset_name, split=split)


def _find_instance(dataset, instance_id: str) -> dict | None:
    for row in dataset:
        if row.get("instance_id") == instance_id:
            return row
    return None


SWEBENCH_VERIFIED_DATASET = "princeton-nlp/SWE-Bench_Verified"


def select_files_by_gold_patch(
    instance_id: str,
    repo_path: Path,
    dataset_name: str = SWEBENCH_VERIFIED_DATASET,
    split: str = "test",
) -> list[Path]:
    dataset = _load_dataset_cached(dataset_name, split)
    row = _find_instance(dataset, instance_id)

    if row is None:
        return []

    patch: str = row.get("patch", "")
    if not patch:
        print(f"Empty 'patch' field for instance {instance_id}.")
        return []

    rel_paths = DIFF_FILE_PATTERN.findall(patch)

    resolved: list[Path] = []
    seen: set[str] = set()
    for rel in rel_paths:
        if rel in seen:
            continue
        seen.add(rel)
        if not rel.endswith(".py"):
            continue
        abs_path = repo_path / rel
        if abs_path.exists():
            resolved.append(abs_path)

    selected = [str(p.relative_to(repo_path)) for p in resolved]
    print(
        f"gold_patch selected {len(resolved)} file(s) for {instance_id}: {selected}"
    )

    return resolved

