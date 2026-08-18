import multiprocessing
import shutil
from concurrent.futures import ProcessPoolExecutor
from typing import Iterable
import uuid
from pathlib import Path
import numpy as np
import json

from strategies.strategy import Strategy
from semantic_transformer.api import (
    apply_transformation,
    collect_candidate_nodes,
)

from agentic_robustness_experiment.utils.utils import (
    collect_python_files,
    available_transformations,
    task_specific_transformations,
)
from agentic_robustness_experiment.utils.task_keywords import extract_task_keywords

# Fallback values, used when no config is supplied. The authoritative values for
# the reported results live in the `spt_config` block of config.yaml.
DEFAULT_NUM_TRANSFORMATIONS = 3
DEFAULT_MUTANT_ID_LENGTH = 12
DEFAULT_FRACTION_OF_CANDIDATES_TO_TRANSFORM = 0.7
DEFAULT_MAX_FILES_FOR_INJECTION_SPTS = 10
DEFAULT_MAX_KEYWORDS_FOR_TASK_SPECIFIC_SPTS = 5


def _transform_file_worker(args: tuple) -> dict | None:
    """Transform a single file in place and return a log entry.

    Top-level function so it is picklable by ProcessPoolExecutor.
    Each worker creates its own RNG from the provided seed, keeping randomness
    independent across files without sharing state with the main process.

    Returns a dict with transformation details (for SPT logging), or None when
    no candidates were found or selected.
    """
    file_path, transformation_name, match_transformation_percentage, rng_seed, src_root, target_name = args
    if transformation_name in {"dead_method_injection", "dead_string_assignment"} and target_name is None:
        return None
    rng = np.random.default_rng(rng_seed)
    source_code = file_path.read_text()
    # target_name is only meaningful for dead_method_injection in candidate collection
    collect_kwargs = {"target_name": target_name} if transformation_name == "dead_method_injection" else {}
    candidates = collect_candidate_nodes(source_code, transformation_name, **collect_kwargs)
    if not candidates:
        return None
    n = len(candidates)
    k = max(1, int(np.ceil(match_transformation_percentage * n)))
    indices = rng.choice(n, size=k, replace=False)
    selected = [candidates[i] for i in indices]
    if not selected:
        return None
    # target_name is only meaningful for KEYWORD_SPECIFIC_TRANSFORMATIONS in apply
    apply_kwargs = {"target_name": target_name} if transformation_name in task_specific_transformations else {}
    file_path.write_text(
        apply_transformation(source_code, selected, transformation_name, **apply_kwargs)
    )
    return {
        "transformation": transformation_name,
        "file": str(Path(file_path).relative_to(src_root)),
        "positions": [{"line": line, "column": col} for line, col in selected],
    }


class RandomSelectionStrategy(Strategy):
    def __init__(self, rng_seed=None, config=None):
        """Create a strategy.

        ``config`` is an optional ``AppConfig``; when given, the SPT sampling
        parameters are read from its ``spt_config`` block instead of the module
        defaults, so the values used for a run are visible in config.yaml.
        """
        self.rng = np.random.default_rng(rng_seed)
        self.num_transformations = getattr(
            config, "num_transformations", DEFAULT_NUM_TRANSFORMATIONS
        )
        self.fraction_of_candidates_to_transform = getattr(
            config,
            "fraction_of_candidates_to_transform",
            DEFAULT_FRACTION_OF_CANDIDATES_TO_TRANSFORM,
        )
        self.max_files_for_injection_spts = getattr(
            config, "max_files_for_injection_spts", DEFAULT_MAX_FILES_FOR_INJECTION_SPTS
        )
        self.max_keywords_for_task_specific_spts = getattr(
            config,
            "max_keywords_for_task_specific_spts",
            DEFAULT_MAX_KEYWORDS_FOR_TASK_SPECIFIC_SPTS,
        )
        self.mutant_id_length = getattr(
            config, "mutant_id_length", DEFAULT_MUTANT_ID_LENGTH
        )

    def select_candidates(self, candidates: list, match_probability=1.0) -> list:
        if not 0 < match_probability <= 1:
            raise ValueError("match_probability must be in (0, 1].")

        n = len(candidates)
        k = max(1, int(np.ceil(match_probability * n)))
        indices = self.rng.choice(n, size=k, replace=False)
        return [candidates[i] for i in indices]

    def transform_directory(
        self,
        src_path: str | Path,
        transformation_name: str,
        exclude_files: set[str] | None = None,
        exclude_patterns: list[str] | None = None,
        file_transformation_probability: float = 1.0,
        match_transformation_percentage: float = 1.0,
        max_workers: int | None = None,
        required_files: list[Path] | None = None,
        target_name: str | None = None,
    ) -> list[dict]:
        """
        Transform inplace all the source (Python) files within a directory.

        File selection is done in the main process (preserving RNG reproducibility).
        The CPU-bound CST parsing and transformation of each selected file is
        dispatched to a ProcessPoolExecutor so multiple files are transformed in
        parallel, bypassing the GIL.

        ``required_files`` is an optional list of absolute paths that are always
        included in the transformation, regardless of ``file_transformation_probability``.

        Returns a list of SPT log entries (one per file that was actually transformed).
        """
        if not isinstance(src_path, Path):
            src_path = Path(src_path)

        if not src_path.is_dir():
            raise ValueError("The source directory must exist.")

        src_files, collection_results = collect_python_files(
            src_path, exclude_files, exclude_patterns
        )
        print(
            f"applyling {transformation_name} transformation with file transformation probability {file_transformation_probability} and match transformation percentage {match_transformation_percentage} in {src_path}"
        )

        selected_files = [
            f for f in src_files if self.rng.random() <= file_transformation_probability
        ]
        # Always include required files (e.g. gold-patch files) that exist in
        # the collected source set, regardless of the random probability.
        if required_files:
            required_set = set(required_files)
            selected_set = set(selected_files)
            selected_files = list(selected_set | (required_set & set(src_files)))

        if not selected_files:
            return []

        if transformation_name in {"dead_string_assignment", "dead_method_injection"}:
            required_set = set(required_files or [])
            non_required = [f for f in selected_files if f not in required_set]
            cap = max(0, self.max_files_for_injection_spts - len(required_set))
            selected_files = list(required_set & set(selected_files)) + non_required[:cap]

        # Each worker gets a unique seed derived from the main RNG so candidate
        # selection inside each file is independent but deterministic.
        seeds = [int(self.rng.integers(0, 2**32)) for _ in selected_files]
        task_args = [
            (f, transformation_name, match_transformation_percentage, s, src_path, target_name)
            for f, s in zip(selected_files, seeds)
        ]

        mp_context = multiprocessing.get_context("forkserver")
        with ProcessPoolExecutor(
            max_workers=max_workers, mp_context=mp_context
        ) as executor:
            results = list(executor.map(_transform_file_worker, task_args))
        return [entry for entry in results if entry is not None]

    def transform_file(
        self,
        source_code: str,
        transformation_name: str,
        match_transformation_percentage=1.0,
    ) -> str:
        candidates = collect_candidate_nodes(source_code, transformation_name)
        if len(candidates) == 0:
            return source_code

        selected_candidates = self.select_candidates(
            candidates, match_transformation_percentage
        )

        if len(selected_candidates) == 0:
            return source_code

        transformed_code = apply_transformation(
            source_code, selected_candidates, transformation_name
        )
        return transformed_code

    def create_mutant_directory(
        self, seed_location: Path, dest_root: Path, mutant_id: str
    ) -> Path:
        mutant_dir = dest_root / mutant_id
        shutil.copytree(seed_location, mutant_dir, dirs_exist_ok=True, symlinks=True)
        return mutant_dir

    def sample_transformations(
        self,
        k: int,
        use_task_specific_spts: bool = True,
    ) -> list[tuple[str, float, float, str | None]]:
        pool = list(available_transformations)
        if not use_task_specific_spts:
            pool = [t for t in pool if t not in task_specific_transformations]

        k = min(k, len(pool))
        transformation_names = self.rng.choice(pool, size=k, replace=False)
        return [
            (
                name,
                float(self.rng.random()),
                self.fraction_of_candidates_to_transform,
                None,
            )
            for name in transformation_names
        ]

    def _assign_target_names(
        self,
        transformations: list[tuple[str, float, float, str | None]],
        keywords: dict,
    ) -> list[tuple[str, float, float, str | None]]:
        """Expand keyword-specific SPTs into one tuple per keyword (up to max_keywords).

        Non-keyword SPTs pass through unchanged. The returned list may be longer
        than the input when a keyword SPT has multiple available keywords.
        """
        result = []
        for name, p_file, match_pct, _ in transformations:
            if name == "dead_string_assignment":
                literals = keywords.get("string_literals", [])
                if literals:
                    n = min(self.max_keywords_for_task_specific_spts, len(literals))
                    chosen = self.rng.choice(len(literals), size=n, replace=False)
                    for i in chosen:
                        result.append((name, p_file, match_pct, str(literals[i])))
                else:
                    result.append((name, p_file, match_pct, None))
            elif name == "dead_method_injection":
                methods = keywords.get("methods", [])
                if methods:
                    n = min(self.max_keywords_for_task_specific_spts, len(methods))
                    chosen = self.rng.choice(len(methods), size=n, replace=False)
                    for i in chosen:
                        result.append((name, p_file, match_pct, str(methods[i]).split(".")[-1]))
                else:
                    result.append((name, p_file, match_pct, None))
            else:
                result.append((name, p_file, match_pct, None))
        return result

    def apply_transformations(
        self,
        mutant_dir: Path,
        transformations: Iterable[tuple[str, float, float, str | None]],
        exclude_patterns: list[str] | None,
        exclude_files: set[str] | None = None,
        max_workers: int | None = None,
        required_files: list[Path] | None = None,
    ) -> list[dict]:
        """Apply all transformations and return aggregated SPT log entries in order."""
        log_entries: list[dict] = []
        for name, p_file, match_pct, target_name in transformations:
            entries = self.transform_directory(
                src_path=mutant_dir,
                transformation_name=name,
                file_transformation_probability=p_file,
                match_transformation_percentage=match_pct,
                exclude_files=exclude_files,
                exclude_patterns=exclude_patterns,
                max_workers=max_workers,
                required_files=required_files,
                target_name=target_name,
            )
            log_entries.extend(entries)
        return log_entries

    def generate_samples(
        self,
        seed_location: str,
        size: int,
        dest_path: str,
        exclude_patterns: list[str] | None = None,
        log_file: str | Path | None = "mutant_transformations.jsonl",
        exclude_files: set[str] | None = None,
        max_workers: int | None = None,
        gold_patch_files: list[Path] | None = None,
        issue_description: str | None = None,
        llm_provider: str = "anthropic",
        llm_model: str = "claude-opus-4-5-20251101",
        llm_base_url: str = "",
        llm_api_key: str = "",
        llm_region: str = "us-east-1",
        use_task_specific_spts: bool = True,
    ) -> list[tuple[str, Path, list[dict]]]:
        """Generate `size` mutant samples.

        ``gold_patch_files`` is an optional list of absolute paths inside
        ``seed_location`` that should always be transformed (regardless of the
        random file-selection probability).  They are remapped to the
        corresponding paths inside each mutant directory automatically.

        When ``use_task_specific_spts=True`` and ``issue_description`` is
        provided, task-specific SPTs (string_literal_splitter,
        dead_string_assignment, dead_method_injection) are included in the
        random sampling pool alongside all other available SPTs, with keywords
        extracted from the issue description used as target_name where needed.
        Set ``use_task_specific_spts=False`` to exclude them from the pool.

        Returns a list of ``(mutant_id, mutant_dir, spt_entries)`` triples where
        ``spt_entries`` is the ordered list of SPT log entries for that sample,
        suitable for passing to ``write_spt_log``.
        """
        keywords: dict | None = None  # extracted lazily, at most once
        keyword_spts = {"dead_string_assignment", "dead_method_injection"}

        mutant_samples = []
        seed_location = Path(seed_location)
        dest_path = Path(dest_path)
        for _ in range(size):
            mutant_id = uuid.uuid4().hex[: self.mutant_id_length]
            mutant_dir = self.create_mutant_directory(
                Path(seed_location), Path(dest_path), mutant_id
            )
            try:
                transformations = self.sample_transformations(
                    self.num_transformations,
                    use_task_specific_spts=use_task_specific_spts,
                )

                needs_keywords = any(t[0] in keyword_spts for t in transformations)
                if needs_keywords and keywords is None and use_task_specific_spts and issue_description:
                    keywords = extract_task_keywords(
                        issue_description,
                        model=llm_model,
                        provider=llm_provider,
                        base_url=llm_base_url,
                        api_key=llm_api_key,
                        region=llm_region,
                    )
                    print(f"Extracted task keywords: {keywords}")

                if needs_keywords and keywords:
                    transformations = self._assign_target_names(transformations, keywords)
                # Remap gold-patch paths from seed_location to mutant_dir so
                # that transform_directory receives valid absolute paths.
                required_files: list[Path] | None = None
                if gold_patch_files:
                    required_files = [
                        mutant_dir / f.relative_to(seed_location)
                        for f in gold_patch_files
                    ]
                spt_entries = self.apply_transformations(
                    mutant_dir,
                    transformations,
                    exclude_patterns,
                    exclude_files=exclude_files,
                    max_workers=max_workers,
                    required_files=required_files,
                )

                if log_file:
                    with open(log_file, "a") as f:
                        log_entry: dict = {
                            "mutant_id": mutant_id,
                            "transformations": [
                                {
                                    "name": t[0],
                                    "p_file": round(t[1], 4),
                                    "match_pct": round(t[2], 4),
                                    **({"target_name": t[3]} if t[3] else {}),
                                }
                                for t in transformations
                            ],
                        }
                        if use_task_specific_spts and keywords:
                            log_entry["task_specific_keywords"] = {
                                "string_literals": keywords.get("string_literals", []),
                                "methods": keywords.get("methods", []),
                            }
                        f.write(json.dumps(log_entry) + "\n")
                mutant_samples.append((mutant_id, mutant_dir, spt_entries))

            except Exception as e:
                shutil.rmtree(mutant_dir, ignore_errors=True)
                raise RuntimeError(
                    f"Failed to create mutant {mutant_id} at {mutant_dir}: {e}"
                ) from e
        return mutant_samples
