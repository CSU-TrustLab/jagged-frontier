import logging
import math
import multiprocessing
import os
import random
from pathlib import Path
from typing import List, Optional, Set, Tuple, Union

from semantic_transformer.api import apply_transformation, collect_candidate_nodes
from semantic_transformer.core.utils import collect_python_files

ENCODING = "utf-8"
TEST_FILE_PATTERNS = {"test_*.py", "*_test.py"}

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

EXCLUDE_DIR_PATTERNS = {
    "pydata__xarray": " **/tests/** **/test/* **/testing/**",
    "astropy__astropy": "**/tests/** **/test/* **/tests/*",
    "django__django": "**/tests/** **/test/** tests/** test/**",
    "sympy__sympy": "**/tests/** **/testing/**",
    "scikit": "**/tests/** **/test/**",
    "matplotlib__matplotlib": "**/tests/** **/test/**",
    "pytest": "**/testing/** testing/**",
    "sphinx": "tests/** **/tests/**",
    "psf__requests": "tests/** **/tests/**",
    "mwaskom__seaborn": "tests/** **/tests/**",
    "pallets__flask": "tests/** **/tests/**",
    "pylint": "tests/** **/tests/** **/testutils/**",
    "instance_ansible__ansible": "test/** **/tests/** **/test/**",
    "instance_internetarchive__openlibrary": "**/tests/** **/test/**",
    "instance_qutebrowser__qutebrowser": "tests/** **/tests/** **/test/**",
}


def _worker_init(seed_offset: int):
    random.seed(os.getpid() + seed_offset)


def _process_file(
    args: Tuple[Path, str, float, Optional[Union[str, List[str]]]],
) -> Tuple[str, str]:
    file_path, transformation_name, candidate_selection, keyword = args

    try:
        source_code = file_path.read_text(encoding=ENCODING)
    except Exception as e:
        logger.warning("Failed to read %s: %s", file_path, e)
        return str(file_path), "read_error"

    try:
        if transformation_name == TransformationName.DEAD_METHHOD_INJECTION.value:
            candidates = collect_candidate_nodes(
                source_code, transformation_name, target_name=keyword
            )
        else:
            candidates = collect_candidate_nodes(source_code, transformation_name)
    except Exception as e:
        logger.warning("Failed to collect candidates in %s: %s", file_path, e)
        return str(file_path), "candidate_error"

    if not candidates:
        return str(file_path), "no_candidates"

    k = max(1, math.ceil(len(candidates) * candidate_selection))
    selected = random.sample(candidates, min(k, len(candidates)))

    try:
        transformed = apply_transformation(source_code, selected, transformation_name, keyword)
    except Exception as e:
        logger.warning("Failed to transform %s: %s", file_path, e)
        return str(file_path), "transform_error"

    try:
        file_path.write_text(transformed, encoding=ENCODING)
    except Exception as e:
        logger.warning("Failed to write %s: %s", file_path, e)
        return str(file_path), "write_error"

    return str(file_path), "success"


def transform_directory_sampled(
    repo_dir: Union[str, Path],
    transformation_name: str,
    file_selection: float = 1.0,
    candidate_selection: float = 1.0,
    keyword: Optional[Union[str, List[str]]] = None,
    num_workers: Optional[int] = None,
    exclude_files: Optional[Set[str]] = None,
    exclude_patterns: Optional[Set[str]] = None,
) -> dict:
    """
    Apply a semantic-preserving transformation to a sampled subset of Python files
    in a repository directory, in-place, using multiprocessing.

    Parameters
    ----------
    repo_dir : str or Path
        Root directory of the repository.
    transformation_name : str
        Transformation identifier (a TransformationName value).
    file_selection : float
        Fraction of eligible files to transform (0.0–1.0]. Rounded up.
    candidate_selection : float
        Fraction of candidate nodes per file to apply the transformation to (0.0–1.0]. Rounded up.
    keyword : str or list of str, optional
        Keyword(s) for keyword-specific transformations (e.g. DEAD_STRING_ASSIGNMENT).
    num_workers : int, optional
        Cap on the number of worker processes. Defaults to cpu_count.
    exclude_files : set of str, optional
        File paths relative to repo_dir to skip.
    exclude_patterns : set of str, optional
        Glob-style patterns for files to skip (merged with default test-file patterns).
    """
    repo_dir = Path(repo_dir)
    if not repo_dir.is_dir():
        raise ValueError(f"repo_dir must be an existing directory: {repo_dir}")

    if not (0.0 < file_selection <= 1.0):
        raise ValueError("file_selection must be in (0.0, 1.0]")
    if not (0.0 < candidate_selection <= 1.0):
        raise ValueError("candidate_selection must be in (0.0, 1.0]")

    all_exclude_patterns = set(TEST_FILE_PATTERNS)
    if exclude_patterns:
        all_exclude_patterns |= set(exclude_patterns)

    files, collection_results = collect_python_files(repo_dir, exclude_files, all_exclude_patterns)
    logger.info("File collection: %s", collection_results)

    if not files:
        logger.warning("No eligible Python files found in %s", repo_dir)
        return {"selected": 0, "success": 0, "no_candidates": 0, "errors": []}

    k = max(1, math.ceil(len(files) * file_selection))
    selected_files = random.sample(files, min(k, len(files)))
    logger.info("Selected %d / %d files for transformation", len(selected_files), len(files))

    cpu_count = multiprocessing.cpu_count()
    workers = cpu_count if num_workers is None else min(num_workers, cpu_count)
    seed_offset = random.randint(0, 10_000)

    task_args = [(f, transformation_name, candidate_selection, keyword) for f in selected_files]

    counts = {"success": 0, "no_candidates": 0, "errors": []}

    with multiprocessing.Pool(
        processes=workers,
        initializer=_worker_init,
        initargs=(seed_offset,),
    ) as pool:
        for file_path_str, status in pool.imap_unordered(_process_file, task_args):
            if status == "success":
                counts["success"] += 1
                logger.info("Transformed: %s", file_path_str)
            elif status == "no_candidates":
                counts["no_candidates"] += 1
            else:
                counts["errors"].append((file_path_str, status))
                logger.warning("Error (%s): %s", status, file_path_str)

    logger.info(
        "Done — transformed: %d, no candidates: %d, errors: %d",
        counts["success"],
        counts["no_candidates"],
        len(counts["errors"]),
    )
    return {"selected": len(selected_files), **counts}


if __name__ == "__main__":
    from semantic_transformer.core.transformation_name import TransformationName

    results = transform_directory_sampled(
        repo_dir="",
        transformation_name=TransformationName.DEAD_METHHOD_INJECTION.value,
        file_selection=1.0,
        candidate_selection=1.0,
        keyword="get_content_length",
        exclude_files=None,
        exclude_patterns=set(EXCLUDE_DIR_PATTERNS.get("sympy__sympy").split()),
    )
    print(results)
