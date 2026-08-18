import fnmatch
import json
import os
from pathlib import Path

from semantic_transformer.core.transformation_name import TransformationName

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
    "instance_internetarchive__openlibrary": "**/tests/** **/test/** openlibrary/tests/** openlibrary/plugins/*/tests/** openlibrary/catalog/*/tests/**",
    "instance_qutebrowser__qutebrowser": "tests/** **/tests/** **/test/**",
}


EXCLUDE_ROOT_FILES: dict[str, set[str]] = {
    "psf__requests": {"test_requests.py"},
}

DATASET_MAPPING = {
    "full": "princeton-nlp/SWE-Bench",
    "verified": "princeton-nlp/SWE-Bench_Verified",
    "lite": "princeton-nlp/SWE-Bench_Lite",
    "multimodal": "princeton-nlp/SWE-Bench_Multimodal",
    "multilingual": "swe-bench/SWE-Bench_Multilingual",
    "smith": "SWE-bench/SWE-smith",
    "_test": "klieret/swe-bench-dummy-test-dataset",
     "pro": "ScaleAI/SWE-bench_Pro",
}

DEFAULT_EXCLUDED_DIRS = {
    ".git",
    ".hg",
    ".svn",
    "venv",
    ".venv",
    "env",
    ".env",
    "ENV",
    "build",
    "dist",
    "__pycache__",
    "*.egg-info",
    ".tox",
    ".nox",
    "site-packages",
    ".idea",
    ".vscode",
    ".ropeproject",
    ".pytest_cache",
    ".mypy_cache",
    ".cache",
    "docs",
    "htmlcov",
    "coverage",
    "tmp",
    "temp",
    "logs",
    "results",
    "datasets",
    "data",
    "notebooks",
}

available_transformations = [element.value for element in TransformationName]
available_transformations.remove("function_name_renamer")
available_transformations.remove("local_variable_renamer")
task_specific_transformations = ['string_literal_splitter', 'dead_string_assignment', 'dead_method_injection']


def get_issue_description(instance_id: str, subset: str, split: str) -> str:
    try:
        from datasets import load_dataset
        from agentic_robustness_experiment.utils.utils import DATASET_MAPPING

        dataset_path = DATASET_MAPPING.get(subset, DATASET_MAPPING["verified"])
        dataset = load_dataset(dataset_path, split=split)
        for row in dataset:
            if row.get("instance_id") == instance_id:
                return row.get("problem_statement", "")
    except Exception:
        print(
            f"Could not load issue description for {instance_id}; proceeding without it."
        )
    return ""


def get_issue_prompt(instance_id: str, subset: str, split: str) -> str:
    """Like get_issue_description, but for SWEBench Pro concatenates requirements and interface."""
    try:
        from datasets import load_dataset
        from agentic_robustness_experiment.utils.utils import DATASET_MAPPING

        dataset_path = DATASET_MAPPING.get(subset, DATASET_MAPPING["verified"])
        dataset = load_dataset(dataset_path, split=split)
        for row in dataset:
            if row.get("instance_id") == instance_id:
                problem_statement = row.get("problem_statement", "")
                if row.get("requirements") or row.get("interface"):
                    requirement = row.get("requirements", "")
                    interface = row.get("interface", "")
                    return f"{problem_statement}\n\nRequirements:\n{requirement}\n\nNew interfaces introduced:\n{interface}"
                return problem_statement
    except Exception:
        print(
            f"Could not load issue prompt for {instance_id}; proceeding without it."
        )
    return ""


def get_swebench_docker_image_name(instance: dict) -> str:
    """Get the image name for a SWEBench instance."""
    image_name = instance.get("image_name", None)
    if image_name is None:
        # Docker doesn't allow double underscore, so we replace them with a magic token
        iid = instance["instance_id"]
        id_docker_compatible = iid.replace("__", "_1776_")
        image_name = f"docker.io/swebench/sweb.eval.x86_64.{id_docker_compatible}:latest".lower()
        # SWE-bench Pro instances provide a dockerhub_tag field pointing to jefzda/sweap-images
        dockerhub_tag = instance.get("dockerhub_tag", None)
        if dockerhub_tag is not None:
            image_name = f"docker.io/jefzda/sweap-images:{dockerhub_tag}"
        else:
            # Docker doesn't allow double underscore, so we replace them with a magic token
            iid = instance["instance_id"]
            id_docker_compatible = iid.replace("__", "_1776_")
            image_name = f"docker.io/swebench/sweb.eval.x86_64.{id_docker_compatible}:latest".lower()
    return image_name

def write_spt_log(spt_entries: list[dict], output_dir: str) -> None:
    """Write a human-readable SPT log to <output_dir>/spt_log.json.

    Each entry records the transformation name, the file it was applied to
    (relative to the repo root), and the positions (line/column) of every
    candidate node that was actually transformed.  Entries are in chronological
    order so the log can be replayed to re-create the same perturbed version.
    """
    os.makedirs(output_dir, exist_ok=True)
    log_path = os.path.join(output_dir, "spt_log.json")
    numbered = [{"order": i + 1, **entry} for i, entry in enumerate(spt_entries)]
    with open(log_path, "w", encoding="utf-8") as fh:
        json.dump(numbered, fh, indent=2)
    print(f"SPT log written to {log_path} ({len(numbered)} entries)")


def collect_python_files(
    dir_path: str | Path,
    exclude_files: set[str] | None = None,
    exclude_patterns: list[str] | None = None,
) -> tuple[list[Path], dict[str, int]]:
    """
    Recursively collect all Python files in the given directory.

    Parameters
    ----------
    dir_path : Union[str, Path]
        Path to the directory.
    exclude_files : Set[str], optional
        Set of file paths, relative to `dir_path`, to exclude.
    exclude_patterns : List[str], optional
        Set of **glob-style** patterns for excluding subdirectories in `dir_path`.

    Returns
    -------
    python_file_paths : List[Path]
        List of the paths of Python files.
     collection_results : Dict[str, int]
        Statistics about the results of the file path collection.
    """
    dir_path = Path(dir_path).resolve()
    exclude_files = exclude_files or set()
    exclude_patterns = exclude_patterns or set()

    collected_files: list[Path] = []
    total_files = 0
    excluded_files = 0
    ignored_files = 0

    for path in dir_path.rglob("*.py"):
        total_files += 1
        relative_path = path.relative_to(dir_path)

        if any(
            part in DEFAULT_EXCLUDED_DIRS or part.startswith(".")
            for part in relative_path.parts
        ):
            ignored_files += 1
            continue

        as_str = str(relative_path)

        if as_str in exclude_files:
            excluded_files += 1
            continue

        if any(fnmatch.fnmatch(as_str, pattern) for pattern in exclude_patterns):
            excluded_files += 1
            continue

        collected_files.append(path)

    collection_results = {
        "total_files": total_files,
        "excluded_files": excluded_files,
        "ignored_files": ignored_files,
    }

    return collected_files, collection_results
