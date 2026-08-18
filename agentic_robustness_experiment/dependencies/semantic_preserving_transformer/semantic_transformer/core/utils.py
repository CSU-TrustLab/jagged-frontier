import builtins
import fnmatch
import keyword
import random
import string
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple, Union

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
try:
    from nltk.corpus import wordnet
except Exception:
    wordnet = None

reserved_keywords = set(keyword.kwlist)
reserved_keywords |= set(dir(builtins))
reserved_keywords |= {
    "self",
    "cls",
    "__init__",
    "__new__",
    "__str__",
    "__repr__",
    "__call__",
    "__iter__",
    "__next__",
    "__enter__",
    "__exit__",
    "__getitem__",
    "__setitem__",
    "__delitem__",
    "__dict__",
    "__class__",
    "__name__",
    "__main__",
    "__file__",
}


def generate_random_name(length: int = 5) -> str:
    """Generate a random lowercase name of given length."""
    letters = string.ascii_lowercase
    return "".join(random.choice(letters) for _ in range(length))


def synonym_or_random_name(original: str, length: int = 5) -> str:
    """
    Try to pick a synonym from wordnet.
    If no such synonym exists or wordnet not available, fallback to random.
    """
    if wordnet:
        try:
            synsets = wordnet.synsets(original)
            lemmas = [lemma.name().replace("-", "_") for s in synsets for lemma in s.lemmas()]
            lemmas = [l for l in set(lemmas) if l.isidentifier() and not keyword.iskeyword(l)]
            if lemmas:
                return random.choice(list(lemmas))
        except Exception:
            pass
    return generate_random_name(length)


def is_reserved(keyword: str) -> bool:
    return keyword in reserved_keywords


def collect_python_files(
    dir_path: Union[str, Path],
    exclude_files: Optional[Set[str]] = None,
    exclude_patterns: Optional[List[str]] = None,
) -> Tuple[List[Path], Dict[str, int]]:
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

    collected_files: List[Path] = []
    total_files = 0
    excluded_files = 0
    ignored_files = 0

    for path in dir_path.rglob("*.py"):
        total_files += 1
        relative_path = path.relative_to(dir_path)

        if any(
            part in DEFAULT_EXCLUDED_DIRS or part.startswith(".") for part in relative_path.parts
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
