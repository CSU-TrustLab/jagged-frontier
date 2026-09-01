"""Utilities for summarizing unified diffs."""


def summarize_diff(diff_text: str) -> dict:
    """Return changed files and added/removed line counts for a unified diff."""
    added = 0
    removed = 0
    files = set()

    for line in diff_text.splitlines():
        if line.startswith("+++ b/"):
            files.add(line.removeprefix("+++ b/").strip())
        elif line.startswith("+") and not line.startswith("+++"):
            added += 1
        elif line.startswith("-") and not line.startswith("---"):
            removed += 1

    return {
        "files_modified": sorted(files),
        "lines_added": added,
        "lines_removed": removed,
    }
