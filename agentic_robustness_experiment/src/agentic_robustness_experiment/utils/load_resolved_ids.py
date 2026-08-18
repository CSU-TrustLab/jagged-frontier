import os


def load_resolved_ids(file_path: str) -> list[str]:
    """
    Reads instance IDs from a file.
    Assumes one ID per line. Skips empty lines and comments (#).
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Resolved IDs file not found at: {file_path}")

    ids = []
    with open(file_path) as f:
        for line in f:
            cleaned_line = line.strip()

            if not cleaned_line or cleaned_line.startswith("#"):
                continue

            ids.append(cleaned_line)

    print(f"Loaded {len(ids)} resolved IDs from {file_path}")
    return ids
