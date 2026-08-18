import json
import re

# Per-instance path prefixes to strip from patches before application.
INSTANCE_STRIP_PREFIXES: dict[str, list[str]] = {
    "psf__requests-1142": ["build/"],
}


def strip_bad_hunks(patch: str, prefixes: list[str]) -> str:
    blocks = re.split(r"(?=^diff --git )", patch, flags=re.MULTILINE)
    return "".join(
        b for b in blocks
        if not any(re.match(rf"diff --git a/{re.escape(p)}", b) for p in prefixes)
    )


def sanitize_preds_json(preds_file: str, instance_id: str) -> None:
    """Strip bad diff hunks from model_patch in preds.json in-place."""
    prefixes = INSTANCE_STRIP_PREFIXES.get(instance_id, [])
    if not prefixes:
        return
    with open(preds_file) as f:
        data = json.load(f)
    if instance_id not in data:
        return
    data[instance_id]["model_patch"] = strip_bad_hunks(
        data[instance_id]["model_patch"], prefixes
    )
    with open(preds_file, "w") as f:
        json.dump(data, f, indent=2)
    print(f"Sanitized preds.json for {instance_id}: stripped prefixes {prefixes}")


def extract_patch(json_file, output_file, instance_id):
    with open(json_file) as f:
        data = json.load(f)

    if instance_id in data:
        patch_content = data[instance_id]["model_patch"]

        with open(output_file, "w") as out:
            out.write(patch_content)

        print(f"Successfully saved patch to {output_file}")
    else:
        print(f"Key {instance_id} not found in JSON.")
