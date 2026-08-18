#!/usr/bin/env python3
"""
Convert a pred.json file (keyed by instance_id) into the list format
expected by swe_bench_pro_eval.py.

Input format:
    {
        "<instance_id>": {
            "instance_id": "...",
            "model_patch": "...",
            "model_name_or_path": "..."
        },
        ...
    }

Output format:
    [
        {"instance_id": "...", "model_patch": "...", "model_name_or_path": "..."},
        ...
    ]

Usage:
    python helper_code/convert_pred_json.py --input traj/claude-opus-4-5-20251101/pred.json --output preds.json
"""

import argparse
import json


def convert(input_path: str, output_path: str) -> None:
    with open(input_path) as f:
        data = json.load(f)

    if isinstance(data, list):
        print(f"Input is already a list ({len(data)} entries), writing as-is.")
        patches = data
    elif isinstance(data, dict):
        patches = []
        for key, value in data.items():
            if isinstance(value, dict):
                entry = {
                    "instance_id": value.get("instance_id", key),
                    "model_patch": value.get("model_patch", value.get("patch", "")),
                }
                if "model_name_or_path" in value:
                    entry["model_name_or_path"] = value["model_name_or_path"]
                patches.append(entry)
            else:
                print(f"Warning: skipping unexpected value type for key '{key}'")
    else:
        raise ValueError(f"Unexpected top-level JSON type: {type(data)}")

    with open(output_path, "w") as f:
        json.dump(patches, f, indent=2)

    print(f"Wrote {len(patches)} patches to {output_path}")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input", required=True, help="Path to the input pred.json file")
    parser.add_argument("--output", required=True, help="Path for the output JSON file")
    args = parser.parse_args()
    convert(args.input, args.output)


if __name__ == "__main__":
    main()
