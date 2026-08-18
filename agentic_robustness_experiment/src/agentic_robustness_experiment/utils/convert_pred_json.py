import json


def convert_entry(entry: dict) -> dict:
    result = {
        "instance_id": entry["instance_id"],
        "patch": entry.get("model_patch", entry.get("patch", "")),
    }
    if "model_name_or_path" in entry:
        result["prefix"] = entry["model_name_or_path"].replace("/", "_").replace("\\", "_")
    return result


def convert_json(path: str) -> None:
    with open(path) as f:
        data = json.load(f)

    if isinstance(data, list):
        converted = [convert_entry(e) for e in data]
    elif isinstance(data, dict):
        converted = [convert_entry({**v, "instance_id": v.get("instance_id", k)}) for k, v in data.items()]
    else:
        raise ValueError(f"Unexpected top-level JSON type: {type(data)}")

    with open(path, "w") as f:
        json.dump(converted, f, indent=2)

    print(f"Wrote {len(converted)} patches to {path}")


if __name__ == "__main__":
    convert_json("traces/instance_ansible__ansible-395e5e20fab9cad517243372fa3c3c5d9e09ab2a-v7eee2454f617569fd6889f2211f75bc02a35f9f8_e010d214/preds.json")
