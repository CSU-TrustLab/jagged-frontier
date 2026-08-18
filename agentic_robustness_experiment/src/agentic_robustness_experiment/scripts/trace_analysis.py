import json
from collections import defaultdict
from pathlib import Path


def extract_issue_id(dir_name):
    parts = dir_name.rsplit("_", 1)

    if len(parts) > 1:
        return parts[0]

    return dir_name


def classify_run(report_dir):
    """Classify a single run's outcome as resolved/unresolved/error/empty.

    Two report formats exist in the wild:
      - opencode/SWE-bench harness: report/<model>.<instance>.json with
        aggregate counters (resolved_instances, error_instances, ...),
        alongside a per-instance report/report.json.
      - minisweagent: only report/report.json, keyed by instance id, with
        booleans (resolved, patch_is_None, patch_successfully_applied, ...).
    If neither is present, the run never produced a report at all (the
    harness itself crashed), which is treated as an error.
    """
    json_files = sorted(report_dir.glob("*.json")) if report_dir.exists() else []
    aggregate_files = [f for f in json_files if f.name != "report.json"]

    if aggregate_files:
        with open(aggregate_files[0]) as f:
            data = json.load(f)

        if data.get("resolved_instances", 0) > 0:
            return "resolved"
        elif data.get("error_instances", 0) > 0:
            return "error"
        elif data.get("empty_patch_instances", 0) > 0:
            return "empty"
        return "unresolved"

    per_instance_file = report_dir / "report.json"
    if per_instance_file.exists():
        with open(per_instance_file) as f:
            data = json.load(f)

        if not data:
            return "error"

        result = next(iter(data.values()))
        if result.get("patch_is_None") or not result.get("patch_exists", True):
            return "empty"
        elif not result.get("patch_successfully_applied", True):
            return "error"
        elif result.get("resolved"):
            return "resolved"
        return "unresolved"

    return "error"


def analyze_traces_grouped(root_dir_name="traces"):
    root_path = Path(root_dir_name)

    if not root_path.exists():
        print(f"Error: Directory '{root_dir_name}' not found.")
        return

    grouped_stats = defaultdict(
        lambda: {"total": 0, "resolved": 0, "unresolved": 0, "error": 0, "empty": 0}
    )

    failures = []

    print(f"Scanning '{root_dir_name}' and grouping by transformation...\n")

    for preds_file in root_path.rglob("preds.json"):
        run_dir = preds_file.parent
        dir_name = run_dir.name

        trans_name = extract_issue_id(dir_name)

        try:
            outcome = classify_run(run_dir / "report")

            stats = grouped_stats[trans_name]
            stats["total"] += 1
            stats[outcome] += 1

            if outcome != "resolved":
                failures.append((trans_name, outcome.upper(), run_dir))

        except Exception as e:
            print(f"Skipping {run_dir}: {e}")

    print(
        f"{'Issue ID':<35} | {'TOTAL':<6} | {'RESOLVED':<8} | {'UNRESOLVED':<10} | {'ERROR':<5} | {'RESOLVE PERCENTAGE':<18}"
    )
    print("-" * 85)

    grand_total = 0
    grand_resolved = 0

    for name, s in sorted(grouped_stats.items()):
        success_rate = (s["resolved"] / s["total"]) * 100 if s["total"] > 0 else 0.0
        print(
            f"{name:<35} | {s['total']:<6} | {s['resolved']:<8} | {s['unresolved']:<10} | {s['error']:<5} | {success_rate:.1f}%"
        )

        grand_total += s["total"]
        grand_resolved += s["resolved"]

    print("-" * 85)
    if grand_total > 0:
        total_rate = (grand_resolved / grand_total) * 100
        print(
            f"{'TOTAL AGGREGATE':<35} | {grand_total:<6} | {grand_resolved:<8} | {grand_total - grand_resolved:<10} | -    | {total_rate:.1f}%"
        )

    print("\n" + "=" * 85)



def extract_instance_id(eval_data, instance_dir):
    """Extract the canonical instance ID from eval_results.json keys."""
    if eval_data:
        return next(iter(eval_data))
    # Fallback: directory name contains instance_id repeated twice; take first half.
    name = instance_dir.name
    parts = name.split("_")
    mid = len(parts) // 2
    return "_".join(parts[:mid])


def analyze_swepro_traces(root_dir_name="swepro_traces/kimi_vanilla"):
    root_path = Path(root_dir_name)

    if not root_path.exists():
        print(f"Error: Directory '{root_dir_name}' not found.")
        return

    print(f"Scanning '{root_dir_name}' for SWEPro traces...\n")

    grouped = defaultdict(lambda: {"total": 0, "resolved": 0, "unresolved": 0, "error": 0})

    for instance_dir in sorted(root_path.iterdir()):
        if not instance_dir.is_dir():
            continue

        eval_file = instance_dir / "report" / "eval_results.json"

        if not eval_file.exists():
            instance_id = instance_dir.name
            grouped[instance_id]["total"] += 1
            grouped[instance_id]["error"] += 1
            continue

        try:
            with open(eval_file) as f:
                data = json.load(f)

            instance_id = extract_instance_id(data, instance_dir)
            grouped[instance_id]["total"] += 1

            if any(v for v in data.values()):
                grouped[instance_id]["resolved"] += 1
            else:
                grouped[instance_id]["unresolved"] += 1

        except Exception as e:
            print(f"Skipping {eval_file}: {e}")
            grouped[instance_dir.name]["total"] += 1
            grouped[instance_dir.name]["error"] += 1

    if not grouped:
        print("No instance directories found.")
        return

    col_w = 85
    print(f"{'Instance ID':<55} | {'RUNS':<4} | {'RESOLVED':<8} | {'UNRESOLVED':<10} | {'ERROR':<5} | {'RATE':<6}")
    print("-" * col_w)

    grand_total = grand_resolved = 0

    for instance_id, s in sorted(grouped.items()):
        rate = s["resolved"] / s["total"] * 100 if s["total"] > 0 else 0.0
        print(
            f"{instance_id[:45]:<55} | {s['total']:<4} | {s['resolved']:<8} | {s['unresolved']:<10} | {s['error']:<5} | {rate:.0f}%"
        )
        grand_total += s["total"]
        grand_resolved += s["resolved"]

    print("-" * col_w)
    total_rate = grand_resolved / grand_total * 100 if grand_total > 0 else 0.0
    grand_unresolved = sum(s["unresolved"] for s in grouped.values())
    grand_error = sum(s["error"] for s in grouped.values())
    print(
        f"{'TOTAL':<55} | {grand_total:<4} | {grand_resolved:<8} | {grand_unresolved:<10} | {grand_error:<5} | {total_rate:.1f}%"
    )
    print("\n" + "=" * col_w)


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Analyze agent trace results.")
    parser.add_argument("traces_dir", help="Path to the traces directory")
    parser.add_argument(
        "--swepro",
        action="store_true",
        help="Analyze SWEPro-format traces (report/eval_results.json with boolean results)",
    )
    args = parser.parse_args()

    if args.swepro:
        analyze_swepro_traces(args.traces_dir)
    else:
        analyze_traces_grouped(args.traces_dir)


if __name__ == "__main__":
    main()
