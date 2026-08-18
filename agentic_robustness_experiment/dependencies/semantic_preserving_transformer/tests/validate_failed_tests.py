import json
import re
from typing import Any, List


def find_entry_point(code_string: str) -> str:
    function_names = re.findall(r"def\s+([a-zA-Z_][a-zA-Z0-9_]*)\s*\(", code_string)

    if function_names:
        return function_names[-1]

    return None


def run_and_compare(
    func1_code: str, func2_code: str, func_name: str, testcases: List[List[Any]]
) -> List[dict]:
    ns1, ns2 = {}, {}

    try:
        exec(func1_code, ns1)
        exec(func2_code, ns2)
    except Exception as e:
        print(f"Error during code execution: {e}")
        return [], []

    func1 = ns1.get(func_name)
    func2 = ns2.get(func_name)

    if not func1 or not func2:
        print(f"Error: Could not find entry point function '{func_name}' in both solutions.")
        return [], []

    results = []
    mismatched_tests = []

    for i, testcase in enumerate(testcases):
        call_args = testcase
        out1, out2 = None, None
        err1, err2 = None, None
        error_message = ""

        try:
            out1 = func1(*call_args)
        except Exception as e:
            err1 = e

        try:
            out2 = func2(*call_args)
        except Exception as e:
            err2 = e

        if err1 is not None:
            error_message = f"Original Code Runtime Error: {type(err1).__name__}: {str(err1)}"
        if err2 is not None:
            error_message += f"Transformed Code Runtime Error: {type(err2).__name__}: {str(err2)}"

        if err1 or err2:
            mismatch_info = {
                "test_input": testcase,
                "error": error_message,
                "original_output": str(err1) if err1 else str(out1),
                "transformed_output": str(err2) if err2 else str(out2),
            }
            mismatched_tests.append(mismatch_info)
            print(f"ERROR (Test {i + 1}): Input={testcases}, {error_message}")
            continue

        match = out1 == out2
        results.append((testcase, out1, out2, match))

        if not match:
            mismatch_info = {
                "test_input": testcase,
                "original_output": str(out1),
                "transformed_output": str(out2),
            }
            mismatched_tests.append(mismatch_info)
            print(f"MISMATCH (Test {i + 1}): Input={testcase}, Original={out1}, Transformed={out2}")

    return mismatched_tests


def validate_json_file(file_path: str):
    try:
        with open(file_path, encoding="utf-8") as f:
            data = json.load(f)
    except FileNotFoundError:
        print(f"Error: File not found at '{file_path}'.")
        return
    except json.JSONDecodeError:
        print(f"Error: Invalid JSON format in '{file_path}'.")
        return

    all_mismatches = []

    for task in data:
        task_id = task.get("taskid", "Unknown Task")
        base_code = task.get("base_solution")
        transformed_code = task.get("transformed_solution")
        failed_tests = task.get("failed_tests", [])

        print("\n========================================================")
        print(f"Validation for Task ID: {task_id}")
        print(f"Total Failed Tests to Re-run: {len(failed_tests)}")

        if not base_code or not transformed_code:
            print("Skipping: Missing base or transformed solution.")
            continue

        func_name = find_entry_point(base_code)
        mismatches = run_and_compare(base_code, transformed_code, func_name, failed_tests)

        if mismatches:
            print(
                f"Summary: {len(mismatches)} tests showed MISMATCHES or ERRORS for Task {task_id}."
            )

            all_mismatches.append(
                {
                    "task_id": task_id,
                    "function_name": func_name,
                    "mismatches": mismatches,
                }
            )
        else:
            print("All re-run failed tests produced matching outputs")

    print("\n")
    print("------------------FINAL MISMATCH SUMMARY----------------------")

    if all_mismatches:
        total_mismatch_count = sum(len(m["mismatches"]) for m in all_mismatches)
        print(f"Total discrepancies found across all tasks: {total_mismatch_count}")

        output_file_name = "mismatch_report.json"
        with open(output_file_name, "w", encoding="utf-8") as f:
            json.dump(all_mismatches, f, indent=4)

        print(f"Detailed report saved to {output_file_name}")

        print("\nTasks with Functional Discrepancies:")
        for m in all_mismatches:
            print(
                f"- {m['task_id']} ('{m['function_name']}'): {len(m['mismatches'])} inputs failed comparison."
            )

    else:
        print("All re-run failed test cases produced identical results.")
    print("-----------------------x--------------------------")


input_json_file = "./humaneval_failure_analysis.json"
validate_json_file(input_json_file)

input_json_file = "./mbpp_failure_analysis.json"
validate_json_file(input_json_file)
