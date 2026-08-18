import json

from evalplus.data import get_mbpp_plus
from tests.parse_human_eval_plus import get_humaneval_merged_solutions


def analyze_report(file_name, output_name="failure_analysis.json", is_humaneval: bool = False):
    data = None
    testcase_passed = 0
    testcase_failed = 0
    report = {"passed": 0, "failed": 0, "report": []}
    try:
        with open(file_name, encoding="utf-8") as f:
            data = json.load(f)
        print("JSON data loaded successfully")
    except FileNotFoundError:
        print(f"Error: The file '{file_name}' was not found.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

    if not data:
        return

    results = data.get("eval")
    testcase_passed = len(results)
    all_failed_testcases = []
    all_testcases = {}
    if is_humaneval:
        all_testcases = get_humaneval_merged_solutions()
    else:
        all_testcases = get_mbpp_plus()

    for task_id in results:
        result = results.get(task_id)
        if result[0].get("base_status") == "fail" or result[0].get("plus_status") == "fail":
            testcase_failed += 1
            failed_testcases = result[0].get("base_fail_tests") + result[0].get("plus_fail_tests")
            solution = result[0].get("solution")
            if is_humaneval:
                cannonical_solution = all_testcases.get(task_id)
            else:
                cannonical_solution = all_testcases.get(task_id).get("canonical_solution")

            jsonObj = {
                "taskid": task_id,
                "base_solution": cannonical_solution,
                "transformed_solution": solution,
                "failed_tests": failed_testcases,
            }
            all_failed_testcases.append(jsonObj)

    testcase_passed -= testcase_failed
    with open(output_name, "w", encoding="utf-8") as outf:
        json.dump(all_failed_testcases, outf, indent=2, ensure_ascii=False)
    print("-------------------Report--------------------")
    print("Total transformation: ", testcase_passed + testcase_failed)
    print("All test cases passed: ", testcase_passed)
    print("Test cases failed: ", testcase_failed)
    print(f"Report written to {output_name}.")


def run_analysis(*, is_humaneval_dataset: bool):
    if is_humaneval_dataset:
        file_name = "./humaneval_eval_results.json"
        analyze_report(
            file_name=file_name,
            output_name="humaneval_failure_analysis.json",
            is_humaneval=True,
        )
    else:
        file_name = "./mbpp_eval_results.json"
        analyze_report(file_name=file_name, output_name="mbpp_failure_analysis.json")


def main():
    run_analysis(is_humaneval_dataset=False)
    run_analysis(is_humaneval_dataset=True)


if __name__ == "__main__":
    main()
