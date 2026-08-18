import argparse
import json

from evalplus.data import get_mbpp_plus
from tests.parse_human_eval_plus import get_humaneval_merged_solutions

from semantic_transformer.api import apply_transformation, collect_candidate_nodes
from semantic_transformer.core.transformation_name import TransformationName


def parse_arguments():
    parser = argparse.ArgumentParser(
        prog="Single-transformation Tester",
        description="Applies a given transformation to the evalplus's `mbpp` and `humaneval` datasets.",
    )
    parser.add_argument(
        "--transformation",
        type=str,
        choices=list(key.value for key in TransformationName),
        help="Transformation to apply",
        default="rename_function_parameters",
        required=True,
    )

    return parser.parse_args()


def run_all_tests_from_mbpp(transformation):
    testcases = get_mbpp_plus()
    with open("mbpp.jsonl", "w", encoding="utf-8") as f:
        for key in testcases.keys():
            testcase = testcases.get(key)
            source_code = testcase.get("canonical_solution")
            transformed_code = transform(source_code, transformation)
            data = {"task_id": key, "solution": transformed_code}
            f.write(json.dumps(data) + "\n")


def run_all_tests_from_human_eval_plus(transformation):
    testcases = get_humaneval_merged_solutions()

    with open("humaneval.jsonl", "w", encoding="utf-8") as f:
        for key in testcases.keys():
            source_code = testcases.get(key)
            transformed_code = transform(source_code, transformation)
            data = {"task_id": key, "solution": transformed_code}
            f.write(json.dumps(data) + "\n")


def transform(source_code: str, transformation_name: str) -> str:
    candidate_coords = collect_candidate_nodes(source_code, transformation_name)
    transformed_code = apply_transformation(source_code, candidate_coords, transformation_name)
    return transformed_code


if __name__ == "__main__":
    args = parse_arguments()
    run_all_tests_from_human_eval_plus(args.transformation)
    run_all_tests_from_mbpp(args.transformation)
