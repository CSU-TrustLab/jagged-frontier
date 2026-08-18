import ast
import json
import textwrap
from typing import Any, Dict, Optional

from evalplus.data import get_human_eval_plus


def clean_field(value: Any) -> str:
    if not value:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        return "\n".join(x for x in value if isinstance(x, str))
    return str(value)


def contains_def_or_class(code: str) -> bool:
    if not code:
        return False
    try:
        tree = ast.parse(code)
    except Exception:
        return False
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            return True
    return False


def get_first_function_node(code: str) -> Optional[ast.FunctionDef]:
    try:
        tree = ast.parse(code)
    except Exception:
        return None
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            return node
    return None


def merge_prompt_and_solution(prompt: str, solution: str) -> str:
    prompt = prompt or ""
    solution = solution or ""
    prompt_lines = prompt.splitlines()

    func = get_first_function_node(prompt)
    if func is None:
        return ""

    header_end_lineno = None
    if func.body:
        first = func.body[0]
        if (
            isinstance(first, ast.Expr)
            and isinstance(getattr(first, "value", None), ast.Constant)
            and isinstance(first.value.value, str)
        ):
            header_end_lineno = getattr(first, "end_lineno", None) or getattr(first, "lineno", None)
    if header_end_lineno is None:
        header_end_lineno = getattr(func, "lineno", 1)

    header_lines = prompt_lines[:header_end_lineno]
    trailing_lines = prompt_lines[header_end_lineno:]

    indent = " " * 4
    dedented_body = textwrap.dedent(solution).rstrip()
    if dedented_body == "":
        merged = "\n".join(prompt_lines) + (
            "\n" if prompt_lines and not prompt_lines[-1].endswith("\n") else ""
        )
        return merged, None

    body_lines = dedented_body.splitlines()
    indented_body = [(indent + ln if ln.strip() else "") for ln in body_lines]

    cleaned_trailing = list(trailing_lines)
    merged_lines = header_lines + indented_body + cleaned_trailing
    merged_code = "\n".join(merged_lines) + "\n"
    return merged_code


def get_humaneval_merged_solutions(validate_syntax: bool = True) -> Dict[str, str]:
    problems = get_human_eval_plus()
    results: Dict[str, str] = {}

    for task_id, prob in problems.items():
        prompt = clean_field(prob.get("prompt"))
        canonical = clean_field(prob.get("canonical_solution"))

        if contains_def_or_class(canonical):
            source_candidate = canonical
            if validate_syntax:
                try:
                    ast.parse(source_candidate)
                except Exception:
                    continue
            results[task_id] = source_candidate
            continue

        function_in_prompt = get_first_function_node(prompt)
        if function_in_prompt is not None and canonical.strip():
            merged = merge_prompt_and_solution(prompt, canonical)
            if len(merged) == 0:
                continue
            if validate_syntax:
                try:
                    ast.parse(merged)
                except Exception:
                    continue
            results[task_id] = merged
            continue

        continue

    return results


def get_humaneval_merged_solutions_in_json(solutions: Dict[str, str]):
    print(solutions.keys())
    with open("human_eval_modified.jsonl", "w", encoding="utf-8") as f:
        for key in solutions.keys():
            solution = solutions.get(key)
            data = {"task_id": key, "original_solution": solution}
        f.write(json.dumps(data) + "\n")


if __name__ == "__main__":
    get_humaneval_merged_solutions_in_json(get_humaneval_merged_solutions())
