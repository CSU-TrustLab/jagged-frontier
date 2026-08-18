<a href="https://github.com/astral-sh/ruff"><img alt="Ruff" src="https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json"></a>
## Overview

**Semantic_Preserving_Transformation** is a Python library for performing **semantic-preserving code transformations**. Built on top of LibCST, it performs structural refactoring while strictly preserving comments, whitespace, and formatting. Each transformation ensures that the functionality of the code remains unchanged while rewriting its structure.

## Available Transformations
| Transformation Name | CLI ID (`-t`) | Description |
| :--- | :--- | :--- |
| If/Else Block Swap | `if_else_swap` | Swaps the `if` and `else` blocks and inverts the condition. |
| For-to-While Conversion | `for_loop_rewriting` | Converts `for` loops into equivalent `while` loops. |
| While Loop Unrolling | `while_loop_unrolling` | Unrolls the body of `while` loops one time |
| Operand Permutation [DO NOT USE/ IN DEVELOPMENT] | `commutative_operand_permuter` | Reorders commutative operands (e.g., `a + b` -> `b + a`). |
| Comparison Swapper | `comparison_swapper` | Swaps comparison sides (e.g., `a < b` -> `b > a`). |
| And Condition Splitter| ` and_condition_splitter`  | Splits an AND condition into equivalent nested if statments|
| If True Wrapper|  `if_true_wrapper` | Wraps indented blocks in statements equivalent to: "If True" |
| Try Except Injector|  `"try_except_injector"` | Wraps indented blocks in Try Except |
| Dead Code Injector|  `"dead_code_injector"` | Inserts a semantically neutral "if False" block inside valid indented blocks|
| Double Negation Injector|  `"double_negation_injector"` | Wraps if and while conditions with 'not not' without changing program semantics|
| String Literal Splitter| `"string_literal_splitter"` | Splits string literals at a random point |
| Dead String Assignment| `"dead_string_assignment"` | Inserts a semantically neutral `<name> = "<keyword>"` assignment before valid statements inside function bodies, using a fresh local name that does not shadow any existing binding |
| Dead Method Injection| `"dead_method_injection"` | Inserts a method definition at the end of valid class |


## Architecture
This library uses a Stateless Batch Pipeline to ensure robustness:

Collection: A visitor scans the source code and identifies valid transformation candidates (returning coordinates).

Selection: Strategies (like random sampling) select targets based on the configured probabilities.

Batch Transformation: All selected transformations are applied in a single pass.

## Installation

### Docker (recommended)

```bash
docker build -t semantic-transformer .
```

Run a transformation on a local directory:

```bash
docker run --rm -v /path/to/repo:/app/target -w /app semantic-transformer \
    python docker_transform_repo.py --src /app/target --transformation if_else_swap
```

### Local

```bash
pip install -e .
```

## Examples

In the examples folder, there is an example of transformation. To execute an example file, run:
```bash
python examples/if_else_swap_demo.py
```

## Testing

#### Quick Evaluation of transformations

We use the `evalplus` library to test the transformation.
First, transform the code of the `mbpp` and `humaneval` datasets with a given transformation function `trans_name` with
```python
python tests/test_transformation.py --transformation trans_name
```
This will create two jsonl files: "humaneval.jsonl" and "mbpp.jsonl".
These files contain the transformed solutions in a format expected by `evalplus`.
Then, run the test cases of the two test suites with:
```bash
evalplus.evaluate --test-details --dataset humaneval --samples humaneval.jsonl
evalplus.evaluate --test-details --dataset mbpp --samples mbpp.jsonl
```

The two resulting evaluation reports can be found at "humaneval_eval_results.json" and "mbpp_eval_results.json"

To analyze and get any failed test cases, run the following script:
```bash
python tests/evaluation_report_analyzer.py
```

The report may show some failed tests.
We need to validate those failed cases in case there is any false positive.
We can do so with the by the `validate_failed_tests.py` script, as shown below:

```bash
python tests/validate_failed_tests.py
```

The report will inform us if there is any true positive cases where the tranformation evaluated has introduced a bug.

### Transformation Stress Testing

For more extensive testing, we also provide a script that validates a given transformation by comparing the testing results any original GitHub repository and its transformed counterpart.
The script handles all the execution pipeline automatically.
Specifically, it performs the following steps:
- Pull the source GitHub repository.
- Install its **Python only** required dependencies (with .venv).
- Execute `pytest` to generate the initial testing results.
- Transform the repository, either inplace or at a specific location.
- Similarly as before, execute `pytest` to the transformed repository.

Once done, the user can then examine and compare the testing results.
Theoritically, the latter should *equivalent*, if not identical.
Here a usage template:
```bash
python tests/test_transformations_on_repo.py \
--repo-url GITHUT_REPO_URL \
--work-dir test_my_transfo_on_repo \
--transformation my_transformation_name \
--inplace # optional
```
**Note:** In some cases, the default python recurssion limit may need to be increased in order for LibCST to properly process deeply nested structures.

