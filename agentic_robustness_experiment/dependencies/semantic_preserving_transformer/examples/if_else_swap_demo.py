from semantic_transformer.api import apply_transformation, collect_candidate_nodes

source_code = """
def test_if_else(x, y):
    if x > 10:
        a = 0
    else:
        a = - 1

    if x > 0:
        a = 1
    elif x < 10:
        a = 2
    else:
        a = 3

    b = 5 if x < 0 else 10
"""

print("=== if else swap===")

candidates = collect_candidate_nodes(source_code, "if_else_swap")
print(f"Found {len(candidates)} candidate nodes.")
apply_transformation(source_code, candidates, "if_else_swap")
