from semantic_transformer.api import apply_transformation, collect_candidate_nodes

source_code = """
def test_if_else(x, y):
    if x > 10:
        a = 0
        print("x is greater than 10")
        b = a^2
        if b > 10:
            print("b is greater than 10")
            c = a + b
        else:
            print("b is less than or equal to 10")
            c = a - b
    else:
        a = - 1
        print("x is less than or equal to 10")
        b = a^2
        print(f"b is {b}")
        print("This is a test of the try-catch injector transformation.")

"""

print("=== try catch injector===")

candidates = collect_candidate_nodes(source_code, "try_except_injector")
print(f"Found {len(candidates)} candidate nodes.")
new_code = apply_transformation(source_code, candidates, "try_except_injector")
print(new_code)
