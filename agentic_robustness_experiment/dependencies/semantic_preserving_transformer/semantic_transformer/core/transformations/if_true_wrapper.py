import random
import secrets
import string
from typing import Set, Tuple

import libcst as cst
import libcst.metadata as metadata
import numpy as np


class IfTrueWrapper(cst.CSTTransformer):
    """
    Wraps an indented block with a condiiton logically equivelant to
    if True:
    """

    METADATA_DEPENDENCIES = (metadata.PositionProvider,)

    def __init__(
        self,
        target_coords: Set[Tuple[int, int]],
        theta: float = 1.0,
        rng_seed: int = secrets.randbits(64),
        expression_complexity: int = 3,
    ):

        super().__init__()

        self.gen = LogicalExpressionExpander(rng_seed)
        self.expression_complexity = expression_complexity

        self.target_coords = target_coords

        if theta > 1.0 or theta <= 0:
            raise ValueError("The injection probability must be between 0 and 1.")
        if rng_seed < 0:
            raise ValueError("The rng seed must be a positive integer.")

        self.rng_seed = rng_seed
        self.theta = theta
        self.rng = np.random.default_rng(self.rng_seed)

    def leave_IndentedBlock(self, original_node, updated_node):

        if not updated_node.body:
            return updated_node

        pos = self.get_metadata(metadata.PositionProvider, original_node.body[0])
        current_coord = (pos.start.line, pos.start.column)

        if current_coord not in self.target_coords:
            return updated_node

        if self.rng.random() > self.theta:
            return updated_node

        expr = cst.parse_expression(self.gen.generate(self.expression_complexity))

        new_if = cst.If(
            test=expr,
            body=updated_node,
        )

        return cst.IndentedBlock(body=[new_if])


class LogicalExpressionExpander:
    """
    This helper class randomly expands logical expressions

     Ex. generate(complexity = 3, expression = "True")
         -> ( ( ( ( False ^ False ) ^ ( not ( 8 <= 26 ) ) ) or ( ( not ( 11 < 4 ) ) and ( 28 > 3 ) ) ) and ( "abop" < "ijnkj" ) and True )
    """

    INT_MAX = 30
    STR_MAX_LEN = 15

    def __init__(self, rngseed):
        self.seed = rngseed
        random.seed(self.seed)

    def generate(self, complexity, expression: str = "True"):
        return self.generate_helper(complexity, "True")

    def generate_helper(self, complexity, curr):
        if complexity == 0:
            return curr
        if complexity == 1:
            return self.last_pass(curr)
        arr = curr.split()
        for i, s in enumerate(arr):
            if s in (self.T, self.F):
                truth = s == self.T
                arr[i] = f"( {random.choice(self.EXPANSIONS[truth])} )"
        curr = " ".join(arr)
        complexity -= 1
        return self.generate_helper(complexity, curr)

    def last_pass(self, curr):
        func = [
            self.int_equality,
            self.str_equality,
            self.atomic_prop,
            self.bitwise_prop,
            self.arithmetic_prop,
        ]

        arr = curr.split()
        for i, s in enumerate(arr):
            if s in (self.T, self.F):
                truth = s == self.T
                arr[i] = random.choice(func)(truth)
        return " ".join(arr)

    def atomic_prop(self, truth):
        return f"( {random.choice(self.EXPANSIONS[truth])} )"

    def int_equality(self, truth):
        op = random.choice([">", ">=", "<", "<=", "=="])
        x = random.randint(0, self.INT_MAX)
        y = random.randint(0, self.INT_MAX)
        return self.build_equality(x, y, op, truth)

    def str_equality(self, truth):
        op = random.choice([">", ">=", "<", "<=", "=="])
        x = f" '{self.random_string(1, self.STR_MAX_LEN)}' "
        y = f" '{self.random_string(1, self.STR_MAX_LEN)}' "
        return self.build_equality(x, y, op, truth)

    def bitwise_prop(self, truth):
        x = random.randint(1, 15)
        y = random.randint(1, 15)
        op = random.choice(["&", "|", "^"])
        res = eval(f"{x} {op} {y}")
        if truth:
            return f"( ({x} {op} {y}) == {res} )"
        else:
            return f"( ({x} {op} {y}) == {res + 1} )"

    def arithmetic_prop(self, truth):
        x = random.randint(1, 100)
        y = random.randint(1, 100)
        z = random.randint(1, 100)
        op = random.choice(["+", "-", "*", "%"])
        equality = random.choice([">", ">=", "<", "<=", "=="])
        expression = f"( {x} {op} {y} {equality} {z} )"
        if eval(expression) == truth:
            return expression
        else:
            return f"( not {expression} )"

    def build_equality(self, x, y, op, truth):
        expression = f"( {x} {op} {y} )"
        if eval(expression) == truth:
            return expression
        else:
            return f"( not {expression} )"

    def random_string(self, n, k):
        length = random.randint(n, k)
        characters = string.ascii_letters + string.digits
        return "".join(random.choice(characters) for _ in range(length))

    T = "True"
    F = "False"

    EXPANSIONS = {
        True: [
            "True and True",
            "True or True",
            "True or False",
            "False or True",
            "not False",
            "True ^ False",
            "False ^ True",
        ],
        False: [
            "False and True",
            "True and False",
            "False or False",
            "not True",
            "False ^ False",
            "True ^ True",
        ],
    }
