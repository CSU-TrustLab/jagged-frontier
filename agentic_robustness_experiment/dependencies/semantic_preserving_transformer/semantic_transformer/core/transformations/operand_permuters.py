from typing import Optional, Set, Tuple

import astroid
import libcst as cst
import libcst.matchers as m
import libcst.metadata as metadata


class CommutativeOperandPermuter(cst.CSTTransformer):
    """
    Simple permuter for all commutative binary operations.

    It does not include commutative comparitive operations such as equal.
    """

    METADATA_DEPENDENCIES = (metadata.PositionProvider,)

    ALLOWED_NUMERIC_PYTYPES = {
        "builtins.bool",
        "builtins.int",
        "builtins.float",
        "builtins.complex",
    }

    ALLOWED_BITWISE_PYTYPES = {
        "builtins.bool",
        "builtins.int",
        "builtins.set",
        "builtins.frozenset",
    }

    def __init__(
        self,
        target_coords: Optional[Set[Tuple[int, int]]] = None,
    ):
        self.target_coords = target_coords
        self.binary_operand_types = (
            m.Add(),
            m.Multiply(),
            m.BitAnd(),
            m.BitOr(),
            m.BitXor(),
        )

    def leave_BinaryOperation(self, original_node, updated_node):
        if not self.is_target_node(original_node):
            return updated_node

        if m.matches(updated_node.operator, m.OneOf(*self.binary_operand_types)):
            if not self.can_swap_operation(updated_node):
                return updated_node
            new_left = updated_node.right
            new_right = updated_node.left
            return updated_node.with_changes(left=new_left, right=new_right)

        else:
            return updated_node

    def leave_BooleanOperation(self, original_node, updated_node):
        return updated_node

    def is_target_node(self, original_node: cst.CSTNode) -> bool:
        if self.target_coords is None:
            return True
        pos = self.get_metadata(metadata.PositionProvider, original_node)
        current_coord = (pos.start.line, pos.start.column)
        return current_coord in self.target_coords

    def const_pytype(self, inferred: astroid.Const) -> Optional[str]:
        value = inferred.value
        if isinstance(value, bool):
            return "builtins.bool"
        if isinstance(value, int):
            return "builtins.int"
        if isinstance(value, float):
            return "builtins.float"
        if isinstance(value, complex):
            return "builtins.complex"
        return None

    def infer_only(self, expression: cst.BaseExpression, allowed_pytypes: Set[str]) -> bool:
        try:
            expr_code = cst.Module([]).code_for_node(expression)
            inferred_nodes = list(astroid.extract_node(expr_code).infer())
        except Exception:
            return False
        if not inferred_nodes:
            return False
        uninferable_singleton = getattr(astroid, "Uninferable", None)
        uninferable_base = getattr(getattr(astroid, "util", None), "UninferableBase", None)
        for inferred in inferred_nodes:
            if uninferable_singleton is not None and inferred is uninferable_singleton:
                return False
            if uninferable_base is not None and isinstance(inferred, uninferable_base):
                return False
            if isinstance(inferred, astroid.Const):
                const_type = self.const_pytype(inferred)
                if const_type is None or const_type not in allowed_pytypes:
                    return False
                continue

            pytype_fn = getattr(inferred, "pytype", None)
            if not callable(pytype_fn):
                return False

            try:
                if pytype_fn() not in allowed_pytypes:
                    return False
            except Exception:
                return False
        return True

    def can_swap_addition(self, node: cst.BinaryOperation) -> bool:
        return self.infer_only(node.left, self.ALLOWED_NUMERIC_PYTYPES) and self.infer_only(
            node.right, self.ALLOWED_NUMERIC_PYTYPES
        )

    def can_swap_multiplication(self, node: cst.BinaryOperation) -> bool:
        return self.infer_only(node.left, self.ALLOWED_NUMERIC_PYTYPES) and self.infer_only(
            node.right, self.ALLOWED_NUMERIC_PYTYPES
        )

    def can_swap_bitwise(self, node: cst.BinaryOperation) -> bool:
        return self.infer_only(node.left, self.ALLOWED_BITWISE_PYTYPES) and self.infer_only(
            node.right, self.ALLOWED_BITWISE_PYTYPES
        )

    def can_swap_operation(self, node: cst.BinaryOperation) -> bool:
        operator = node.operator
        if isinstance(operator, cst.Add):
            return self.can_swap_addition(node)
        if isinstance(operator, cst.Multiply):
            return self.can_swap_multiplication(node)
        if isinstance(operator, (cst.BitAnd, cst.BitOr, cst.BitXor)):
            return self.can_swap_bitwise(node)
        return False
