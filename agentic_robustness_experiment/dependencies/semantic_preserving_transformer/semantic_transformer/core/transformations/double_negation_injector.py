from typing import Set, Tuple

import libcst as cst
from libcst.metadata import PositionProvider


class DoubleNegationInjector(cst.CSTTransformer):
    METADATA_DEPENDENCIES = (PositionProvider,)

    def __init__(self, target_coords: Set[Tuple[int, int]]):
        super().__init__()
        self.target_coords = target_coords

    def _coord(self, node):
        pos = self.get_metadata(PositionProvider, node)
        return (pos.start.line, pos.start.column)

    def _should_skip(self, test_expr: cst.BaseExpression) -> bool:
        if isinstance(test_expr, cst.Name) and test_expr.value in {"True", "False"}:
            return True

        if isinstance(test_expr, cst.UnaryOperation) and isinstance(test_expr.operator, cst.Not):
            return True

        return False

    def _wrap_double_not(self, expr: cst.BaseExpression) -> cst.BaseExpression:
        parenthesized = cst.ensure_type(
            expr.with_changes(
                lpar=[cst.LeftParen()],
                rpar=[cst.RightParen()],
            ),
            cst.BaseExpression,
        )

        return cst.UnaryOperation(
            operator=cst.Not(),
            expression=cst.UnaryOperation(
                operator=cst.Not(),
                expression=parenthesized,
            ),
        )

    def leave_If(self, original_node, updated_node):
        coord = self._coord(original_node)

        if coord not in self.target_coords:
            return updated_node

        if self._should_skip(updated_node.test):
            return updated_node

        return updated_node.with_changes(test=self._wrap_double_not(updated_node.test))

    def leave_While(self, original_node, updated_node):
        coord = self._coord(original_node)

        if coord not in self.target_coords:
            return updated_node

        if self._should_skip(updated_node.test):
            return updated_node

        return updated_node.with_changes(test=self._wrap_double_not(updated_node.test))
