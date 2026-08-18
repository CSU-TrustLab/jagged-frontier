from typing import List, Optional, Tuple

import libcst as cst
import libcst.metadata as metadata


class DoubleNegationVisitor(cst.CSTVisitor):
    METADATA_DEPENDENCIES = (metadata.PositionProvider,)

    def __init__(self):
        self.candidates: List[Tuple[int, int]] = []

    def _should_skip(self, expr: cst.BaseExpression) -> bool:
        if isinstance(expr, cst.Name) and expr.value in {"True", "False"}:
            return True
        if isinstance(expr, cst.UnaryOperation) and isinstance(expr.operator, cst.Not):
            return True

        return False

    def visit_If(self, node: cst.If) -> Optional[bool]:
        if self._should_skip(node.test):
            return

        pos = self.get_metadata(metadata.PositionProvider, node)
        self.candidates.append((pos.start.line, pos.start.column))

    def visit_While(self, node: cst.While) -> Optional[bool]:
        if self._should_skip(node.test):
            return

        pos = self.get_metadata(metadata.PositionProvider, node)
        self.candidates.append((pos.start.line, pos.start.column))
