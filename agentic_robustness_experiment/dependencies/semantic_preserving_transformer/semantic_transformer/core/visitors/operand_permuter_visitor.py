from typing import List, Tuple

import libcst as cst
import libcst.matchers as m
import libcst.metadata as metadata

BINARY_OPERAND_TYPES = (
    m.Add(),
    m.Multiply(),
    m.BitAnd(),
    m.BitOr(),
    m.BitXor(),
)


class OperandPermuterVisitor(cst.CSTVisitor):
    METADATA_DEPENDENCIES = (metadata.PositionProvider,)

    def __init__(self):
        self.candidates: List[Tuple[int, int]] = []

    def visit_BinaryOperation(self, node: cst.BinaryOperation):
        if m.matches(node.operator, m.OneOf(*BINARY_OPERAND_TYPES)):
            pos = self.get_metadata(metadata.PositionProvider, node)
            self.candidates.append((pos.start.line, pos.start.column))
