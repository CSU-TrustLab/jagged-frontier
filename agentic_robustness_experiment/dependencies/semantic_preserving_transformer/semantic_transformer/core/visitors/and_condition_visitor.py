from typing import List, Optional, Tuple

import libcst as cst
import libcst.metadata as metadata


class AndConditionVisitor(cst.CSTVisitor):
    METADATA_DEPENDENCIES = (metadata.PositionProvider,)

    def __init__(self):
        self.candidates: List[Tuple[int, int]] = []

    def visit_If(self, node: cst.If) -> Optional[bool]:
        if isinstance(node.test, cst.BooleanOperation) and isinstance(node.test.operator, cst.And):
            pos = self.get_metadata(metadata.PositionProvider, node)
            self.candidates.append((pos.start.line, pos.start.column))
