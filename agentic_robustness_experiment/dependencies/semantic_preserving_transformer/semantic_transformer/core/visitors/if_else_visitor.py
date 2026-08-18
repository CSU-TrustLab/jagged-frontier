from typing import List, Optional, Tuple

import libcst as cst
import libcst.metadata as metadata


class IfElseVisitor(cst.CSTVisitor):
    METADATA_DEPENDENCIES = (metadata.PositionProvider,)

    def __init__(self):
        self.candidates: List[Tuple[int, int]] = []

    def visit_If(self, node: cst.If) -> Optional[bool]:
        if node.orelse is not None and not isinstance(node.orelse, cst.If):
            pos = self.get_metadata(metadata.PositionProvider, node)
            self.candidates.append((pos.start.line, pos.start.column))

    def visit_IfExp(self, node: cst.IfExp) -> Optional[bool]:
        if node.orelse is not None:
            pos = self.get_metadata(metadata.PositionProvider, node)
            self.candidates.append((pos.start.line, pos.start.column))
