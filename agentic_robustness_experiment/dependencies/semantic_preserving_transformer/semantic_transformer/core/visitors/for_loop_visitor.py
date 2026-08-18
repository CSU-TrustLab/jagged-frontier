from typing import List, Optional, Tuple

import libcst as cst
import libcst.metadata as metadata


class ForLoopVisitor(cst.CSTVisitor):
    METADATA_DEPENDENCIES = (metadata.PositionProvider,)

    def __init__(self):
        self.candidates: List[Tuple[int, int]] = []

    def visit_For(self, node: cst.For) -> Optional[bool]:
        if getattr(node, "is_async", False):
            return
        if isinstance(node.body, cst.IndentedBlock):
            pos = self.get_metadata(metadata.PositionProvider, node)
            self.candidates.append((pos.start.line, pos.start.column))
