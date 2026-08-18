from typing import List, Optional, Tuple

import libcst as cst
import libcst.metadata as metadata


class SingleStatementVisitor(cst.CSTVisitor):
    METADATA_DEPENDENCIES = (metadata.PositionProvider,)

    def __init__(self):
        self.candidates: List[Tuple[int, int]] = []

    def visit_SimpleStatementLine(self, node: cst.SimpleStatementLine) -> Optional[bool]:
        pos = self.get_metadata(metadata.PositionProvider, node)
        self.candidates.append((pos.start.line, pos.start.column))
