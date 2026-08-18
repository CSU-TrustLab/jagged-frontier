from typing import List, Optional, Tuple

import libcst as cst
import libcst.metadata as metadata


class IndentedBlockVisitor(cst.CSTVisitor):
    METADATA_DEPENDENCIES = (metadata.PositionProvider,)

    def __init__(self, min_size: int = 5, max_size: int = 100):
        super().__init__()
        self.candidates: List[Tuple[int, int]] = []
        self.min_size = min_size
        self.max_size = max_size

    def _measure_block_size(self, node: cst.IndentedBlock) -> int:
        """O(1) exact line count using pre-computed Position metadata."""
        pos = self.get_metadata(metadata.PositionProvider, node)
        return pos.end.line - pos.start.line + 1

    def visit_IndentedBlock(self, node: cst.IndentedBlock) -> Optional[bool]:
        if not node.body:
            return None

        # Skips docstrings
        first_stmt = node.body[0]
        body = node.body
        if isinstance(first_stmt, cst.SimpleStatementLine) and (len(first_stmt.body) == 1):
            stmt = first_stmt.body[0]
            if isinstance(stmt, cst.Expr) and isinstance(
                stmt.value,
                (cst.SimpleString, cst.ConcatenatedString, cst.FormattedString),
            ):
                body = node.body[1:]

        if not body:
            return None

        block_size = self._measure_block_size(node)
        if self.min_size <= block_size <= self.max_size:
            pos = self.get_metadata(metadata.PositionProvider, body[0])
            self.candidates.append((pos.start.line, pos.start.column))
