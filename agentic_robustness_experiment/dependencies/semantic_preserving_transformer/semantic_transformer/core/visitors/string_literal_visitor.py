from typing import List, Optional, Tuple

import libcst as cst
import libcst.metadata as metadata


class StringLiteralVisitor(cst.CSTVisitor):
    METADATA_DEPENDENCIES = (
        metadata.PositionProvider,
        metadata.ParentNodeProvider,
    )

    def __init__(self, min_length: int = 4):
        super().__init__()
        self.candidates: List[Tuple[int, int]] = []
        self.min_length = min_length

    def visit_SimpleString(self, node: cst.SimpleString) -> Optional[bool]:

        if (
            self.has_prefix(node)
            or self.is_triple_quoted(node)
            or self.get_content_length(node) < self.min_length
            or self.is_docstring(node)
            or self.is_annotation(node)
            or self.is_inside_concat(node)
            or self.is_inside_switch(node)
            or self.is_attribute_target(node)
            or self.is_inside_binop_non_add(node)
        ):
            return None

        pos = self.get_metadata(metadata.PositionProvider, node)
        self.candidates.append((pos.start.line, pos.start.column))

    def is_docstring(self, node: cst.SimpleString) -> bool:
        parent = self.get_metadata(metadata.ParentNodeProvider, node)
        if not isinstance(parent, cst.Expr):
            return False
        expr_parent = self.get_metadata(metadata.ParentNodeProvider, parent)
        if not isinstance(expr_parent, cst.SimpleStatementLine):
            return False
        stmt_parent = self.get_metadata(metadata.ParentNodeProvider, expr_parent)
        if not isinstance(stmt_parent, (cst.IndentedBlock, cst.Module)):
            return False
        body = stmt_parent.body
        return len(body) > 0 and body[0] is expr_parent

    def is_annotation(self, node: cst.SimpleString) -> bool:
        current = node
        for _ in range(10):
            parent = self.get_metadata(metadata.ParentNodeProvider, current)
            if isinstance(parent, cst.Annotation):
                return True
            if isinstance(parent, cst.Module):
                break
            current = parent
        return False

    def is_inside_concat(self, node: cst.SimpleString) -> bool:
        parent = self.get_metadata(metadata.ParentNodeProvider, node)
        return isinstance(parent, cst.ConcatenatedString)

    def is_inside_switch(self, node: cst.SimpleString) -> bool:
        current = node
        for i in range(10):
            parent = self.get_metadata(metadata.ParentNodeProvider, current)
            if isinstance(
                parent, (cst.MatchValue, cst.MatchMapping, cst.MatchOrElement, cst.MatchCase)
            ):
                return True
            if isinstance(parent, cst.Module):
                break
            current = parent
        return False

    def is_attribute_target(self, node: cst.SimpleString) -> bool:
        parent = self.get_metadata(metadata.ParentNodeProvider, node)
        if isinstance(parent, cst.Attribute) and parent.value is node:
            return True
        return False

    def is_inside_binop_non_add(self, node: cst.SimpleString) -> bool:
        parent = self.get_metadata(metadata.ParentNodeProvider, node)
        if isinstance(parent, cst.BinaryOperation):
            if not isinstance(parent.operator, cst.Add):
                return True
        return False

    def get_content_length(self, node: cst.SimpleString) -> int:
        raw = node.value
        i = 0
        while i < len(raw) and raw[i] in "bBrRuU":
            i += 1
        rest = raw[i:]
        if rest.startswith('"""') or rest.startswith("'''"):
            return len(rest) - 6
        elif rest.startswith('"') or rest.startswith("'"):
            return len(rest) - 2
        return 0

    def has_prefix(self, node: cst.SimpleString) -> bool:
        raw = node.value
        return len(raw) > 0 and raw[0] in "bBrRuU"

    def is_triple_quoted(self, node: cst.SimpleString) -> bool:
        raw = node.value
        i = 0
        while i < len(raw) and raw[i] in "bBrRuU":
            i += 1
        rest = raw[i:]
        return rest.startswith('"""') or rest.startswith("'''")
