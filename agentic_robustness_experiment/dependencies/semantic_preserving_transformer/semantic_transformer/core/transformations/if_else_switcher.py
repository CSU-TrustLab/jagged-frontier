from typing import Set, Tuple

import libcst as cst
import libcst.metadata as metadata


class IfElseSwitcher(cst.CSTTransformer):
    METADATA_DEPENDENCIES = (metadata.PositionProvider,)

    def __init__(self, target_coords: Set[Tuple[int, int]]):
        super().__init__()
        self.target_coords = target_coords

    def _switch(self, node: cst.If):
        negated_test = cst.UnaryOperation(
            operator=cst.Not(),
            expression=node.test.with_changes(lpar=[cst.LeftParen()], rpar=[cst.RightParen()]),
        )

        new_if_body = node.orelse.body
        new_else_body = cst.Else(body=node.body)

        return node.with_changes(
            test=negated_test,
            body=new_if_body,
            orelse=new_else_body,
            whitespace_before_test=cst.SimpleWhitespace(" "),
        )

    def leave_If(self, original_node, updated_node):
        pos = self.get_metadata(metadata.PositionProvider, original_node)
        current_coord = (pos.start.line, pos.start.column)

        if current_coord not in self.target_coords:
            return updated_node

        if updated_node.orelse and isinstance(updated_node.orelse, cst.Else):
            return self._switch(updated_node)

        else:
            return updated_node

    def leave_IfExp(self, original_node, updated_node):
        pos = self.get_metadata(metadata.PositionProvider, original_node)
        current_coord = (pos.start.line, pos.start.column)

        if current_coord not in self.target_coords:
            return updated_node

        if not updated_node.orelse:
            raise ValueError("Malformed IfExp detected.")

        negated_test = cst.UnaryOperation(
            operator=cst.Not(),
            expression=updated_node.test.with_changes(
                lpar=[cst.LeftParen()], rpar=[cst.RightParen()]
            ),
        )

        new_body = updated_node.orelse.with_changes(lpar=[cst.LeftParen()], rpar=[cst.RightParen()])
        new_orelse = updated_node.body.with_changes(lpar=[cst.LeftParen()], rpar=[cst.RightParen()])

        return updated_node.with_changes(test=negated_test, body=new_body, orelse=new_orelse)
