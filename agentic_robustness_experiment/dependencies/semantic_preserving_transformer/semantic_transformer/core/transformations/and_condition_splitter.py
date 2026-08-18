from typing import Set, Tuple

import libcst as cst
import libcst.metadata as metadata


class AndConditionSplitter(cst.CSTTransformer):
    """
    Replaces AND conditions with nested if statments
    """

    METADATA_DEPENDENCIES = (metadata.PositionProvider,)

    def __init__(self, target_coords: Set[Tuple[int, int]]):
        super().__init__()
        self.target_coords = target_coords

    def leave_If(self, original_node, updated_node):

        pos = self.get_metadata(metadata.PositionProvider, original_node)
        current_coord = (pos.start.line, pos.start.column)

        if current_coord not in self.target_coords:
            return updated_node

        if not (
            isinstance(updated_node.test, cst.BooleanOperation)
            and isinstance(updated_node.test.operator, cst.And)
        ):
            return updated_node

        left_test = updated_node.test.left.with_changes(
            lpar=[cst.LeftParen()], rpar=[cst.RightParen()]
        )
        right_test = updated_node.test.right.with_changes(
            lpar=[cst.LeftParen()], rpar=[cst.RightParen()]
        )

        inner_if = cst.If(test=right_test, body=updated_node.body, orelse=updated_node.orelse)

        outer_if = cst.If(
            test=left_test,
            body=cst.IndentedBlock(body=[inner_if]),
            orelse=updated_node.orelse,
        )

        return outer_if
