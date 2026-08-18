from typing import Set, Tuple

import libcst as cst
import libcst.metadata as metadata


class WhileLoopUnroller(cst.CSTTransformer):
    METADATA_DEPENDENCIES = (metadata.PositionProvider,)

    def __init__(self, target_coords: Set[Tuple[int, int]]):
        super().__init__()
        self.target_coords = target_coords

    def leave_While(self, original_node, updated_node):
        pos = self.get_metadata(metadata.PositionProvider, original_node)
        current_coord = (pos.start.line, pos.start.column)

        if current_coord not in self.target_coords:
            return updated_node

        outer_while_body = updated_node.body

        inner_while_body = cst.While(
            test=updated_node.test,
            body=outer_while_body,
        )

        new_outer_statements = []
        new_outer_statements.extend(outer_while_body.body)
        new_outer_statements.append(inner_while_body)
        new_outer_statements.append(cst.SimpleStatementLine(body=[cst.Break()]))

        new_block = outer_while_body.with_changes(body=new_outer_statements)

        return updated_node.with_changes(body=new_block, orelse=updated_node.orelse)
