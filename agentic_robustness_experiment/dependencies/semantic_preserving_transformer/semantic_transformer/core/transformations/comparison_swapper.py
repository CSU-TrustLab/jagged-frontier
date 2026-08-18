from typing import Set, Tuple

import libcst as cst
import libcst.metadata as metadata


class ComparisonSwapper(cst.CSTTransformer):
    """
    Simple transformation that only **swaps** comparison operands.

    """

    METADATA_DEPENDENCIES = (metadata.PositionProvider,)

    def __init__(self, target_coords: Set[Tuple[int, int]]):
        self.target_coords = target_coords
        self.swap_dict = {
            cst.LessThan: cst.GreaterThan,
            cst.GreaterThan: cst.LessThan,
            cst.LessThanEqual: cst.GreaterThanEqual,
            cst.GreaterThanEqual: cst.LessThanEqual,
        }

    def leave_Comparison(self, original_node, updated_node):
        pos = self.get_metadata(metadata.PositionProvider, original_node)
        current_coord = (pos.start.line, pos.start.column)

        if current_coord not in self.target_coords:
            return updated_node

        target = updated_node.comparisons[0]

        left = updated_node.left
        right = target.comparator

        new_op = self.swap_dict[type(target.operator)]()

        return updated_node.with_changes(
            left=right,
            comparisons=[target.with_changes(operator=new_op, comparator=left)],
        )
