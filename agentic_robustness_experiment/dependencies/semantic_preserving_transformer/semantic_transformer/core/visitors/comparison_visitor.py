from typing import List, Tuple

import libcst as cst
import libcst.matchers as m
import libcst.metadata as metadata


class ComparisonVisitor(cst.CSTVisitor):
    METADATA_DEPENDENCIES = (metadata.PositionProvider,)

    def __init__(self):
        self.candidates: List[Tuple[int, int]] = []
        self.ops = (
            m.LessThan(),
            m.GreaterThan(),
            m.LessThanEqual(),
            m.GreaterThanEqual(),
        )

    def visit_Comparison(self, node: cst.Comparison):
        if m.matches(
            node,
            m.Comparison(
                left=m.DoNotCare(),
                comparisons=(
                    m.ComparisonTarget(
                        operator=m.OneOf(*self.ops),
                        comparator=m.DoNotCare(),
                    ),
                ),
            ),
        ):
            pos = self.get_metadata(metadata.PositionProvider, node)
            self.candidates.append((pos.start.line, pos.start.column))
