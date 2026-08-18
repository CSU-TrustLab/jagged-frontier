import libcst as cst
from libcst.metadata import PositionProvider


class DeadCodeInjector(cst.CSTTransformer):
    METADATA_DEPENDENCIES = (PositionProvider,)

    def __init__(self, target_coords=None):
        super().__init__()
        self.target_coords = target_coords or set()

    def leave_IndentedBlock(self, original_node, updated_node):

        if not original_node.body:
            return updated_node

        original_body = list(original_node.body)
        insert_index = 0

        first_stmt = original_body[0]
        if isinstance(first_stmt, cst.SimpleStatementLine) and len(first_stmt.body) == 1:
            stmt = first_stmt.body[0]
            if isinstance(stmt, cst.Expr) and isinstance(
                stmt.value,
                (cst.SimpleString, cst.ConcatenatedString, cst.FormattedString),
            ):
                insert_index = 1

        if insert_index >= len(original_body):
            return updated_node

        target_stmt = original_body[insert_index]
        pos = self.get_metadata(PositionProvider, target_stmt)
        coord = (pos.start.line, pos.start.column)

        if coord not in self.target_coords:
            return updated_node

        dead_code = cst.If(
            test=cst.Name("False"),
            body=cst.IndentedBlock(
                body=[
                    cst.SimpleStatementLine(
                        body=[
                            cst.Assign(
                                targets=[
                                    cst.AssignTarget(target=cst.Name("_semantic_preserving_block"))
                                ],
                                value=cst.Integer("0"),
                            )
                        ]
                    )
                ]
            ),
        )

        updated_body = list(updated_node.body)

        new_body = updated_body[:insert_index] + [dead_code] + updated_body[insert_index:]

        return updated_node.with_changes(body=new_body)
