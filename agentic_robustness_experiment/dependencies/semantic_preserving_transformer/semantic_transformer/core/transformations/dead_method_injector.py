from typing import Set, Tuple

import libcst as cst
import libcst.metadata as metadata


class DeadMethodTransformer(cst.CSTTransformer):
    METADATA_DEPENDENCIES = (metadata.PositionProvider,)

    def __init__(self, target_coords: Set[Tuple[int, int]], target_name: str):
        super().__init__()
        self.target_coords = target_coords
        self.target_name = target_name

    def leave_ClassDef(
        self,
        original_node: cst.ClassDef,
        updated_node: cst.ClassDef,
    ) -> cst.ClassDef:
        pos = self.get_metadata(metadata.PositionProvider, original_node)
        if (pos.start.line, pos.start.column) not in self.target_coords:
            return updated_node

        if not isinstance(updated_node.body, cst.IndentedBlock):
            return updated_node

        for stmt in updated_node.body.body:
            if isinstance(stmt, cst.FunctionDef) and stmt.name.value == self.target_name:
                return updated_node

        method = self._build_method()
        new_body_stmts = list(updated_node.body.body) + [method]
        new_body = updated_node.body.with_changes(body=new_body_stmts)
        return updated_node.with_changes(body=new_body)

    def _build_method(self) -> cst.FunctionDef:
        guard_inner = cst.IndentedBlock(
            body=[
                cst.SimpleStatementLine(
                    body=[
                        cst.Assign(
                            targets=[cst.AssignTarget(target=cst.Name("value"))],
                            value=cst.SimpleString(f'"{_escape(self.target_name)}"'),
                        )
                    ]
                ),
                cst.SimpleStatementLine(body=[cst.Return(value=cst.Name("None"))]),
            ]
        )
        guard = cst.If(test=cst.Name("False"), body=guard_inner)

        params = cst.Parameters(
            params=[cst.Param(name=cst.Name("self"))],
            star_arg=cst.Param(name=cst.Name("args")),
            star_kwarg=cst.Param(name=cst.Name("kwargs")),
        )

        return cst.FunctionDef(
            name=cst.Name(self.target_name),
            params=params,
            body=cst.IndentedBlock(body=[guard]),
            leading_lines=[cst.EmptyLine()],
        )


def _escape(s: str) -> str:
    return s.replace("\\", "\\\\").replace('"', '\\"')
