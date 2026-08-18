import itertools
from typing import List, Optional, Set, Tuple
from uuid import uuid4

import libcst as cst
import libcst.metadata as metadata

NAME_POOL = (
    "result",
    "value",
    "data",
    "tmp",
    "buf",
    "cached",
    "entry",
    "item",
    "record",
    "payload",
    "label",
    "tag",
    "msg",
    "info",
    "state",
    "ctx",
    "meta",
    "token",
    "prefix",
    "suffix",
    "marker",
    "ident",
    "ref",
    "current",
    "previous",
    "candidate",
    "resolved",
    "parsed",
    "raw",
    "snapshot",
    "handle",
    "frame",
    "cursor",
    "name_",
    "value_",
    "tmp_",
    "result_",
    "data_",
    "key_",
)


class StringFloodTransformer(cst.CSTTransformer):
    """
    Inserts `<name> = "<keyword>"` immediately BEFORE each
    statement whose (line, col) is in `target_coords`.

    The name is chosen per-function. When we enter a
    FunctionDef, we compute the set of identifiers used
    anywhere in its scope. We then pick, for each injection in that
    function, the first NAME_POOL entry not in that reserved set.
    Fallback to a hash-suffixed name if the pool is
    exhausted.
    """

    METADATA_DEPENDENCIES = (metadata.PositionProvider,)

    def __init__(self, target_coords: Set[Tuple[int, int]], target_name: str):
        super().__init__()
        self.target_coords = target_coords
        self.keyword = target_name
        # Stack of (function_coord, reserved_names, chosen_so_far)
        self._fn_stack: List[Tuple[Tuple[int, int], Set[str], Set[str]]] = []

    def visit_FunctionDef(self, node: cst.FunctionDef) -> Optional[bool]:
        pos = self.get_metadata(metadata.PositionProvider, node)
        fn_coordinate = (pos.start.line, pos.start.column)
        reserved = _collect_used_names(node)
        self._fn_stack.append((fn_coordinate, reserved, set()))
        return True

    def leave_FunctionDef(
        self,
        original_node: cst.FunctionDef,
        updated_node: cst.FunctionDef,
    ) -> cst.FunctionDef:
        if self._fn_stack:
            self._fn_stack.pop()
        return updated_node

    def leave_IndentedBlock(
        self,
        original_node: cst.IndentedBlock,
        updated_node: cst.IndentedBlock,
    ) -> cst.IndentedBlock:
        if not self._fn_stack or len(original_node.body) != len(updated_node.body):
            return updated_node

        new_body = []
        any_inserted = False
        for original_child, updated_child in zip(original_node.body, updated_node.body):
            cp = self.get_metadata(metadata.PositionProvider, original_child)
            if (cp.start.line, cp.start.column) in self.target_coords:
                new_body.append(self._build_assign(self._pick_name()))
                any_inserted = True
            new_body.append(updated_child)

        if not any_inserted:
            return updated_node

        return updated_node.with_changes(body=new_body)

    def _pick_name(self) -> str:
        _, reserved, chosen = self._fn_stack[-1]
        for candidate in NAME_POOL:
            if candidate not in reserved and candidate not in chosen:
                chosen.add(candidate)
                return candidate

        combined = reserved | chosen
        name = "tmp_" + uuid4().hex[:6]
        while name in combined:
            name = "tmp_" + uuid4().hex[:6]
        chosen.add(name)
        return name

    def _build_assign(self, name: str) -> cst.SimpleStatementLine:
        return cst.SimpleStatementLine(
            body=[
                cst.Assign(
                    targets=[cst.AssignTarget(target=cst.Name(name))],
                    value=cst.SimpleString(f'"{self.keyword}"'),
                )
            ]
        )


def _collect_used_names(fn: cst.FunctionDef) -> Set[str]:
    params = fn.params
    # Collecting all the parameter names
    names = {
        p.name.value
        for p in itertools.chain(params.params, params.posonly_params, params.kwonly_params)
    }
    for p in (params.star_arg, params.star_kwarg):
        if isinstance(p, cst.Param):
            names.add(p.name.value)

    collector = NameCollector()
    fn.body.visit(collector)
    names.update(collector.names)
    return names


class NameCollector(cst.CSTVisitor):
    """Every Name.value in a subtree, skipping nested scopes."""

    def __init__(self):
        self.names = set()

    def visit_Name(self, node: cst.Name) -> Optional[bool]:
        self.names.add(node.value)
        return False

    def visit_FunctionDef(self, node: cst.FunctionDef) -> Optional[bool]:
        self.names.add(node.name.value)
        return False

    def visit_ClassDef(self, node: cst.ClassDef) -> Optional[bool]:
        self.names.add(node.name.value)
        return False
