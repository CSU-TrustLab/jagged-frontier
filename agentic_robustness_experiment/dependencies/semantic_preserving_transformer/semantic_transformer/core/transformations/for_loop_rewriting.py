import uuid
from typing import Set, Tuple

import libcst as cst
import libcst.metadata as metadata
from libcst import FlattenSentinel

from semantic_transformer.core.binding_analysis import BindingAnalyzer


class _BreakFlagInjector(cst.CSTTransformer):
    def __init__(self, flag_name: str):
        super().__init__()
        self.flag_name = flag_name

    def leave_Break(self, original_node, updated_node):
        assign_stmt = cst.Assign(
            targets=[cst.AssignTarget(target=cst.Name(self.flag_name))],
            value=cst.Name("False"),
        )
        return FlattenSentinel([assign_stmt, updated_node])

    def visit_For(self, node):
        return False

    def visit_While(self, node):
        return False

    def visit_AsyncFor(self, node):
        return False


class ForToWhileTransformer(cst.CSTTransformer):
    METADATA_DEPENDENCIES = (metadata.PositionProvider, metadata.ParentNodeProvider)

    REQUIRED_BUILTINS = ("iter", "next", "StopIteration")

    def __init__(self, target_coords: Set[Tuple[int, int]]):
        super().__init__()
        self.target_coords = target_coords
        self._bindings = BindingAnalyzer(
            lambda node: self.get_metadata(metadata.ParentNodeProvider, node)
        )

    def _unique(self, base: str) -> str:
        return f"__{base}_{uuid.uuid4().hex[:8]}"

    def leave_For(self, original_node: cst.For, updated_node: cst.For):
        pos = self.get_metadata(metadata.PositionProvider, original_node)
        current_coord = (pos.start.line, pos.start.column)

        if current_coord not in self.target_coords:
            return updated_node


        if self._bindings.shadowed_any(original_node, self.REQUIRED_BUILTINS):
            return updated_node

        iter_name = self._unique("for_iter")
        flag_name = self._unique("for_else_flag")

        iterator_node = updated_node.iter

        # If the iterator is a Tuple and lacks parentheses (implicit tuple),
        # we MUST add them. Otherwise, iter(a, b) is interpreted as iter(callable, sentinel)
        # instead of iter((a, b)).
        if isinstance(iterator_node, cst.Tuple) and not iterator_node.lpar:
            iterator_node = iterator_node.with_changes(
                lpar=[cst.LeftParen()], rpar=[cst.RightParen()]
            )

        iter_assign = cst.SimpleStatementLine(
            body=[
                cst.Assign(
                    targets=[cst.AssignTarget(target=cst.Name(iter_name))],
                    value=cst.Call(func=cst.Name("iter"), args=[cst.Arg(value=iterator_node)]),
                )
            ]
        )
        iter_assign = iter_assign.with_changes(leading_lines=original_node.leading_lines)

        flag_init = cst.SimpleStatementLine(
            body=[
                cst.Assign(
                    targets=[cst.AssignTarget(target=cst.Name(flag_name))],
                    value=cst.Name("True"),
                )
            ]
        )

        next_call = cst.Call(func=cst.Name("next"), args=[cst.Arg(value=cst.Name(iter_name))])
        assign_target_stmt = cst.SimpleStatementLine(
            body=[
                cst.Assign(
                    targets=[cst.AssignTarget(target=updated_node.target)],
                    value=next_call,
                )
            ]
        )

        except_block = cst.ExceptHandler(
            type=cst.Name("StopIteration"),
            body=cst.IndentedBlock(body=[cst.SimpleStatementLine(body=[cst.Break()])]),
        )
        try_node = cst.Try(
            body=cst.IndentedBlock(body=[assign_target_stmt]),
            handlers=[except_block],
        )

        injector = _BreakFlagInjector(flag_name)
        new_body_block = updated_node.body.visit(injector)

        while_body_list = [try_node]
        while_body_list.extend(new_body_block.body)

        while_block = cst.While(
            test=cst.Name("True"),
            body=cst.IndentedBlock(body=while_body_list),
        )

        replaced_nodes = [iter_assign, flag_init, while_block]

        if original_node.orelse:
            if_block = cst.If(test=cst.Name(flag_name), body=original_node.orelse.body, orelse=None)
            replaced_nodes.append(if_block)

        return FlattenSentinel(replaced_nodes)
