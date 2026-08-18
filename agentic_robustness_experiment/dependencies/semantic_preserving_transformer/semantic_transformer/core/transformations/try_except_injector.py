from typing import Set, Tuple

import libcst as cst
import libcst.metadata as metadata

from semantic_transformer.core.binding_analysis import BindingAnalyzer


class TryExceptInjector(cst.CSTTransformer):
    METADATA_DEPENDENCIES = (metadata.PositionProvider, metadata.ParentNodeProvider)

    #: Builtins the injected handler resolves at the injection site.
    REQUIRED_BUILTINS = ("Exception",)

    def __init__(self, target_coords: Set[Tuple[int, int]]):
        super().__init__()
        self.target_coords = target_coords
        self.skipped_shadowed = 0
        self._bindings = BindingAnalyzer(
            lambda node: self.get_metadata(metadata.ParentNodeProvider, node)
        )

    def _inject_try_except(self, node: cst.IndentedBlock):
        except_handler = cst.ExceptHandler(
            type=cst.Name("Exception"),
            body=cst.IndentedBlock(body=[cst.SimpleStatementLine(body=[cst.Raise()])]),
        )
        indented_block = cst.IndentedBlock(node.body)
        try_except = cst.Try(body=indented_block, handlers=[except_handler])
        return node.with_changes(body=[try_except])

    def leave_IndentedBlock(self, original_node, updated_node):
        pos = self.get_metadata(metadata.PositionProvider, original_node)

        if (pos.start.line, pos.start.column) not in self.target_coords:
            return updated_node

        # A rebound `Exception` makes the handler itself raise `TypeError:
        # catching classes that do not inherit from BaseException`.
        if self._bindings.shadowed_any(original_node, self.REQUIRED_BUILTINS):
            self.skipped_shadowed += 1
            return updated_node

        except_handler = cst.ExceptHandler(
            type=cst.Name("Exception"),
            body=cst.IndentedBlock(body=[cst.SimpleStatementLine(body=[cst.Raise()])]),
        )
        indented_block = cst.IndentedBlock(updated_node.body)
        try_except = cst.Try(body=indented_block, handlers=[except_handler])
        return updated_node.with_changes(body=[try_except])
