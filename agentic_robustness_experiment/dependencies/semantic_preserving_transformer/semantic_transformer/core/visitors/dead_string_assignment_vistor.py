from enum import Enum
from typing import List, Optional, Tuple

import libcst as cst
import libcst.metadata as metadata


class FRAME_REFLECTION_NAMES(Enum):
    LOCALS = "locals"
    VARS = "vars"


_FRAME_ACCESS_ATTRS = frozenset(
    {
        "_getframe",
        "currentframe",
        "stack",
        "getouterframes",
        "getinnerframes",
        "trace",
    }
)

_IMPLICIT_LOCALS_CALLABLES = frozenset({"exec", "eval"})


class _FrameReflectionScanner(cst.CSTVisitor):
    """Scans a single function body for patterns that expose the
    caller's local variables.

    After visiting, `self.uses_frame_reflection` is True iff the
    scanned body is unsafe for injection.
    """

    def __init__(self):
        self.uses_frame_reflection = False

    def visit_FunctionDef(self, node: cst.FunctionDef) -> Optional[bool]:
        return False

    def visit_ClassDef(self, node: cst.ClassDef) -> Optional[bool]:
        return False

    def visit_Lambda(self, node: cst.Lambda) -> Optional[bool]:
        return False

    def visit_Call(self, node: cst.Call) -> Optional[bool]:
        if self._is_frame_reflection_call(node):
            self.uses_frame_reflection = True
            return False
        return True

    @staticmethod
    def _is_frame_reflection_call(call: cst.Call) -> bool:
        func = call.func

        if isinstance(func, cst.Name):
            name = func.value
            if name == FRAME_REFLECTION_NAMES.LOCALS.value:
                return True
            if name == FRAME_REFLECTION_NAMES.VARS.value:
                # vars() == locals(); vars(obj) is safe.
                return len(call.args) == 0
            if name in _IMPLICIT_LOCALS_CALLABLES:
                has_explicit_locals = len(call.args) >= 3 or any(
                    isinstance(a.keyword, cst.Name)
                    and a.keyword.value == FRAME_REFLECTION_NAMES.LOCALS.value
                    for a in call.args
                )
                return not has_explicit_locals
            return False

        # Attribute call: sys._getframe(), inspect.currentframe(), etc.
        if isinstance(func, cst.Attribute) and isinstance(func.attr, cst.Name):
            return func.attr.value in _FRAME_ACCESS_ATTRS

        return False


class StringFloodVisitor(cst.CSTVisitor):
    """
    Collects every valid (line, col) position at which a dead string
    assignment can be inserted to flood grep for a keyword.

    Each candidate is the (line, col) of an existing statement inside
    some FunctionDef's body — the transformer will insert a new
    assignment IMMEDIATELY BEFORE that statement.

    We do NOT emit positions for:
      - The docstring itself (inserting before it would demote
        `__doc__`).
      - Statements that come after an unconditional terminator
        (return/raise/break/continue) in the same block — inserting
        there creates unreachable code. Unreachable code is technically
        semantic-preserving but stylistically suspicious and
        lint-flagged.
      - Any function which accesses local symbol table information
        is excluded from selection
    """

    METADATA_DEPENDENCIES = (metadata.PositionProvider,)

    def __init__(self):
        self.candidates: List[Tuple[int, int]] = []

    def visit_FunctionDef(self, node: cst.FunctionDef) -> Optional[bool]:
        if not isinstance(node.body, cst.IndentedBlock):
            return False

        scanner = _FrameReflectionScanner()
        node.body.visit(scanner)
        if scanner.uses_frame_reflection:
            return True

        self._enumerate_block(node.body)
        return True

    def _enumerate_block(self, block: cst.IndentedBlock) -> None:
        """Emit one candidate per valid statement in this block, then
        recurse into compound-statement sub-blocks within the same
        function scope."""
        statements = block.body
        if not statements:
            return

        starting_idx = 1 if self._is_docstring_stmt(statements[0]) else 0
        collecting = True

        for i, statement in enumerate(statements):
            if collecting:
                if i >= starting_idx:
                    pos = self.get_metadata(metadata.PositionProvider, statement)
                    self.candidates.append((pos.start.line, pos.start.column))
                if self._is_unconditional_terminator(statement):
                    collecting = False
            self._recurse_compound(statement)

    def _should_enumerate(self, body: cst.CSTNode) -> None:
        if isinstance(body, cst.IndentedBlock):
            self._enumerate_block(body)

    def _recurse_compound(self, statement: cst.CSTNode) -> None:
        """Descend into compound-statement bodies within the same
        function scope."""
        if isinstance(statement, (cst.FunctionDef, cst.ClassDef)):
            return

        if isinstance(statement, cst.If):
            self._should_enumerate(statement.body)
            orelse_statement = statement.orelse
            while isinstance(orelse_statement, cst.If):
                self._should_enumerate(orelse_statement.body)
                orelse_statement = orelse_statement.orelse
            if isinstance(orelse_statement, cst.Else):
                self._should_enumerate(orelse_statement.body)

        elif isinstance(statement, (cst.For, cst.While)):
            self._should_enumerate(statement.body)
            if statement.orelse:
                self._should_enumerate(statement.orelse.body)

        elif isinstance(statement, cst.With):
            self._should_enumerate(statement.body)

        elif isinstance(statement, cst.Try):
            self._should_enumerate(statement.body)
            for handler in statement.handlers:
                self._should_enumerate(handler.body)
            if statement.orelse:
                self._should_enumerate(statement.orelse.body)
            if statement.finalbody:
                self._should_enumerate(statement.finalbody.body)

    @staticmethod
    def _is_docstring_stmt(stmt: cst.CSTNode) -> bool:
        if not isinstance(stmt, cst.SimpleStatementLine) or len(stmt.body) != 1:
            return False
        body = stmt.body[0]
        return isinstance(body, cst.Expr) and isinstance(
            body.value, (cst.SimpleString, cst.ConcatenatedString)
        )

    @staticmethod
    def _is_unconditional_terminator(stmt: cst.CSTNode) -> bool:
        if not isinstance(stmt, cst.SimpleStatementLine) or not stmt.body:
            return False
        last_node = stmt.body[-1]
        return isinstance(last_node, (cst.Return, cst.Raise, cst.Break, cst.Continue))
