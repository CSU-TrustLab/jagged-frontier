import ast
import random
from typing import List, Optional, Set, Tuple

import libcst as cst
import libcst.metadata as metadata


class StringLiteralSplitter(cst.CSTTransformer):
    METADATA_DEPENDENCIES = (metadata.PositionProvider,)

    def __init__(
        self,
        target_coords: Optional[Set[Tuple[int, int]]] = None,
        target_name: Optional[List[str]] = None,
    ):
        super().__init__()
        self.target_coords = target_coords or set()
        self.target_strings = target_name

    def leave_SimpleString(self, original_node, updated_node):

        pos = self.get_metadata(metadata.PositionProvider, original_node)

        if (pos.start.line, pos.start.column) not in self.target_coords:
            return updated_node

        quote_char = updated_node.value[0]

        string = ast.literal_eval(updated_node.value)

        if self.target_strings is not None and string not in self.target_strings:
            return updated_node

        if len(string) < 2:
            return updated_node

        split_loc = random.randint(1, len(string) - 1)

        return cst.BinaryOperation(
            left=cst.SimpleString(make_string_literal(string[:split_loc], quote_char)),
            operator=cst.Add(),
            right=cst.SimpleString(make_string_literal(string[split_loc:], quote_char)),
            lpar=[cst.LeftParen()],
            rpar=[cst.RightParen()],
        )


def make_string_literal(s: str, quote: str = "'") -> str:
    s = s.replace("\\", "\\\\")
    s = s.replace(quote, "\\" + quote)
    s = s.replace("\n", "\\n")
    s = s.replace("\r", "\\r")
    s = s.replace("\t", "\\t")
    s = s.replace("\0", "\\0")
    return quote + s + quote
