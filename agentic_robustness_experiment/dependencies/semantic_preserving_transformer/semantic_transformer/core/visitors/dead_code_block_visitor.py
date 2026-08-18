from semantic_transformer.core.visitors.indented_block_visitor import (
    IndentedBlockVisitor,
)


class DeadCodeBlockVisitor(IndentedBlockVisitor):
    def __init__(self):
        super().__init__(min_size=1, max_size=100)
