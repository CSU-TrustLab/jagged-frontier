import libcst as cst
import libcst.metadata as metadata


class LocalVariableVisitor(cst.CSTVisitor):
    METADATA_DEPENDENCIES = (metadata.PositionProvider,)

    def __init__(self):
        self.candidates = []

    def visit_Assign(self, node: cst.Assign):
        pos = self.get_metadata(metadata.PositionProvider, node)

        # collect ONLY assignment nodes
        self.candidates.append((pos.start.line, pos.start.column))
