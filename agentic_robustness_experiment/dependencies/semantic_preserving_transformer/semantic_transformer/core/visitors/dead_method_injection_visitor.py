from typing import List, Optional, Tuple

import libcst as cst
import libcst.metadata as metadata

RISKY_BASE_NAMES = frozenset(
    {
        # Pydantic
        "BaseModel",
        "GenericModel",
        # Django / peewee / SQLAlchemy
        # Marshmallow / Django REST Framework
        "Schema",
        "Serializer",
        # MongoEngine
        "Document",
        "EmbeddedDocument",
        # Standard-library enum and typed containers
        "Enum",
        "IntEnum",
        "StrEnum",
        "Flag",
        "IntFlag",
        "NamedTuple",
        "TypedDict",
        # typing.Protocol
        "Protocol",
        # ABC
        "ABC",
        "ABCMeta",
    }
)

_ATTR_DISPATCH_DUNDERS = frozenset(
    {
        "__getattr__",
        "__getattribute__",
        "__setattr__",
        "__delattr__",
    }
)


class DeadMethodVisitor(cst.CSTVisitor):
    METADATA_DEPENDENCIES = (metadata.PositionProvider,)

    def __init__(self, target_name: str):
        self.target_name = target_name
        self.candidates: List[Tuple[int, int]] = []

        self._target_is_dunder = (
            target_name is not None
            and target_name.startswith("__")
            and target_name.endswith("__")
            and len(target_name) >= 4
        )

    def visit_ClassDef(self, node: cst.ClassDef) -> Optional[bool]:
        if self._target_is_dunder:
            return True

        if self._has_member(node, self.target_name):
            return True
        if self._is_sentinel_class(node):
            return True
        if self._has_risky_base(node):
            return True
        if self._has_custom_metaclass(node):
            return True
        if self._is_property_namespace(node):
            return True
        if self._has_attr_dispatch_override(node):
            return True

        pos = self.get_metadata(metadata.PositionProvider, node)
        self.candidates.append((pos.start.line, pos.start.column))
        return True  # still descend for nested classes

    @staticmethod
    def _has_member(cls: cst.ClassDef, name: str) -> bool:
        if not isinstance(cls.body, cst.IndentedBlock):
            return False
        for stmt in cls.body.body:
            if isinstance(stmt, cst.FunctionDef) and stmt.name.value == name:
                return True
            if isinstance(stmt, cst.SimpleStatementLine):
                for small in stmt.body:
                    if isinstance(small, cst.Assign):
                        for tgt in small.targets:
                            t = tgt.target
                            if isinstance(t, cst.Name) and t.value == name:
                                return True
                    elif isinstance(small, cst.AnnAssign):
                        if isinstance(small.target, cst.Name) and small.target.value == name:
                            return True
        return False

    @staticmethod
    def _is_property_namespace(cls: cst.ClassDef) -> bool:
        """All non-dunder methods are decorated (e.g. @property, @memoize_property).
        Injecting a plain method into such a class breaks any code that iterates
        its attributes expecting only property-produced instances."""
        if not isinstance(cls.body, cst.IndentedBlock):
            return False
        has_any = False
        for stmt in cls.body.body:
            if not isinstance(stmt, cst.FunctionDef):
                continue
            name = stmt.name.value
            if name.startswith("__") and name.endswith("__"):
                continue
            if not stmt.decorators:
                return False
            has_any = True
        return has_any

    @staticmethod
    def _has_attr_dispatch_override(cls: cst.ClassDef) -> bool:
        """Class directly defines one of the attribute-dispatch magic methods,
        meaning its attribute namespace is explicitly controlled."""
        return any(DeadMethodVisitor._has_member(cls, name) for name in _ATTR_DISPATCH_DUNDERS)

    @staticmethod
    def _is_sentinel_class(cls: cst.ClassDef) -> bool:
        if not isinstance(cls.body, cst.IndentedBlock):
            return False
        real = [s for s in cls.body.body if not isinstance(s, cst.EmptyLine)]
        if len(real) != 1:
            return False
        stmt = real[0]
        if isinstance(stmt, cst.SimpleStatementLine) and len(stmt.body) == 1:
            small = stmt.body[0]
            if isinstance(small, cst.Pass):
                return True
            if isinstance(small, cst.Expr) and isinstance(small.value, cst.Ellipsis):
                return True
        return False

    def _has_risky_base(self, cls: cst.ClassDef) -> bool:
        for base in cls.bases:
            name = self._base_simple_name(base.value)
            if name and name in RISKY_BASE_NAMES:
                return True
        return False

    @staticmethod
    def _has_custom_metaclass(cls: cst.ClassDef) -> bool:
        for kw in cls.keywords:
            if isinstance(kw.keyword, cst.Name) and kw.keyword.value == "metaclass":
                return True
        return False

    @staticmethod
    def _base_simple_name(expr: cst.BaseExpression) -> Optional[str]:
        if isinstance(expr, cst.Name):
            return expr.value
        if isinstance(expr, cst.Attribute) and isinstance(expr.attr, cst.Name):
            return expr.attr.value
        if isinstance(expr, cst.Subscript):
            return DeadMethodVisitor._base_simple_name(expr.value)
        return None
