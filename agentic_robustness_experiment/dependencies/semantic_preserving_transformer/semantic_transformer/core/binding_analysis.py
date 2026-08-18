from collections import Counter
from collections.abc import Iterable, Iterator
from typing import Callable, Optional

import libcst as cst

#: Nodes that introduce a Python scope.
SCOPE_NODES = (cst.Module, cst.FunctionDef, cst.ClassDef, cst.Lambda)

#: These parse as ``Name`` nodes but cannot be rebound in Python 3.
UNBINDABLE_NAMES = frozenset({"True", "False", "None"})


def _target_names(node: Optional[cst.CSTNode]) -> Iterator[str]:
    if isinstance(node, cst.Name):
        yield node.value # type: ignore
    elif isinstance(node, (cst.Tuple, cst.List)):
        for element in node.elements: # type: ignore
            yield from _target_names(element.value)
    elif isinstance(node, cst.StarredElement):
        yield from _target_names(node.value) # type: ignore
    # `self.x` and `a[0]` targets bind no local name.


class _ScopeBindingCollector(cst.CSTVisitor):
    def __init__(self, scope: cst.CSTNode):
        super().__init__()
        self._scope = scope
        self.counts: Counter = Counter()

    def _add(self, node: Optional[cst.CSTNode]) -> None:
        self.counts.update(_target_names(node))

    def _add_params(self, params: cst.Parameters) -> None:
        for param in (*params.posonly_params, *params.params, *params.kwonly_params):
            self.counts[param.name.value] += 1

        for star in (params.star_arg, params.star_kwarg):
            if isinstance(star, cst.Param):
                self.counts[star.name.value] += 1

    # --- scope boundaries -------------------------------------------------
    def visit_FunctionDef(self, node: cst.FunctionDef) -> bool:
        if node is not self._scope:
            self.counts[node.name.value] += 1
            return False
        self._add_params(node.params)
        return True

    def visit_ClassDef(self, node: cst.ClassDef) -> bool:
        if node is not self._scope:
            self.counts[node.name.value] += 1
            return False
        return True

    def visit_Lambda(self, node: cst.Lambda) -> bool:
        if node is not self._scope:
            return False
        self._add_params(node.params)
        return True

    # A comprehension/genexp loop variable is scoped to the comprehension
    # itself
    def visit_ListComp(self, node: cst.ListComp) -> bool:
        return node is self._scope

    def visit_SetComp(self, node: cst.SetComp) -> bool:
        return node is self._scope

    def visit_DictComp(self, node: cst.DictComp) -> bool:
        return node is self._scope

    def visit_GeneratorExp(self, node: cst.GeneratorExp) -> bool:
        return node is self._scope

    # --- binding statements -----------------------------------------------

    def visit_Assign(self, node: cst.Assign) -> None:
        for target in node.targets:
            self._add(target.target)

    def visit_AnnAssign(self, node: cst.AnnAssign) -> None:
        self._add(node.target)

    def visit_AugAssign(self, node: cst.AugAssign) -> None:
        self._add(node.target)

    def visit_NamedExpr(self, node: cst.NamedExpr) -> None:
        self._add(node.target)

    def visit_For(self, node: cst.For) -> None:
        self._add(node.target)

    def visit_CompFor(self, node: cst.CompFor) -> None:
        self._add(node.target)

    def visit_WithItem(self, node: cst.WithItem) -> None:
        if node.asname is not None:
            self._add(node.asname.name)

    def visit_ExceptHandler(self, node: cst.ExceptHandler) -> None:
        if node.name is not None:
            self._add(node.name.name)

    def visit_ExceptStarHandler(self, node: "cst.ExceptStarHandler") -> None:
        if node.name is not None:
            self._add(node.name.name)

    def visit_MatchAs(self, node: "cst.MatchAs") -> None:
        self._add(node.name)

    def visit_MatchStar(self, node: "cst.MatchStar") -> None:
        self._add(node.name)

    def visit_MatchMapping(self, node: "cst.MatchMapping") -> None:
        self._add(node.rest)

    def visit_Import(self, node: cst.Import) -> None:
        self._add_imports(node.names)

    def visit_ImportFrom(self, node: cst.ImportFrom) -> None:
        if not isinstance(node.names, cst.ImportStar):
            self._add_imports(node.names)

    def _add_imports(self, aliases: Iterable[cst.ImportAlias]) -> None:
        for alias in aliases:
            if alias.asname is not None:
                self._add(alias.asname.name)
            elif isinstance(alias.name, cst.Name):
                self.counts[alias.name.value] += 1
            elif isinstance(alias.name, cst.Attribute):
                # `import a.b.c` binds the root package name `a`.
                root = alias.name
                while isinstance(root, cst.Attribute):
                    root = root.value
                self._add(root)

    def visit_Global(self, node: cst.Global) -> None:
        # The name resolves to a module global rather than the builtin.
        self.counts.update(item.name.value for item in node.names)

    def visit_Nonlocal(self, node: cst.Nonlocal) -> None:
        self.counts.update(item.name.value for item in node.names)


class _FreeNameCollector(cst.CSTVisitor):
    def __init__(self):
        super().__init__()
        self.names: set[str] = set()

    def visit_Name(self, node: cst.Name) -> None:
        if node.value not in UNBINDABLE_NAMES:
            self.names.add(node.value)

    def visit_Attribute(self, node: cst.Attribute) -> bool:
        # `x.next` looks up `x`, not `next`.
        node.value.visit(self)
        return False

    def visit_Arg(self, node: cst.Arg) -> bool:
        # `f(next=1)` looks up `f`, not `next`.
        node.value.visit(self)
        return False


def binding_counts_in_scope(scope: cst.CSTNode) -> dict[str, int]:
    collector = _ScopeBindingCollector(scope)
    scope.visit(collector)
    return dict(collector.counts)


def bindings_in_scope(scope: cst.CSTNode) -> set[str]:
    return set(binding_counts_in_scope(scope))


def scopes_in(module: cst.Module) -> list[cst.CSTNode]:
    collector = _ScopeCollector()
    module.visit(collector)
    return [module, *collector.scopes]


class _ScopeCollector(cst.CSTVisitor):
    def __init__(self):
        super().__init__()
        self.scopes: list[cst.CSTNode] = []

    def visit_FunctionDef(self, node: cst.FunctionDef) -> None:
        self.scopes.append(node)

    def visit_ClassDef(self, node: cst.ClassDef) -> None:
        self.scopes.append(node)

    def visit_Lambda(self, node: cst.Lambda) -> None:
        self.scopes.append(node)


def free_names(node: cst.CSTNode) -> set[str]:
    collector = _FreeNameCollector()
    node.visit(collector)
    return collector.names


class BindingAnalyzer:
    def __init__(self, get_parent: Callable[[cst.CSTNode], cst.CSTNode]):
        self._get_parent = get_parent
        self._bindings_cache = {}

    def enclosing_scopes(self, node: cst.CSTNode) -> list[cst.CSTNode]:
        scopes = []
        current = node

        while True:
            try:
                current = self._get_parent(current)
            except KeyError:  # the module has no parent
                break

            if not isinstance(current, SCOPE_NODES):
                continue

            # A class body is only in the lookup chain for code written
            # directly in it; methods and nested classes skip straight past it.
            if isinstance(current, cst.ClassDef) and scopes:
                continue

            scopes.append(current)

        return scopes

    def bindings_in_scope(self, scope: cst.CSTNode) -> set[str]:
        cached = self._bindings_cache.get(scope)
        if cached is None:
            cached = bindings_in_scope(scope)
            self._bindings_cache[scope] = cached
        return cached

    def bound_names(self, node: cst.CSTNode) -> set[str]:
        names: set[str] = set()
        for scope in self.enclosing_scopes(node):
            names |= self.bindings_in_scope(scope)
        return names

    def is_shadowed(self, node: cst.CSTNode, name: str) -> bool:
        return any(name in self.bindings_in_scope(scope) for scope in self.enclosing_scopes(node))

    def shadowed_any(self, node: cst.CSTNode, names: Iterable[str]) -> bool:
        names = set(names)
        if not names:
            return False
        return bool(names & self.bound_names(node))
