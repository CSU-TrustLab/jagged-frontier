import builtins
import keyword
import random
import string

import libcst as cst
import libcst.metadata as metadata
from nltk.corpus import wordnet as wn

from semantic_transformer.core.binding_analysis import binding_counts_in_scope, bindings_in_scope
from semantic_transformer.core.utils import is_reserved


def random_name(length=5):
    return "".join(random.choices(string.ascii_lowercase, k=length))


def wordnet_name(name):
    try:
        for syn in wn.synsets(name):
            for lemma in syn.lemmas():
                candidate = lemma.name().lower()
                if candidate.isalpha() and candidate != name and len(candidate) >= 3:
                    return candidate
    except Exception:
        pass
    return None


class _Collector(cst.CSTVisitor):
    """Flags names that must not be renamed because of how the module uses them."""

    METADATA_DEPENDENCIES = (metadata.ParentNodeProvider,)

    def __init__(self):
        super().__init__()
        self.unsafe_vars = set()
        self.all_names = set()

    def visit_Name(self, node):
        self.all_names.add(node.value)

        parent = self.get_metadata(metadata.ParentNodeProvider, node)

        if isinstance(parent, cst.Attribute):
            self.unsafe_vars.add(node.value)

        if isinstance(parent, cst.Call) and parent.func is node:
            self.unsafe_vars.add(node.value)

        if isinstance(parent, cst.Arg):
            self.unsafe_vars.add(node.value)

        if isinstance(parent, cst.Subscript) and parent.value is node:
            self.unsafe_vars.add(node.value)

        # A name in an import statement is part of a module path, not a local.
        # Renaming `fields` in `from .fields import X` invents a module that
        # does not exist and kills the importing package outright.
        if isinstance(parent, (cst.Import, cst.ImportFrom, cst.ImportAlias, cst.Decorator)):
            self.unsafe_vars.add(node.value)

    def visit_Import(self, node):
        for name in node.names:
            if isinstance(name, cst.ImportAlias):
                if isinstance(name.name, cst.Name):
                    imported = name.asname.name.value if name.asname else name.name.value

                elif isinstance(name.name, cst.Attribute):
                    imported = name.asname.name.value if name.asname else name.name.attr.value

                else:
                    continue

                self.unsafe_vars.add(imported)

    def visit_ImportFrom(self, node):
        if isinstance(node.names, cst.ImportStar):
            return

        for name in node.names:
            if isinstance(name, cst.ImportAlias):
                if name.asname:
                    imported = name.asname.name.value
                else:
                    if isinstance(name.name, cst.Name):
                        imported = name.name.value
                    else:
                        continue

                self.unsafe_vars.add(imported)

    def visit_FunctionDef(self, node):
        self.unsafe_vars.add(node.name.value)

        for param in node.params.params:
            self.unsafe_vars.add(param.name.value)
        for param in node.params.kwonly_params:
            self.unsafe_vars.add(param.name.value)
        for param in node.params.posonly_params:
            self.unsafe_vars.add(param.name.value)

        if node.params.star_arg:
            star = node.params.star_arg
            if isinstance(star, cst.Param):
                self.unsafe_vars.add(star.name.value)
            elif isinstance(star, str):
                self.unsafe_vars.add(star)

        if node.params.star_kwarg:
            kw = node.params.star_kwarg
            if isinstance(kw, cst.Param):
                self.unsafe_vars.add(kw.name.value)
            elif isinstance(kw, str):
                self.unsafe_vars.add(kw)

    def visit_Assign(self, node):
        for t in node.targets:
            if isinstance(t.target, cst.Name):
                name = t.target.value

                # Aliasing: after `a = b` the two names travel together.
                if isinstance(node.value, cst.Name):
                    self.unsafe_vars.add(name)
                    self.unsafe_vars.add(node.value.value)

                if isinstance(node.value, cst.Call):
                    self.unsafe_vars.add(name)

    def visit_For(self, node):
        if isinstance(node.target, cst.Name):
            self.unsafe_vars.add(node.target.value)
        elif isinstance(node.target, cst.Tuple):
            for elt in node.target.elements:
                if isinstance(elt.value, cst.Name):
                    self.unsafe_vars.add(elt.value.value)

    def visit_Global(self, node):
        # A `global`/`nonlocal` name is shared with a binding in another scope,
        # which a scope-local rename would not reach.
        self.unsafe_vars.update(item.name.value for item in node.names)

    def visit_Nonlocal(self, node):
        self.unsafe_vars.update(item.name.value for item in node.names)


class _Renamer(cst.CSTTransformer):
    METADATA_DEPENDENCIES = (metadata.ParentNodeProvider,)

    def __init__(self, renames_by_scope):
        super().__init__()
        self.renames_by_scope = renames_by_scope
        self._active = [{}]

    def _enter_scope(self, node):
        inherited = self._active[-1]
        owned = self.renames_by_scope.get(node)

        if not inherited and not owned:
            self._active.append(inherited)
            return

        active = dict(inherited)

        # Whatever this scope binds itself is a different variable.
        for name in bindings_in_scope(node):
            active.pop(name, None)

        if owned:
            active.update(owned)

        self._active.append(active)

    def visit_FunctionDef(self, node):
        self._enter_scope(node)
        return True

    def leave_FunctionDef(self, original_node, updated_node):
        self._active.pop()
        return updated_node

    def visit_ClassDef(self, node):
        self._enter_scope(node)
        return True

    def leave_ClassDef(self, original_node, updated_node):
        self._active.pop()
        return updated_node

    def visit_Lambda(self, node):
        self._enter_scope(node)
        return True

    def leave_Lambda(self, original_node, updated_node):
        self._active.pop()
        return updated_node

    def visit_Import(self, node):
        return False

    def visit_ImportFrom(self, node):
        return False

    def leave_Name(self, original_node, updated_node):
        new_name = self._active[-1].get(original_node.value)
        if new_name is None:
            return updated_node

        parent = self.get_metadata(metadata.ParentNodeProvider, original_node)

        # `obj.name` is an attribute, `f(name=...)` is a keyword, and the names
        # in a `global`/`nonlocal` statement refer to a binding somewhere else.
        if isinstance(parent, cst.Attribute) and parent.attr is original_node:
            return updated_node
        if isinstance(parent, cst.Arg) and parent.keyword is original_node:
            return updated_node
        if isinstance(parent, (cst.NameItem, cst.Param)):
            return updated_node

        return updated_node.with_changes(value=new_name)


_COMPREHENSION_NODES = (cst.ListComp, cst.SetComp, cst.DictComp, cst.GeneratorExp)


def _enclosing_scope(node, parent_map):
    """Return the innermost scope containing ``node``."""
    current = node
    while current in parent_map:
        current = parent_map[current]
        if isinstance(current, _COMPREHENSION_NODES):
            return current
        if isinstance(current, (cst.FunctionDef, cst.Lambda, cst.ClassDef, cst.Module)):
            return current
    return None


class LocalVariableRenamer(cst.CSTTransformer):
    METADATA_DEPENDENCIES = (metadata.PositionProvider,)

    def __init__(self, target_coords: list[tuple[int, int]]):
        super().__init__()
        self.target_coords = target_coords

    def leave_Module(self, original_node, updated_node):
        # One wrapper, kept identity-stable, so the scopes found during
        # analysis are the same objects the renamer sees while visiting.
        wrapper = cst.MetadataWrapper(updated_node, unsafe_skip_copy=True)

        collector = _Collector()
        wrapper.visit(collector)

        position_map = wrapper.resolve(metadata.PositionProvider)
        parent_map = wrapper.resolve(metadata.ParentNodeProvider)

        # Each selected variable is tied to the scope its assignment sits in,
        # so the rename can be confined to that scope.
        selected = set()

        for node, pos in position_map.items():
            if not isinstance(node, cst.Name):
                continue

            if (pos.start.line, pos.start.column) not in self.target_coords:
                continue

            scope = _enclosing_scope(node, parent_map)
            if isinstance(scope, cst.FunctionDef):
                selected.add((scope, node.value))

        counts_by_scope = {}
        renames_by_scope = {}
        used_names = set()

        def in_source_order(item):
            """Order renames by where they occur, so a rerun assigns the same names."""
            scope, name = item
            start = position_map[scope].start
            return (start.line, start.column, name)

        for scope, name in sorted(selected, key=in_source_order):
            if name in dir(builtins):
                continue

            if name in collector.unsafe_vars:
                continue

            if not (name.isalpha() and len(name) >= 3):
                continue

            if keyword.iskeyword(name) or is_reserved(name):
                continue

            if scope not in counts_by_scope:
                counts_by_scope[scope] = binding_counts_in_scope(scope)

            # More than one binding in the scope means control flow decides
            # which one a reference sees; leave those alone.
            if counts_by_scope[scope].get(name) != 1:
                continue

            new_name = wordnet_name(name)

            # A new name that already means something in this module would
            # capture references instead of just relabelling one variable.
            while (
                not new_name
                or new_name in used_names
                or new_name in collector.all_names
                or new_name in dir(builtins)
                or keyword.iskeyword(new_name)
            ):
                new_name = random_name()

            renames_by_scope.setdefault(scope, {})[name] = new_name
            used_names.add(new_name)

        return wrapper.visit(_Renamer(renames_by_scope))
