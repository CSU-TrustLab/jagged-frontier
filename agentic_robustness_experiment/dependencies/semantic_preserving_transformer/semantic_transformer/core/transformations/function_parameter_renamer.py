from typing import Dict, List, Optional

import libcst as cst
from libcst import metadata

from semantic_transformer.core.scope import Scope
from semantic_transformer.core.utils import is_reserved


class FunctionParameterRenamer(cst.CSTTransformer):
    METADATA_DEPENDENCIES = (
        metadata.ParentNodeProvider,
        metadata.QualifiedNameProvider,
    )

    def __init__(self, length=5, should_use_synonyms=False):
        self.scope_stack: List[Scope] = [Scope()]
        self.length = length
        self.should_use_synonyms = should_use_synonyms
        self.function_param_maps: Dict[str, Dict[str, str]] = {}

    @property
    def current_scope(self) -> Scope:
        return self.scope_stack[-1]

    def visit_FunctionDef(self, node: cst.FunctionDef) -> Optional[bool]:
        function_scope = Scope(parent=self.current_scope)
        self.scope_stack.append(function_scope)
        qualifiedNameMetadata = self.get_metadata(metadata.QualifiedNameProvider, node)
        qualifiedName = list(qualifiedNameMetadata)[0].name

        regular_args = [param.name.value for param in node.params.params]
        posonly_args = [param.name.value for param in node.params.posonly_params]
        kwonly_args = [param.name.value for param in node.params.kwonly_params]
        try:
            variable_postional_arg = [node.params.star_arg.name.value]
        except AttributeError:
            variable_postional_arg = []
        try:
            variable_keyword_arg = [node.params.star_kwarg.name.value]
        except AttributeError:
            variable_keyword_arg = []
        all_args = (
            regular_args
            + posonly_args
            + kwonly_args
            + variable_keyword_arg
            + variable_postional_arg
        )

        for arg in all_args:
            if not is_reserved(arg):
                function_scope.get_or_create_name(
                    arg,
                    length=self.length,
                    should_use_synonyms=self.should_use_synonyms,
                )

        if node.name and isinstance(node.name, cst.Name):
            self.function_param_maps[qualifiedName] = dict(function_scope.rename_map)

        return True

    def leave_FunctionDef(
        self, original_node: cst.FunctionDef, updated_node: cst.FunctionDef
    ) -> cst.FunctionDef:
        self.scope_stack.pop()
        return updated_node

    def leave_Param(self, original_node: cst.Param, updated_node: cst.Param) -> cst.Param:
        if original_node.name and isinstance(original_node.name, cst.Name):
            name_str = original_node.name.value
            if name_str in self.current_scope.rename_map:
                return updated_node.with_changes(
                    name=cst.Name(self.current_scope.rename_map[name_str])
                )
        return updated_node

    def leave_Name(self, original_node: cst.Name, updated_node: cst.Name) -> cst.Name:
        name_str = original_node.value
        parent = self.get_metadata(metadata.ParentNodeProvider, original_node)

        if isinstance(parent, cst.Attribute):
            if parent.attr is original_node:
                return updated_node

        if is_reserved(name_str):
            return updated_node
        if name_str in self.current_scope.rename_map:
            return updated_node.with_changes(value=self.current_scope.rename_map[name_str])
        return updated_node

    def leave_Call(self, original_node: cst.Call, updated_node: cst.Call) -> cst.Call:
        qualifiedNameMetadata = self.get_metadata(metadata.QualifiedNameProvider, original_node)
        qualifiedName = list(qualifiedNameMetadata)[0].name
        func = original_node.func
        callee_name = None
        if isinstance(func, cst.Name):
            callee_name = func.value
        elif isinstance(func, cst.Attribute) and isinstance(func.attr, cst.Name):
            callee_name = func.attr.value

        if not callee_name:
            return updated_node

        param_map = self.function_param_maps.get(qualifiedName)
        if not param_map:
            return updated_node

        new_args = []
        changed = False
        for arg in updated_node.args:
            if isinstance(arg.keyword, cst.Name) and arg.keyword.value in param_map:
                transformed_keyword = cst.Name(param_map[arg.keyword.value])
                new_args.append(arg.with_changes(keyword=transformed_keyword))
                changed = True
                continue
            new_args.append(arg)

        if changed:
            return updated_node.with_changes(args=new_args)
        return updated_node
