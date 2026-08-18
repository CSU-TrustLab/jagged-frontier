from __future__ import annotations


class QueryStringBuilder:
    def __init__(self, language: str = "python") -> None:
        self.language = language

    def line_variable_access_query(self, file_name: str, line_number: int) -> str:
        return (
            f"import {self.language}\n"
            f"\n"
            f"string role(Name name) {{\n"
            f'  if name.isDefinition() then result = "write" else result = "read"\n'
            f"}}\n"
            f"\n"
            f"from Name name, Variable v\n"
            f"where\n"
            f"  v = name.getVariable() and\n"
            f"  name.getLocation().getStartLine() = {line_number} and\n"
            f'  name.getLocation().getFile().getBaseName() = "{file_name}"\n'
            f"select\n"
            f"  v.getId() as varName,\n"
            f"  role(name) as role\n"
        )

    def line_calls_function_query(self, file_name: str, line_number: int) -> str:
        return (
            f"import {self.language}\n"
            f"\n"
            f"from Call call, Location loc\n"
            f"where\n"
            f"  loc = call.getLocation() and\n"
            f"  loc.getStartLine() = {line_number} and\n"
            f'  loc.getFile().getBaseName() = "{file_name}"\n'
            f"select\n"
            f"  call.getFunc().toString() as callExpression,\n"
            f"  call.getLocation().getStartColumn() as col\n"
        )

    def data_flows_locally_query(self, file_name: str, source_line: int, target_line: int) -> str:
        return (
            f"import {self.language}\n"
            f"import semmle.python.dataflow.new.DataFlow\n"
            f"\n"
            f"from DataFlow::Node source, DataFlow::Node target\n"
            f"where\n"
            f"  DataFlow::localFlow(source, target) and\n"
            f'  source.getLocation().getFile().getBaseName() = "{file_name}" and\n'
            f'  target.getLocation().getFile().getBaseName() = "{file_name}" and\n'
            f"  source.getLocation().getStartLine() = {source_line} and\n"
            f"  target.getLocation().getStartLine() = {target_line}\n"
            f"select\n"
            f"  source.toString() as sourceNode,\n"
            f"  source.getLocation().getStartLine() as sourceLine,\n"
            f"  target.toString() as targetNode,\n"
            f"  target.getLocation().getStartLine() as targetLine\n"
        )

    def points_to_same_object_query(self, file_name: str, name1: str, name2: str) -> str:
        return (
            f"import {self.language}\n"
            f"import semmle.python.dataflow.new.DataFlow\n"
            f"\n"
            f"from DataFlow::Node source, DataFlow::Node n1, DataFlow::Node n2\n"
            f"where\n"
            f"  DataFlow::localFlow(source, n1) and\n"
            f"  DataFlow::localFlow(source, n2) and\n"
            f"  n1 != n2 and\n"
            f'  n1.asExpr().(Name).getId() = "{name1}" and\n'
            f'  n2.asExpr().(Name).getId() = "{name2}" and\n'
            f'  n1.getLocation().getFile().getBaseName() = "{file_name}" and\n'
            f'  n2.getLocation().getFile().getBaseName() = "{file_name}"\n'
            f"select\n"
            f"  n1.asExpr().(Name).getId() as firstName,\n"
            f"  n1.getLocation().getStartLine() as firstLine,\n"
            f"  n2.asExpr().(Name).getId() as secondName,\n"
            f"  n2.getLocation().getStartLine() as secondLine,\n"
            f"  source.toString() as sharedSource\n"
        )
