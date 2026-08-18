import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import jedi
import libcst as cst
from jedi.api.classes import Name
from libcst import metadata


class FunctionToRename:
    """
    Implementation of function to rename.

    Attributes
    ----------
    name : str
        Name of the function.
    line : int, optional
        Line number (starting from 1) of its definition.
    column : int, optional
        Column index (starting from 0) of its definition.
    module_path : Path, optional
        Path of the module of its definition.

    Limitations
    -----------
    - Possible "collisions" in case of identical function names across multiple files.
    - The current renaming operator is very simple (WIP).
    """

    def __init__(
        self,
        name: str,
        line: Optional[int] = None,
        column: Optional[int] = None,
        module_path: Optional[Path] = None,
    ):
        self.name = name
        self.line = line
        self.column = column
        self.module_path = module_path
        self._rename()

    def _rename(self):
        name_as_lst = list(self.name)
        name_as_lst.reverse()
        self.new_name = self.name + "".join(name_as_lst)

    @property
    def pos(self):
        if self.line and self.column:
            return (self.line, self.column)
        else:
            return None

    def __str__(self):
        return f"{self.name} {self.line, self.column}"


class FunctionNameCollector(cst.CSTVisitor):
    """
    Visitor collector of all the names of defined functions in the **global** scope.
    """

    METADATA_DEPENDENCIES = (
        metadata.ParentNodeProvider,
        metadata.PositionProvider,
    )

    def __init__(self, file_path: Optional[Union[str, Path]] = None):
        super().__init__()
        self.def_func_names = set()  # type: set[FunctionToRename]
        self.file_path = Path(file_path).resolve() if file_path else None

    def visit_FunctionDef(self, node):
        parent = self.get_metadata(metadata.ParentNodeProvider, node)
        # only Module level (Global)
        if isinstance(parent, cst.Module):
            # position of the function's name
            pos = self.get_metadata(metadata.PositionProvider, node.name)
            self.def_func_names.add(
                FunctionToRename(
                    name=node.name.value,
                    line=pos.start.line,
                    column=pos.start.column,
                    module_path=self.file_path,
                )
            )


def collect_changes_with_jedi(script: jedi.Script, func_names: List[FunctionToRename]):
    """
    Compute all the changes to do for a source (Python) file.
    This done by a double looping process that, iterating over the function to rename:
    - Finding all the names that match the targeted function's one.
    - For each matched name, compare its definition.
        A change is added *only if* the latter matches the targeted function's one.
        In particular, if **none** or **more than one** definition(s) is found by jedi, the name is discarded.

    Parameters
    ----------
    script : jedi.Script
        Script of the file to process.
    func_names : List[FunctionToRename]
        List of functions to rename.

    Returns
    -------
    changes : List[Tuple[int, int, str, str]]
        A list of changes.
    match_results : Dict[FunctionToRename, Dict[str, int]]
        Match results per function to rename.
        Each result consists of a dictionary with the entries "unsafe", "matched", "unmatched".
    """
    names = script.get_names(all_scopes=True, definitions=True, references=True)

    changes = []  # type: List[Tuple[int, int, str, str]]
    match_results = {}  # type: Dict[FunctionToRename, Dict[str, int]]

    def _match_function_to_rename(target_name: FunctionToRename, name: Name):
        if (target_name.pos is None) and (target_name.module_path is None):
            raise ValueError("NOT ENOUGH INFORMATION FOUND IN TARGET FUNCTION.")

        matched = True
        if target_name.module_path:
            matched &= target_name.module_path == name.module_path

        if not target_name.pos:
            warnings.warn("LOOSE NAME MATCHING: ONLY BASED ON FILE_PATH.", RuntimeWarning)
            return matched

        # early return in case of different files...
        if not matched:
            return matched
        else:
            return target_name.pos == (name.line, name.column)

    for func in func_names:
        # find all the candidate names
        matched_names = [n for n in names if n.name == func.name]
        match_result = {"matched": 0, "unmatched": 0, "unsafe": 0}
        for matched_name in matched_names:
            defs = script.goto(
                line=matched_name.line, column=matched_name.column, follow_imports=True
            )
            # discards the name in case of doubt about its definition (zero or multiple)
            if len(defs) == 1:
                definition = defs[0]  # type: Name
                # compares the definition of the matched name with the targeted function
                if _match_function_to_rename(func, definition):
                    changes.append(
                        (
                            matched_name.line,
                            matched_name.column,
                            matched_name.name,
                            func.new_name,
                        )
                    )
                    match_result["matched"] += 1
                else:
                    match_result["unmatched"] += 1
            else:
                match_result["unsafe"] += 1
        match_results[func] = match_result
    return changes, match_results


def refractor_code(
    code: str,
    changes: Dict[int, Tuple[int, str, str]],
    check: bool = False,
) -> str:
    lines = code.splitlines(keepends=True)
    new_lines = []

    # for now, we don't assume that the changes are sorted
    # later, it will be quicker to sort them (or assuming so) and iterate over the dictionary's data?
    for line_idx, line in enumerate(lines, start=1):
        line_changes = changes.get(line_idx, None)
        if not line_changes:
            new_line = line
        else:
            # applies changes
            i = 0
            new_line = ""
            for col, name, new_name in line_changes:
                # adds the unchanged characters
                new_line += line[i:col]
                # input checking (unoptimized for now)
                if check:
                    if not line[col:].startswith(name):
                        raise ValueError(
                            f"CHANGE ERROR AT LINE {line_idx}: found '{line[col : col + len(new_name)]}' instead of '{name}'."
                        )
                new_line += new_name
                # moves to the next
                i = col + len(name)
            # /!\ adds the possible end of the line /!\
            new_line += line[i:]
        new_lines.append(new_line)
    return "".join(new_lines)


def refractor_file(
    src_file: Path,
    changes: Dict[int, Tuple[int, str, str]],
    check: bool = False,
) -> str:
    return refractor_code(src_file.read_text(), changes, check)


class FunctionNameRenamer:
    """
    Class that applies the function names renaming to a single file.
    """

    def transform_code(self, code: str) -> str:
        tree = cst.parse_module(code)
        wrapper = cst.MetadataWrapper(tree)
        visitor = FunctionNameCollector(file_path=None)
        wrapper.visit(visitor)

        if len(visitor.def_func_names) == 0:
            return tree.code

        func_names = set()  # type: set[FunctionToRename]
        for e in visitor.def_func_names:
            func_names.add(e)

        script = jedi.Script(code=code)
        changes, _match_results = collect_changes_with_jedi(
            script,
            list(func_names),
        )

        changes = sorted(changes, key=lambda t: t[0])
        line_changes = {k: [] for k in set([c[0] for c in changes])}  # type: Dict[int, List[Tuple[int, str, str]]]
        for change in changes:
            line_changes[change[0]].append(change[1:])
        for k, v in line_changes.items():
            line_changes[k] = sorted(v, key=lambda t: t[0])
        transformed_code = refractor_code(code, line_changes, check=True)
        return transformed_code


if __name__ == "__main__":
    # Example of usage on an entire directory
    from semantic_transformer.core.utils import collect_python_files

    src = Path("path_to_example_repository")
    dest = Path("out")
    dest.mkdir(parents=True, exist_ok=True)

    python_files, _ = collect_python_files(src)

    func_names = set()  # type: set[FunctionToRename]

    for path in python_files:
        node = cst.parse_module(path.read_text())
        wrapper = cst.MetadataWrapper(node)
        visitor = FunctionNameCollector()
        wrapper.visit(visitor)

        if len(visitor.def_func_names) != 0:
            def_func_names = [func_name.name for func_name in visitor.def_func_names]
            for e in visitor.def_func_names:
                func_names.add(e)

    # processes each file *at once*
    project = jedi.Project(path=src)
    for path in python_files:
        transformed_path = Path(dest) / path.relative_to(src)
        transformed_path.parent.mkdir(parents=True, exist_ok=True)

        script = jedi.Script(path=path.resolve(), project=project)
        changes, _match_results = collect_changes_with_jedi(script, func_names)

        # not necessary
        changes = sorted(changes, key=lambda t: t[0])

        line_changes = {k: [] for k in set([c[0] for c in changes])}  # type: Dict[int, List[Tuple[int, str, str]]]

        for change in changes:
            line_changes[change[0]].append(change[1:])

        # /!\ sorts by column index
        for k, v in line_changes.items():
            line_changes[k] = sorted(v, key=lambda t: t[0])

        transformed_code = refractor_file(path, line_changes, check=True)
        transformed_path.write_text(transformed_code)
