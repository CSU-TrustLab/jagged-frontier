from typing import Dict, Optional, Set

from semantic_transformer.core.utils import generate_random_name, synonym_or_random_name


class Scope:
    def __init__(self, parent: Optional["Scope"] = None):
        self.parent = parent
        self.rename_map: Dict[str, str] = {}
        self.used_names: Set[str] = set()

    def get_or_create_name(self, var: str, length=5, should_use_synonyms=False) -> str:
        if var not in self.rename_map:
            new_name = self.generate_new_name(var, length, should_use_synonyms)
            self.rename_map[var] = new_name
        return self.rename_map[var]

    def generate_new_name(self, variable_name: str, length: int, should_use_synonyms: bool) -> str:
        while True:
            if should_use_synonyms:
                new_random_name = synonym_or_random_name(variable_name, length)
            else:
                new_random_name = generate_random_name(length)
            if new_random_name not in self.used_names:
                self.used_names.add(new_random_name)
                return new_random_name
