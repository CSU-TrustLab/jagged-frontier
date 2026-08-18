from abc import ABC, abstractmethod
from pathlib import Path


class Strategy(ABC):
    @abstractmethod
    def select_candidates(self, candidates: list) -> list:
        pass

    @abstractmethod
    def generate_samples(
        self, seed_location: str, size: int, dest_path: str
    ) -> list[tuple[str, Path] | tuple[str, Path, list[dict]]]:
        pass

    @abstractmethod
    def transform_directory(
        self,
        src_path: str | Path,
        transformation_name: str,
        exclude_files: set[str] | None = None,
        exclude_patterns: list[str] | None = None,
        **kwargs,
    ):
        pass

    @abstractmethod
    def transform_file(
        self,
        source_code: str,
        transformation_name: str,
        **kwargs,
    ) -> str:
        pass
