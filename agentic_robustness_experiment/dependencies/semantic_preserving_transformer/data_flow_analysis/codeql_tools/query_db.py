from __future__ import annotations

import csv
import subprocess
from pathlib import Path

QUERY_OUTPUT_DIR = Path(__file__).resolve().parent.parent / "query_results"


class CodeQLQueryRunner:
    def __init__(
        self,
        database_path: str | Path,
        codeql_executable: str = "codeql",
        output_dir: Path = QUERY_OUTPUT_DIR,
    ) -> None:
        self.database_path = Path(database_path).resolve()
        self.codeql_executable = codeql_executable
        self.output_dir = output_dir

    def run_query(self, query_string: str) -> list[list[str]]:

        ql_path = self._write_query_to_file(query_string)

        bqrs_path = self.output_dir / ql_path.with_suffix(".bqrs").name
        csv_path = self.output_dir / ql_path.with_suffix(".csv").name

        subprocess.run(
            [
                self.codeql_executable,
                "query",
                "run",
                str(ql_path),
                "--database",
                str(self.database_path),
                "--output",
                str(bqrs_path),
                "--additional-packs",
                str(Path.home() / ".codeql" / "packages"),
            ],
            check=True,
        )
        subprocess.run(
            [
                self.codeql_executable,
                "bqrs",
                "decode",
                str(bqrs_path),
                "--format=csv",
                "--output",
                str(csv_path),
            ],
            check=True,
        )

        with csv_path.open(newline="", encoding="utf-8") as fh:
            rows = list(csv.reader(fh))

        for p in (ql_path, bqrs_path, csv_path):
            p.unlink(missing_ok=True)

        return rows

    def _write_query_to_file(self, query_string: str) -> Path:
        self._ensure_qlpack()
        ql_path = self.output_dir / "query.ql"
        ql_path.write_text(query_string, encoding="utf-8")
        return ql_path

    def _ensure_qlpack(self) -> None:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        qlpack = self.output_dir / "qlpack.yml"
        if not qlpack.exists():
            qlpack.write_text(
                'name: custom-queries\nversion: 0.0.0\ndependencies:\n  codeql/python-all: "*"\n',
                encoding="utf-8",
            )
