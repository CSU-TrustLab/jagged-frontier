from __future__ import annotations

import subprocess
from pathlib import Path


class CodeQLDatabaseBuilder:
    def __init__(
        self,
        repo_path: str,
        database_path: str,
        language: str = "python",
        codeql_executable: str = "codeql",
    ) -> None:

        self.repo_path = Path(repo_path).resolve()
        self.database_path = Path(database_path).resolve()
        self.language = language
        self.codeql_executable = codeql_executable

    def create_db(self) -> None:
        self.database_path.parent.mkdir(parents=True, exist_ok=True)

        cmd = [
            self.codeql_executable,
            "database",
            "create",
            str(self.database_path),
            f"--language={self.language}",
            "--source-root",
            str(self.repo_path),
            "--overwrite",
        ]

        subprocess.run(cmd, check=True)

        print(f"Database created at: {self.database_path}")
