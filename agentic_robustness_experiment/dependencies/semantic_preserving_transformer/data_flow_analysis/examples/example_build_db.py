from __future__ import annotations

from pathlib import Path

from data_flow_analysis.codeql_tools import CodeQLDatabaseBuilder

if __name__ == "__main__":
    root = Path(__file__).resolve().parent.parent.parent
    builder = CodeQLDatabaseBuilder(
        repo_path=str(root / "examples"),
        database_path=str(root / "data_flow_analysis" / "repo_databases" / "examples_db"),
        language="python",
    )
    builder.create_db()
    print(f"Database created at: {builder.database_path}")
