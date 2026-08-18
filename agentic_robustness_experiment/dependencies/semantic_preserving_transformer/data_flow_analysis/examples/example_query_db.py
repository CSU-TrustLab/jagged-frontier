import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from data_flow_analysis.codeql_tools import CodeQLQueryRunner, QueryStringBuilder

DB_PATH = Path(__file__).resolve().parent.parent / "repo_databases" / "examples_db"
FILE_NAME = "if_else_swap_demo.py"

if __name__ == "__main__":
    runner = CodeQLQueryRunner(DB_PATH)
    qb = QueryStringBuilder()
    for line in (5, 10):
        query = qb.line_variable_access_query(FILE_NAME, line)
        rows = runner.run_query(query)
        print(f"Line {line}: {rows}")
