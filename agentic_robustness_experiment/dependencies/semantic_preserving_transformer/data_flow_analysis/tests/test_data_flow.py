"""Tests for the data_flow_analysis pipeline.

These tests require CodeQL and must be run inside the Docker container:
    docker run --rm -v "${PWD}:/app" -w /app semantic-transformer pytest data_flow_analysis/tests/
"""

from __future__ import annotations

from pathlib import Path

import pytest

from data_flow_analysis.codeql_tools import CodeQLQueryRunner, QueryStringBuilder

ROOT = Path(__file__).resolve().parent.parent
DB_PATH = ROOT / "repo_databases" / "test_db"
FILE_NAME = "sample.py"


@pytest.fixture(scope="module")
def runner():
    return CodeQLQueryRunner(DB_PATH)


@pytest.fixture(scope="module")
def qb():
    return QueryStringBuilder()


class TestLineVariableAccessQuery:
    def test_write(self, runner, qb):
        rows = runner.run_query(qb.line_variable_access_query(FILE_NAME, 2))
        data = rows[1:]  # skip header
        var_names = [r[0] for r in data]
        roles = [r[1] for r in data]
        assert "x" in var_names
        assert "write" in roles

    def test_read_and_write(self, runner, qb):
        rows = runner.run_query(qb.line_variable_access_query(FILE_NAME, 3))
        data = rows[1:]
        pairs = {(r[0], r[1]) for r in data}
        assert ("y", "write") in pairs
        assert ("x", "read") in pairs


class TestLineCallsFunctionQuery:
    def test_call_detected(self, runner, qb):
        rows = runner.run_query(qb.line_calls_function_query(FILE_NAME, 4))
        data = rows[1:]
        assert len(data) >= 1
        call_exprs = [r[0] for r in data]
        assert any("print" in c for c in call_exprs)


class TestPointsToSameObjectQuery:
    def test_alias_detected(self, runner, qb):
        rows = runner.run_query(qb.points_to_same_object_query(FILE_NAME, "a", "b"))
        data = rows[1:]
        assert len(data) >= 1

    def test_no_alias(self, runner, qb):
        rows = runner.run_query(qb.points_to_same_object_query(FILE_NAME, "x", "a"))
        data = rows[1:]
        assert len(data) == 0


class TestDataFlowsLocallyQuery:
    def test_flow_x_to_y(self, runner, qb):
        """x is defined on line 2 and read on line 3 (y = x + 1)."""
        rows = runner.run_query(qb.data_flows_locally_query(FILE_NAME, 2, 3))
        data = rows[1:]
        assert len(data) >= 1

    def test_flow_y_to_print(self, runner, qb):
        """y is defined on line 3 and read on line 4 (print(y))."""
        rows = runner.run_query(qb.data_flows_locally_query(FILE_NAME, 3, 4))
        data = rows[1:]
        assert len(data) >= 1

    def test_flow_a_to_b(self, runner, qb):
        """a is defined on line 5 and read on line 6 (b = a)."""
        rows = runner.run_query(qb.data_flows_locally_query(FILE_NAME, 5, 6))
        data = rows[1:]
        assert len(data) >= 1

    def test_no_flow_x_to_print(self, runner, qb):
        """x (line 2) is not used on line 4, so no direct flow."""
        rows = runner.run_query(qb.data_flows_locally_query(FILE_NAME, 2, 4))
        data = rows[1:]
        assert len(data) == 0
