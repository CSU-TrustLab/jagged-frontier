import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from ai_analysis.checks import run_deterministic_checks
from ai_analysis.input import _clean_issue_description, discover_cases, load_case
from ai_analysis.pipeline import (
    _compact_enriched_context,
    _generate_validated,
    _prompt_events,
    _repair_trajectory_response,
    run_ai_assisted_pipeline,
)
from ai_analysis.filtering import (
    MAX_PROMPT_CHARS,
    _build_prompt_with_budget,
    _validate_triage_response,
    run_ai_filter_pipeline,
)
from ai_analysis.validator import validate_clue_analysis, validate_trajectory_analysis
from llm import LLMBackend


REPO_ROOT = Path(__file__).resolve().parents[1]
SAMPLE_ROOT = REPO_ROOT / "tests" / "fixtures" / "sample_trajectories"


class FakeBackend:
    def generate_json(self, prompt):
        if "extract contextual clues" in prompt:
            return {
                "clues": [
                    {
                        "id": "C0",
                        "category": "B1",
                        "type": "file_path",
                        "quote": "sampleproj/parser.py",
                        "value": "sampleproj/parser.py",
                        "role": "localization",
                        "signal_strength": "high",
                        "confidence": 1.0,
                    }
                ],
                "summary": {
                    "primary_categories": ["B"],
                    "solution_leakage": "low",
                    "summary": "The issue names the affected file.",
                },
            }

        if "Let me start" in prompt:
            event_id = "E2"
            quote = "Let me start"
            unclassified = ["E0", "E1", "E3", "E4", "E5"]
        else:
            event_id = "E1"
            quote = "I'll start"
            unclassified = ["E0", "E2"]
        evidence = [{"event_id": event_id, "quote": quote, "why": "The agent begins localization."}]
        return {
            "nodes": [
                {
                    "id": "N0",
                    "phase": "Localization",
                    "title": "Locate parser implementation",
                    "summary": "The agent starts from the file named in the issue.",
                    "intent": "Inspect the affected parser.",
                    "outcome": "The parser implementation is inspected.",
                    "resources": ["sampleproj/parser.py"],
                    "event_ids": [event_id],
                    "evidence": evidence,
                    "confidence": 0.9,
                }
            ],
            "unclassified_event_ids": unclassified,
        }


class FakeFilterBackend:
    def generate_json(self, prompt):
        marker = "COMPACT TRAJECTORY EVENTS\n"
        start = prompt.index(marker) + len(marker)
        end = prompt.index("\n\nReturn strict JSON only", start)
        events = json.loads(prompt[start:end])
        evidence_event = next(
            (event for event in events if isinstance(event.get("content"), str) and event["content"].strip()),
            events[0],
        )
        event_id = evidence_event["id"]
        source = evidence_event.get("content") or evidence_event.get("content_start") or ""
        words = source.strip().split()
        quote = " ".join(words[: min(4, len(words))]) if words else "event"

        include = "miniswe" in prompt.lower()
        priority = "high" if include else "low"
        score = 84 if include else 41
        tags = ["strong_iteration_signal", "metadata_alignment"] if include else ["straightforward_pass"]
        return {
            "include": include,
            "priority": priority,
            "confidence": 0.86 if include else 0.62,
            "score": score,
            "tags": tags,
            "rationale": "The trajectory has meaningful validation and debugging evidence.",
            "metadata_evidence": [
                "Evaluator and trajectory-level test signals are informative.",
                "Patch/test behavior provides discriminative value."
            ],
            "trajectory_evidence": [
                {
                    "event_id": event_id,
                    "quote": quote,
                    "why": "Shows substantial actionable behavior."
                }
            ]
        }


class InputDiscoveryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cases, cls.errors = discover_cases(str(SAMPLE_ROOT))

    def test_discovers_both_supported_formats(self):
        self.assertEqual(self.errors, [])
        self.assertEqual({case["format"] for case in self.cases}, {"mini-swe-agent", "opencode"})

    def test_matches_detailed_per_case_results(self):
        self.assertEqual(len(self.cases), 2)
        for case in self.cases:
            self.assertEqual(case["result"]["status"], "matched")
            self.assertIs(case["result"]["resolved"], True)
            self.assertIn("FAIL_TO_PASS", case["result"]["tests_status"])

    def test_normalizes_opencode_tool_parts(self):
        case = next(case for case in self.cases if case["format"] == "opencode")
        tool_events = [event for event in case["events"] if event["kind"] == "tool"]
        self.assertEqual(len(tool_events), 1)
        self.assertIn("sed -n", tool_events[0]["content"])
        self.assertIn("sampleproj/parser.py", tool_events[0]["content"])

    def test_missing_result_does_not_borrow_sibling_result(self):
        with tempfile.TemporaryDirectory() as input_dir:
            root = Path(input_dir)
            for case_name in ("case-a", "case-b"):
                case_dir = root / case_name
                case_dir.mkdir()
                trajectory = {
                    "trajectory_format": "mini-swe-agent-1",
                    "instance_id": case_name,
                    "messages": [{"role": "user", "content": f"Issue for {case_name}"}],
                }
                (case_dir / f"{case_name}.traj.json").write_text(json.dumps(trajectory))
            (root / "case-b" / "result.json").write_text(
                json.dumps({"case-b": {"resolved": True}})
            )

            cases, errors = discover_cases(input_dir)
            self.assertEqual(errors, [])
            by_id = {case["case_id"]: case for case in cases}
            self.assertEqual(by_id["case-a"]["result"]["status"], "missing")
            self.assertEqual(by_id["case-b"]["result"]["status"], "matched")

    def test_opencode_info_issue_description_is_parsed(self):
        with tempfile.TemporaryDirectory() as input_dir:
            case_dir = Path(input_dir) / "case-oc"
            case_dir.mkdir()
            trajectory = {
                "info": {
                    "id": "ses_test",
                    "issue_description": '"## Title \\nFix the \\"parser\\" bug\\n\\nDetails here."',
                },
                "messages": [
                    {
                        "info": {"role": "assistant"},
                        "parts": [{"type": "text", "text": "I'll explore the code."}],
                    }
                ],
            }
            (case_dir / "ses_test.traj.json").write_text(json.dumps(trajectory))
            (case_dir / "result.json").write_text(json.dumps({"case-oc": {"resolved": True}}))

            case = load_case(case_dir / "ses_test.traj.json", Path(input_dir).resolve())
            self.assertEqual(case["format"], "opencode")
            self.assertIn('## Title', case["issue_description"])
            self.assertIn('Fix the "parser" bug', case["issue_description"])
            self.assertNotIn("\\n", case["issue_description"])
            self.assertIsNone(case["issue_event_id"])

    def test_clean_issue_description_variants(self):
        self.assertEqual(_clean_issue_description('"line1\\nline2"'), "line1\nline2")
        self.assertEqual(_clean_issue_description("plain text issue"), "plain text issue")
        self.assertEqual(_clean_issue_description(None), "")
        self.assertEqual(_clean_issue_description(""), "")


class DeterministicCheckTests(unittest.TestCase):
    def test_detects_git_reversion_tests_and_resources(self):
        case = {
            "events": [
                {
                    "id": "E0",
                    "role": "assistant",
                    "kind": "tool",
                    "timestamp": 1.0,
                    "duration_seconds": None,
                    "tool_status": "completed",
                    "tool_input": {"command": "cat src/parser.py && git reset --hard HEAD~1"},
                    "content": "TOOL: bash\nINPUT:\ncat src/parser.py && git reset --hard HEAD~1",
                },
                {
                    "id": "E1",
                    "role": "assistant",
                    "kind": "tool",
                    "timestamp": 2.0,
                    "duration_seconds": None,
                    "tool_status": "completed",
                    "tool_input": {"command": "pytest tests/test_parser.py"},
                    "content": "TOOL: bash\nINPUT:\npytest tests/test_parser.py\nOUTPUT:\n1 passed",
                },
            ],
            "issue_event_id": None,
            "submission": None,
            "result": {"tests_status": {}},
            "spt": {},
        }
        checks = run_deterministic_checks(case)
        self.assertEqual(checks["git"]["reversion_count"], 1)
        self.assertEqual(checks["tests"]["run_count"], 1)
        self.assertEqual(checks["tests"]["events"][0]["outcome"], "passed")
        self.assertIn("src/parser.py", checks["resources"]["unique_resources"])

    def test_detects_test_edit_retest_cycle_and_target_access(self):
        case = {
            "events": [
                {
                    "id": "E0", "role": "assistant", "kind": "tool", "timestamp": 1.0,
                    "duration_seconds": 1.0, "tool_status": "completed",
                    "tool_input": {"command": "pytest tests/test_parser.py"},
                    "content": "pytest tests/test_parser.py\n1 failed",
                },
                {
                    "id": "E1", "role": "assistant", "kind": "tool", "timestamp": 2.0,
                    "duration_seconds": 1.0, "tool_status": "completed",
                    "tool_input": {"command": "apply_patch src/parser.py"},
                    "content": "apply_patch src/parser.py\n*** Update File: src/parser.py\n-old\n+new",
                },
                {
                    "id": "E2", "role": "assistant", "kind": "tool", "timestamp": 3.0,
                    "duration_seconds": 1.0, "tool_status": "completed",
                    "tool_input": {"command": "pytest tests/test_parser.py"},
                    "content": "pytest tests/test_parser.py\n1 passed",
                },
            ],
            "issue_event_id": None,
            "submission": "diff --git a/src/parser.py b/src/parser.py\n+++ b/src/parser.py\n+new",
            "metadata": {"exit_status": "Submitted"},
            "result": {"resolved": True, "tests_status": {}},
            "spt": {},
        }
        checks = run_deterministic_checks(case)
        self.assertEqual(checks["iterations"]["test_edit_cycle_count"], 1)
        self.assertEqual(checks["iterations"]["test_edit_cycles"][0]["retest_outcome"], "passed")
        self.assertEqual(checks["resources"]["first_target_event"], "E1")
        self.assertEqual(checks["result_reconciliation"]["outcome"], "resolved")

    def test_git_global_options_and_read_only_commands(self):
        case = {
            "events": [
                {
                    "id": "E0", "role": "assistant", "kind": "tool", "timestamp": 1.0,
                    "duration_seconds": None, "tool_status": "completed",
                    "tool_input": {"command": "git -C repo status && git -C repo reset --hard HEAD~1"},
                    "content": "git -C repo status && git -C repo reset --hard HEAD~1",
                }
            ],
            "issue_event_id": None, "submission": None, "metadata": {},
            "result": {"tests_status": {}}, "spt": {},
        }
        checks = run_deterministic_checks(case)
        self.assertEqual(checks["git"]["command_count"], 2)
        self.assertEqual(checks["git"]["reversion_count"], 1)


class ValidationTests(unittest.TestCase):
    def test_missing_clue_quote_is_flagged_not_fatal(self):
        result = validate_clue_analysis(
            {
                "clues": [
                    {
                        "id": "C0", "category": "B1", "type": "file_path",
                        "quote": "missing.py", "value": "missing.py", "role": "localization",
                        "signal_strength": "high", "confidence": 1.0,
                    }
                ],
                "summary": {
                    "primary_categories": ["B"], "solution_leakage": "none",
                    "summary": "A file path is present.",
                },
            },
            "The issue affects present.py",
        )
        clue = result["clues"][0]
        self.assertIsNone(clue["start"])
        self.assertFalse(clue["exact_match"])
        self.assertIn("match_warning", clue)

    def test_clue_quote_recovered_case_and_whitespace_insensitively(self):
        issue = "The function\n  parse_stream in Sampleproj/Parser.py fails."
        result = validate_clue_analysis(
            {
                "clues": [
                    {
                        "id": "C0", "category": "B1", "type": "file_path",
                        "quote": "sampleproj/parser.py", "value": "sampleproj/parser.py",
                        "role": "localization", "signal_strength": "high", "confidence": 1.0,
                    },
                    {
                        "id": "C1", "category": "B2", "type": "function_name",
                        "quote": "parse_stream", "value": "parse_stream",
                        "role": "localization", "signal_strength": "high", "confidence": 1.0,
                    },
                ],
                "summary": {
                    "primary_categories": ["B"], "solution_leakage": "none",
                    "summary": "File and function.",
                },
            },
            issue,
        )
        by_id = {clue["id"]: clue for clue in result["clues"]}
        # Case-folded match recovered to the exact source span.
        self.assertEqual(issue[by_id["C0"]["start"]:by_id["C0"]["end"]], "Sampleproj/Parser.py")
        self.assertFalse(by_id["C0"]["exact_match"])
        # Verbatim match stays exact.
        self.assertTrue(by_id["C1"]["exact_match"])
        self.assertNotIn("match_warning", by_id["C1"])

    def test_trajectory_quotes_must_exist_in_event(self):
        analysis = {
            "nodes": [
                {
                    "id": "N0",
                    "phase": "Localization",
                    "title": "Locate code",
                    "summary": "Locate code.",
                    "intent": "Find the code.",
                    "outcome": "Code found.",
                    "resources": [],
                    "event_ids": ["E0"],
                    "evidence": [{"event_id": "E0", "quote": "fabricated"}],
                    "confidence": 1.0,
                }
            ],
            "rubric_assessment": {},
        }
        with self.assertRaisesRegex(ValueError, "not an exact substring"):
            validate_trajectory_analysis(analysis, [{"id": "E0", "content": "real evidence"}])

    def test_all_events_must_be_accounted_for(self):
        evidence = [{"event_id": "E0", "quote": "real", "why": "direct evidence"}]
        assessment = {
            "status": "partial", "answer": "Limited.", "evidence": evidence,
            "confidence": 0.5, "limitations": "Short trace.",
        }
        analysis = {
            "nodes": [{
                "id": "N0", "phase": "Localization", "title": "Locate", "summary": "Locate.",
                "intent": "Find code.", "outcome": "Found code.", "resources": [],
                "event_ids": ["E0"], "evidence": evidence, "confidence": 1.0,
            }],
            "unclassified_event_ids": [],
            "rubric_assessment": {
                key: dict(assessment) for key in (
                    "spt_awareness", "clue_prioritization", "localization",
                    "planning_implementation", "validation", "failure_analysis",
                )
            },
        }
        with self.assertRaisesRegex(ValueError, "not accounted for"):
            validate_trajectory_analysis(
                analysis,
                [{"id": "E0", "content": "real"}, {"id": "E1", "content": "other"}],
            )

    def test_clue_taxonomy_code_is_strict(self):
        with self.assertRaisesRegex(ValueError, "Invalid clue category"):
            validate_clue_analysis(
                {"clues": [{
                    "id": "C0", "category": "A9", "type": "header", "quote": "Issue",
                    "role": "structure", "signal_strength": "high", "confidence": 1.0,
                }], "summary": {}},
                "Issue",
            )


class PipelineReliabilityTests(unittest.TestCase):
    def test_repairs_common_structured_output_mistakes(self):
        response = {
            "nodes": [
                {
                    "id": "N0",
                    "phase": "Localization | Debugging",
                    "title": "Inspect parser",
                    "summary": "The parser is inspected.",
                    "intent": "Understand the parser.",
                    "outcome": "The implementation is found.",
                    "resources": ["src/parser.py"],
                    "event_ids": [],
                    "evidence": [
                        {
                            "event_id": "E0",
                            "quote": "Inspect src/parser.py now",
                            "why": "Shows localization.",
                        }
                    ],
                    "confidence": 0.9,
                }
            ],
            "rubric_assessment": {},
        }
        events = [{"id": "E0", "content": "Inspect\n  src/parser.py now"}]
        repaired = _repair_trajectory_response(response, events)
        self.assertEqual(repaired["nodes"][0]["phase"], "Localization")
        self.assertEqual(repaired["nodes"][0]["event_ids"], ["E0"])
        self.assertEqual(
            repaired["nodes"][0]["evidence"][0]["quote"],
            "Inspect\n  src/parser.py now",
        )
        self.assertEqual(repaired["unclassified_event_ids"], [])
        self.assertEqual(len(repaired["normalization_warnings"]), 2)

    def test_node_evidence_recovered_case_and_punctuation_insensitively(self):
        response = {
            "nodes": [
                {
                    "id": "N0",
                    "phase": "Debugging",
                    "title": "Root cause",
                    "summary": "Identifies the cause.",
                    "intent": "Explain the failure.",
                    "outcome": "Cause found.",
                    "resources": [],
                    "event_ids": ["E0"],
                    "evidence": [
                        {
                            "event_id": "E0",
                            # Case-folded and straight-quote version of the source.
                            "quote": "the parser raises 'truncatederror' early",
                            "why": "States the root cause.",
                        }
                    ],
                    "confidence": 0.9,
                }
            ],
        }
        events = [{"id": "E0", "content": "The parser raises \u2018TruncatedError\u2019 early in the loop."}]
        repaired = _repair_trajectory_response(response, events)
        recovered = repaired["nodes"][0]["evidence"][0]["quote"]
        # Snapped to the exact source span so strict validation passes.
        self.assertEqual(recovered, "The parser raises \u2018TruncatedError\u2019 early")
        validate_trajectory_analysis(repaired, events)

    def test_unmatched_node_evidence_is_flagged_not_fatal(self):
        response = {
            "nodes": [
                {
                    "id": "N0",
                    "phase": "Debugging",
                    "title": "Root cause",
                    "summary": "Identifies the cause.",
                    "intent": "Explain the failure.",
                    "outcome": "Cause found.",
                    "resources": [],
                    "event_ids": ["E0"],
                    "evidence": [
                        {
                            "event_id": "E0",
                            "quote": "a sentence that never appears in the event",
                            "why": "Paraphrased by the model.",
                        }
                    ],
                    "confidence": 0.9,
                }
            ],
        }
        events = [{"id": "E0", "content": "The real content of this event is different."}]
        repaired = _repair_trajectory_response(response, events)
        # The case is not rejected; the evidence is kept and flagged.
        validated = validate_trajectory_analysis(repaired, events)
        ev = validated["nodes"][0]["evidence"][0]
        self.assertFalse(ev["exact_match"])
        self.assertIn("match_warning", ev)

    def test_overlapping_and_unordered_nodes_are_repaired(self):
        def node(node_id, event_ids, quote_event):
            return {
                "id": node_id,
                "phase": "Localization",
                "title": node_id,
                "summary": "s",
                "intent": "i",
                "outcome": "o",
                "resources": [],
                "event_ids": list(event_ids),
                "evidence": [{"event_id": quote_event, "quote": "x", "why": "w"}],
                "confidence": 0.9,
            }

        events = [{"id": f"E{i}", "content": "x"} for i in range(4)]
        response = {
            "nodes": [
                # Out of order (later events first) and overlaps E1.
                node("N1", ["E1", "E2", "E3"], "E3"),
                node("N0", ["E0", "E1"], "E0"),
            ]
        }
        repaired = _repair_trajectory_response(response, events)
        # Reordered chronologically and de-overlapped, so strict validation passes.
        validate_trajectory_analysis(repaired, events)
        ids = [n["id"] for n in repaired["nodes"]]
        self.assertEqual(ids, ["N0", "N1"])
        n1 = next(n for n in repaired["nodes"] if n["id"] == "N1")
        self.assertEqual(n1["event_ids"], ["E2", "E3"])  # E1 trimmed (claimed by N0)
        self.assertTrue(repaired["normalization_warnings"])

    def test_bedrock_token_alias_is_accepted(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            config_path = Path(temp_dir) / "model.yaml"
            config_path.write_text(
                "provider: bedrock\nmodel: test-model\nregion: us-east-1\n",
                encoding="utf-8",
            )
            environment = {
                key: value
                for key, value in os.environ.items()
                if key not in {"AWS_BEARER_TOKEN_BEDROCK", "AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY"}
            }
            environment["AWS_TOKEN_BEDROCK"] = "test-token"
            with patch.dict(os.environ, environment, clear=True):
                backend = LLMBackend(str(config_path))
                self.assertEqual(backend.bedrock_bearer_token, "test-token")
                self.assertEqual(os.environ["AWS_BEARER_TOKEN_BEDROCK"], "test-token")

    def test_generation_retries_invalid_output(self):
        class FlakyBackend:
            def __init__(self):
                self.calls = 0

            def generate_json(self, prompt):
                self.calls += 1
                return {"valid": self.calls >= 3}

        backend = FlakyBackend()

        def validator(value):
            if not value["valid"]:
                raise ValueError("not valid yet")
            return value

        value, errors = _generate_validated(backend, "prompt", validator)
        self.assertTrue(value["valid"])
        self.assertEqual(backend.calls, 3)
        self.assertEqual(len(errors), 2)

    def test_prompt_event_content_respects_budget(self):
        events = [
            {"id": f"E{index}", "content": "x" * 100, "role": "assistant", "kind": "message"}
            for index in range(5)
        ]
        prompt_events = _prompt_events(events, character_budget=100)
        visible = sum(
            len(event.get("content", ""))
            + len(event.get("content_start", ""))
            + len(event.get("content_end", ""))
            for event in prompt_events
        )
        self.assertLessEqual(visible, 100)

    def test_triage_validator_accepts_quote_from_truncated_spans(self):
        response = {
            "include": True,
            "priority": "high",
            "confidence": 0.8,
            "score": 77,
            "tags": ["interesting_failure"],
            "rationale": "Useful for manual review.",
            "metadata_evidence": ["Has failures."],
            "trajectory_evidence": [
                {"event_id": "E0", "quote": "tail quote", "why": "Evidence from visible span."}
            ],
        }
        compact_events = [
            {
                "id": "E0",
                "role": "assistant",
                "kind": "tool",
                "content_start": "beginning",
                "content_end": "tail quote",
            }
        ]
        validated = _validate_triage_response(response, compact_events)
        self.assertTrue(validated["include"])

    def test_filter_prompt_builder_enforces_character_budget(self):
        case = {
            "case_id": "c1",
            "case_name": "case-1",
            "format": "mini-swe-agent",
            "trajectory_path": "dataset/case-1/run.traj.json",
            "issue_description": "x" * 30000,
            "result": {
                "status": "matched",
                "resolved": False,
                "tests_status": {
                    "FAIL_TO_PASS": {
                        "success": [],
                        "failure": ["t"] * 500,
                    }
                },
            },
            "spt": {
                "metadata_available": True,
                "entry_count": 50,
                "applied": True,
                "data": [{"k": "v"}] * 500,
            },
            "metadata": {"model": "m", "provider": "p"},
            "events": [
                {
                    "id": f"E{i}",
                    "role": "assistant",
                    "kind": "message",
                    "content": "y" * 4000,
                }
                for i in range(40)
            ],
        }
        deterministic = {
            "event_count": 40,
            "tests": {
                "run_count": 12,
                "failure_count": 8,
                "rerun_count": 3,
                "evaluation_counts": {},
                "events": [
                    {
                        "event_id": f"E{i}",
                        "command": "pytest tests/test_x.py -k long_test_name",
                        "outcome": "failed",
                    }
                    for i in range(20)
                ],
            },
            "patches": {
                "iteration_count": 12,
                "events": [
                    {
                        "event_id": f"E{i}",
                        "files_modified": [f"src/file_{j}.py" for j in range(80)],
                        "additions": 30,
                        "deletions": 10,
                        "hunks": 5,
                    }
                    for i in range(20)
                ],
                "repeatedly_edited_files": {f"src/file_{i}.py": ["E1", "E2"] for i in range(40)},
                "submitted_patch": {"files_modified": [f"src/f_{i}.py" for i in range(200)]},
            },
            "iterations": {
                "test_edit_cycle_count": 7,
                "test_edit_cycles": [{"failed_test_event_id": "E1"}] * 100,
                "git_reversion_event_ids": [f"E{i}" for i in range(100)],
                "repeated_file_edits": {f"src/r_{i}.py": i for i in range(80)},
                "backtracking_signal_count": 20,
            },
            "errors": {
                "count": 18,
                "events": [
                    {
                        "event_id": f"E{i}",
                        "matched": "Traceback",
                        "snippet": "z" * 3000,
                        "tool_status": "error",
                    }
                    for i in range(40)
                ],
            },
            "result_reconciliation": {"outcome": "unresolved"},
            "spt": {"metadata_available": True, "entry_count": 50, "applied": True},
            "git": {"reversion_count": 4},
            "resources": {"unique_resource_count": 12},
        }

        prompt, compact_events, _ = _build_prompt_with_budget(case, deterministic)
        self.assertLessEqual(len(prompt), MAX_PROMPT_CHARS)
        self.assertTrue(compact_events)

    def test_compact_enriched_context_omits_heavy_payloads(self):
        case = {
            "case_id": "case-1",
            "case_name": "case-1",
            "format": "mini-swe-agent",
            "issue_description": "x" * 4000,
            "result": {
                "status": "matched",
                "resolved": True,
                "tests_status": {"FAIL_TO_PASS": {"success": ["t1"], "failure": []}},
            },
            "spt": {
                "metadata_available": True,
                "entry_count": 12,
                "applied": True,
                "data": [
                    {
                        "order": i,
                        "transformation": "comparison_swapper",
                        "file": f"src/mod_{i % 3}.py",
                        "positions": [{"line": i + 1, "column": 5}],
                    }
                    for i in range(100)
                ],
            },
            "metadata": {"model": "m", "provider": "p", "cost": 0.3},
        }
        deterministic = {
            "event_count": 10,
            "tests": {"run_count": 2, "failure_count": 1, "rerun_count": 1},
            "patches": {"iteration_count": 2},
            "iterations": {"backtracking_signal_count": 1, "git_reversion_event_ids": [f"E{i}" for i in range(50)]},
            "errors": {"count": 1},
            "result_reconciliation": {"outcome": "resolved"},
        }
        payload = _compact_enriched_context(case, deterministic)
        self.assertIn("issue_description_excerpt", payload["case"])
        self.assertNotIn("data", payload["spt"])
        self.assertIn("spt_mutations_summary", payload)
        self.assertEqual(payload["spt_mutations_summary"]["total_entries"], 100)
        self.assertTrue(payload["spt_mutations_summary"]["files_touched"])
        self.assertTrue(payload["spt_mutations_summary"]["transformations"])
        self.assertLessEqual(len(payload["case"]["issue_description_excerpt"]), 1200 + 18)
        self.assertLessEqual(len(payload["deterministic_summary"]["git_reversion_event_ids"]), 20)


class EndToEndTests(unittest.TestCase):
    def test_sample_directory_generates_json_and_html(self):
        with tempfile.TemporaryDirectory() as output_dir:
            summary = run_ai_assisted_pipeline(
                str(SAMPLE_ROOT),
                output_dir,
                str(REPO_ROOT / "config" / "model_config.yaml"),
                backend=FakeBackend(),
            )
            self.assertEqual(summary["completed"], 2)
            self.assertEqual(summary["failed"], 0)
            self.assertTrue((Path(output_dir) / "summary.json").is_file())
            self.assertTrue((Path(output_dir) / "summary.csv").is_file())
            self.assertTrue((Path(output_dir) / "index.html").is_file())
            for case in summary["cases"]:
                case_dir = Path(output_dir) / case["output_directory"]
                self.assertTrue((case_dir / "analysis.json").is_file())
                report = (case_dir / "report.html").read_text(encoding="utf-8")
                self.assertIn("Issue description", report)
                self.assertIn("Trajectory graph", report)
                self.assertIn('class="explorer"', report)
                self.assertIn("showNode(", report)
                self.assertIn('id="report-data"', report)
                self.assertIn('data-tab="issue"', report)
                self.assertIn('data-tab="graph"', report)
                self.assertIn('data-tab="raw"', report)
                self.assertIn('data-tab="rules"', report)
                self.assertIn('data-tab="spt"', report)
                self.assertIn('data-tab="patch"', report)
                self.assertIn('data-tab="manual"', report)
                self.assertIn('id="eval-form"', report)
                self.assertIn("saveEval", report)
                self.assertIn("jumpToEvent", report)
                self.assertIn('class="rawcard"', report)
                self.assertIn("Issue keyword grep", report)
                self.assertIn("SPT metadata", report)
                self.assertIn("SPT-trajectory overlap", report)
                self.assertIn("Submitted patch", report)
                self.assertNotIn("Manual evaluation rubric", report)
                self.assertNotIn("<pr_description>", report)
                self.assertNotIn("<issue>", report)
                artifact = json.loads((case_dir / "analysis.json").read_text(encoding="utf-8"))
                self.assertEqual(artifact["schema_version"], "ai-trajectory-analysis-v1")
                self.assertNotIn("rubric", artifact)
                self.assertIn("git_commands", artifact["rule_based_results"])
                self.assertIn("keyword_grep", artifact["rule_based_results"])
                self.assertIn("phase_counts", artifact["rule_based_results"])
                self.assertIn("classification_coverage", artifact["data_quality"])

    def test_resume_skips_completed_cases(self):
        config = str(REPO_ROOT / "config" / "model_config.yaml")
        with tempfile.TemporaryDirectory() as output_dir:
            first = run_ai_assisted_pipeline(
                str(SAMPLE_ROOT), output_dir, config, backend=FakeBackend()
            )
            self.assertEqual(first["completed"], 2)

            class ExplodingBackend:
                def generate_json(self, prompt):
                    raise AssertionError("resume should not call the model for completed cases")

            second = run_ai_assisted_pipeline(
                str(SAMPLE_ROOT), output_dir, config, backend=ExplodingBackend(), resume=True
            )
            self.assertEqual(second["completed"], 2)
            self.assertEqual(second["skipped"], 2)
            self.assertEqual(second["failed"], 0)
            self.assertTrue(all(case.get("skipped") for case in second["cases"]))

    def test_ai_assisted_enriched_context_flag_is_recorded(self):
        config = str(REPO_ROOT / "config" / "model_config.yaml")
        with tempfile.TemporaryDirectory() as output_dir:
            summary = run_ai_assisted_pipeline(
                str(SAMPLE_ROOT),
                output_dir,
                config,
                backend=FakeBackend(),
                enriched_prompt_context=True,
            )
            self.assertEqual(summary["completed"], 2)
            for case in summary["cases"]:
                case_dir = Path(output_dir) / case["output_directory"]
                artifact = json.loads((case_dir / "analysis.json").read_text(encoding="utf-8"))
                self.assertTrue(artifact["data_quality"]["enriched_prompt_context_enabled"])

    def test_ai_assisted_issue_only_mode(self):
        config = str(REPO_ROOT / "config" / "model_config.yaml")
        with tempfile.TemporaryDirectory() as output_dir:
            summary = run_ai_assisted_pipeline(
                str(SAMPLE_ROOT),
                output_dir,
                config,
                backend=FakeBackend(),
                run_issue_prompt=True,
                run_trajectory_prompt=False,
            )
            self.assertEqual(summary["completed"], 2)
            for case in summary["cases"]:
                case_dir = Path(output_dir) / case["output_directory"]
                artifact = json.loads((case_dir / "analysis.json").read_text(encoding="utf-8"))
                self.assertEqual(artifact["nodes"], [])
                self.assertTrue(artifact["unclassified_event_ids"])
                self.assertTrue(artifact["data_quality"]["issue_prompt_enabled"])
                self.assertFalse(artifact["data_quality"]["trajectory_prompt_enabled"])

    def test_ai_assisted_trajectory_only_mode(self):
        config = str(REPO_ROOT / "config" / "model_config.yaml")
        with tempfile.TemporaryDirectory() as output_dir:
            summary = run_ai_assisted_pipeline(
                str(SAMPLE_ROOT),
                output_dir,
                config,
                backend=FakeBackend(),
                run_issue_prompt=False,
                run_trajectory_prompt=True,
            )
            self.assertEqual(summary["completed"], 2)
            for case in summary["cases"]:
                case_dir = Path(output_dir) / case["output_directory"]
                artifact = json.loads((case_dir / "analysis.json").read_text(encoding="utf-8"))
                self.assertTrue(artifact["nodes"])
                self.assertFalse(artifact["issue_clues"]["clues"])
                self.assertFalse(artifact["data_quality"]["issue_prompt_enabled"])
                self.assertTrue(artifact["data_quality"]["trajectory_prompt_enabled"])

    def test_ai_assisted_artifact_contains_spt_hypothesis_fields(self):
        config = str(REPO_ROOT / "config" / "model_config.yaml")
        with tempfile.TemporaryDirectory() as output_dir:
            summary = run_ai_assisted_pipeline(
                str(SAMPLE_ROOT),
                output_dir,
                config,
                backend=FakeBackend(),
                enriched_prompt_context=True,
            )
            self.assertEqual(summary["completed"], 2)
            for case in summary["cases"]:
                case_dir = Path(output_dir) / case["output_directory"]
                artifact = json.loads((case_dir / "analysis.json").read_text(encoding="utf-8"))
                self.assertIn("spt_impact_hypothesis", artifact)
                self.assertIn("spt_hypothesis_present", artifact["data_quality"])

    def test_filter_mode_generates_ranked_shortlist_outputs(self):
        config = str(REPO_ROOT / "config" / "model_config.yaml")
        with tempfile.TemporaryDirectory() as output_dir:
            summary = run_ai_filter_pipeline(
                str(SAMPLE_ROOT),
                output_dir,
                config,
                backend=FakeFilterBackend(),
            )
            self.assertEqual(summary["completed"], 2)
            self.assertEqual(summary["failed"], 0)
            self.assertEqual(summary["selected"], 1)
            self.assertTrue((Path(output_dir) / "filter_summary.json").is_file())
            self.assertTrue((Path(output_dir) / "filter_shortlist.csv").is_file())

            case_dirs = [
                case["output_directory"]
                for case in summary["cases"]
                if case.get("status") == "completed"
            ]
            for case_dir in case_dirs:
                triage_path = Path(output_dir) / "filter_cases" / case_dir / "triage.json"
                self.assertTrue(triage_path.is_file())
                payload = json.loads(triage_path.read_text(encoding="utf-8"))
                self.assertIn("triage", payload)
                self.assertIn("features", payload)

    def test_filter_mode_resume_skips_existing_triage(self):
        config = str(REPO_ROOT / "config" / "model_config.yaml")
        with tempfile.TemporaryDirectory() as output_dir:
            first = run_ai_filter_pipeline(
                str(SAMPLE_ROOT), output_dir, config, backend=FakeFilterBackend()
            )
            self.assertEqual(first["completed"], 2)

            class ExplodingFilterBackend:
                def generate_json(self, prompt):
                    raise AssertionError("resume should not call the model for completed cases")

            second = run_ai_filter_pipeline(
                str(SAMPLE_ROOT),
                output_dir,
                config,
                backend=ExplodingFilterBackend(),
                resume=True,
            )
            self.assertEqual(second["failed"], 0)
            self.assertEqual(second["skipped"], 2)


if __name__ == "__main__":
    unittest.main()