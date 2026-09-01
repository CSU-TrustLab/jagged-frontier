"""Prompt templates for AI-assisted trajectory analysis."""


ISSUE_CLUE_PROMPT = """
Role: You extract contextual clues from software issue descriptions.

Analyze only the issue description below. Do not use outside knowledge and do
not infer clues from a solution trajectory.

ISSUE DESCRIPTION
-----------------
{ISSUE_DESCRIPTION}
-----------------

Taxonomy:
A1 section headers; A2 Requirements block; A3 New interfaces block; A4 title.
B1 file paths; B2 functions/methods; B3 classes/types; B4 constants/config;
B5 source permalinks; B6 line locations; B7 signatures/type annotations.
C1 reproduction steps; C2 reproduction code; C3 expected-versus-actual;
C4 program output or before/after behavior.
D1 stack traces; D2 quoted errors/warnings; D3 symptom-only diagnostics.
E1 suggested diff; E2 implementation snippet; E3 prescriptive requirement;
E4 desired outcome or migration steps.
F1 versions/dependencies; F2 OS/platform; F3 configuration; F4 version boundary.
G1 issue/PR references; G2 documentation links; G3 commits/revisions.
H1 labels; H2 issue/component tags.
I1 out-of-band benchmark fields explicitly included in the supplied text.

Return strict JSON only, with this schema:
{{
  "clues": [
    {{
      "id": "C0",
      "category": "B1",
      "type": "file_path",
      "quote": "an exact non-empty substring copied from the issue",
      "value": "normalized useful value",
      "role": "localization | reproduction | diagnosis | solution | constraint | environment | structure | metadata",
      "signal_strength": "low | medium | high",
      "confidence": 0.0
    }}
  ],
  "summary": {{
    "primary_categories": ["B", "E"],
    "solution_leakage": "none | low | medium | high",
    "summary": "short factual summary"
  }}
}}

Rules:
- Every quote must be copied exactly from the issue description.
- Extract all meaningful clues, but do not duplicate the same span.
- Use confidence values from 0 to 1.
- Return no Markdown and no prose outside the JSON object.
"""


TRAJECTORY_ANALYSIS_PROMPT = """
Role: You convert a software-engineering agent trajectory into a phase-oriented
graph. Do not judge the agent or answer any evaluation questions. Only segment
the trajectory into nodes and attach exact supporting evidence.

PHASES
- Localization: finding where to look by searching or reading code.
- Debugging: interpreting behavior and determining root cause.
- Planning: deciding an implementation or validation strategy.
- Patching: changing repository code or tests.
- Validation: running checks and observing results.
- Recovery: explicitly changing direction after failure, including a rollback.
- General: relevant activity that fits no other phase.

ISSUE CLUES
{ISSUE_CLUES_JSON}

NORMALIZED EVENTS
{EVENTS_JSON}

OPTIONAL ENRICHED CONTEXT
{ENRICHED_CONTEXT_JSON}

Return strict JSON only, with this schema:
{{
  "nodes": [
    {{
      "id": "N0",
      "phase": "Localization | Debugging | Planning | Patching | Validation | Recovery | General",
      "title": "short title",
      "summary": "high-level summary",
      "intent": "what the agent was trying to accomplish",
      "outcome": "what happened, or unknown",
      "resources": ["files, tests, commands, or other concrete resources"],
      "event_ids": ["E2", "E3"],
      "evidence": [
        {{
          "event_id": "E2",
          "quote": "exact substring copied from that event's content",
          "why": "why this supports the node"
        }}
      ],
      "confidence": 0.0
    }}
  ],
  "unclassified_event_ids": ["E0", "E1"]{SPT_HYPOTHESIS_SCHEMA_SUFFIX}
}}

Rules:
- Preserve chronology. Nodes must be ordered by their first event ID.
- Every supplied event ID must appear exactly once: either in one node's
  event_ids or in unclassified_event_ids. Never reuse an event across nodes.
- Build a fine-grained, Graphectory-style graph. Create a distinct node for each
  atomic step the agent takes: a reasoning or planning step, a tool action, and
  an observed result are each normally their own node. Prefer more nodes over
  fewer so the graph reflects the real sequence of work.
- Only merge two events into one node when they are inseparable, such as a single
  action and the immediate output it produced with no intervening reasoning.
- Only truly non-substantive events belong in unclassified_event_ids: the system
  instruction, the verbatim issue-description restatement, empty events, and pure
  telemetry. Any event where the agent reasons, searches, reads, edits, runs a
  command, or observes a result must become a node.
- A later return to a phase must remain a separate node.
- Use only event IDs supplied in NORMALIZED EVENTS.
- Use OPTIONAL ENRICHED CONTEXT only as supporting metadata. It does not replace
  evidence from NORMALIZED EVENTS.
{SPT_HYPOTHESIS_RULES}
- Every evidence quote must be an exact, non-empty substring of the referenced
  event's content. Never paraphrase inside quote fields.
- Long events may expose content_start and content_end instead of content. Quotes
  may be copied from either visible span; do not quote omitted middle content.
- Return no Markdown and no prose outside the JSON object.
"""


TRAJECTORY_FILTER_PROMPT = """
Role: You are triaging software-engineering trajectories for manual evaluation.

Goal:
- Decide whether this trajectory is worth human evaluation now.
- Use both trajectory evidence and provided metadata signals.

Signals to consider (non-exhaustive):
- SPT metadata (availability, entry count, applied flag)
- Evaluator outcome (resolved, tests_status groups)
- Patch/application signals
- Test execution behavior (runs, failures, reruns)
- Patch iteration count and repeated edits
- Error events, reversions, and backtracking

CASE METADATA
{CASE_METADATA_JSON}

DETERMINISTIC SIGNALS
{DETERMINISTIC_SIGNALS_JSON}

COMPACT TRAJECTORY EVENTS
{EVENTS_JSON}

Return strict JSON only, with this schema:
{
  "include": true,
  "priority": "high | medium | low",
  "confidence": 0.0,
  "score": 0,
  "tags": ["brief labels explaining why this case is useful"],
  "rationale": "2-5 sentences grounded in concrete evidence",
  "metadata_evidence": [
    "specific metadata signal and why it matters"
  ],
  "trajectory_evidence": [
    {
      "event_id": "E3",
      "quote": "exact non-empty substring copied from that event's content",
      "why": "why this quote supports your triage decision"
    }
  ]
}

Rules:
- score must be an integer from 0 to 100.
- confidence must be between 0 and 1.
- If include is true, priority cannot be low.
- Use only event IDs from COMPACT TRAJECTORY EVENTS.
- Each quote must be an exact substring of its referenced event content.
- Prefer trajectory evidence that illustrates behavior quality (iteration, debugging quality,
  validation discipline, or unusual failure modes) rather than generic setup chatter.
- Return no Markdown and no prose outside the JSON object.
"""