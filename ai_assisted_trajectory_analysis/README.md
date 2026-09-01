# AI-Assisted Trajectory Analysis

This repository is centered on AI-assisted trajectory analysis for agent runs.
The primary workflow turns directories of trajectory cases into evidence-grounded
HTML reports and machine-readable JSON outputs for review and downstream analysis.

## Table of Contents

- [What This Mode Produces](#what-this-mode-produces)
- [Installation](#installation)
- [Configuration](#configuration)
- [Credentials](#credentials)
- [Input Directory Layout](#input-directory-layout)
- [Run AI-Assisted Analysis](#run-ai-assisted-analysis)
- [Quick Filtering Mode](#quick-filtering-mode)
- [Outputs](#outputs)
- [What Is Extracted](#what-is-extracted)
- [Result Matching Rules](#result-matching-rules)
- [Troubleshooting](#troubleshooting)
- [Limitations](#limitations)

---

## What This Mode Produces

For each discovered case trajectory, AI-assisted mode produces:

- A self-contained interactive `report.html` with four tabs:
  1. Issue description (highlighted extracted keywords)
  2. Phase graph (clickable nodes with evidence snippets)
  3. Raw trajectory (normalized event timeline)
  4. Rule based results (git/test/edit/error summaries)
- A machine-readable `analysis.json` with extracted nodes, checks, clues, and quality metadata

For the full batch it also produces:

- `index.html`
- `summary.json`
- `summary.csv`

Reports are offline-viewable (embedded CSS/JS).

---

## Installation

### 1. Clone

```bash
git clone https://github.com/wafflesandpancakes19/trajectory-analysis.git
cd trajectory-analysis
```

### 2. Create and activate virtual environment

```bash
python3 -m venv trajectory-analysis
source trajectory-analysis/bin/activate
# Windows PowerShell:
# .\\trajectory-analysis\\Scripts\\activate
```

The shell prompt should show `(trajectory-analysis)`. You can confirm which
interpreter is active with `python -c "import sys; print(sys.executable)"`.
Use `source`, not `conda activate`: this is a standard Python virtual
environment rather than a Conda environment.

### 3. Install dependencies

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### 4. Verify

```bash
python main.py --help
python -m unittest discover -s tests
```

The test suite uses synthetic fixtures and fake model backends. It runs fully
offline and does not require AWS or OpenAI credentials.

---

## Configuration

LLM settings are in `config/model_config.yaml`:

```yaml
provider: bedrock
model: global.anthropic.claude-sonnet-4-6
region: us-east-1
temperature: 0.0
max_tokens: 32000
timeout: 600
num_retries: 2
```

Bedrock note: some models require inference profile IDs (for example
`global.anthropic.claude-sonnet-4-6`) instead of bare on-demand IDs.

---

## Credentials

Set credentials in the same terminal session used to run commands.

### AWS Bedrock

Bearer token option:

```bash
export AWS_TOKEN_BEDROCK="your-bedrock-bearer-token"
```

Or AWS credential chain:

```bash
export AWS_ACCESS_KEY_ID="..."
export AWS_SECRET_ACCESS_KEY="..."
export AWS_SESSION_TOKEN="..."   # temporary credentials only
export AWS_DEFAULT_REGION="us-east-1"
```

### OpenAI

```bash
export OPENAI_API_KEY="your-key"
```

---

## Input Directory Layout

Input must be a directory containing case subdirectories. Each case should have
one or more `*.traj.json` files and its own result JSON.

```text
dataset/
├── case-a/
│   ├── run.traj.json
│   ├── result.json
│   ├── spt_log.json        # optional
│   └── patch.diff          # optional
└── case-b/
    ├── session.traj.json
    └── report/
        └── report.json
```

Rules:

- Trajectory files must end with `.traj.json`
- Discovery is recursive
- mini-swe-agent and OpenCode formats can be mixed
- Results are matched only within the same case directory

Recognized result names include:

- `result.json`, `results.json`
- `report.json`
- `eval_results.json`, `eval_results_detailed.json`
- `*_output.json` under `report/`

Optional adjacent files:

- `spt_log.json`
- `preds.json`
- `patch.diff`

---

## Run AI-Assisted Analysis

```bash
python main.py \
  --ai-assisted \
  --input dataset/ \
  --output_dir outputs/ai-analysis \
  --config config/model_config.yaml
```

### Small one-case smoke run

Set the credentials for the configured provider first. The following command
uses the smallest trajectory currently included in the artifact and makes two
LLM requests: one for issue clues and one for the trajectory graph.

```bash
SAMPLE_CASE="ai_assisted_data/manual_evaluation_trajectories/instance_ansible__ansible-e0c91af45fa9af575d10fd3e724ebc59d2b2d6ac-v30a923fb5c164d6cd18280c02422f75e611e8fb2_7e00d69ea474"

python main.py \
  --ai-assisted \
  --input "$SAMPLE_CASE" \
  --output_dir outputs/smoke-test \
  --config config/model_config.yaml
```

On success, `outputs/smoke-test/` contains the case report plus batch-level
`index.html`, `summary.json`, and `summary.csv`. The command exits nonzero and
records the case as failed if provider credentials are unavailable.

### AI-assisted flags

| Argument | Required | Description |
|---|---|---|
| `--ai-assisted` | Yes | Enables AI-assisted mode (input must be a directory) |
| `--ai-assisted-enriched-context` | No | Adds compact result/SPT/deterministic metadata to trajectory graph prompt |
| `--ai-assisted-only-issue-prompt` | No | Run issue keyword extraction only |
| `--ai-assisted-only-trajectory-prompt` | No | Run trajectory graph extraction only |
| `--input` | Yes | Root directory containing per-case subdirectories |
| `--output_dir` | Yes | Destination for reports and artifacts |
| `--config` | Yes | Path to YAML model config |
| `--error_log` | No | Failure log path (default `error_log.txt`) |
| `--resume` | No | Skip cases that already have completed outputs |

---

## Quick Filtering Mode

Use filtering when you need a fast shortlist before deeper manual evaluation.

```bash
python main.py \
  --ai-filter \
  --input dataset/ \
  --output_dir outputs/ai-filter \
  --config config/model_config.yaml
```

Filtering generates:

- `filter_summary.json`
- `filter_shortlist.csv`
- `filter_cases/<case>/triage.json`

`--resume` is supported and skips cases that already have `triage.json`.

---

## Outputs

```text
outputs/ai-analysis/
├── index.html
├── summary.json
├── summary.csv
├── case-a/
│   ├── report.html
│   └── analysis.json
└── case-b/
    ├── report.html
    └── analysis.json
```

---

## What Is Extracted

`analysis.json` includes:

- `case`: case identity, issue metadata, model metadata, result metadata, SPT info
- `events`: normalized trajectory events
- `nodes`: phase graph nodes with intent/outcome/resources/evidence
- `unclassified_event_ids`: context events outside classified work
- `phase_statistics`: per-phase counts and segment pattern
- `deterministic_checks`: git/resources/patches/tests/errors/iteration checks
- `issue_clues`: extracted issue keywords (A-I taxonomy)
- `rule_based_results`: git commands, issue-keyword grep, phase counts
- `data_quality`: match status, retry errors, coverage stats

Evidence snippets are validated against raw trajectory text.

---

## Result Matching Rules

Each trajectory is matched to a result in the same case directory, in order:

1. Embedded `instance_id` in trajectory
2. Instance key inside result file
3. Case directory name
4. Single unambiguous result record in case

If matching remains ambiguous, the case is flagged with a data-quality error.

---

## Troubleshooting

| Symptom | Fix |
|---|---|
| `Bedrock credentials are unavailable` | Set `AWS_TOKEN_BEDROCK` or AWS credential chain in the active terminal |
| `on-demand throughput isn't supported ... inference profile` | Use profile-prefixed model ID, for example `global.anthropic.claude-sonnet-4-6` |
| `No module named 'tenacity'` | Run `pip install -r requirements.txt` |
| `--ai-assisted requires --input to be a directory` | Pass a directory, not a file |
| `No supported *.traj.json files found` | Ensure files end in `.traj.json` under input root |
| Missing or ambiguous result | Ensure each case has a recognizable result file |

Per-case failures append to `--error_log` and do not stop the full batch.

---

## Limitations

- Repository-wide file coverage cannot be computed from trajectory logs alone
- SPT causality and excess-step comparisons require additional baseline data
- Timestamp granularity differs across trajectory formats, so timing is secondary
- Runs make real LLM calls and incur provider costs
