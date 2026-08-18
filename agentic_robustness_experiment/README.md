# Robustness Evaluation of Code Agents via Semantic-Preserving Transformations

The framework measures how robust coding
agents are to **semantics-preserving transformations (SPTs)**. It rewrites a
repository's source code in ways that doesnot change its behaviour, then checks
whether the agent still solves the same SWE-bench Verified / SWE-Bench-Pro task.

Each agent run happens in a fresh container, and every mutation applied to every
sample is logged with file and line positions.

- [What is included](#what-is-included)
- [Requirements](#requirements)
- [Installation](#installation)
- [Configuring the model](#configuring-the-model)
- [AWS Bedrock credentials](#aws-bedrock-credentials)
- [Serving a local model](#serving-a-local-model)
- [Smoke test](#smoke-test)
- [Running experiments](#running-experiments)
- [Analyzing results](#analyzing-results)
- [Running inside Docker (optional)](#running-inside-docker-optional)
- [Configuration reference](#configuration-reference)
- [Scope and limitations](#scope-and-limitations)
- [Repository layout](#repository-layout)

---

## What is included

Everything needed to run the pipeline. All three dependencies are **bundled** under
`dependencies/` and installed from there, so setup needs no source-repository access
and no credentials:

| Path | Contents |
|---|---|
| `src/agentic_robustness_experiment/` | Experiment controller: container lifecycle, agent invocation, evaluation |
| `src/strategies/` | Mutant-sampling strategy (random) |
| `dependencies/semantic_preserving_transformer/` | The SPT library that performs the source-to-source rewrites |
| `dependencies/mini-swe-agent/` | Agent scaffold under test |
| `dependencies/swebench/` | SWE-bench evaluation harness |
| `swebench_pro_evaluation/` | SWE-Bench-Pro grading harness, per-instance Dockerfiles and run scripts |
| `config/` | Main config, agent configs, instance lists, example alternatives |
| `analysis/compute_metadata.py` | Turns a baseline run and a mutated run into the reported numbers |

**Result data is not included.** This artifact is code only; traces are regenerated
by running the pipeline. See [Scope and limitations](#scope-and-limitations) for what
that means for exact reproduction.

## Requirements

- **Linux**, with **Docker** usable by the current user. Docker is required because
  each agent run happens inside the task's own SWE-bench instance container — this
  is core to the method, not a packaging choice.
- **Python 3.10+**.
- **Disk**: room for one SWE-bench instance image per instance evaluated (typically
  1–3 GB each). They accumulate unless `--cleanup-images` is passed.
- **An LLM endpoint**, either:
  - **hosted** — AWS Bedrock or Anthropic credentials. No GPU required; this is the
    easiest path; or
  - **local** — 2 GPUs with enough combined memory for a 27B model at
    tensor-parallel degree 2 (see [Serving a local model](#serving-a-local-model)).

No GPU is needed for the framework itself. The model is reached over HTTP.

## Installation

Install directly on the host. There is no separate build step and nothing is
downloaded from a source repository — all three dependencies ship in
`dependencies/`.

```bash
# 1. (Recommended) create a virtual environment
python3 -m venv .venv && source .venv/bin/activate

# 2. Install this package
pip install -e .

# 3. Install the bundled dependencies and fill in the absolute paths in config.yaml
python3 src/agentic_robustness_experiment/scripts/setup.py --config config.yaml
```

Step 3 does exactly two things: it runs `pip install -e` on each directory under
`dependencies/`, and it rewrites the path fields in `config.yaml` to point at this
checkout. Re-run it if you move the repository. It never accesses the network and
never needs credentials.

If you would rather work in a container, see
[Running inside Docker (optional)](#running-inside-docker-optional) — but it is
genuinely optional.

## Configuring the model

The same model settings go in **three** places:

| File | Block | Used by |
|---|---|---|
| `config.yaml` | `llm_config` | The controller, for task-keyword extraction |
| `config/swebench.yaml` | `model` | The agent, on SWE-bench instances |
| `config/swepro.yaml` | `model` | The agent, on SWE-Bench-Pro instances |

**[config/examples/README.md](config/examples/README.md) has complete,
copy-pasteable blocks for all three providers.** Copy the pair for your provider
into the files above and change nothing else. A Bedrock example:

```yaml
# config.yaml
llm_config:
  provider: bedrock
  model: us.anthropic.claude-opus-4-5-20251101-v1:0
  base_url: ''
  api_key: ''
  region: us-east-1
```

```yaml
# config/swebench.yaml and config/swepro.yaml
model:
  model_name: bedrock/us.anthropic.claude-opus-4-5-20251101-v1:0
  cost_tracking: "ignore_errors"
  model_kwargs:
    temperature: 1.0
    drop_params: true
```

Note the agent's `model_name` takes a `bedrock/` prefix while `config.yaml`'s
`model` does not — the provider is a separate field there. Model calls go through
[litellm](https://docs.litellm.ai/docs/providers), so any provider it supports
works.

Then verify before committing to a long run:

```bash
python3 tools/test_model.py
```

It prints `Success` and exits 0 when the model replies.

## AWS Bedrock credentials

Credentials are read from the standard AWS chain and should **not** be written into
the config files. Leave `api_key: ''` and use one of:

**Option A — an existing AWS profile** (recommended for a workstation):

```bash
aws configure --profile my-profile      # one time; prompts for key, secret, region
export AWS_PROFILE=my-profile
```

**Option B — environment variables**:

```bash
export AWS_ACCESS_KEY_ID=...
export AWS_SECRET_ACCESS_KEY=...
export AWS_SESSION_TOKEN=...            # only if using temporary credentials
export AWS_DEFAULT_REGION=us-east-1
```

**Option C — a Bedrock API key** (short-lived bearer token):

```bash
export AWS_BEARER_TOKEN_BEDROCK=...
```

Before your first run, confirm three things:

1. **The region matches.** `region` in `config.yaml` and `AWS_DEFAULT_REGION` should
   agree, and the model must be available in that region.
2. **Model access is enabled.** In the AWS console: Bedrock → Model access → request
   access for the model. This is per-account and per-region, and is the most common
   cause of `AccessDeniedException`.
3. **The model ID is right.** IDs beginning `us.`, `eu.`, or `apac.` are
   cross-region inference profiles and are usually what you want for Claude models.

## Serving a local model

Only needed for the open-weights results; skip this if you are using a hosted API.
Any OpenAI-compatible server works as long as
`api_base` in the agent configs matches its host and port.

The reported open-weights results used vLLM:

```bash
docker run -d --gpus '"device=0,1"' --network host \
  --ipc=host \
  --security-opt=label=disable \
  -e NCCL_DEBUG=INFO \
  -e NCCL_NVLS_ENABLE=0 \
  -e NCCL_P2P_DISABLE=1 \
  -v "$HOME/.cache:/root/.cache" \
  vllm/vllm-openai:latest \
  --model Qwen/Qwen3.6-27B \
  --disable-custom-all-reduce \
  --host 0.0.0.0 --port 8003 \
  --tensor-parallel-size 2 \
  --gpu-memory-utilization 0.90 \
  --max-num-seqs 4 \
  --reasoning-parser qwen3 \
  --enable-auto-tool-choice \
  --tool-call-parser qwen3_coder
```

Notes:

- **Two GPUs**, tensor-parallel degree 2. Set `--gpus` to whichever devices are free
  on your machine; the value above assumes devices 0 and 1.
- `NCCL_P2P_DISABLE=1`, `NCCL_NVLS_ENABLE=0`, and `--disable-custom-all-reduce` work
  around peer-to-peer transport issues on multi-GPU hosts. They cost some throughput
  but are safe to keep; drop them if your interconnect handles P2P cleanly.
- `--max-num-seqs 4` bounds concurrent sequences. Raise it for throughput if the GPUs
  have headroom.
- `-v "$HOME/.cache:/root/.cache"` persists the Hugging Face model cache so the
  weights are downloaded only once.
- `--port 8003` must match `api_base` in `config/swebench.yaml` and
  `config/swepro.yaml`.
- The `qwen3` reasoning parser and `qwen3_coder` tool-call parser are required for
  the agent to parse this model's responses correctly.

Sampling temperature (`1.0`), the step limit (250), and the cost limit (3.0) are set
in `config/swebench.yaml` and `config/swepro.yaml`. They are applied client-side by
the agent, so they are independent of the vLLM launch command.

## Smoke test

Before a long run, check the whole pipeline on a single instance with a single
sample:

```bash
python3 src/agentic_robustness_experiment/experiment.py \
    --ids config/instances/smoke.txt \
    --subset verified \
    --samples 1 \
    --output-dir traces/smoke
```

Success looks like a directory under `traces/smoke/` containing `patch.diff`,
`preds.json`, `spt_log.json`, and a `report/` with the evaluation result. The first
run is slow because it pulls the instance's Docker image.

## Running experiments

```bash
python3 src/agentic_robustness_experiment/experiment.py \
    --ids config/instances/swebench_verified.txt \
    --subset verified \
    --samples 20 \
    --output-dir traces/perturbed_run
```

The corresponding unperturbed baseline runs the same instances the same number of
times with no SPTs applied:

```bash
python3 src/agentic_robustness_experiment/experiment.py \
    --ids config/instances/swebench_verified.txt \
    --subset verified \
    --samples 10 \
    --unperturbed \
    --output-dir traces/unperturbed_run
```

Instance sets in `config/instances/`:

| File | Instances | Subset |
|---|---|---|
| `smoke.txt` | 1 | `verified` — fast end-to-end check |
| `swebench_pro.txt` | 26 | `pro` |
| `swebench_verified.txt` | 28 | `verified` |

### Pipeline

For each instance, the controller performs Extract → Mutate → Test:

1. **Seed extraction** — start a temporary container for the instance and copy the
   original source out to local storage.
2. **Mutant generation** — copy the seed N times and apply randomly sampled SPTs to
   each copy. Files touched by the gold patch are always transformed, so the
   mutations are guaranteed to reach the code the task is about.
3. **Isolated execution** — for every mutant: boot a fresh container from the base
   image, inject the mutant, run the agent, evaluate with the standard harness, then
   destroy the container. No state carries between runs.
4. **Logging** — the patch, the agent trajectory, and the exact SPTs applied
   (`spt_log.json`) are written to the trace directory.

### Useful flags

| Flag | Effect |
|---|---|
| `--samples N` | Mutants per instance |
| `--workers N` | Parallel instance workers |
| `--unperturbed` | Run the clean seed N times (baseline) |
| `--cleanup-images` | Remove instance images after each instance (saves disk) |
| `--no-cleanup-seed`, `--no-cleanup-transformed` | Keep intermediate code for inspection |
| `--poll-quota` | Wait for API quota to reset instead of failing on rate limits |
| `--prune` | Prune dangling Docker resources after each instance (use with care on shared hosts) |

## Analyzing results


### `analysis/compute_metadata.py`

The single analysis entry point. It takes one **baseline** run and one
**mutated** run, makes a single pass over every sample in both, and emits the
per-instance metadata, and the statistics behind the
reported numbers. Both sides are treated as many runs per instance (~20 each),
so steps, cost, and file coverage are reported as mean / standard deviation
rather than as single values.

```bash
python3 analysis/compute_metadata.py \
    --baseline traces/unperturbed_run \
    --mutated  traces/perturbed_run \
    --output insights/run1
```

Inputs may be `.tar.gz` archives, `.zip` archives, or already-extracted trace
directories; `--mode auto` (the default) detects which per path. For a `.zip`
holding several experiments, `--baseline-subpath` / `--mutated-subpath` select
the subtree to analyze (e.g. `minisweagent/swebench/qwen/vanilla`).

**What it computes**

- **Per-sample status** — each sample directory is classified `resolved`,
  `unresolved`, `empty`, `error`, or `unknown` from its `report/`
  (both the SWE-Bench-Pro `eval_results.json` and the legacy SWE-bench report
  schemas are understood).
- **Effort metrics** — steps, cost, files touched, and patch files touched, read
  from the agent trajectory. Both the mini-swe-agent and opencode trajectory
  formats are handled.
- **Stability score and degradation** — resolve rate per instance for baseline
  and mutant, and their difference in percentage points (positive = the
  perturbation hurt).
- **Statistics** — per-instance Newcombe CIs on the difference of two
  proportions, bootstrap CIs on mean degradation and on step/cost inflation
  (all runs and resolved-only), a fixed-issue binomial-redraw CI. `--n-boot` and `--seed` control the
  resampling.

**Outputs** (with `--output insights/run1`):

| File | Contents |
|---|---|
| `run1.csv` | Per-instance aggregates: baseline/mutant stability, degradation, status breakdown, mean±sd steps, cost and file coverage |
| `run1_detailed.csv` | One row per sample run, no aggregation |
| `run1_spt_correlation.csv` | Per-transformation failure rate with/without and adversarial lift |
| `run1_spt_combinations.csv` | Per-combination sample count and failure rate |
| `run1_statistics.json` | Bootstrap and permutation results: degradation CI, sign-flip p-value, step/cost delta CIs |
| `run1_degradation_ci.csv` | Per-instance Newcombe CIs on the degradation |
| `run1_errors.log` | No-op mutants, instances with too few baseline samples, baseline-only instances |

A summary of the same numbers is printed to stdout at the end of the run.

## Running inside Docker (optional)

Nothing here requires the controller to run in a container — the
[Installation](#installation) steps work directly on any Linux host with Python
3.10+ and Docker. The provided `dockerfile` is a convenience for getting a known-good
Python environment, and is entirely optional.

If you do want it:

```bash
docker build -t agentic-robustness-experiment:v1 .

docker run --rm -it \
  --shm-size=1g --ulimit memlock=-1 --ulimit stack=67108864 \
  --network=host \
  -v /var/run/docker.sock:/var/run/docker.sock \
  -v "$(pwd):/workspace/agentic-robustness-experiment" \
  agentic-robustness-experiment:v1
```

Then run the same [Installation](#installation) step 3 and
[Configuring the model](#configuring-the-model) steps inside the container.

Two details matter if you use it:

- `-v /var/run/docker.sock:/var/run/docker.sock` lets the controller start the
  sibling instance containers it needs. Without it nothing will run.
- `--network=host` lets the controller reach a model server running on the host's
  `localhost`. If you use a hosted API instead, you can drop it.

To pass credentials in, add `-e AWS_PROFILE=...` (with
`-v "$HOME/.aws:/root/.aws:ro"`) or `-e ANTHROPIC_API_KEY=...` to the `docker run`
command rather than writing keys into the config files.

## Configuration reference

Only a few fields need your attention. Everything else is either filled in
automatically or is an experimental setting that should stay as shipped.

### You must set

| File | Field | Notes |
|---|---|---|
| `config.yaml` | `llm_config.provider` | `openai`, `anthropic`, or `bedrock` |
| `config.yaml` | `llm_config.model` | Model identifier |
| `config.yaml` | `llm_config.base_url` | Endpoint for `openai`; `''` for anthropic/bedrock |
| `config.yaml` | `llm_config.api_key` | Leave `''` to read credentials from the environment |
| `config.yaml` | `llm_config.region` | Bedrock only |
| `config/swebench.yaml` | `model` | Agent model block — SWE-bench |
| `config/swepro.yaml` | `model` | Agent model block — SWE-Bench-Pro |

See [config/examples/README.md](config/examples/README.md) for exact blocks.

### You may want to change

| File | Field | Default | Notes |
|---|---|---|---|
| `config.yaml` | `agent_config.agent_scaffold` | `miniswe` | `miniswe` or `opencode` |
| `config.yaml` | `agent_config.agent_trace_dir` | `traces` | Output root; `--output-dir` overrides it per run |
| `config.yaml` | `environment.timeout` | `120` | Seconds before one command in the instance container is killed |
| `config.yaml` | `spt_config.*` | see below | Mutation parameters; changing them changes the experiment |

`spt_config` holds the mutation settings used for the reported results:
`num_transformations: 3`, `fraction_of_candidates_to_transform: 0.7`,
`max_files_for_injection_spts: 10`, `max_keywords_for_task_specific_spts: 5`.

### Filled in for you

`paths.*`, `agent_config.agent_executable`, `agent_config.agent_swebench_config`,
`agent_config.swebench_pro_config`, and `swebench_config.*` are absolute paths
written by `setup.py`. Re-run it after moving the repository:

```bash
python3 src/agentic_robustness_experiment/scripts/setup.py --config config.yaml --skip-install
```

Note that `setup.py` rewrites `config.yaml` through `yaml.safe_dump`, which keeps
every value but strips the explanatory comments. This section and
[docs/REPRODUCING.md](docs/REPRODUCING.md) are the durable reference.

## Scope and limitations

**Runs are not bit-for-bit reproducible, by design of the experiment rather than
oversight.** Two sources of randomness are deliberately left free:

1. **Mutant sampling is unseeded.** `RandomSelectionStrategy` draws a fresh
   `numpy` generator per run and mutant identifiers come from `uuid4`, so a re-run
   produces a *different* random sample of transformations.
2. **Agent decoding is stochastic** (`temperature: 1.0`), so even an identical
   mutant can yield a different trajectory and patch.

Consequently, re-running will not reproduce individual trace directories or
per-sample outcomes. What reproduces is the **aggregate effect** — the difference in
resolve rate between perturbed and unperturbed runs over N samples per instance.

Other constraints worth noting:

- Evaluation pulls third-party Docker images (SWE-bench official images; SWE-Bench-Pro
  images from a Docker Hub account not controlled by the authors). Availability is
  outside this artifact's control.
- SPTs apply to Python source. SWE-Bench-Pro instances in other languages are
  evaluated but not mutated at the source level.

## Repository layout

```
config.yaml                    Main config: paths, environment, SPT parameters, model
config/
  swebench.yaml                Agent config for SWE-bench (prompts, limits, model)
  swepro.yaml                  Agent config for SWE-Bench-Pro
  instances/                   Instance ID lists per experiment
  examples/README.md           Copy-pasteable model blocks for each provider
src/agentic_robustness_experiment/
  experiment.py                Entry point: the Extract -> Mutate -> Test loop
  scripts/setup.py             Install bundled dependencies, resolve config paths
  scripts/trace_analysis.py    Classify run outcomes
  utils/                       Docker, evaluation, config, file selection helpers
src/strategies/                Mutant-sampling strategy (random)
analysis/
  compute_metadata.py          Baseline vs mutant analysis: metadata, SPT correlation, statistics
tools/test_model.py            Model connectivity check
swebench_pro_evaluation/       SWE-Bench-Pro grading harness and per-instance assets
dependencies/                  Bundled SPT library, agent scaffold, evaluator
dockerfile                     Optional containerized environment
docs/REPRODUCING.md            Paper table/figure -> command mapping
```
