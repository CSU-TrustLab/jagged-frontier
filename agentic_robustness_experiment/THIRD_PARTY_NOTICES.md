# Third-party notices

This artifact bundles and depends on third-party material. Each component remains
under its own license.

## Vendored source (`dependencies/`)

These are included in full so the artifact installs without network access. Each is
a *fork*, modified for this work.

| Component | Path | License | Notes |
|---|---|---|---|
| mini-swe-agent | `dependencies/mini-swe-agent/` | See `dependencies/mini-swe-agent/LICENSE.md` | Agent scaffold under test. Fork with additions for injecting perturbed source and recording traces. |
| SWE-bench | `dependencies/swebench/` | See `dependencies/swebench/LICENSE` | Evaluation harness. Fork adapted for the perturbed-code workflow. |
| semantic-preserving-transformer | `dependencies/semantic_preserving_transformer/` | MIT (declared in its `pyproject.toml`) | The SPT library; authored for this work. Depends on `libcst`, `nltk`, `jedi`, `astroid`, `evalplus`. |


## Benchmark datasets

Task data is downloaded from Hugging Face at run time and is not redistributed here:

| Subset flag | Dataset |
|---|---|
| `verified` | `princeton-nlp/SWE-Bench_Verified` |
| `pro` | `ScaleAI/SWE-bench_Pro` |


## SWE-Bench-Pro evaluation assets (`swebench_pro_evaluation/`)

Derived from the SWE-Bench-Pro benchmark and its evaluation harness:

- `sweap_eval_full_v2.jsonl` — instance metadata (problem statements,
  FAIL_TO_PASS / PASS_TO_PASS test lists, image names), derived from the
  SWE-Bench-Pro dataset release.
- `dockerfiles/` and `run_scripts/`  — per-instance build
  and grading scripts, generated from that metadata.
- `swe_bench_pro_eval.py`, `helper_code/` — grading harness.


## External Docker images

Evaluation pulls prebuilt container images that are **not** distributed with this
artifact and are hosted by third parties:

- **SWE-bench instance images** — `swebench/sweb.eval.x86_64.*` from Docker Hub,
  published by the SWE-bench maintainers.
- **SWE-Bench-Pro instance images** — pulled from the Docker Hub account configured
  as `swebench_pro_eval_config.dockerhub_username` in `config.yaml` (currently
  `jefzda`), and referenced in `sweap_eval_full_v2.jsonl` by an Amazon ECR path.

If an image is withdrawn or retagged, the affected instances cannot be evaluated.

## Python dependencies

Installed from PyPI at setup time, each under its own license: `anthropic`, `openai`,
`datasets`, `docker`, `numpy`, and `pyyaml`. See `pyproject.toml`. `litellm` is
pulled in transitively by mini-swe-agent and handles the model API calls.

## Models

No model weights are distributed. Reported results use a locally served
open-weights model and, hosted APIs (AWS Bedrock / Anthropic).

