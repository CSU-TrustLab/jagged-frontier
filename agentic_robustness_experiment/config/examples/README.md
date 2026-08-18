# Model configuration examples

Three files need the same model settings:

| File | Block to edit | Used by |
|---|---|---|
| `config.yaml` | `llm_config` | The controller, for task-keyword extraction |
| `config/swebench.yaml` | `model` | The agent, on SWE-bench instances |
| `config/swepro.yaml` | `model` | The agent, on SWE-Bench-Pro instances |

Pick the provider you are using below, copy the two blocks into the right files,
and change nothing else. Model calls go through
[litellm](https://docs.litellm.ai/docs/providers), so any provider it supports
works — these three are the ones used for the paper.

Verify with `python3 tools/test_model.py` before starting a run.

---

## 1. Local OpenAI-compatible server (vLLM, SGLang, llama.cpp, Ollama, …)

Used for the open-weights results. Start the server first — see "Serving a local
model" in the README — then match `base_url` / `api_base` to its host and port.

**`config.yaml`**

```yaml
llm_config:
  provider: openai
  model: Qwen/Qwen3.6-27B
  base_url: http://localhost:8003/v1/
  api_key: ''
  region: us-east-1
```

**`config/swebench.yaml` and `config/swepro.yaml`**

```yaml
model:
  model_name: Qwen/Qwen3.6-27B
  cost_tracking: "ignore_errors"
  model_kwargs:
    api_base: "http://localhost:8003/v1/"
    api_key: "EMPTY"
    custom_llm_provider: "openai"
    temperature: 1.0
    drop_params: true
```

Notes:

- `api_key: "EMPTY"` is a placeholder. Local servers ignore it.
- `model_name` must match the `--model` the server was launched with.
- `temperature: 1.0` is the reported setting.
---

## 2. AWS Bedrock

Used for the hosted-model results. Credentials come from the standard AWS chain,
never from these files — see "AWS Bedrock credentials" in the README.

**`config.yaml`**

```yaml
llm_config:
  provider: bedrock
  model: us.anthropic.claude-opus-4-5-20251101-v1:0
  base_url: ''
  api_key: ''
  region: us-east-1
```

**`config/swebench.yaml` and `config/swepro.yaml`**

```yaml
model:
  model_name: bedrock/us.anthropic.claude-opus-4-5-20251101-v1:0
  cost_tracking: "ignore_errors"
  model_kwargs:
    temperature: 1.0
    drop_params: true
```

Notes:

- The agent's `model_name` needs the `bedrock/` prefix; `config.yaml`'s `model`
  does not (the provider is given separately there).
- No `api_base` and no `api_key`. Setting either breaks Bedrock's SigV4 signing.
- The model ID must be available **in your region** and enabled for your account.
  Cross-region inference profiles start with `us.`, `eu.`, or `apac.`.

---

## 3. Anthropic API

**`config.yaml`**

```yaml
llm_config:
  provider: anthropic
  model: claude-opus-4-5-20251101
  base_url: ''
  api_key: ''
  region: us-east-1
```

**`config/swebench.yaml` and `config/swepro.yaml`**

```yaml
model:
  model_name: anthropic/claude-opus-4-5-20251101
  cost_tracking: "ignore_errors"
  model_kwargs:
    temperature: 1.0
    drop_params: true
```

Notes:

- Leave `api_key` empty in both files and export `ANTHROPIC_API_KEY` instead, so
  the key is never written to disk.

---

## Keeping keys out of the config files

Every provider above reads its credentials from the environment when `api_key` is
left empty:

```bash
export ANTHROPIC_API_KEY=sk-ant-...       # Anthropic
export AWS_PROFILE=my-profile             # Bedrock (or use an IAM role)
```

This is the recommended approach. If you do put a key in a config file, be careful
not to commit or redistribute it.
