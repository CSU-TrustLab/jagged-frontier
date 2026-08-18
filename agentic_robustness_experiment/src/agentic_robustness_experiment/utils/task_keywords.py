
import json
import os
import re
from typing import Literal, Optional

_SYSTEM_PROMPT = """You are a code fault-localization assistant for a security research project.

You will be given a SWE-bench Verified task description. Extract two categories of \
"localization keywords" — tokens an agent would likely grep/search for when trying to \
find the relevant code in the project's source tree.

The guiding question for every candidate is:
"If I grep for this exact token in the project's source code, would I expect to \
find the line that defines or uses it?"
If the answer is no or uncertain, EXCLUDE it.

Categories:

1. **methods** — function or method names explicitly mentioned in the description.
   - Bare name only (e.g. "save"), UNLESS the description uses a fully-qualified
     form, in which case keep the most-qualified form that appears.
       "the save() method fails"                          → "save"
       "Model.save() fails"                               → "Model.save"
       "django.db.models.Model.save() fails"              → "django.db.models.Model.save"
   - Do not include parentheses.
   - Include functions named in tracebacks (they are defined in source).

2. **string_literals** — exact string values that are hardcoded in the project's \
   source code and would help pinpoint the relevant file or line.
   Strong candidates (INCLUDE):
   - Hardcoded format-string fragments, file extension lists, regex patterns,
     sentinel values, registered format names, magic identifiers.
   - Error/warning messages raised by the project itself (not by the interpreter).
     Weak candidates (EXCLUDE):
   - Python interpreter-generated error messages ("tuple index out of range",
     "list index out of range", "object has no attribute 'X'", "unsupported
     operand type", etc.) — these are produced by CPython, not written in the
     project's source.
   - Example or placeholder values the user invented to reproduce the bug
     (e.g. "bububu.ecsv", "test.txt", "foo", "/tmp/x"). If a filename or value
     appears only inside a reproduction snippet and looks made-up, exclude it.
   - Generic single-word arguments in reproduction code ("write", "read", "r",
     "w", "rb") unless the description explicitly identifies the string as a
     sentinel constant defined in the project.
   - Variable names, parameter names, attribute names, stack-trace file paths,
     line numbers, version strings.
   - Include the string exactly as shown, stripping only the outermost
     surrounding quotes the description used to delimit it.

General rules:
- Only extract items genuinely present in the description text.
- Return an empty list when a category has no good candidates. An empty list
  is strongly preferred over a list of weak candidates.
- Output ONLY the structured result; do not add prose.

Worked examples:

Example A —
  Description: "calling identify_format('write', Table, 'bububu.ecsv',
  None, [], {}) raises IndexError: tuple index out of range in is_fits, which
  checks filepath.lower().endswith(('.fits', '.fits.gz', '.fit', '.fit.gz',
  '.fts', '.fts.gz'))"
  →
  methods:         ["identify_format", "is_fits"]
  string_literals: []

  Rejected and why:
    "IndexError: tuple index out of range" — interpreter-generated
    "bububu.ecsv"                          — user-invented reproduction value
    "write"                                — generic argument string

Example B — 
Description: Hi,
It seems like that request.get always adds 'content-length' header to the request.
I think that the right behavior is not to add this header automatically in GET requests or add the possibility to not send it.
For example http://amazon.com returns 503 for every get request that contains 'content-length' header.
Thanks,
Oren
→
methods:         ["request.get"]
string_literals: ["content-length"]
"""

_JSON_SCHEMA = {
    "type": "object",
    "properties": {
        "methods": {"type": "array", "items": {"type": "string"}},
        "string_literals": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["methods", "string_literals"],
    "additionalProperties": False,
}

_EXTRACTION_TOOL = {
    "name": "report_keywords",
    "description": (
        "Report the fault-localization keywords extracted from the issue description."
    ),
    "input_schema": _JSON_SCHEMA,
}

_USER_TEMPLATE = (
    "Extract fault-localization keywords from this issue description:\n\n{desc}"
)

_FENCE_RE = re.compile(r"^\s*```(?:json)?\s*|\s*```\s*$", re.MULTILINE)


def _failed(reason: str) -> dict:
    print(f"[task_keywords] WARNING: {reason}")
    return {"methods": [], "string_literals": [], "_parse_failed": True}


def _normalize(parsed: dict) -> dict:
    methods = parsed.get("methods", []) or []
    literals = parsed.get("string_literals", []) or []
    if not isinstance(methods, list) or not isinstance(literals, list):
        return _failed(f"unexpected shape from model: {parsed!r}")
    return {
        "methods": [str(m) for m in methods],
        "string_literals": [str(s) for s in literals],
    }


def _parse_response(raw: str) -> dict:
    raw = _FENCE_RE.sub("", raw).strip()
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        return _failed(f"could not parse model output as JSON: {raw!r}")
    if not isinstance(parsed, dict):
        return _failed(f"expected JSON object, got: {parsed!r}")
    return _normalize(parsed)


def _extract_anthropic(
    issue_description: str,
    model: str = "claude-opus-4-5-20251101",
    api_key: Optional[str] = None,
) -> dict:
    try:
        import anthropic
    except ImportError as exc:
        raise ImportError(
            "The 'anthropic' package is required for Anthropic models. "
            "Install it with: pip install anthropic"
        ) from exc

    client = anthropic.Anthropic(
        api_key=api_key or os.environ.get("ANTHROPIC_API_KEY"),
        max_retries=3,
    )

    try:
        response = client.messages.create(
            model=model,
            max_tokens=2048,
            temperature=0,
            system=_SYSTEM_PROMPT,
            tools=[_EXTRACTION_TOOL],
            tool_choice={"type": "tool", "name": "report_keywords"},
            messages=[
                {
                    "role": "user",
                    "content": _USER_TEMPLATE.format(desc=issue_description),
                }
            ],
        )
    except anthropic.APIError as exc:
        return _failed(f"Anthropic API error: {exc!r}")

    if response.stop_reason == "max_tokens":
        return _failed("response truncated at max_tokens; consider increasing limit")

    tool_block = next(
        (b for b in response.content if getattr(b, "type", None) == "tool_use"),
        None,
    )
    if tool_block is None:
        return _failed(f"no tool_use block in response: {response.content!r}")

    return _normalize(tool_block.input or {})


def _extract_openai_compatible(
    issue_description: str,
    model: str,
    base_url: str | None,
    api_key: str | None,
) -> dict:
    """Call any OpenAI-compatible endpoint (Qwen, Ollama, vLLM, etc.)."""
    try:
        from openai import OpenAI, OpenAIError
    except ImportError as exc:
        raise ImportError(
            "The 'openai' package is required for OpenAI-compatible providers. "
            "Install it with: pip install openai"
        ) from exc

    resolved_key = api_key or os.environ.get("OPENAI_API_KEY", "EMPTY")
    client = OpenAI(api_key=resolved_key, base_url=base_url, max_retries=3)

    common_kwargs = dict(
        model=model,
        temperature=0,
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": _SYSTEM_PROMPT},
            {
                "role": "user",
                "content": _USER_TEMPLATE.format(desc=issue_description),
            },
        ],
    )

    try:
        try:
            response = client.chat.completions.create(
                max_completion_tokens=32768, **common_kwargs
            )
        except TypeError:
            response = client.chat.completions.create(
                max_tokens=32768, **common_kwargs
            )
    except OpenAIError as exc:
        return _failed(f"OpenAI-compatible API error: {exc!r}")

    choice = response.choices[0]
    if getattr(choice, "finish_reason", None) == "length":
        return _failed("response truncated at token limit; consider increasing limit")

    raw = choice.message.content or "{}"
    return _parse_response(raw)


def _extract_bedrock(
    issue_description: str,
    model: str,
    region: str = "us-east-1",
) -> dict:
    """Call AWS Bedrock Converse API (supports Claude, Kimi K2, Minimax, etc.)."""
    try:
        import boto3
        from botocore.exceptions import ClientError
    except ImportError as exc:
        raise ImportError(
            "The 'boto3' package is required for Bedrock models. "
            "Install it with: pip install boto3"
        ) from exc

    client = boto3.client("bedrock-runtime", region_name=region)

    tool_config = {
        "tools": [
            {
                "toolSpec": {
                    "name": _EXTRACTION_TOOL["name"],
                    "description": _EXTRACTION_TOOL["description"],
                    "inputSchema": {"json": _JSON_SCHEMA},
                }
            }
        ],
        "toolChoice": {"tool": {"name": _EXTRACTION_TOOL["name"]}},
    }

    messages = [
        {
            "role": "user",
            "content": [{"text": _USER_TEMPLATE.format(desc=issue_description)}],
        }
    ]

    try:
        response = client.converse(
            modelId=model,
            system=[{"text": _SYSTEM_PROMPT}],
            messages=messages,
            toolConfig=tool_config,
            inferenceConfig={"maxTokens": 2048, "temperature": 0},
        )
    except ClientError as exc:
        return _failed(f"Bedrock API error: {exc!r}")

    if response.get("stopReason") == "max_tokens":
        return _failed("response truncated at max_tokens; consider increasing limit")

    output_content = response.get("output", {}).get("message", {}).get("content", [])
    for block in output_content:
        tool_use = block.get("toolUse")
        if tool_use and tool_use.get("name") == _EXTRACTION_TOOL["name"]:
            return _normalize(tool_use.get("input") or {})

    return _failed(f"no toolUse block in Bedrock response: {output_content!r}")


def extract_task_keywords(
    issue_description: str,
    model: str = "claude-opus-4-5-20251101",
    provider: Literal["anthropic", "openai", "bedrock"] = "anthropic",
    base_url: Optional[str] = None,
    api_key: Optional[str] = None,
    region: str = "us-east-1",
) -> dict:
    """Extract method names and string literals from a SWEbench task description.

    Parameters
    ----------
    issue_description:
        The raw problem statement / bug report text.
    model:
        Model identifier. Examples:
        - Anthropic: ``"claude-opus-4-5-20251101"``
        - vLLM/OpenAI-compatible: ``"Qwen/Qwen3-7B"``
        - Bedrock: ``"anthropic.claude-3-5-sonnet-20241022-v2:0"``, ``"amazon.nova-pro-v1:0"``
    provider:
        ``"anthropic"`` — direct Anthropic API, uses ``ANTHROPIC_API_KEY``.
        ``"openai"``    — OpenAI-compatible endpoint (vLLM localhost, etc.);
                          set ``base_url`` for non-OpenAI servers.
        ``"bedrock"``   — AWS Bedrock Converse API (Claude, Kimi K2, Minimax, etc.);
                          uses boto3 default credential chain (env vars, ~/.aws, IAM role).
    base_url:
        Base URL for OpenAI-compatible servers, e.g. ``"http://localhost:8000/v1"``.
        Only used when ``provider="openai"``.
    api_key:
        Explicit API key. Falls back to ``ANTHROPIC_API_KEY`` / ``OPENAI_API_KEY``
        environment variables when omitted. Not used for ``provider="bedrock"``.
    region:
        AWS region for Bedrock. Only used when ``provider="bedrock"``.

    Returns
    -------
    dict
        ``{"methods": [...], "string_literals": [...]}`` on success.
        On any failure, additionally contains ``"_parse_failed": True`` so
        callers can distinguish a genuinely empty extraction from an error.
    """
    if not issue_description or not issue_description.strip():
        return {"methods": [], "string_literals": []}

    if provider == "anthropic":
        return _extract_anthropic(issue_description, model=model, api_key=api_key)
    if provider == "openai":
        return _extract_openai_compatible(issue_description, model, base_url, api_key)
    if provider == "bedrock":
        return _extract_bedrock(issue_description, model=model, region=region)
    raise ValueError(
        f"Unsupported provider: {provider!r}. Choose 'anthropic', 'openai', or 'bedrock'."
    )
