import json
import os
import subprocess

from agentic_robustness_experiment.utils.utils import get_issue_prompt

OPENCODE_BIN = "/usr/local/bin/opencode"
OPENCODE_INSTALL_DIR = "/usr/local/bin"

def run_agent_on_instance(
    config,
    instance_id,
    subset="verified",
    split="test",
    output_folder="./",
    container=None,
):
    if container is None:
        raise ValueError("The OpenCode scaffold requires a running container.")
    
    #Install and configure agent
    os.makedirs(output_folder, exist_ok=True)

    workdir = config.get_cwd(subset)
    model = f"{config.llm_provider}/{config.llm_model}"

    install_opencode(container)
    env, model = config_agent(
        model,
        config.llm_api_key,
        base_url=getattr(config, "llm_base_url", None),
        container=container,
    )

    #Run agent on issue
    issue_description = get_issue_prompt(instance_id, subset, split)
    # print(issue_description)
    prompt = (
        """
        You are an expert software engineer tasked with fixing a bug in a GitHub repository.


        ## Task
        Fix the issue below. Make minimal, focused changes that directly address the problem.

        ## Instructions
        1. First, explore the codebase to understand the project structure and locate relevant files
        2. Analyze the issue carefully to understand what needs to be fixed
        3. Make the necessary code changes to fix the bug
        4. Ensure your changes are minimal and focused - only modify what is necessary
        5. Do NOT add new tests or modify existing test files
        6. Do NOT modify any configuration files unless absolutely necessary for the fix
        7. After making changes, verify they address the issue described
        
        
        ## Important Notes
        - The repository has already been cloned and checked out to the correct commit
        - You have full access to read and modify files in the repository
        - Focus on producing a clean, minimal patch that fixes the described issue
        - If the issue mentions specific files or line numbers, start your investigation there


        """
        f"<issue>\n{issue_description or instance_id}\n</issue>"
    )
    agent_run_cmd = [OPENCODE_BIN, "run", "--model", model, "--dangerously-skip-permissions", "--print-logs", prompt]

    print(f"Running OpenCode on instance {instance_id} in container {container.name}...")

    exit_code, output = container.exec_run(agent_run_cmd, workdir=workdir, environment=env, demux=True)
    stdout = (output[0] or b"") if isinstance(output, tuple) else output
    stderr = (output[1] or b"") if isinstance(output, tuple) else b""
    decoded = (stdout + stderr).decode(errors="ignore")
    os.makedirs(output_folder, exist_ok=True)
    with open(os.path.join(output_folder, "opencode.log"), "w") as f:
        f.write(decoded)

    if exit_code != 0:
        raise subprocess.CalledProcessError(exit_code, " ".join(agent_run_cmd[:-1] + ["<prompt>"]), output=output) # type: ignore
    
    #Output appropriate files
    patch = get_patch(container, workdir)
    build_preds(model, instance_id, container, output_folder, patch)

    session_id = get_last_session_id(container, workdir, env)
    export_traj(session_id, output_folder, container=container, workdir=workdir, env=env)


def install_opencode(container):
    exit_code, _ = container.exec_run(["test", "-x", OPENCODE_BIN])
    if exit_code == 0:
        return

    install_cmd = [
        "/bin/bash",
        "-lc",
        "(curl -fsSL https://opencode.ai/install | bash"
        " || (wget -qO- https://opencode.ai/install | bash))"
        f' && cp "$HOME/.opencode/bin/opencode" {OPENCODE_BIN}'
        f" && chmod 755 {OPENCODE_BIN}"
        f" && test -x {OPENCODE_BIN}",
    ]
    exit_code, output = container.exec_run(install_cmd)
    if exit_code != 0:
        raise RuntimeError( f"Failed to install OpenCode in container: {output.decode(errors='ignore')}" )
    
    print("OpenCode installed.")


_AWS_CREDENTIAL_VARS = (
    "AWS_ACCESS_KEY_ID",
    "AWS_SECRET_ACCESS_KEY",
    "AWS_DEFAULT_REGION",
    "AWS_REGION",
)


def config_agent(model, api_key, base_url=None, container=None):
    provider, _, model_id = model.partition("/")

    if provider == "bedrock":
        env = {k: v for k in _AWS_CREDENTIAL_VARS if (v := os.environ.get(k))}
        if not env.get("AWS_DEFAULT_REGION") and not env.get("AWS_REGION"):
            raise RuntimeError(
                "AWS_DEFAULT_REGION (or AWS_REGION) must be set in the host environment for Bedrock."
            )
        if not env.get("AWS_ACCESS_KEY_ID") or not env.get("AWS_SECRET_ACCESS_KEY"):
            raise RuntimeError(
                "AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY must be set in the host environment for Bedrock."
            )
        # OpenCode uses "amazon-bedrock" as its provider name, not "bedrock"
        return env, f"amazon-bedrock/{model_id}"

    env_var = {
        "anthropic": "ANTHROPIC_API_KEY",
        "openai": "OPENAI_API_KEY",
        "google": "GEMINI_API_KEY",
        "openrouter": "OPENROUTER_API_KEY",
    }.get(provider, f"{provider.upper().replace('-', '_')}_API_KEY")

    env = {env_var: api_key or "EMPTY"}

    opencode_config = {
        "$schema": "https://opencode.ai/config.json",
        "permission": {
            "webfetch": "deny",
            "websearch": "deny",
        },
    }

    if base_url and container is not None:
        base_url = _resolve_base_url_for_container(base_url)
        if provider == "openai":
            provider = "local-openai"
            model = f"{provider}/{model_id}"

        opencode_config["provider"] = {
            provider: {
                "npm": "@ai-sdk/openai-compatible",
                "options": {
                    "baseURL": base_url.rstrip("/"),
                    "apiKey": api_key or "EMPTY",
                },
                "models": {model_id: {"name": model_id}},
            }
        }

    if container is not None:
        config_json = json.dumps(opencode_config, indent=2)
        write_cmd = [
            "/bin/bash",
            "-c",
            "mkdir -p /root/.config/opencode"
            " && cat > /root/.config/opencode/opencode.json <<'OPENCODE_EOF'\n"
            f"{config_json}\n"
            "OPENCODE_EOF",
        ]
        exit_code, output = container.exec_run(write_cmd)
        if exit_code != 0:
            raise RuntimeError(
                f"Failed to write opencode config in container: {output.decode(errors='ignore')}"
            )
    return env, model


def get_patch(container, workdir):
    cmd = [
        "/bin/bash",
        "-c",
        "git add -A . && git -c core.fileMode=false diff --cached",
    ]
    exit_code, output = container.exec_run(cmd, workdir=workdir)

    if exit_code != 0:
        raise RuntimeError(f"git diff failed: {output.decode(errors='ignore')}")
    patch = output.decode(errors="ignore")
    if patch and not patch.endswith("\n"):
        patch += "\n"
    return patch


def build_preds(model,
                instance_id,
                container,
                output_folder,
                patch):
    preds = {
        instance_id: {
            "model_name_or_path": model,
            "instance_id": instance_id,
            "model_patch": patch,
        }
    }
    os.makedirs(output_folder, exist_ok=True)
    preds_path = os.path.join(output_folder, "preds.json")
    with open(preds_path, "w") as f:
        json.dump(preds, f, indent=2)
    print(f"Saved predictions to {preds_path}")


def get_last_session_id(container, workdir, env):
    cmd = [OPENCODE_BIN, "session", "list", "--format", "json", "-n", "1"]

    exit_code, output = container.exec_run(cmd, workdir=workdir, environment=env, demux=True)
    stdout = (output[0] or b"") if isinstance(output, tuple) else output

    if exit_code != 0:
        raise RuntimeError(f"Could not list OpenCode sessions: {stdout.decode(errors='ignore')}")

    sessions = json.loads(stdout.decode(errors="ignore"))
    if not sessions:
        raise RuntimeError("No OpenCode sessions found.")
    return sessions[0]["id"]


def export_traj(session_id, output_folder, container=None, workdir=None, env=None):
    cmd = [OPENCODE_BIN, "export", session_id]

    exit_code, output = container.exec_run(cmd, workdir=workdir, environment=env, demux=True) # type: ignore
    stdout = (output[0] or b"") if isinstance(output, tuple) else output

    if exit_code != 0:
        raise RuntimeError(f"Failed to export OpenCode trajectory: {stdout.decode(errors='ignore')}")

    os.makedirs(output_folder, exist_ok=True)
    traj_path = os.path.join(output_folder, f"{session_id}.traj.json")

    with open(traj_path, "w") as f:
        f.write(stdout.decode(errors="ignore"))

    print(f"Saved trajectory to {traj_path}")


def _resolve_base_url_for_container(base_url):
    return base_url.replace("localhost", "host.docker.internal").replace("127.0.0.1", "host.docker.internal")
