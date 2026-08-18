"""Install the bundled dependencies and write their absolute paths into config.yaml.

All three dependencies ship inside ``dependencies/``, so this never touches the
network and needs no credentials:

    python3 src/agentic_robustness_experiment/scripts/setup.py --config config.yaml

It does two things:

1. ``pip install -e`` each directory under ``dependencies/``.
2. Rewrite the absolute paths in the given config file so they point at this
   checkout, wherever it happens to live.

Run it again any time you move the repository.
"""

import argparse
import os
import subprocess
import sys

import yaml

# Bundled dependencies, in install order. `exec_relpath` is the entry point that
# gets written into the config file.
DEPENDENCIES = {
    "mini-swe-agent": {
        "target_subdir": "dependencies/mini-swe-agent",
        "exec_relpath": "src/minisweagent/run/extra/swebench.py",
    },
    "swebench": {
        "target_subdir": "dependencies/swebench",
        "exec_relpath": "swebench.harness.run_evaluation",
    },
    "semantic-preserving-transformer": {
        "target_subdir": "dependencies/semantic_preserving_transformer",
        "exec_relpath": "semantic_transformer.cli",
    },
}


def run(cmd, cwd=None, check=True):
    print(f"> {' '.join(cmd)} (cwd={cwd or os.getcwd()})")
    subprocess.run(cmd, cwd=cwd, check=check)


def install_python_deps(name, target):
    """Install one bundled dependency in editable mode."""
    pyproject = os.path.join(target, "pyproject.toml")
    req_txt = os.path.join(target, "requirements.txt")
    if os.path.exists(pyproject):
        print(f"[{name}] installing (editable) from {target}")
        run([sys.executable, "-m", "pip", "install", "-e", target])
    elif os.path.exists(req_txt):
        print(f"[{name}] installing requirements from {req_txt}")
        run([sys.executable, "-m", "pip", "install", "-r", req_txt])
    else:
        print(f"[{name}] no pyproject.toml or requirements.txt found — skipping")


def update_yaml_paths(mapping_yaml_path, repo_exec_paths):
    """Point the config's absolute paths at this checkout."""
    with open(mapping_yaml_path) as f:
        data = yaml.safe_load(f) or {}

    data.setdefault("agent_config", {})
    data.setdefault("swebench_config", {})
    data.setdefault("paths", {})

    sections = {
        "agent_config": (
            "agent_executable",
            "agent_swebench_config",
            "swebench_pro_config",
            "agent_trace_dir",
        ),
        "swebench_config": ("swebench_executable", "swebench_directory"),
        "paths": ("transformation_bin", "seed_storage", "transformed_storage"),
    }
    for section, keys in sections.items():
        for key in keys:
            if key in repo_exec_paths:
                data[section][key] = repo_exec_paths[key]

    with open(mapping_yaml_path, "w") as f:
        yaml.safe_dump(data, f, sort_keys=False)
    print(f"\nUpdated paths in {mapping_yaml_path}")


def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--config",
        default="config.yaml",
        help="Config file whose paths should be rewritten (default: %(default)s).",
    )
    parser.add_argument(
        "--skip-install",
        action="store_true",
        help="Only rewrite the config paths; do not run pip.",
    )
    args = parser.parse_args()

    if not os.path.exists(args.config):
        parser.error(
            f"Config file not found: {args.config}. Run this from the repository root."
        )

    missing = [
        name
        for name, info in DEPENDENCIES.items()
        if not os.path.isdir(info["target_subdir"])
    ]
    if missing:
        parser.error(
            "These bundled dependencies are missing from dependencies/: "
            f"{', '.join(missing)}. They ship with this repository — re-extract the "
            "archive if they were lost."
        )

    # Paths that depend only on where this checkout lives.
    repo_exec_paths = {
        "seed_storage": os.path.abspath("seed_storage"),
        "transformed_storage": os.path.abspath("transformed_storage"),
        "agent_swebench_config": os.path.abspath("config/swebench.yaml"),
        "swebench_pro_config": os.path.abspath("config/swepro.yaml"),
        "agent_trace_dir": os.path.abspath("traces"),
    }

    for name, info in DEPENDENCIES.items():
        target = os.path.abspath(info["target_subdir"])
        if args.skip_install:
            print(f"[{name}] skipping install (--skip-install)")
        else:
            install_python_deps(name, target)

        exec_abs = os.path.abspath(os.path.join(target, info["exec_relpath"]))
        if name == "mini-swe-agent":
            repo_exec_paths["agent_executable"] = exec_abs
        elif name == "swebench":
            repo_exec_paths["swebench_executable"] = exec_abs
            repo_exec_paths["swebench_directory"] = target
        elif name == "semantic-preserving-transformer":
            repo_exec_paths["transformation_bin"] = exec_abs

    update_yaml_paths(args.config, repo_exec_paths)
    print(
        "\nSetup complete. Next: set your model in config.yaml, config/swebench.yaml\n"
        "and config/swepro.yaml (see the README), then check connectivity with:\n"
        "  python3 tools/test_model.py"
    )


if __name__ == "__main__":
    main()
