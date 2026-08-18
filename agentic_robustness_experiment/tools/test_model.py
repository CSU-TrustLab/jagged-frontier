import argparse
import sys
from pathlib import Path

import litellm
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = REPO_ROOT / "config" / "swebench.yaml"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG,
        help="Agent config to read the model block from (default: %(default)s).",
    )
    args = parser.parse_args()

    config_path = args.config if args.config.is_absolute() else REPO_ROOT / args.config
    if not config_path.exists():
        print(f"Error: {config_path} not found.")
        return 1

    with open(config_path) as f:
        config = yaml.safe_load(f) or {}

    model_cfg = config.get("model", {})
    model_name = model_cfg.get("model_name")
    if not model_name:
        print(f"Error: model_name not found in {config_path}.")
        return 1

    model_kwargs = model_cfg.get("model_kwargs", {})

    print(f"Testing model: {model_name}")
    if model_kwargs.get("api_base"):
        print(f"Using api_base: {model_kwargs['api_base']}")

    try:
        print("Sending test request via litellm...")
        response = litellm.completion(
            model=model_name,
            messages=[
                {
                    "role": "user",
                    "content": "Hello! Reply with 'Connection Successful' if you can read this.",
                }
            ],
            **{"temperature": 0.0, **model_kwargs},
        )
        print("\n--- Success! ---")
        print(f"Response: {response.choices[0].message.content}")
        return 0
    except Exception as e:
        print("\n--- Failed to call model ---")
        print(f"Error: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
