"""Command-line entry point for AI-assisted trajectory analysis."""

import argparse
import os


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate AI-assisted trajectory reports or a review shortlist."
    )
    parser.add_argument(
        "--input",
        required=True,
        help="Root directory containing trajectory cases",
    )
    parser.add_argument("--output_dir", required=True, help="Output directory")
    parser.add_argument("--config", required=True, help="Path to model configuration YAML")

    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument(
        "--ai-assisted",
        action="store_true",
        help="Generate evidence-grounded JSON and interactive HTML reports",
    )
    mode.add_argument(
        "--ai-filter",
        action="store_true",
        help="Run lightweight LLM triage to shortlist trajectories for manual evaluation",
    )

    parser.add_argument(
        "--ai-assisted-enriched-context",
        action="store_true",
        help="Enrich the trajectory prompt with compact result, SPT, and deterministic metadata",
    )
    stage = parser.add_mutually_exclusive_group()
    stage.add_argument(
        "--ai-assisted-only-issue-prompt",
        action="store_true",
        help="Run only issue keyword extraction and skip trajectory graph generation",
    )
    stage.add_argument(
        "--ai-assisted-only-trajectory-prompt",
        action="store_true",
        help="Run only trajectory graph generation and skip issue keyword extraction",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Skip cases with completed outputs",
    )
    parser.add_argument(
        "--error_log",
        default="error_log.txt",
        help="Path for per-case failure logs (default: error_log.txt)",
    )
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    if not os.path.isdir(args.input):
        parser.error("--input must be a directory")

    ai_only_flags = (
        args.ai_assisted_enriched_context
        or args.ai_assisted_only_issue_prompt
        or args.ai_assisted_only_trajectory_prompt
    )
    if ai_only_flags and not args.ai_assisted:
        parser.error("--ai-assisted-* flags require --ai-assisted")

    os.makedirs(args.output_dir, exist_ok=True)

    if args.ai_assisted:
        from ai_analysis import run_ai_assisted_pipeline

        print("Running AI-assisted trajectory analysis...")
        try:
            summary = run_ai_assisted_pipeline(
                args.input,
                args.output_dir,
                args.config,
                error_log=args.error_log,
                resume=args.resume,
                enriched_prompt_context=args.ai_assisted_enriched_context,
                run_issue_prompt=not args.ai_assisted_only_trajectory_prompt,
                run_trajectory_prompt=not args.ai_assisted_only_issue_prompt,
            )
        except Exception as exc:
            print(f"AI-assisted analysis could not start: {exc}")
            return 1

        print(
            "AI-assisted analysis complete: "
            f"{summary['completed']} completed, {summary['failed']} failed."
        )
        return 1 if summary["failed"] else 0

    from ai_analysis import run_ai_filter_pipeline

    print("Running AI trajectory filter...")
    try:
        summary = run_ai_filter_pipeline(
            args.input,
            args.output_dir,
            args.config,
            error_log=args.error_log,
            resume=args.resume,
        )
    except Exception as exc:
        print(f"AI trajectory filter could not start: {exc}")
        return 1

    print(
        f"AI filtering complete: {summary['selected']} selected, "
        f"{summary['completed']} completed, {summary['failed']} failed."
    )
    return 1 if summary["failed"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
