import argparse
import os
import shutil
import sys
import time
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from uuid import uuid4

from agentic_robustness_experiment.utils.evaluate import evaluate_patch, evaluate_patch_pro
from agentic_robustness_experiment.utils.agent_actions import run_agent_on_instance
from agentic_robustness_experiment.utils.config_loader import AppConfig
from agentic_robustness_experiment.utils.create_diff_file import extract_patch, sanitize_preds_json
from agentic_robustness_experiment.utils.docker_actions import (
    check_git_status_in_container,
    commit_transformed_code_in_container,
    copy_testbed_from_container,
    copy_transformed_code_into_testbed,
    get_image_name_for_instance,
    remove_image,
    prune_global,
    stop_container,
    start_container,
)
from agentic_robustness_experiment.utils.file_selection import select_files_by_gold_patch
from agentic_robustness_experiment.utils.load_resolved_ids import load_resolved_ids
from agentic_robustness_experiment.utils.utils import (
    DATASET_MAPPING,
    EXCLUDE_DIR_PATTERNS,
    EXCLUDE_ROOT_FILES,
    write_spt_log,
    get_issue_description,
)
from strategies.random_selection_strategy import RandomSelectionStrategy

DEFAULT_NUMBER_OF_SAMPLES = 1
DEFAULT_MAX_WORKERS = 4


REPO_ROOT = Path(__file__).resolve().parents[2]
TEST_MODEL_SCRIPT = REPO_ROOT / "tools" / "test_model.py"


def wait_for_quota():
    """Poll the model endpoint until the quota is available."""
    print(f"Checking model quota availability via {TEST_MODEL_SCRIPT}...")
    while True:
        try:
            result = subprocess.run(
                [sys.executable, str(TEST_MODEL_SCRIPT)],
                capture_output=True, text=True,
            )
            output = result.stdout + result.stderr
            if result.returncode == 0 and "Success" in output:
                print("Model is ready! Quota available.")
                break
            else:
                print(f"Model returned an error or rate-limit testing quota. Waiting 5 minutes before retrying...")
                time.sleep(300)
        except Exception as e:
            print(f"Error polling model: {e}. Waiting 5 minutes before retrying...")
            time.sleep(300)


def run_and_evaluate_single_instance(
    config: AppConfig, instance_id: str, container, subset="verified", split="test", sample_location=None
):
    output_folder_path = f"{config.agent_trace_dir}/{container.name}"
    run_agent_on_instance(
        config,
        instance_id,
        subset,
        split,
        output_folder=output_folder_path,
        container=container,
    )

    preds_file = f"{output_folder_path}/preds.json"
    sanitize_preds_json(preds_file, instance_id)
    extract_patch(
        preds_file,
        f"{output_folder_path}/patch.diff",
        instance_id,
    )
    transformed_dir = sample_location 
    dataset_name = DATASET_MAPPING.get(subset, DATASET_MAPPING["verified"])

    if subset == "pro":
        evaluate_patch_pro(
            config,
            preds_file=preds_file,
            output_dir=f"{output_folder_path}/report/",
            transformed_dir=transformed_dir,
        )
    else:
        evaluate_patch(
            config,
            container,
            f"{output_folder_path}/preds.json",
            run_id=container.name,
            report_dir=f"{output_folder_path}/report/",
            transformed_dir=transformed_dir,
            dataset_name=dataset_name,
        )

    return container.name


def process_single_sample(
    config: AppConfig,
    instance_id: str,
    seed_location: str,
    exclude_patterns: list[str],
    exclude_files: set[str],
    subset: str,
    split: str,
    inner_workers: int = 1,
    cleanup_transformed: bool = True,
    cleanup_containers: bool = False,
    poll_quota: bool = False,
    issue_description: str = "",
) -> str:
    """Generate one sample, write its SPT log, run the agent, and evaluate it."""
    dataset_name = DATASET_MAPPING.get(subset, DATASET_MAPPING["verified"])
    gold_patch_files = select_files_by_gold_patch(
        instance_id=instance_id,
        repo_path=Path(seed_location),
        dataset_name=dataset_name,
        split=split,
    )
    strategy = RandomSelectionStrategy(config=config)
    sample_locations = strategy.generate_samples(
        seed_location, 1, config.transformed_storage, exclude_patterns, log_file=None,
        exclude_files=exclude_files,
        max_workers=inner_workers,
        gold_patch_files=gold_patch_files,
        issue_description=issue_description,
        llm_provider=config.llm_provider,
        llm_model=config.llm_model,
        llm_base_url=config.llm_base_url,
        llm_api_key=config.llm_api_key,
        llm_region=config.llm_region,
        use_task_specific_spts=True,
    )
    sample_id, sample_location, spt_entries = sample_locations[0]
    print(f"Generated sample: {sample_id} at {sample_location}")

    container = start_container(instance_id, config, subset, split, sample_id)
    try:
        src_path = f"{sample_location}/."
        dest_path = f"{container.id}:{config.get_cwd(subset)}/" # type: ignore
        copy_transformed_code_into_testbed(src_path, dest_path)
        check_git_status_in_container(container,config, subset)
        commit_transformed_code_in_container(container, config, subset, exclude_patterns)

        output_folder_path = f"{config.agent_trace_dir}/{container.name}" # type: ignore
        write_spt_log(spt_entries, output_folder_path)

        while True:
            try:
                run_and_evaluate_single_instance(config, instance_id, container, subset, split, sample_location)
                break
            except subprocess.CalledProcessError as e:
                if not poll_quota:
                    raise
                agent_log_name = "opencode.log" if config.agent_scaffold == "opencode" else "minisweagent.log"
                log_file_path = os.path.join(output_folder_path, agent_log_name)
                preds_file_path = os.path.join(output_folder_path, "preds.json")
                is_rate_limit = False
                # Only check for rate-limit if the agent actually produced a preds file
                # (missing preds = agent crashed before finish, e.g. container gone)
                if os.path.exists(preds_file_path) and os.path.exists(log_file_path):
                    with open(log_file_path, "r", encoding="utf-8") as f:
                        log_content = f.read().lower()
                        if "429" in log_content or "rate limit" in log_content or "throttling" in log_content:
                            is_rate_limit = True
                
                if is_rate_limit:
                    print(f"Rate limit hit for {sample_id}. Pausing thread and waiting for daily quota to reset...")
                    wait_for_quota()
                    print(f"Quota restored. Retrying evaluation for {sample_id}...")
                else:
                    if not os.path.exists(preds_file_path):
                        print(f"Agent did not produce preds.json for {sample_id} — agent likely crashed (container issue?). Not retrying.")
                    raise
    except Exception as e:
        print(f"Error processing sample {sample_id}: {e}")
        raise
    finally:
        stop_container(container, remove=cleanup_containers)
        if cleanup_transformed and sample_location and os.path.exists(sample_location):
            print(f"Cleaning up transformed storage: {sample_location}")
            shutil.rmtree(sample_location, ignore_errors=True)

    return sample_id


def process_unperturbed_sample(
    config: AppConfig,
    instance_id: str,
    seed_location: str | None,
    subset: str,
    split: str,
    cleanup_containers: bool = False,
    poll_quota: bool = False,
) -> str:
    """Run the agent on the unperturbed seed (no SPTs applied)."""
    sample_id = f"unperturbed-{uuid4().hex[:8]}"
    container = start_container(instance_id, config, subset, split, sample_id)
    try:
        output_folder_path = f"{config.agent_trace_dir}/{container.name}" # type: ignore
        write_spt_log([], output_folder_path)

        while True:
            try:
                run_and_evaluate_single_instance(config, instance_id, container, subset, split, seed_location)
                break
            except subprocess.CalledProcessError as e:
                if not poll_quota:
                    raise
                agent_log_name = "opencode.log" if config.agent_scaffold == "opencode" else "minisweagent.log"
                log_file_path = os.path.join(output_folder_path, agent_log_name)
                preds_file_path = os.path.join(output_folder_path, "preds.json")
                is_rate_limit = False
                if os.path.exists(preds_file_path) and os.path.exists(log_file_path):
                    with open(log_file_path, "r", encoding="utf-8") as f:
                        log_content = f.read().lower()
                        if "429" in log_content or "rate limit" in log_content or "throttling" in log_content:
                            is_rate_limit = True
                if is_rate_limit:
                    print(f"Rate limit hit for {sample_id}. Pausing thread and waiting for daily quota to reset...")
                    wait_for_quota()
                    print(f"Quota restored. Retrying evaluation for {sample_id}...")
                else:
                    if not os.path.exists(preds_file_path):
                        print(f"Agent did not produce preds.json for {sample_id} — agent likely crashed. Not retrying.")
                    raise
    except Exception as e:
        print(f"Error processing unperturbed sample {sample_id}: {e}")
        raise
    finally:
        stop_container(container, remove=cleanup_containers)

    return sample_id


def run_experiment(
    instance_ids: list[str],
    subset: str,
    split: str,
    config: AppConfig,
    max_workers: int = DEFAULT_MAX_WORKERS,
    number_of_samples: int = DEFAULT_NUMBER_OF_SAMPLES,
    cleanup_image: bool = False,
    resume: bool = False,
    cleanup_seed: bool = True,
    cleanup_transformed: bool = True,
    cleanup_containers: bool = False,
    poll_quota: bool = False,
    prune: bool = False,
    unperturbed: bool = False,
):
    os.makedirs(config.agent_trace_dir, exist_ok=True)
    os.makedirs(config.transformed_storage, exist_ok=True)
    os.makedirs(config.seed_storage, exist_ok=True)
    
    error_log_path = Path(config.agent_trace_dir) / "errors.log"
    
    def log_error(msg):
        with open(error_log_path, "a") as f:
            f.write(msg + "\n")

    if not resume:
        print("Cleaning up storage for a fresh run...")
        shutil.rmtree(config.transformed_storage, ignore_errors=True)
        shutil.rmtree(config.seed_storage, ignore_errors=True)
        os.makedirs(config.transformed_storage, exist_ok=True)
        os.makedirs(config.seed_storage, exist_ok=True)
    else:
        print("Resuming experiment. Skipping already processed instances.")

    inner_workers = max(1, (os.cpu_count() or 1) // max_workers)
    for instance_id in instance_ids:
        num_to_run = number_of_samples
        if resume and os.path.exists(config.agent_trace_dir):
            existing_traces = [d for d in os.listdir(config.agent_trace_dir) if d.startswith(instance_id)]
            num_existing = len(existing_traces)
            if num_existing >= number_of_samples:
                print(f"Skipping {instance_id}: {num_existing} traces already exist.")
                continue
            num_to_run = number_of_samples - num_existing
            print(f"Resuming {instance_id}: {num_existing} traces exist, running {num_to_run} more.")

        docker_image_name = get_image_name_for_instance(instance_id, subset, split) if cleanup_image else None
        if not unperturbed:
            container = start_container(instance_id, config, subset, split)
            seed_location = copy_testbed_from_container(config, container, subset)
            exclude_patterns = EXCLUDE_DIR_PATTERNS.get(
                instance_id.split("-")[0], ""
            ).split()
            exclude_files = EXCLUDE_ROOT_FILES.get(instance_id.split("-")[0], set())
            issue_description = get_issue_description(instance_id, subset, split)
            stop_container(container, remove=cleanup_containers)
        else:
            seed_location = None
            exclude_patterns = []
            exclude_files = set()

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            if unperturbed:
                futures = {
                    executor.submit(
                        process_unperturbed_sample,
                        config,
                        instance_id,
                        seed_location,
                        subset,
                        split,
                        cleanup_containers,
                        poll_quota,
                    ): i + (number_of_samples - num_to_run)
                    for i in range(num_to_run)
                }
            else:
                assert seed_location is not None
                futures = {
                    executor.submit(
                        process_single_sample,
                        config,
                        instance_id,
                        seed_location,
                        exclude_patterns,
                        exclude_files,
                        subset,
                        split,
                        inner_workers,
                        cleanup_transformed,
                        cleanup_containers,
                        poll_quota,
                        issue_description,
                    ): i + (number_of_samples - num_to_run)
                    for i in range(num_to_run)
                }
            for future in as_completed(futures):
                sample_index = futures[future]
                try:
                    sample_id = future.result()
                    print(f"Completed sample {sample_index + 1}/{number_of_samples}: {sample_id}")
                except Exception as e:
                    import traceback
                    error_msg = f"\n--- Error for Instance {instance_id}, Sample {sample_index + 1} ---\n"
                    error_msg += traceback.format_exc()
                    error_msg += f"Summary: {e}\n"
                    error_msg += "-------------------------------------------\n"
                    
                    log_error(error_msg)
                    print(f"Sample {sample_index + 1} failed: {e}. Check {error_log_path} for details.")

        if cleanup_seed and seed_location and os.path.exists(seed_location):
            print(f"Cleaning up seed storage for instance {instance_id}: {seed_location}")
            shutil.rmtree(seed_location, ignore_errors=True)

        if cleanup_image and docker_image_name:
            remove_image(docker_image_name)

        if prune:
            prune_global()


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Random-mutation agent robustness experiment for SWE-bench instances.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--config",
        default="config.yaml",
        help="Path to the YAML configuration file.",
    )
    parser.add_argument(
        "--ids",
        required=True,
        help="File containing SWE-bench instance IDs (one per line).",
    )
    parser.add_argument(
        "--subset",
        default="verified",
        help="Dataset subset name (e.g. verified, lite, pro).",
    )
    parser.add_argument(
        "--split",
        default="test",
        help="Dataset split.",
    )
    parser.add_argument(
        "--samples",
        type=int,
        default=DEFAULT_NUMBER_OF_SAMPLES,
        help="Number of mutant samples to generate per instance.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=DEFAULT_MAX_WORKERS,
        help="Number of parallel worker threads.",
    )
    parser.add_argument(
        "--cleanup-images",
        action="store_true",
        help="Remove Docker images after processing each instance.",
    )
    parser.add_argument(
        "--continue",
        dest="resume",
        action="store_true",
        help="Resume experiment by skipping already processed instances.",
    )
    parser.add_argument(
        "--output-dir",
        help="Override the agent_trace_dir from config.",
    )
    parser.add_argument(
        "--no-cleanup-seed",
        action="store_true",
        help="Do not remove seed_storage after processing an instance.",
    )
    parser.add_argument(
        "--no-cleanup-transformed",
        action="store_true",
        help="Do not remove transformed_storage after processing a sample.",
    )
    parser.add_argument(
        "--cleanup-containers",
        action="store_true",
        help="Remove Docker containers after processing.",
    )
    parser.add_argument(
        "--poll-quota",
        dest="poll_quota",
        action="store_true",
        help="Poll the model endpoint at startup and on rate limit errors, waiting for quota to reset before retrying.",
    )
    parser.add_argument(
        "--prune",
        action="store_true",
        help="Prune unused Docker containers and dangling images after each instance. Use with caution on shared machines.",
    )
    parser.add_argument(
        "--unperturbed",
        action="store_true",
        help="Run the unperturbed seed (no SPTs applied) n times instead of generating mutant samples.",
    )
    return parser


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()

    config = AppConfig(args.config)
    if args.output_dir:
        config.agent_trace_dir = args.output_dir
        print(f"Overriding trace directory to: {config.agent_trace_dir}")

    # Wait for daily quota to be available before starting script execution
    if args.poll_quota:
        wait_for_quota()

    instance_ids = load_resolved_ids(args.ids)

    run_experiment(
        instance_ids=instance_ids,
        subset=args.subset,
        split=args.split,
        config=config,
        max_workers=args.workers,
        number_of_samples=args.samples,
        cleanup_image=args.cleanup_images,
        resume=args.resume,
        cleanup_seed=not args.no_cleanup_seed,
        cleanup_transformed=not args.no_cleanup_transformed,
        cleanup_containers=args.cleanup_containers,
        poll_quota=args.poll_quota,
        prune=args.prune,
        unperturbed=args.unperturbed,
    )


if __name__ == "__main__":
    main()
