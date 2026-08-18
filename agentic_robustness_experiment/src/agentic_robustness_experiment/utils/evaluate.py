import os
import subprocess

from agentic_robustness_experiment.utils.config_loader import AppConfig
from agentic_robustness_experiment.utils.convert_pred_json import convert_json


def evaluate_patch(
    config: AppConfig,
    container,
    preds_file,
    workers=1,
    run_id="agentic-robustness-experiment",
    report_dir="./reports",
    transformed_dir=None,
    dataset_name="princeton-nlp/SWE-bench_Verified",
):
    preds_file = os.path.abspath(preds_file)
    report_dir = os.path.abspath(report_dir)
    base_cmd = [
        "cd",
        config.swebench_directory,
        "&&",
        "python",
        "-m",
        "swebench.harness.run_evaluation",
        "--dataset_name",
        dataset_name,
        "--predictions_path",
        preds_file,
        "--max_workers",
        str(workers),
        "--run_id",
        run_id,
        "--report_dir",
        report_dir,
    ]

    if transformed_dir is not None:
        base_cmd += ["--transform_dir", os.path.abspath(str(transformed_dir))]

    evaluate_patch_cmd = " ".join(str(x) for x in base_cmd)
    print(f"Evaluating patch with command: {evaluate_patch_cmd}")
    try:
        output = subprocess.run(
            evaluate_patch_cmd,
            shell=True,
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        print(output.stdout)
    except subprocess.CalledProcessError as e:
        print(f"Evaluation failed with exit status {e.returncode}")
        print("STDOUT:")
        print(e.stdout)
        print("STDERR:")
        print(e.stderr)
        raise


def evaluate_patch_pro(
    config: AppConfig,
    preds_file,
    output_dir="./reports",
    workers=1,
    transformed_dir=None,
):
    convert_json(preds_file)

    cmd = [
        "python",
        config.swepro_eval_script,
        f"--raw_sample_path={config.swepro_raw_sample_path}",
        f"--patch_path={preds_file}",
        f"--output_dir={output_dir}",
        f"--scripts_dir={config.swepro_scripts_dir}",
        f"--num_workers={workers}",
        f"--dockerhub_username={config.swepro_dockerhub_username}",
        "--use_local_docker",
    ]

    if transformed_dir is not None:
        cmd.append(f"--transform_dir={transformed_dir}")

    evaluate_cmd = " ".join(cmd)
    print(f"Evaluating SWE-bench Pro patch with command: {evaluate_cmd}")
    output = subprocess.run(evaluate_cmd, shell=True, check=True)
    print(output)
