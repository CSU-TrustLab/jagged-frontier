import os
import random
import re
import subprocess
import uuid
import time
from datasets import load_dataset
import docker

from agentic_robustness_experiment.utils.config_loader import AppConfig
from agentic_robustness_experiment.utils.utils import (
    DATASET_MAPPING,
    get_swebench_docker_image_name,
)

client = docker.from_env()
api = client.api


def run_container(
    image: str,
    name: str,
    workdir: str,
    command: str = "sleep 2h",
    env: dict[str, str] | None = None,
    detach: bool = True,
    remove: bool = True,
    cpu_shares: int | None = None,
    mem_limit: str | None = None,
):
    container = client.containers.run(
        image=image,
        command=command,
        name=name,
        working_dir=workdir,
        environment=env,
        detach=detach,
        remove=remove,
        cpu_shares=cpu_shares,
        mem_limit=mem_limit,
        
        extra_hosts={"host.docker.internal": "host-gateway"},
    )
    return container


def exec_in_container(
    container,
    cmd: list[str],
    workdir: str | None = None,
    stream_output: bool = True,
    timeout: int | None = None,
):
    exec_id = api.exec_create(container.id, cmd, workdir=workdir, tty=False)
    sock = api.exec_start(exec_id, stream=stream_output, demux=True)
    if not stream_output:
        return api.exec_inspect(exec_id)["ExitCode"], b""
    output_bytes = bytearray()
    for out_err in sock:
        if out_err is None:
            continue
        out, err = out_err
        if out:
            output_bytes.extend(out)
            print(out.decode(errors="ignore"), end="")
        if err:
            output_bytes.extend(err)
            print(err.decode(errors="ignore"), end="")
    inspect = api.exec_inspect(exec_id)
    return inspect["ExitCode"], bytes(output_bytes)


def restart_container(container):
    container.restart()


def stop_container(container, timeout: int = 5, remove: bool = False):
    try:
        container.stop(timeout=timeout)
    except Exception as e:
        print(f"Warning: Failed to stop container {container.name} gracefully: {e}")
        if remove:
            print(f"Forcing removal of container {container.name}...")
            try:
                container.remove(force=True)
                return
            except Exception as remove_err:
                print(f"Error forcing removal of container {container.name}: {remove_err}")

    if remove:
        try:
            container.remove()
            print(f"Removed container: {container.name}")
        except Exception as e:
            # If auto_remove=True was set, it might already be gone (404) 
            # or the removal might already be in progress (409)
            err_str = str(e).lower()
            if "not found" in err_str or "404" in err_str or "409" in err_str or "already in progress" in err_str:
                pass
            else:
                print(f"Warning: Failed to remove container {container.name}: {e}")


def remove_image(image_name: str) -> None:
    try:
        client.images.remove(image_name, force=True)
        print(f"Removed Docker image: {image_name}")
    except docker.errors.ImageNotFound:
        print(f"Image not found, skipping removal: {image_name}")
    except docker.errors.APIError as e:
        print(f"Warning: Could not remove image {image_name}: {e}")


def prune_global() -> None:
    """Reclaim disk space by pruning unused containers and images."""
    try:
        # Prune containers first to free up images
        pruned_containers = client.containers.prune()
        if pruned_containers.get("ContainersDeleted"):
            print(f"Pruned {len(pruned_containers['ContainersDeleted'])} containers.")
        
        # Prune only dangling (untagged) images
        pruned_images = client.images.prune(filters={"dangling": True})
        reclaimed = pruned_images.get("SpaceReclaimed", 0)
        print(f"Global prune (dangling only) finished. Reclaimed: {reclaimed / (1024**3):.2f} GB")
    except Exception as e:
        print(f"Warning: Global prune failed: {e}")


def run_docker_copy(src_path: str, dest_path: str):
    cmd = ["docker", "cp", src_path, dest_path]
    print(f"Running command: {' '.join(cmd)}")

    try:
        result = subprocess.run(
            cmd, capture_output=True, text=True, check=True, timeout=120
        )
    except subprocess.CalledProcessError as e:
        print(
            f"Error during 'docker cp'.\nReturn Code: {e.returncode}\nStderr: {e.stderr}"
        )
        raise RuntimeError(f"docker cp failed: {e.stderr}")
    except subprocess.TimeoutExpired as e:
        print(f"Timeout during 'docker cp': {e}")
        raise RuntimeError("docker cp timed out")

    print(f"Successfully copied from '{src_path}' to '{dest_path}'")


def copy_testbed_from_container(config, container, subset="verified") -> str:
    destination_path = f"{config.seed_storage}/{container.name}/"
    os.makedirs(destination_path, exist_ok=True)
    container_source_path = f"{container.id}:{config.get_cwd(subset)}/."
    run_docker_copy(container_source_path, destination_path)
    return destination_path


def copy_transformed_code_into_testbed(src_path: str, dest_path: str):
    run_docker_copy(src_path, dest_path)


def commit_transformed_code_in_container(
    container,
    config,
    subset,
    exclude_patterns: list[str] | None = None,
):
    print("Committing transformed code in the container...")
    pathspecs = ["."] + [f":!{p}" for p in (exclude_patterns or [])]
    pathspec_str = " ".join(f"'{p}'" for p in pathspecs)
    git_cfg = "-c user.email=root@container -c user.name=root"
    cmd = f"/bin/bash -c 'git {git_cfg} add -- {pathspec_str} && git {git_cfg} commit -m \"Applied transformation\"'"

    exit_code, output = container.exec_run(
        cmd,
        workdir=config.get_cwd(subset),
    )

    if exit_code != 0:
        print(f"Error: {output.decode('utf-8')}")
    else:
        print("Success:", output.decode("utf-8"))


def check_git_status_in_container(container,config,subset):
    cmd = '/bin/bash -c "git status --porcelain"'

    exit_code, output = container.exec_run(
        cmd,
        workdir=config.get_cwd(subset),
    )

    if exit_code != 0:
        print(f"Error checking git status: {output.decode('utf-8')}")
        return None
    else:
        status_output = output.decode("utf-8").strip()
        if status_output:
            print("Uncommitted changes detected:\n", status_output)
            return False
        else:
            print("No uncommitted changes.")
            return True


def filter_instances(
    instances: list[dict],
    *,
    instance_id: str,
    slice_spec: str = "",
    shuffle: bool = False,
) -> list[dict]:
    if shuffle:
        instances = sorted(instances.copy(), key=lambda x: x["instance_id"])
        random.seed(None)
        random.shuffle(instances)
    before_filter = len(instances)
    instances = [
        instance
        for instance in instances
        if re.match(instance_id, instance["instance_id"])
    ]
    if (after_filter := len(instances)) != before_filter:
        print(f"Test Instance: {instance_id} loaded")
    if slice_spec:
        values = [int(x) if x else None for x in slice_spec.split(":")]
        instances = instances[slice(*values)]
        if (after_slice := len(instances)) != before_filter:
            print(f"Instance slice: {before_filter} -> {after_slice} instances")
    return instances


_DATASET_CACHE = {}

def get_dataset(path: str, split: str):
    key = (path, split)
    if key not in _DATASET_CACHE:
        print(f"Loading dataset {path}, split {split}...")
        _DATASET_CACHE[key] = list(load_dataset(path, split=split))
    return _DATASET_CACHE[key]

def get_image_name_for_instance(instance_id: str, subset: str, split: str) -> str | None:
    dataset_path = DATASET_MAPPING.get(subset, subset)
    instances = get_dataset(dataset_path, split=split)
    instance = filter_instances(instances, instance_id=instance_id)
    if not instance:
        return None
    return get_swebench_docker_image_name(instance[0])


def start_container(
    instance_id: str, config: AppConfig, subset="verified", split="test", sample_id=None
):
    dataset_path = DATASET_MAPPING.get(subset, subset)
    instances = get_dataset(dataset_path, split=split)
    print(instances[0]["instance_id"])
    instance = filter_instances(instances, instance_id=instance_id)
    if not instance:
        raise ValueError(f"Instance {instance_id} not found in {dataset_path}")
    print(f"Found {len(instance)} matching instances for ID '{instance_id}'")
    docker_image_name = get_swebench_docker_image_name(instance[0])
    print(f"Starting container for instance {instance_id} using image {docker_image_name}...")
    container_name = f"{instance_id}_{sample_id or uuid.uuid4().hex[:8]}"
    sleep_cmd = (
        ["-c", f"sleep 2h"]
        if subset == "pro"
        else ["sleep", "2h"]
    )
    container = run_container(
        image=docker_image_name,
        name=container_name,
        workdir=config.get_cwd(subset),
        command=sleep_cmd,
    )
    poll_interval = 0.5
    elapsed = 0
    while container.status != "running" and elapsed < config.timeout:
        container.reload()
        time.sleep(poll_interval)
        elapsed += poll_interval

    if container.status != "running":
        raise RuntimeError("Container did not start in time")
    print(container.status)
    return container
