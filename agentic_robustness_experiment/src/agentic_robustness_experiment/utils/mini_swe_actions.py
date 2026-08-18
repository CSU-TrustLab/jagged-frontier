import subprocess


def run_agent_on_instance(
    config,
    instance_id,
    subset="verified",
    split="test",
    output_folder="./",
    container=None,
):
    agent_config = config.get_agent_config(subset)
    if container:
        agent_run_cmd = f"python {config.agent_executable} --filter {instance_id} --subset {subset} --split {split} --config {agent_config} --redo-existing -o {output_folder} --container-id {container.id}"
    else:
        agent_run_cmd = f"python {config.agent_executable} --filter {instance_id} --subset {subset} --split {split} --config {agent_config} --redo-existing -o {output_folder}"

    print(f"Running agent with command: {agent_run_cmd}")
    output = subprocess.run(agent_run_cmd, shell=True, check=True)
