from agentic_robustness_experiment.utils import mini_swe_actions
from agentic_robustness_experiment.utils import open_code_actions


def run_agent_on_instance(
    config,
    instance_id,
    subset="verified",
    split="test",
    output_folder="./",
    container=None,
):
    scaffold = config.agent_scaffold

    if scaffold == "opencode":
        return open_code_actions.run_agent_on_instance(config,instance_id,subset=subset,split=split,output_folder=output_folder,container=container)
    
    elif scaffold == "miniswe":
        return mini_swe_actions.run_agent_on_instance(config,instance_id,subset=subset,split=split,output_folder=output_folder,container=container)
    
    else:
        raise ValueError(f"Unknown agent_scaffold")

