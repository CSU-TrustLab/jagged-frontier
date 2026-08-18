import os

import yaml


class AppConfig:
    def __init__(self, config_path=None):
        if config_path is None:
            raise ValueError("A valid config_path must be provided.")
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Config file not found at: {config_path}")

        print(f"Loading configuration from {config_path}...")

        with open(config_path) as file:
            config_data = yaml.safe_load(file)

        paths = config_data.get("paths", {})
        self.transformation_library_bin = paths.get("transformation_bin")
        self.seed_storage = paths.get("seed_storage")
        self.transformed_storage = paths.get("transformed_storage")
        self.agent_config = config_data.get("agent_config", {})
        self.agent_scaffold = self.agent_config.get("agent_scaffold", "miniswe")
        self.agent_executable = self.agent_config.get("agent_executable")
        self.agent_swebench_config = self.agent_config.get("agent_swebench_config")
        self.swebench_pro_config = self.agent_config.get("swebench_pro_config")
        self.agent_trace_dir = self.agent_config.get("agent_trace_dir")

        self.swebench_config = config_data.get("swebench_config", {})
        self.swebench_directory = self.swebench_config.get("swebench_directory")
        self.swebench_executable = self.swebench_config.get("swebench_executable")

        swepro_eval = config_data.get("swebench_pro_eval_config", {})
        self.swepro_eval_script = swepro_eval.get("eval_script")
        self.swepro_raw_sample_path = swepro_eval.get("raw_sample_path")
        self.swepro_scripts_dir = swepro_eval.get("scripts_dir")
        self.swepro_dockerhub_username = swepro_eval.get("dockerhub_username")

        env = config_data.get("environment", {})
        self.cwd = env.get("container_cwd")
        self.swepro_container_cwd = env.get("swepro_container_cwd")
        self.timeout = env.get("timeout", 120)

        # SPT sampling parameters. Defaults match the values used for the
        # reported results, so an older config without this block behaves the
        # same as before.
        spt = config_data.get("spt_config", {})
        self.num_transformations = spt.get("num_transformations", 3)
        self.fraction_of_candidates_to_transform = spt.get(
            "fraction_of_candidates_to_transform", 0.7
        )
        self.max_files_for_injection_spts = spt.get("max_files_for_injection_spts", 10)
        self.max_keywords_for_task_specific_spts = spt.get(
            "max_keywords_for_task_specific_spts", 5
        )
        self.mutant_id_length = spt.get("mutant_id_length", 12)

        llm_config = config_data.get("llm_config", {})
        self.llm_provider = llm_config.get("provider", "anthropic")
        self.llm_model = llm_config.get("model", "claude-opus-4-5-20251101")
        self.llm_base_url = llm_config.get("base_url", "")
        self.llm_api_key = llm_config.get("api_key", "")
        self.llm_region = llm_config.get("region", "us-east-1")

    def get_cwd(self, subset: str) -> str:
        return self.swepro_container_cwd if subset == "pro" else self.cwd

    def get_agent_config(self, subset: str) -> str:
        return self.swebench_pro_config if subset == "pro" else self.agent_swebench_config

    def display(self):
        print("--- Configuration Loaded ---")
        print(f"transformation_library_bin: {self.transformation_library_bin}")
        print(f"seed_storage: {self.seed_storage}")
        print(f"agent_executable: {self.agent_executable}")
        print(f"agent_swebench_config: {self.agent_swebench_config}")
        print(f"agent_trace_dir: {self.agent_trace_dir}")
        print(f"cwd (container_cwd): {self.cwd}")
        print(f"timeout (timeout_seconds): {self.timeout}")
        print(f"num_transformations: {self.num_transformations}")
        print(
            "fraction_of_candidates_to_transform: "
            f"{self.fraction_of_candidates_to_transform}"
        )
        print(f"max_files_for_injection_spts: {self.max_files_for_injection_spts}")
        print(
            "max_keywords_for_task_specific_spts: "
            f"{self.max_keywords_for_task_specific_spts}"
        )
        print("----------------------------")
