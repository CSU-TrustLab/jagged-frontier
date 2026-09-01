import yaml
import json
import os
import boto3
from litellm import completion
from dotenv import load_dotenv
import re


load_dotenv()


class LLMBackend:

    def __init__(self, config_path: str):

        with open(config_path) as f:
            config = yaml.safe_load(f)

        self.provider = config["provider"]
        self.model = config["model"]

        self.temperature = config.get("temperature", 0.0)
        self.max_tokens = config.get("max_tokens", 4096)
        self.timeout = config.get("timeout", 120)
        self.num_retries = config.get("num_retries", 2)
        self.bedrock_bearer_token = (
            os.getenv("AWS_BEARER_TOKEN_BEDROCK")
            or os.getenv("AWS_TOKEN_BEDROCK")
        )
        if self.bedrock_bearer_token and not os.getenv("AWS_BEARER_TOKEN_BEDROCK"):
            # LiteLLM uses AWS_BEARER_TOKEN_BEDROCK. Accept the shorter name
            # used by this project without exposing or copying it to disk.
            os.environ["AWS_BEARER_TOKEN_BEDROCK"] = self.bedrock_bearer_token
        self.region = (
            config.get("region")
            or os.getenv("AWS_REGION_NAME")
            or os.getenv("AWS_REGION")
            or os.getenv("AWS_DEFAULT_REGION")
            or boto3.Session().region_name
        )

        if self.provider == "bedrock":
            self._validate_bedrock_configuration()

        self.model = self._format_model_name()

    def _validate_bedrock_configuration(self):
        if not self.region:
            raise ValueError(
                "Bedrock region is not configured. Set region in the model YAML "
                "or set AWS_DEFAULT_REGION."
            )
        if self.bedrock_bearer_token:
            return
        if boto3.Session(region_name=self.region).get_credentials() is None:
            raise ValueError(
                "Bedrock credentials are unavailable. Set AWS_TOKEN_BEDROCK or "
                "AWS_BEARER_TOKEN_BEDROCK, or configure the standard AWS credential "
                "chain before running the analysis."
            )

    def _format_model_name(self):

        if self.provider == "bedrock":
            return f"bedrock/{self.model}"

        if self.provider == "openai":
            return self.model

        raise ValueError(f"Unsupported provider: {self.provider}")

    def extract_json(self, content):

        # remove markdown code blocks
        content = re.sub(r"```json", "", content)
        content = re.sub(r"```", "", content)

        # find first json object
        start = content.find("{")
        end = content.rfind("}")

        if start == -1 or end == -1:
            raise ValueError("No JSON object found in model output")

        json_str = content[start:end+1]

        try:
            return json.loads(json_str)
        except json.JSONDecodeError as e:
            print("---- RAW MODEL OUTPUT ----")
            print(content)
            print("--------------------------")
            raise e

    def generate_json(self, prompt: str):
        
        provider_options = {}
        if self.provider == "bedrock":
            provider_options["aws_region_name"] = self.region
            if self.bedrock_bearer_token:
                provider_options["api_key"] = self.bedrock_bearer_token

        response = completion(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            timeout=self.timeout,
            num_retries=self.num_retries,
            **provider_options,
        )
        
        content = response.choices[0].message.content.strip()

        #UNCOMMENT FOR DEBUGGING MODEL OUTPUT
        #print("\nMODEL OUTPUT:\n")
        #print(content)
        #print("\nEND OUTPUT\n")

        try:
            return self.extract_json(content)
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON returned by model: {e}")