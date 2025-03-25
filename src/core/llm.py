import os
from dataclasses import dataclass
from enum import Enum
from functools import cache
from typing import TypeAlias

from dotenv import find_dotenv
from langchain_aws import ChatBedrock

from core import settings
from schema.models import AWSModelName

_ = find_dotenv(".env")


@dataclass(frozen=True)
class ModelConfig:
    """Configuration for model parameters."""

    temperature: float = 0.5
    max_tokens: int = 4096
    top_p: float = 1.0
    top_k: int = 250
    stop_sequences: list[str] = ("\n\nHuman:",)


class ModelError(Exception):
    """Base exception for model-related errors."""

    pass


class UnsupportedModelError(ModelError):
    """Raised when an unsupported model is requested."""

    pass


class ConfigurationError(ModelError):
    """Raised when there's an issue with configuration."""

    pass


_MODEL_TABLE = {
    AWSModelName.BEDROCK_CLAUDE_3_5_SONNET_V1: "anthropic.claude-3-5-haiku-20241022-v1:0",
    AWSModelName.BEDROCK_CLAUDE_3_5_SONNET_V2: "us.anthropic.claude-3-5-sonnet-20241022-v2:0",
}

ModelT: TypeAlias = ChatBedrock


def get_aws_credentials() -> tuple[str, str]:
    profile = os.getenv("AWS_PROFILE", None)
    region = os.getenv("AWS_REGION", "us-west-2")

    if not profile:
        raise ConfigurationError("AWS_PROFILE environment variable is not set")

    return profile, region


def create_model_kwargs(config: ModelConfig | None = None) -> dict:
    """
    Returns:
        dict: Model configuration parameters
    """
    if config is None:
        config = ModelConfig()

    return {
        "temperature": config.temperature,
        "max_tokens": config.max_tokens,
        "top_p": config.top_p,
        "top_k": config.top_k,
        "stop_sequences": config.stop_sequences,
    }


@cache
def get_model(model_name: AWSModelName, config: ModelConfig | None = None) -> ModelT:
    """
    Get a ChatBedrock model instance.

    Args:
        model_name: The AWS model name to use
        config: Optional model configuration parameters

    Returns:
        ModelT: Configured ChatBedrock instance

    Raises:
        UnsupportedModelError: If the requested model is not supported
        ConfigurationError: If AWS credentials are missing or invalid
    """
    api_model_name = _MODEL_TABLE.get(model_name)
    if not api_model_name:
        raise UnsupportedModelError(f"Unsupported model: {model_name}")

    try:
        profile, region = get_aws_credentials()
        model_kwargs = create_model_kwargs(config)

        return ChatBedrock(
            model_id=api_model_name,
            credentials_profile_name=profile,
            region_name=region,
            # model_kwargs=model_kwargs,
        )

    except ConfigurationError as e:
        raise ConfigurationError(f"AWS configuration error: {str(e)}")
    except Exception as e:
        raise ModelError(f"Error creating model {model_name}: {str(e)}")


# _MODEL_TABLE = {
#     AWSModelName.BEDROCK_CLAUDE_3_5_SONNET_V1: "anthropic.claude-3-5-haiku-20241022-v1:0",
#     AWSModelName.BEDROCK_CLAUDE_3_5_SONNET_V2: "us.us.us.anthropic.claude-3-5-sonnet-20241022-v2:0",
# }

# ModelT: TypeAlias = ChatBedrock


# @cache
# def get_model(model_name: AWSModelName, /) -> ModelT:
#     api_model_name = _MODEL_TABLE.get(model_name)
#     if not api_model_name:
#         raise ValueError(f"Unsupported model: {model_name}")

#     model_kwargs = {
#         "temperature": 0.5,
#         "max_tokens": 4096,
#         "top_p": 1,
#         "top_k": 250,
#         "stop_sequences": ["\n\nHuman:"],
#     }

#     try:
#         if model_name in AWSModelName and getattr(settings, "USE_AWS_BEDROCK", False):
#             return ChatBedrock(
#                 model_id=api_model_name,
#                 credentials_profile_name=settings.AWS_PROFILE,
#                 region_name=settings.AWS_REGION,
#                 model_kwargs=model_kwargs,
#             )

#         else:
#             return ChatBedrock(
#                 model_id=model_name,
#                 region_name=getattr(settings, "AWS_REGION", "us-west-2"),
#                 credentials_profile_name=getattr(
#                     settings, "AWS_PROFILE", "AWS_AI_Access-814020624271"
#                 ),
#                 model_kwargs=model_kwargs,
#             )

#     except Exception as e:
#         raise ValueError(f"Unsupported model: {model_name}")
