from functools import cache
from typing import TypeAlias

from langchain_aws import ChatBedrock

from core import settings
from schema.models import AWSModelName

_MODEL_TABLE = {
    AWSModelName.BEDROCK_CLAUDE_3_5_SONNET: "anthropic.claude-3-5-haiku-20241022-v1:0",
}

ModelT: TypeAlias = ChatBedrock


@cache
def get_model(model_name: AWSModelName, /) -> ModelT:
    api_model_name = _MODEL_TABLE.get(model_name)
    if not api_model_name:
        raise ValueError(f"Unsupported model: {model_name}")

    model_kwargs = {
        "temperature": 0.5,
        "max_tokens": 4096,
        "top_p": 1,
        "top_k": 250,
        "stop_sequences": ["\n\nHuman:"],
    }

    try:
        if model_name in AWSModelName and getattr(settings, "USE_AWS_BEDROCK", False):
            return ChatBedrock(
                model_id=api_model_name,
                credentials_profile_name=settings.AWS_PROFILE,
                region_name=settings.AWS_REGION,
                model_kwargs=model_kwargs,
            )

        else:
            return ChatBedrock(
                model_id=model_name,
                region_name=getattr(settings, "AWS_REGION", "us-west-2"),
                credentials_profile_name=getattr(
                    settings, "AWS_PROFILE", "AWS_AI_Access-814020624271"
                ),
                model_kwargs=model_kwargs,
            )

    except Exception as e:
        raise ValueError(f"Unsupported model: {model_name}")
