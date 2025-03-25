from enum import StrEnum, auto
from typing import TypeAlias


class Provider(StrEnum):
    AWS = auto()


class AWSModelName(StrEnum):
    """https://docs.aws.amazon.com/bedrock/latest/userguide/models-supported.html"""

    BEDROCK_CLAUDE_3_5_SONNET_V1 = "anthropic.claude-3-5-sonnet-20240620-v1:0"
    BEDROCK_CLAUDE_3_5_SONNET_V2 = "us.anthropic.claude-3-5-sonnet-20241022-v2:0"


AllModelEnum: TypeAlias = AWSModelName
