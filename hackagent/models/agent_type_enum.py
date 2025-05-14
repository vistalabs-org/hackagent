from enum import Enum


class AgentTypeEnum(str, Enum):
    GOOGLE_ADK = "GOOGLE_ADK"
    LITELMM = "LITELMM"
    OPENAI_SDK = "OPENAI_SDK"
    OTHER = "OTHER"
    UNKNOWN = "UNKNOWN"

    def __str__(self) -> str:
        return str(self.value)
