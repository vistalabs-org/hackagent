"""Contains shared errors types that can be raised from API functions"""


class HackAgentError(Exception):
    """Base exception for all HackAgent library specific errors."""

    pass


class ApiError(HackAgentError):
    """Represents an error returned by the API or an issue with API communication."""

    pass


class UnexpectedStatusError(ApiError):
    """Raised when an API response has an unexpected HTTP status code."""

    def __init__(self, status_code: int, content: bytes):
        self.status_code = status_code
        self.content = content
        super().__init__(
            f"Unexpected status code: {status_code}, content: {content.decode('utf-8', errors='replace')}"
        )


__all__ = [
    "HackAgentError",
    "ApiError",
    "UnexpectedStatusError",
]
