"""A client library for accessing HackAgent API"""

from .client import AuthenticatedClient, Client
from .agent import HackAgent
from .errors import HackAgentError, ApiError, UnexpectedStatusError
from .models import Agent, Prompt, Result, Run
from .logger import setup_package_logging

setup_package_logging()


__all__ = (
    "AuthenticatedClient",
    "Client",
    "HackAgent",
    "HackAgentError",
    "ApiError",
    "UnexpectedStatusError",
    "Agent",
    "Prompt",
    "Result",
    "Run",
)
