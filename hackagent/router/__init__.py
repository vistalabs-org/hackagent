"""Main router logic for dispatching requests to appropriate agent adapters."""

from .router import AgentRouter
from .adapters import (
    ADKAgentAdapter,
)  # This makes it easy to access adapters via router module

__all__ = [
    "AgentRouter",
    "ADKAgentAdapter",  # Exporting specific adapters for convenience
]
