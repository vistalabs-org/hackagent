import abc
from typing import Any, Dict


class BaseAttack(abc.ABC):
    """
    Abstract base class for black-box attacks against language models.
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initializes the attack with configuration parameters.

        Args:
            config: A dictionary containing configuration settings for the attack.
        """
        self.config = config
        self._validate_config()
        self._setup()

    def _validate_config(self):
        """
        Validates the provided configuration.
        Subclasses can override this to enforce specific config requirements.
        """
        if not isinstance(self.config, dict):
            raise TypeError("Configuration must be a dictionary.")
        # Add more specific validation in subclasses as needed
        pass

    def _setup(self):
        """
        Performs any necessary setup based on the configuration.
        Subclasses can override this for specific setup tasks (e.g., loading models, data).
        """
        # Add setup logic in subclasses (e.g., logging, directories)
        pass

    @abc.abstractmethod
    def run(self, **kwargs: Any) -> Any:
        """
        Executes the attack logic.

        Args:
            **kwargs: Attack-specific arguments (e.g., input prompts, goals, dataset).

        Returns:
            Attack-specific results (e.g., adversarial examples, success metrics, report).
        """
        pass
