"""
CLI Configuration Management

Handles configuration loading from environment variables, files, and command line arguments.
"""

import os
import json
from pathlib import Path
from typing import Optional
from dataclasses import dataclass, asdict


@dataclass
class CLIConfig:
    """CLI configuration management with multiple sources"""

    api_key: Optional[str] = None
    base_url: str = "https://hackagent.dev"
    config_file: Optional[str] = None
    verbose: int = 0
    output_format: str = "table"  # table, json, csv

    def __post_init__(self):
        """Load configuration from various sources in priority order"""
        # Store default values
        defaults = {
            "api_key": None,
            "base_url": "https://hackagent.dev",
            "output_format": "table",
            "verbose": 0,
        }

        # Determine which values were explicitly set (different from defaults)
        self._cli_overrides = set()
        for key, default_value in defaults.items():
            current_value = getattr(self, key)
            if current_value != default_value:
                self._cli_overrides.add(key)

        # Load from sources in order: env vars, then config file
        self._load_from_env()
        if self.config_file:
            self._load_from_file(self.config_file)
        else:
            self._load_default_config()

    def _load_from_env(self):
        """Load from environment variables"""
        # Only load from env if not explicitly set via CLI args
        if not self.api_key:
            self.api_key = os.getenv("HACKAGENT_API_KEY")

        # Base URL is always hardcoded to official endpoint
        # if os.getenv('HACKAGENT_BASE_URL'):
        #     self.base_url = os.getenv('HACKAGENT_BASE_URL')

        # Only load output_format from env if it's still the default value
        if "output_format" not in self._cli_overrides and os.getenv(
            "HACKAGENT_OUTPUT_FORMAT"
        ):
            self.output_format = os.getenv("HACKAGENT_OUTPUT_FORMAT")

    def _load_from_file(self, config_path: str):
        """Load from configuration file (JSON or YAML)"""
        path = Path(config_path)
        if not path.exists():
            return

        try:
            with open(path) as f:
                if path.suffix.lower() in [".yaml", ".yml"]:
                    try:
                        import yaml

                        config_data = yaml.safe_load(f)
                    except ImportError:
                        raise ImportError(
                            "PyYAML required for YAML config files. Install with: pip install pyyaml"
                        )
                else:
                    config_data = json.load(f)

                # Update values based on precedence: CLI args > Config file > Env vars > Defaults
                # But never override base_url - it's always the official endpoint
                for key, value in config_data.items():
                    if hasattr(self, key) and key != "base_url":
                        # Never override CLI arguments
                        if key in self._cli_overrides:
                            continue

                        # For api_key, also check if it's None (from env vars)
                        if key == "api_key":
                            if self.api_key is None:
                                setattr(self, key, value)
                        else:
                            setattr(self, key, value)
        except Exception as e:
            raise ValueError(f"Failed to load config file {config_path}: {e}")

    def _load_default_config(self):
        """Load from default config file"""
        default_config = Path.home() / ".hackagent" / "config.json"
        if default_config.exists():
            self._load_from_file(str(default_config))

    def save(self, path: Optional[str] = None):
        """Save configuration to file"""
        if not path:
            config_dir = Path.home() / ".hackagent"
            config_dir.mkdir(parents=True, exist_ok=True)
            path = config_dir / "config.json"

        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            # Don't save None values, verbose level, or base_url (always hardcoded)
            config_dict = {
                k: v
                for k, v in asdict(self).items()
                if v is not None and k not in ["verbose", "config_file", "base_url"]
            }
            json.dump(config_dict, f, indent=2)

    def validate(self):
        """Validate configuration"""
        if not self.api_key:
            raise ValueError(
                "API key is required. Set HACKAGENT_API_KEY environment variable, "
                "use --api-key flag, or run 'hackagent config set --api-key YOUR_KEY'"
            )

        if not self.base_url:
            raise ValueError("Base URL is required")

    @property
    def default_config_path(self) -> Path:
        """Get the default configuration file path"""
        return Path.home() / ".hackagent" / "config.json"
