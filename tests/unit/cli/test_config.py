"""
Unit tests for CLI configuration functionality.
"""

import pytest
import json
import tempfile
from pathlib import Path
from unittest.mock import patch

from hackagent.cli.config import CLIConfig


class TestCLIConfig:
    """Test CLI configuration management"""

    def test_default_config(self):
        """Test default configuration values"""
        # Mock the default config file to not exist and clear env vars
        with (
            patch("pathlib.Path.exists", return_value=False),
            patch.dict("os.environ", {}, clear=True),
        ):
            config = CLIConfig()

            assert config.api_key is None
            assert config.base_url == "https://hackagent.dev"
            assert config.verbose == 0
            assert config.output_format == "table"

    def test_env_variable_loading(self):
        """Test loading from environment variables"""
        with (
            patch.dict(
                "os.environ",
                {
                    "HACKAGENT_API_KEY": "test-key",
                    "HACKAGENT_BASE_URL": "https://test.example.com",
                    "HACKAGENT_OUTPUT_FORMAT": "json",
                },
            ),
            patch("pathlib.Path.exists", return_value=False),
        ):
            config = CLIConfig()

            assert config.api_key == "test-key"
            # Note: base_url is hardcoded and doesn't load from env
            assert config.base_url == "https://hackagent.dev"
            assert config.output_format == "json"

    def test_config_file_loading(self):
        """Test loading from configuration file"""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            config_data = {
                "api_key": "file-key",
                "base_url": "https://file.example.com",
                "output_format": "csv",
            }
            json.dump(config_data, f)
            config_file = f.name

        try:
            # Clear env vars to ensure we're only testing file loading
            with patch.dict("os.environ", {}, clear=True):
                config = CLIConfig(config_file=config_file)

                assert config.api_key == "file-key"
                # Note: base_url is hardcoded and doesn't load from config file
                assert config.base_url == "https://hackagent.dev"
                assert config.output_format == "csv"
        finally:
            Path(config_file).unlink()

    def test_cli_args_override(self):
        """Test that CLI arguments override other sources"""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            config_data = {
                "api_key": "file-key",
                "base_url": "https://file.example.com",
            }
            json.dump(config_data, f)
            config_file = f.name

        try:
            with patch.dict("os.environ", {"HACKAGENT_API_KEY": "env-key"}):
                config = CLIConfig(
                    config_file=config_file,
                    api_key="cli-key",
                    base_url="https://cli.example.com",
                )

                # CLI args should take precedence
                assert config.api_key == "cli-key"
                assert config.base_url == "https://cli.example.com"
        finally:
            Path(config_file).unlink()

    def test_save_config(self):
        """Test saving configuration to file"""
        with tempfile.TemporaryDirectory() as temp_dir:
            config_path = Path(temp_dir) / "test_config.json"

            config = CLIConfig(
                api_key="test-key",
                base_url="https://test.example.com",
                output_format="json",
            )

            config.save(str(config_path))

            assert config_path.exists()

            # Load and verify
            with open(config_path) as f:
                saved_data = json.load(f)

            assert saved_data["api_key"] == "test-key"
            # Note: base_url is not saved as it's always hardcoded
            assert "base_url" not in saved_data
            assert saved_data["output_format"] == "json"

    def test_validate_success(self):
        """Test successful validation"""
        config = CLIConfig(api_key="valid-key", base_url="https://example.com")

        # Should not raise an exception
        config.validate()

    def test_validate_missing_api_key(self):
        """Test validation failure with missing API key"""
        with (
            patch("pathlib.Path.exists", return_value=False),
            patch.dict("os.environ", {}, clear=True),
        ):
            config = CLIConfig(base_url="https://example.com")

            with pytest.raises(ValueError, match="API key is required"):
                config.validate()

    def test_validate_missing_base_url(self):
        """Test validation failure with missing base URL"""
        config = CLIConfig(api_key="test-key", base_url="")

        with pytest.raises(ValueError, match="Base URL is required"):
            config.validate()

    def test_default_config_path(self):
        """Test default configuration path"""
        with patch.dict("os.environ", {}, clear=True):
            config = CLIConfig()

            expected_path = Path.home() / ".hackagent" / "config.json"
            assert config.default_config_path == expected_path

    def test_yaml_config_loading(self):
        """Test loading YAML configuration"""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            yaml_content = """
api_key: yaml-key
base_url: https://yaml.example.com
output_format: table
"""
            f.write(yaml_content)
            config_file = f.name

        try:
            # This should work if PyYAML is available
            try:
                import importlib.util

                yaml_spec = importlib.util.find_spec("yaml")
                if yaml_spec is not None:
                    # Clear env vars to ensure we're only testing file loading
                    with patch.dict("os.environ", {}, clear=True):
                        config = CLIConfig(config_file=config_file)

                    assert config.api_key == "yaml-key"
                    # Note: base_url is hardcoded and doesn't load from config file
                    assert config.base_url == "https://hackagent.dev"
                    assert config.output_format == "table"
            except ImportError:
                # PyYAML not available, should raise appropriate error
                with pytest.raises(ImportError, match="PyYAML required"):
                    CLIConfig(config_file=config_file)
        finally:
            Path(config_file).unlink()

    def test_invalid_json_config(self):
        """Test handling of invalid JSON configuration"""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            f.write("invalid json content")
            config_file = f.name

        try:
            with patch.dict("os.environ", {}, clear=True):
                with pytest.raises(ValueError, match="Failed to load config file"):
                    CLIConfig(config_file=config_file)
        finally:
            Path(config_file).unlink()

    def test_nonexistent_config_file(self):
        """Test handling of non-existent configuration file"""
        # Should not raise an error, just continue with defaults
        with patch.dict("os.environ", {}, clear=True):
            config = CLIConfig(config_file="/nonexistent/config.json")

            # Should use defaults
            assert config.base_url == "https://hackagent.dev"
            assert config.output_format == "table"
