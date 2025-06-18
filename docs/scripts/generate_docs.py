#!/usr/bin/env python3
"""
HackAgent Documentation Generator

Generates API documentation from PyPI versions using Poetry and pydoc-markdown.
"""

import argparse
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

try:
    import requests
except ImportError:
    print("Installing required dependencies...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "requests"])
    import requests


def get_latest_version():
    """Get the latest version from PyPI."""
    try:
        response = requests.get("https://pypi.org/pypi/hackagent/json", timeout=30)
        response.raise_for_status()
        data = response.json()
        return data["info"]["version"]
    except Exception as e:
        print(f"Warning: Could not fetch latest version from PyPI: {e}")
        return "0.2.4"  # fallback


def get_current_version():
    """Get current version from local pyproject.toml."""
    try:
        # Get project root from script location
        script_dir = Path(__file__).parent
        project_root = script_dir.parent.parent

        result = subprocess.run(
            [
                "poetry",
                "run",
                "python",
                "-c",
                "import toml; "
                "data = toml.load('pyproject.toml'); "
                "print(data['tool']['poetry']['version'])",
            ],
            capture_output=True,
            text=True,
            cwd=project_root,
        )
        if result.returncode == 0:
            return result.stdout.strip()
    except Exception as e:
        print(f"Warning: Could not get local version: {e}")
    return "0.2.4"  # fallback


def check_requirements():
    """Check if Poetry is installed."""
    try:
        subprocess.run(["poetry", "--version"], capture_output=True, check=True)
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("❌ Poetry not found. Install with:")
        print("curl -sSL https://install.python-poetry.org | python3 -")
        return False


def create_pydoc_config(output_dir):
    """Create pydoc-markdown configuration."""
    config = f"""
loaders:
  - type: python
    modules: 
      - hackagent.agent
      - hackagent.client
      - hackagent.errors
      - hackagent.models
      - hackagent.router
      - hackagent.attacks.strategies
      - hackagent.attacks.base
      - hackagent.vulnerabilities.prompts

processors:
  - type: filter
    documented_only: true
    skip_empty_modules: true
  - type: smart
  - type: crossref

renderer:
  type: docusaurus
  docs_base_path: {output_dir}
  sidebar_top_level_label: "🔗 API Reference"
  sidebar_top_level_module: null
"""

    with tempfile.NamedTemporaryFile(mode="w", suffix=".yml", delete=False) as f:
        f.write(config.strip())
        return f.name


def run_command(cmd, cwd=None, description=None, exit_on_error=True):
    """Run a command with proper error handling."""
    if description:
        print(f"📦 {description}...")

    try:
        result = subprocess.run(
            cmd, cwd=cwd, check=True, capture_output=True, text=True
        )
        return result
    except subprocess.CalledProcessError as e:
        if exit_on_error:
            print(f"❌ Command failed: {' '.join(cmd)}")
            print(f"Error: {e.stderr}")
            sys.exit(1)
        else:
            return None


def generate_docs(version):
    """Generate documentation for the specified version."""
    print(f"🚀 Generating documentation for hackagent v{version}...")

    # Check requirements
    if not check_requirements():
        sys.exit(1)

    # Setup paths - script is in hackagent/docs/scripts/, so project root is 2 levels up
    script_dir = Path(__file__).parent
    project_root = script_dir.parent.parent
    output_dir = project_root / "docs/api-reference"
    docs_dir = project_root / "docs/docs"

    # Verify we're in the right directory (should have pyproject.toml and hackagent/ subdirectory)
    if (
        not (project_root / "pyproject.toml").exists()
        or not (project_root / "hackagent").exists()
    ):
        print(f"❌ Cannot find hackagent project structure from {script_dir}")
        print(f"❌ Expected pyproject.toml and hackagent/ directory in {project_root}")
        sys.exit(1)

    # Clean and create output directory
    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Install dependencies
    result = run_command(
        ["poetry", "install", "--with", "docs"],
        cwd=project_root,
        description="Installing dependencies",
        exit_on_error=False,
    )

    if result is None:
        # Fallback: install dependencies individually
        print("📦 Installing docs dependencies individually...")
        run_command(
            [
                "poetry",
                "add",
                "--group",
                "docs",
                "pydoc-markdown[docusaurus]",
                "toml",
                "packaging",
                "requests",
            ],
            cwd=project_root,
            description="Installing docs dependencies",
        )

    # Install specific hackagent version if not local
    # Skip installing if we're already in the hackagent project
    if version != "local" and version != get_current_version():
        print(
            f"📦 Note: Using local hackagent instead of v{version} (cannot install over self)"
        )
    elif version != "local":
        print(f"📦 Using current hackagent v{version}")

    # Create pydoc config
    config_file = create_pydoc_config(str(output_dir.relative_to(project_root)))

    try:
        # Generate documentation
        run_command(
            ["poetry", "run", "pydoc-markdown", config_file],
            cwd=project_root,
            description="Generating documentation",
        )

        # Create index file
        index_content = f"""---
sidebar_position: 1
---

# API Reference v{version}

This documentation was generated from hackagent v{version}.

"""

        (output_dir / "index.md").write_text(index_content)

        # Copy generated files to docs directory for Docusaurus
        if (output_dir / "reference").exists():
            # Move reference files to correct location
            for item in (output_dir / "reference").iterdir():
                if item.is_file():
                    shutil.copy2(item, output_dir)
                elif item.is_dir():
                    if (output_dir / item.name).exists():
                        shutil.rmtree(output_dir / item.name)
                    shutil.copytree(item, output_dir / item.name)
            shutil.rmtree(output_dir / "reference")

        # Copy API docs to main docs directory
        api_index_path = output_dir / "index.md"
        if api_index_path.exists():
            shutil.copy2(api_index_path, docs_dir / "api-index.md")

        hackagent_dir = output_dir / "hackagent"
        if hackagent_dir.exists():
            docs_hackagent_dir = docs_dir / "hackagent"
            if docs_hackagent_dir.exists():
                shutil.rmtree(docs_hackagent_dir)
            shutil.copytree(hackagent_dir, docs_hackagent_dir)

        print(f"✅ Documentation generated in {output_dir}")
        print(f"✅ Documentation copied to {docs_dir}")
        print("\n🔧 To view documentation:")
        print("  cd docs && npm start")

    finally:
        # Cleanup
        try:
            os.unlink(config_file)
        except OSError:
            pass


def main():
    parser = argparse.ArgumentParser(description="HackAgent Documentation Generator")
    parser.add_argument("-v", "--version", help="Specific PyPI version (e.g., 0.2.4)")
    parser.add_argument(
        "-l", "--latest", action="store_true", help="Use latest PyPI version (default)"
    )
    parser.add_argument(
        "-c", "--current", action="store_true", help="Use current local version"
    )

    args = parser.parse_args()

    if args.current:
        version = "local"
    elif args.version:
        version = args.version
    else:
        version = get_latest_version()

    generate_docs(version)


if __name__ == "__main__":
    main()
