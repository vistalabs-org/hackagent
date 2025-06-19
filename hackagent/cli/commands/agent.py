"""
Agent Commands

Manage AI agents registered with HackAgent.
"""

import click
from rich.console import Console
from rich.table import Table

from hackagent.cli.config import CLIConfig
from hackagent.cli.utils import (
    handle_errors,
    display_success,
    display_info,
    display_warning,
    get_agent_type_enum,
    confirm_action,
)

console = Console()


@click.group()
def agent():
    """🤖 Manage AI agents"""
    # Show logo when agent commands are used
    _show_logo_once()


def _show_logo_once():
    """Show the logo once per session"""
    if not hasattr(_show_logo_once, "_shown"):
        from hackagent.utils import display_hackagent_splash

        display_hackagent_splash()
        _show_logo_once._shown = True


@agent.command()
@click.pass_context
@handle_errors
def list(ctx):
    """List registered agents"""

    cli_config: CLIConfig = ctx.obj["config"]
    cli_config.validate()

    try:
        from hackagent.client import AuthenticatedClient
        from hackagent.api.agent import agent_list

        # Initialize client
        client = AuthenticatedClient(
            base_url=cli_config.base_url, token=cli_config.api_key, prefix="Api-Key"
        )

        with console.status("[bold green]Fetching agents..."):
            response = agent_list.sync_detailed(client=client)

        if response.status_code == 200 and response.parsed:
            agents = response.parsed.results

            if not agents:
                display_info("No agents found")
                return

            table = Table(
                title=f"Registered Agents ({len(agents)})",
                show_header=True,
                header_style="bold cyan",
            )
            table.add_column("ID", style="dim")
            table.add_column("Name", style="cyan")
            table.add_column("Type", style="green")
            table.add_column("Endpoint", style="yellow")
            table.add_column("Created", style="dim")

            for agent_obj in agents:
                # Format creation date
                created = "Unknown"
                if hasattr(agent_obj, "created_at") and agent_obj.created_at:
                    try:
                        from datetime import datetime

                        if isinstance(agent_obj.created_at, datetime):
                            created = agent_obj.created_at.strftime("%Y-%m-%d %H:%M")
                        else:
                            created = str(agent_obj.created_at)[:16]
                    except (AttributeError, ValueError, TypeError):
                        created = str(agent_obj.created_at)[:16]

                table.add_row(
                    str(agent_obj.id)[:8] + "...",
                    agent_obj.name or "Unnamed",
                    agent_obj.agent_type.value
                    if hasattr(agent_obj.agent_type, "value")
                    else str(agent_obj.agent_type),
                    agent_obj.endpoint or "Not specified",
                    created,
                )

            console.print(table)

        else:
            raise click.ClickException(
                f"Failed to fetch agents: Status {response.status_code}"
            )

    except Exception as e:
        raise click.ClickException(f"Failed to list agents: {e}")


@agent.command()
@click.option("--name", required=True, help="Agent name")
@click.option(
    "--type",
    "agent_type",
    type=click.Choice(["google-adk", "litellm"]),
    required=True,
    help="Agent type",
)
@click.option("--endpoint", required=True, help="Agent endpoint URL")
@click.option("--description", help="Agent description")
@click.option("--metadata", help="Additional metadata as JSON string")
@click.pass_context
@handle_errors
def create(ctx, name, agent_type, endpoint, description, metadata):
    """Create a new agent"""

    cli_config: CLIConfig = ctx.obj["config"]
    cli_config.validate()

    # Convert agent type
    agent_type_enum = get_agent_type_enum(agent_type)

    # Parse metadata if provided
    metadata_dict = {}
    if metadata:
        try:
            import json

            metadata_dict = json.loads(metadata)
        except json.JSONDecodeError as e:
            raise click.ClickException(f"Invalid JSON metadata: {e}")

    try:
        from hackagent.client import AuthenticatedClient
        from hackagent.api.agent import agent_create
        from hackagent.models.agent_request import AgentRequest

        # Initialize client
        client = AuthenticatedClient(
            base_url=cli_config.base_url, token=cli_config.api_key, prefix="Api-Key"
        )

        # Get organization ID from API key
        from hackagent.api.key import key_list

        keys_response = key_list.sync_detailed(client=client)

        if keys_response.status_code != 200 or not keys_response.parsed:
            raise click.ClickException("Failed to get organization information")

        organization_id = None
        current_token = cli_config.api_key
        for key_obj in keys_response.parsed.results:
            if current_token.startswith(key_obj.prefix):
                organization_id = key_obj.organization
                break

        if not organization_id:
            raise click.ClickException("Could not determine organization ID")

        # Create agent request
        agent_request = AgentRequest(
            name=name,
            agent_type=agent_type_enum,
            endpoint=endpoint,
            description=description or "Agent managed by CLI",
            metadata=metadata_dict,
            organization=organization_id,
        )

        with console.status(f"[bold green]Creating agent '{name}'..."):
            response = agent_create.sync_detailed(client=client, body=agent_request)

        if response.status_code == 201 and response.parsed:
            agent_obj = response.parsed
            display_success(f"✅ Agent '{name}' created successfully")

            # Display agent details
            _display_agent_details(agent_obj)

        else:
            error_msg = "Unknown error"
            if response.content:
                try:
                    import json

                    error_data = json.loads(response.content.decode())
                    error_msg = str(error_data)
                except (json.JSONDecodeError, UnicodeDecodeError, AttributeError):
                    error_msg = response.content.decode()

            raise click.ClickException(f"Failed to create agent: {error_msg}")

    except Exception as e:
        raise click.ClickException(f"Failed to create agent: {e}")


@agent.command()
@click.argument("agent_id")
@click.pass_context
@handle_errors
def show(ctx, agent_id):
    """Show detailed information about an agent"""

    cli_config: CLIConfig = ctx.obj["config"]
    cli_config.validate()

    try:
        from hackagent.client import AuthenticatedClient
        from hackagent.api.agent import agent_retrieve

        client = AuthenticatedClient(
            base_url=cli_config.base_url, token=cli_config.api_key, prefix="Api-Key"
        )

        with console.status(f"[bold green]Fetching agent {agent_id}..."):
            response = agent_retrieve.sync_detailed(client=client, id=agent_id)

        if response.status_code == 200 and response.parsed:
            agent_obj = response.parsed
            _display_agent_details(agent_obj, detailed=True)
        else:
            raise click.ClickException(f"Agent not found or access denied: {agent_id}")

    except Exception as e:
        raise click.ClickException(f"Failed to fetch agent: {e}")


@agent.command()
@click.argument("agent_id")
@click.option("--name", help="New agent name")
@click.option("--endpoint", help="New agent endpoint")
@click.option("--description", help="New agent description")
@click.option("--metadata", help="New metadata as JSON string")
@click.pass_context
@handle_errors
def update(ctx, agent_id, name, endpoint, description, metadata):
    """Update an existing agent"""

    cli_config: CLIConfig = ctx.obj["config"]
    cli_config.validate()

    # Check if any updates provided
    if not any([name, endpoint, description, metadata]):
        display_info("No updates specified")
        return

    try:
        from hackagent.client import AuthenticatedClient
        from hackagent.api.agent import agent_partial_update
        from hackagent.models.patched_agent_request import PatchedAgentRequest

        client = AuthenticatedClient(
            base_url=cli_config.base_url, token=cli_config.api_key, prefix="Api-Key"
        )

        # Parse metadata if provided
        metadata_dict = None
        if metadata:
            try:
                import json

                metadata_dict = json.loads(metadata)
            except json.JSONDecodeError as e:
                raise click.ClickException(f"Invalid JSON metadata: {e}")

        # Create update request with only provided fields
        update_data = {}
        if name:
            update_data["name"] = name
        if endpoint:
            update_data["endpoint"] = endpoint
        if description:
            update_data["description"] = description
        if metadata_dict is not None:
            update_data["metadata"] = metadata_dict

        patch_request = PatchedAgentRequest(**update_data)

        with console.status(f"[bold green]Updating agent {agent_id}..."):
            response = agent_partial_update.sync_detailed(
                client=client, id=agent_id, body=patch_request
            )

        if response.status_code == 200 and response.parsed:
            agent_obj = response.parsed
            display_success("✅ Agent updated successfully")
            _display_agent_details(agent_obj)
        else:
            raise click.ClickException(
                f"Failed to update agent: Status {response.status_code}"
            )

    except Exception as e:
        raise click.ClickException(f"Failed to update agent: {e}")


@agent.command()
@click.argument("agent_id")
@click.option("--confirm", is_flag=True, help="Skip confirmation prompt")
@click.pass_context
@handle_errors
def delete(ctx, agent_id, confirm):
    """Delete an agent"""

    cli_config: CLIConfig = ctx.obj["config"]
    cli_config.validate()

    if not confirm:
        if not confirm_action(
            f"Delete agent {agent_id}? This action cannot be undone."
        ):
            display_info("Agent deletion cancelled")
            return

    try:
        from hackagent.client import AuthenticatedClient
        from hackagent.api.agent import agent_destroy

        client = AuthenticatedClient(
            base_url=cli_config.base_url, token=cli_config.api_key, prefix="Api-Key"
        )

        with console.status(f"[bold red]Deleting agent {agent_id}..."):
            response = agent_destroy.sync_detailed(client=client, id=agent_id)

        if response.status_code == 204:
            display_success(f"✅ Agent {agent_id} deleted successfully")
        else:
            raise click.ClickException(
                f"Failed to delete agent: Status {response.status_code}"
            )

    except Exception as e:
        raise click.ClickException(f"Failed to delete agent: {e}")


def _display_agent_details(agent_obj, detailed: bool = False) -> None:
    """Display detailed information about an agent"""

    # Basic info table
    table = Table(
        title=f"Agent: {agent_obj.name}", show_header=True, header_style="bold cyan"
    )
    table.add_column("Property", style="cyan")
    table.add_column("Value", style="green")

    table.add_row("ID", str(agent_obj.id))
    table.add_row("Name", agent_obj.name or "Unnamed")
    table.add_row(
        "Type",
        agent_obj.agent_type.value
        if hasattr(agent_obj.agent_type, "value")
        else str(agent_obj.agent_type),
    )
    table.add_row("Endpoint", agent_obj.endpoint or "Not specified")
    table.add_row("Description", agent_obj.description or "No description")

    # Format dates
    if hasattr(agent_obj, "created_at") and agent_obj.created_at:
        try:
            from datetime import datetime

            if isinstance(agent_obj.created_at, datetime):
                created = agent_obj.created_at.strftime("%Y-%m-%d %H:%M:%S")
            else:
                created = str(agent_obj.created_at)
        except (AttributeError, ValueError, TypeError):
            created = str(agent_obj.created_at)
        table.add_row("Created", created)

    if hasattr(agent_obj, "updated_at") and agent_obj.updated_at:
        try:
            from datetime import datetime

            if isinstance(agent_obj.updated_at, datetime):
                updated = agent_obj.updated_at.strftime("%Y-%m-%d %H:%M:%S")
            else:
                updated = str(agent_obj.updated_at)
        except (AttributeError, ValueError, TypeError):
            updated = str(agent_obj.updated_at)
        table.add_row("Updated", updated)

    console.print(table)

    # Show metadata if present and detailed view requested
    if detailed and hasattr(agent_obj, "metadata") and agent_obj.metadata:
        console.print("\n[bold cyan]📋 Metadata:")
        try:
            import json

            metadata_str = json.dumps(agent_obj.metadata, indent=2)
            console.print(f"[dim]{metadata_str}")
        except (json.JSONDecodeError, TypeError, AttributeError):
            console.print(f"[dim]{agent_obj.metadata}")

    # Show organization info if available
    if hasattr(agent_obj, "organization") and agent_obj.organization:
        console.print(f"\n[dim]Organization ID: {agent_obj.organization}")


@agent.command()
@click.argument("agent_name")
@click.pass_context
@handle_errors
def test(ctx, agent_name):
    """Test connection to an agent

    This command attempts to establish a connection with the specified agent
    to verify it's accessible and responding.
    """

    cli_config: CLIConfig = ctx.obj["config"]
    cli_config.validate()

    try:
        # First, find the agent by name
        from hackagent.client import AuthenticatedClient
        from hackagent.api.agent import agent_list

        client = AuthenticatedClient(
            base_url=cli_config.base_url, token=cli_config.api_key, prefix="Api-Key"
        )

        with console.status(f"[bold green]Looking up agent '{agent_name}'..."):
            response = agent_list.sync_detailed(client=client)

        if response.status_code != 200 or not response.parsed:
            raise click.ClickException("Failed to fetch agents list")

        # Find agent by name
        target_agent = None
        for agent_obj in response.parsed.results:
            if agent_obj.name == agent_name:
                target_agent = agent_obj
                break

        if not target_agent:
            raise click.ClickException(f"Agent '{agent_name}' not found")

        display_info(
            f"Found agent: {target_agent.name} ({target_agent.agent_type.value})"
        )
        display_info(f"Endpoint: {target_agent.endpoint}")

        # Test basic connectivity
        with console.status(
            f"[bold green]Testing connection to {target_agent.endpoint}..."
        ):
            import requests
            import time

            start_time = time.time()
            try:
                # Try a basic HTTP request to the endpoint
                response = requests.get(
                    target_agent.endpoint,
                    timeout=10,
                    headers={"User-Agent": "HackAgent-CLI/0.2.4"},
                )
                duration = time.time() - start_time

                if response.status_code < 500:
                    display_success(
                        f"✅ Connection successful (HTTP {response.status_code}) in {duration:.2f}s"
                    )
                else:
                    display_warning(
                        f"⚠️ Server error (HTTP {response.status_code}) in {duration:.2f}s"
                    )

            except requests.exceptions.Timeout:
                display_warning("⚠️ Connection timeout after 10s")
            except requests.exceptions.ConnectionError:
                display_warning("⚠️ Connection failed - agent may not be running")
            except Exception as e:
                display_warning(f"⚠️ Connection test failed: {e}")

        # Additional agent-specific tests could be added here
        display_info("💡 Use 'hackagent attack' commands to perform security tests")

    except Exception as e:
        raise click.ClickException(f"Failed to test agent: {e}")
