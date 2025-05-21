# Copyright 2025 - Vista Labs. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from rich.console import Console
from rich.panel import Panel
from rich.text import Text
import logging
import os
from typing import Optional, Union
from dotenv import load_dotenv, find_dotenv

from hackagent.models import AgentTypeEnum

logger = logging.getLogger(__name__)


HACKAGENT = """
██╗  ██╗ █████╗  ██████╗██╗  ██╗            
██║  ██║██╔══██╗██╔════╝██║ ██╔╝            
███████║███████║██║     █████╔╝             
██╔══██║██╔══██║██║     ██╔═██╗             
██║  ██║██║  ██║╚██████╗██║  ██╗            
╚═╝  ╚═╝╚═╝  ╚═╝ ╚═════╝╚═╝  ╚═╝            
                                            
 █████╗  ██████╗ ███████╗███╗   ██╗████████╗
██╔══██╗██╔════╝ ██╔════╝████╗  ██║╚══██╔══╝
███████║██║  ███╗█████╗  ██╔██╗ ██║   ██║   
██╔══██║██║   ██║██╔══╝  ██║╚██╗██║   ██║   
██║  ██║╚██████╔╝███████╗██║ ╚████║   ██║   
╚═╝  ╚═╝ ╚═════╝ ╚══════╝╚═╝  ╚═══╝   ╚═╝                                               
"""


def display_hackagent_splash():
    """Displays the HackAgent splash screen using the pre-defined ASCII art."""
    console = Console()

    # Create a Text object from the HACKAGENT string
    title_content = Text(HACKAGENT, style="bold dark_red")

    splash_panel = Panel(
        title_content,
        border_style="red",
        padding=(2, 2),
        expand=False,
    )

    console.print(splash_panel)
    console.print()


def resolve_agent_type(agent_type_input: Union[AgentTypeEnum, str]) -> AgentTypeEnum:
    """Resolves the agent type from a string or AgentTypeEnum member."""
    if isinstance(agent_type_input, str):
        try:
            # Convert to uppercase and replace hyphens with underscores for enum matching
            return AgentTypeEnum[agent_type_input.upper().replace("-", "_")]
        except KeyError:
            logger.warning(
                f"Invalid agent_type string: '{agent_type_input}'. Falling back to UNKNOWN. "
                f"Valid types are: {[member.name for member in AgentTypeEnum]}"
            )
            return AgentTypeEnum.UNKNOWN
    elif isinstance(agent_type_input, AgentTypeEnum):
        return agent_type_input
    else:
        logger.warning(
            f"Invalid agent_type type: {type(agent_type_input)}. Falling back to UNKNOWN."
        )
        return AgentTypeEnum.UNKNOWN


def resolve_api_token(
    direct_api_key_param: Optional[str], env_file_path: Optional[str]
) -> str:
    """Resolves the API token from the direct api_key parameter or environment variables."""
    if direct_api_key_param is not None:
        logger.debug("Using API token provided directly via 'api_key' parameter.")
        return direct_api_key_param

    # If direct_api_key_param is None, attempt to load from environment.
    logger.debug(
        "API token not provided via 'api_key' parameter, attempting to load from environment."
    )
    dotenv_to_load = env_file_path or find_dotenv(usecwd=True)

    if dotenv_to_load:
        logger.debug(f"Loading .env file from: {dotenv_to_load}")
        load_dotenv(dotenv_to_load)
    else:
        logger.debug("No .env file found to load.")

    api_token_resolved = os.getenv("HACKAGENT_API_KEY")

    if not api_token_resolved:
        error_message = (
            "API token not provided via 'api_key' parameter, "
            "and not found in HACKAGENT_API_KEY environment variable "
            "(after attempting to load .env)."
        )
        raise ValueError(error_message)
    logger.debug("Using API token from HACKAGENT_API_KEY environment variable.")
    return api_token_resolved
