"""
Module for getting complete responses from prefixes using target LLM.
"""

import asyncio
import pandas as pd
import os
import logging
import uuid
from typing import Dict, Optional, Any
from dataclasses import dataclass
from rich.progress import (
    Progress,
    BarColumn,
    TextColumn,
    TimeRemainingColumn,
    MofNCompleteColumn,
    SpinnerColumn,
)
from hackagent.client import AuthenticatedClient
from hackagent.router.router import AgentRouter, AgentTypeEnum


@dataclass
class CompletionConfig:
    """Configuration for getting completions using an Agent via AgentRouter."""

    agent_name: str  # A descriptive name for this agent configuration
    agent_type: AgentTypeEnum  # Type of agent (ADK, LiteLLM, etc.)
    organization_id: int  # Organization ID for backend agent registration
    model_id: str  # General model identifier (e.g., "claude-2", "gpt-4", "ADK")
    agent_endpoint: str  # API endpoint for the agent service (e.g., ADK's base URL, LiteLLM's API base if applicable)
    agent_metadata: Optional[Dict[str, Any]] = (
        None  # For ADK: {'adk_app_name': 'app_name'}; For LiteLLM: {'name': 'model_string', 'api_key': '...', ...}
    )

    batch_size: int = 1  # Remains, but actual batching for API calls might be handled differently or by adapter
    max_new_tokens: int = 256
    temperature: float = 1.0
    n_samples: int = 25
    surrogate_attack_prompt: str = ""  # Remains for LiteLLM type agents
    request_timeout: int = 120
    # api_key removed, should be in agent_metadata for LiteLLM if needed by adapter
    # adk_app_name removed, should be in agent_metadata for ADK


class PrefixCompleter:
    """Class for getting completions from prefixes using a target LLM via AgentRouter."""

    def __init__(self, client: AuthenticatedClient, config: CompletionConfig):
        """Initialize the completer with config and an AuthenticatedClient."""
        self.client = client
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.api_key = (
            None  # Remains for LiteLLM type agents if API key is directly managed
        )

        # API key loading for LiteLLM (if specified in metadata)
        if (
            self.config.agent_type == AgentTypeEnum.LITELMM
            and self.config.agent_metadata
            and "api_key" in self.config.agent_metadata
        ):
            api_key = self.config.agent_metadata["api_key"]
            self.api_key = os.environ.get(api_key)
            if not self.api_key:
                self.logger.warning(
                    f"Environment variable {api_key} for LiteLLM API key not set."
                )

        # Initialize AgentRouter
        # The router handles backend agent registration and adapter instantiation.
        # Operational config for the adapter can be passed here if needed,
        # otherwise, it's taken from backend_agent.metadata or the adapter's defaults.
        adapter_op_config = {}
        if self.config.agent_type == AgentTypeEnum.LITELMM:
            # For LiteLLM, ensure 'name' (model string) is available for the adapter
            if self.config.agent_metadata and "name" in self.config.agent_metadata:
                adapter_op_config["name"] = self.config.agent_metadata["name"]
            else:
                # Fallback or error if model_id itself isn't the direct model string
                # This depends on how LiteLLMAgentAdapter expects 'name'
                adapter_op_config["name"] = (
                    self.config.model_id
                )  # Assuming model_id can be the litellm model string
                self.logger.warning(
                    f"LiteLLM 'name' (model string) not found in agent_metadata, using model_id '{self.config.model_id}'. Ensure this is correct."
                )
            if self.api_key:  # Pass API key if loaded
                adapter_op_config["api_key"] = self.api_key
            if self.config.agent_endpoint:  # Pass API base if specified
                adapter_op_config["endpoint"] = self.config.agent_endpoint
            adapter_op_config["max_new_tokens"] = self.config.max_new_tokens
            adapter_op_config["temperature"] = self.config.temperature
            # Potentially other LiteLLM params like 'top_p' if needed by adapter

        self.agent_router = AgentRouter(
            client=self.client,
            name=self.config.agent_name,  # Name for backend agent registration
            agent_type=self.config.agent_type,
            organization_id=self.config.organization_id,
            endpoint=self.config.agent_endpoint,  # Endpoint of the actual agent service
            metadata=self.config.agent_metadata,
            adapter_operational_config=adapter_op_config,
            overwrite_metadata=True,  # Or False, depending on desired behavior
        )
        # The agent's unique registration key (backend agent ID)
        # Assuming the AgentRouter's _agent_registry has one entry after init for a single agent.
        if not self.agent_router._agent_registry:
            raise RuntimeError(
                "AgentRouter did not register any agent upon initialization."
            )
        self.agent_registration_key = list(self.agent_router._agent_registry.keys())[0]

        self.logger.info(
            f"PrefixCompleter initialized for agent '{self.config.agent_name}' "
            f"(Type: {self.config.agent_type.value}, Backend ID: {self.agent_registration_key}) "
            f"via AgentRouter."
        )

    def expand_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """Expand dataframe to include multiple samples per prefix"""
        expanded_rows = []
        self.logger.info(
            f"Expanding DataFrame for {self.config.n_samples} samples per prefix."
        )
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            MofNCompleteColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.1f}%"),
            TimeRemainingColumn(),
        ) as progress_bar:
            task = progress_bar.add_task("[cyan]Expanding samples...", total=len(df))
            for _, row in df.iterrows():
                for sample_id in range(self.config.n_samples):
                    expanded_row = row.to_dict()
                    expanded_row["sample_id"] = sample_id
                    expanded_row["completion"] = (
                        ""  # Placeholder for the generated part
                    )
                    expanded_rows.append(expanded_row)
                progress_bar.update(task, advance=1)

        return pd.DataFrame(expanded_rows)

    async def get_completions(self, df: pd.DataFrame) -> pd.DataFrame:
        """Get completions for all prefixes in dataframe using the configured AgentRouter."""
        self.logger.info(
            f"Starting completions for {len(df)} unique prefixes with {self.config.n_samples} samples each."
        )
        expanded_df = self.expand_dataframe(df)

        if "target" in expanded_df.columns:
            expanded_df.rename(columns={"target": "prefix"}, inplace=True)
            self.logger.debug("Renamed 'target' column to 'prefix'.")
        if "target_ce_loss" in expanded_df.columns:
            expanded_df.rename(columns={"target_ce_loss": "prefix_nll"}, inplace=True)
            self.logger.debug("Renamed 'target_ce_loss' column to 'prefix_nll'.")

        if "prefix" not in expanded_df.columns or "goal" not in expanded_df.columns:
            raise ValueError(
                "Input DataFrame must contain 'prefix' and 'goal' columns."
            )

        adk_session_id: Optional[str] = None
        adk_user_id: Optional[str] = None
        if self.config.agent_type == AgentTypeEnum.GOOGLE_ADK:
            adk_session_id = str(uuid.uuid4())
            adk_user_id = f"completer_user_{adk_session_id[:8]}"
            self.logger.info(
                f"Generated ADK session_id: {adk_session_id} and user_id: {adk_user_id} for this batch."
            )
            # ADK session creation is now handled by the ADKAgentAdapter internally per request if needed,
            # or managed based on session_id persistence by the adapter.

        tasks = []
        for index, row in expanded_df.iterrows():
            goal = row["goal"]
            prefix_text = row["prefix"]
            # Pass adk_session_id and adk_user_id if ADK, they will be None otherwise
            tasks.append(
                self._execute_completion_request(
                    goal, prefix_text, index, adk_session_id, adk_user_id
                )
            )

        self.logger.info(f"Gathering {len(tasks)} completion requests...")
        detailed_completion_results = await asyncio.gather(
            *tasks, return_exceptions=True
        )
        self.logger.info("All completion requests processed.")

        # Process results, handling potential exceptions from asyncio.gather
        processed_results = []
        for i, result in enumerate(detailed_completion_results):
            if isinstance(result, Exception):
                self.logger.error(
                    f"Exception during completion request for original index {i}: {result}",
                    exc_info=result,
                )
                processed_results.append(
                    {
                        "generated_text": f"[ERROR: Async Task Exception - {type(result).__name__}]",
                        "request_payload": None,
                        "response_status_code": None,
                        "response_headers": None,
                        "response_body_raw": None,
                        "adk_events_list": None,
                        "error_message": str(result),
                    }
                )
            else:
                processed_results.append(result)

        detailed_completion_results = processed_results

        if len(detailed_completion_results) == len(expanded_df):
            expanded_df["generated_text_only"] = [
                res.get("generated_text") for res in detailed_completion_results
            ]
            expanded_df["request_payload"] = [
                res.get("request_payload") for res in detailed_completion_results
            ]
            expanded_df["response_status_code"] = [
                res.get("response_status_code") for res in detailed_completion_results
            ]
            expanded_df["response_headers"] = [
                res.get("response_headers") for res in detailed_completion_results
            ]
            expanded_df["response_body_raw"] = [
                res.get("response_body_raw") for res in detailed_completion_results
            ]
            expanded_df["adk_events_list"] = [
                res.get("adk_events_list") for res in detailed_completion_results
            ]
            expanded_df["completion_error_message"] = [
                res.get("error_message") for res in detailed_completion_results
            ]
        else:
            self.logger.error(
                f"Mismatch between detailed_completion_results ({len(detailed_completion_results)}) and rows ({len(expanded_df)}). Padding with error indicators."
            )
            num_missing = len(expanded_df) - len(detailed_completion_results)
            error_padding = [
                {
                    "generated_text": "[ERROR: Length Mismatch]",
                    "request_payload": None,
                    "response_status_code": None,
                    "response_headers": None,
                    "response_body_raw": None,
                    "adk_events_list": None,
                    "error_message": "Length Mismatch",
                }
            ] * num_missing
            padded_results = detailed_completion_results + error_padding
            expanded_df["generated_text_only"] = [
                res.get("generated_text") for res in padded_results
            ]
            expanded_df["request_payload"] = [
                res.get("request_payload") for res in padded_results
            ]
            expanded_df["response_status_code"] = [
                res.get("response_status_code") for res in padded_results
            ]
            expanded_df["response_headers"] = [
                res.get("response_headers") for res in padded_results
            ]
            expanded_df["response_body_raw"] = [
                res.get("response_body_raw") for res in padded_results
            ]
            expanded_df["adk_events_list"] = [
                res.get("adk_events_list") for res in padded_results
            ]
            expanded_df["completion_error_message"] = [
                res.get("error_message") for res in padded_results
            ]

        self.logger.info(
            f"Finished getting completions for {len(expanded_df)} total samples."
        )
        return expanded_df

    async def _execute_completion_request(
        self,
        goal: str,
        prefix: str,
        index: int,
        adk_session_id: Optional[str],
        adk_user_id: Optional[str],
    ) -> Dict:
        """Helper method to get completion via AgentRouter."""
        request_data: Dict[str, Any] = {"timeout": self.config.request_timeout}
        interaction_result: Dict[str, Any] = {}
        generated_text_specific = ""
        error_message_str = None

        try:
            if self.config.agent_type == AgentTypeEnum.GOOGLE_ADK:
                if not adk_session_id or not adk_user_id:
                    self.logger.error(
                        f"ADK agent type selected, but session_id or user_id is missing for index {index}."
                    )
                    raise ValueError(
                        "ADK session_id and user_id are required for ADK agent type."
                    )

                request_data.update(
                    {
                        "prompt_text": prefix,
                        "session_id": adk_session_id,
                        "user_id": adk_user_id,
                        # ADKAgentAdapter specific params if any, e.g., 'max_output_tokens'
                        # 'max_output_tokens': self.config.max_new_tokens # Example, ADKAdapter needs to support this
                    }
                )
                # self.logger.debug(f"ADK request for index {index}: {request_data}")

            elif self.config.agent_type == AgentTypeEnum.LITELMM:
                formatted_goal = goal
                if self.config.surrogate_attack_prompt:
                    try:
                        # Ensure prefix is lstripped for surrogate prompt to avoid leading spaces if any
                        formatted_goal += self.config.surrogate_attack_prompt.format(
                            prefix=prefix.lstrip()
                        )
                    except Exception as fmt_e:
                        self.logger.warning(
                            f"Failed to format surrogate prompt for goal at index {index}, using original goal. Error: {fmt_e}"
                        )

                messages = [
                    {"role": "user", "content": formatted_goal},
                    {
                        "role": "assistant",
                        "content": prefix,
                    },  # LiteLLM expects the prefix as an assistant message
                ]
                request_data.update(
                    {
                        "messages": messages,
                        "max_tokens": self.config.max_new_tokens,  # Standard LiteLLM param
                        "temperature": self.config.temperature,
                        "top_p": 1.0,  # Default, can be made configurable
                    }
                )
                # self.logger.debug(f"LiteLLM request for index {index} (first message content): {messages[0]['content'][:100]}...")
            else:
                raise NotImplementedError(
                    f"Agent type {self.config.agent_type} not supported by _execute_completion_request."
                )

            # Make the call through the AgentRouter
            # self.logger.info(f"Routing request for agent key {self.agent_registration_key} index {index}")
            adapter_response = await self.agent_router.route_request(
                registration_key=self.agent_registration_key, request_data=request_data
            )
            # self.logger.info(f"Adapter response for index {index}: {adapter_response}")

            # Process adapter_response
            # Expected keys from adapters (ADKAgentAdapter, LiteLLMAgentAdapter):
            # - 'generated_text': The core model output
            # - 'error_message': String if an error occurred, else None
            # - 'raw_request': The request payload sent to the actual agent
            # - 'raw_response_status': Status code from the agent HTTP call
            # - 'raw_response_headers': Headers from the agent HTTP call
            # - 'raw_response_body': Raw body from the agent HTTP call
            # - 'adapter_specific_events': e.g., ADK events list

            error_message_str = adapter_response.get("error_message")

            if error_message_str:
                self.logger.warning(
                    f"Adapter reported error for index {index}: {error_message_str}"
                )
                generated_text_specific = f"[ERROR: Adapter - {error_message_str}]"
            else:
                final_text_from_adapter = adapter_response.get("generated_text", "")
                if self.config.agent_type == AgentTypeEnum.GOOGLE_ADK:
                    # ADK adapter should ideally return the full text including prefix.
                    # If it returns only completion, this logic is fine. If it returns full, we strip.
                    # Assuming ADKAgentAdapter's 'generated_text' is the full text.
                    if final_text_from_adapter.startswith(prefix):
                        generated_text_specific = final_text_from_adapter[len(prefix) :]
                    else:
                        # This might happen if ADK output is unexpected or if adapter already stripped prefix
                        self.logger.warning(
                            f"ADK response for index {index} did not start with the prefix as expected. "
                            f"Prefix: '{prefix[:50]}...', Response: '{final_text_from_adapter[:100]}...'. "
                            f"Using full response or adapter's stripped version."
                        )
                        generated_text_specific = (
                            final_text_from_adapter  # Or some indicator of mismatch
                        )
                elif self.config.agent_type == AgentTypeEnum.LITELMM:
                    # LiteLLMAgentAdapter should directly return the completion part
                    generated_text_specific = final_text_from_adapter
                else:
                    generated_text_specific = final_text_from_adapter  # Fallback

            interaction_result = {
                "generated_text": generated_text_specific,
                "request_payload": adapter_response.get("raw_request"),
                "response_status_code": adapter_response.get("raw_response_status"),
                "response_headers": adapter_response.get("raw_response_headers"),
                "response_body_raw": adapter_response.get("raw_response_body"),
                "adk_events_list": (
                    adapter_response.get("adapter_specific_events")
                    if self.config.agent_type == AgentTypeEnum.GOOGLE_ADK
                    else None
                ),
                "error_message": error_message_str,
            }

        except Exception as e:
            self.logger.error(
                f"Error in _execute_completion_request for index {index} (Agent: {self.config.agent_name}): {e}",
                exc_info=True,
            )
            error_message_str = (
                f"Internal Completer Error: {type(e).__name__}: {str(e)}"
            )
            interaction_result = {
                "generated_text": f"[ERROR: {error_message_str}]",
                "request_payload": request_data,  # Log what we tried to send
                "response_status_code": None,
                "response_headers": None,
                "response_body_raw": None,
                "adk_events_list": None,
                "error_message": error_message_str,
            }

        return interaction_result

    # _get_adk_completion and _get_litellm_completion are now removed and replaced by _execute_completion_request
    # __del__ method removed as no explicit cleanup was being done that's still relevant.
