import logging
from typing import Any, Dict, Type, Optional, Union
from uuid import UUID

from hackagent.router.base import Agent
from hackagent.router.adapters import ADKAgentAdapter
from hackagent.router.adapters.litellm_adapter import LiteLLMAgentAdapter
from hackagent.client import AuthenticatedClient
from hackagent.models import (
    AgentTypeEnum,
    Agent as BackendAgentModel,
    AgentRequest,
    PatchedAgentRequest,
    UserAPIKey,
)
from ..types import Unset, UNSET
from hackagent.api.agent import agent_list, agent_create, agent_partial_update
from hackagent.api.key import key_list

logger = logging.getLogger(__name__)

# --- Agent Type to Adapter Mapping ---
AGENT_TYPE_TO_ADAPTER_MAP: Dict[AgentTypeEnum, Type[Agent]] = {
    AgentTypeEnum.GOOGLE_ADK: ADKAgentAdapter,
    AgentTypeEnum.LITELMM: LiteLLMAgentAdapter,
    # AgentTypeEnum.OPENAI: OpenAIAgentAdapter, # Example for future
    # Add other agent types and their corresponding adapters here
}


class AgentRouter:
    """
    Manages a single agent's configuration and routes requests to its adapter.

    The router is initialized with the details of an agent, registers it with the
    backend (if not already present or if metadata needs an update), and instantiates
    the appropriate adapter. It then uses this adapter for request routing.
    """

    def _fetch_organization_id(self) -> UUID:
        """Fetches and returns the organization ID (UUID) associated with the API key.
        Raises RuntimeError if not found or if the organization attribute is not a UUID.
        """
        try:
            logger.debug(
                "AgentRouter: Attempting to retrieve Organization ID by listing API keys..."
            )
            keys_response = key_list.sync_detailed(client=self.client)

            if (
                keys_response.status_code == 200
                and keys_response.parsed
                and keys_response.parsed.results
            ):
                current_token = self.client.token
                key_results: list[UserAPIKey] = keys_response.parsed.results
                for key_obj in key_results:
                    if current_token.startswith(key_obj.prefix):
                        if hasattr(key_obj, "organization") and isinstance(
                            key_obj.organization, UUID
                        ):
                            logger.info(
                                f"AgentRouter: Successfully determined Organization ID: {key_obj.organization} from key prefix '{key_obj.prefix}'."
                            )
                            return key_obj.organization
                        else:
                            org_type = (
                                type(key_obj.organization).__name__
                                if hasattr(key_obj, "organization")
                                else "Missing"
                            )
                            logger.warning(
                                f"AgentRouter: Key prefix '{key_obj.prefix}' matched, but 'organization' is not UUID (type: {org_type}). Skipping for Org ID."
                            )
                logger.error(
                    f"AgentRouter: No API key found with a valid Organization (UUID) for token prefix '{current_token[:8]}...'."
                )
                raise RuntimeError(
                    "AgentRouter: Could not determine Organization ID (UUID) from API keys."
                )
            elif keys_response.parsed and not keys_response.parsed.results:
                logger.error(
                    "AgentRouter: API key list empty. Cannot find Organization ID."
                )
                raise RuntimeError(
                    "AgentRouter: API key list empty for Organization ID retrieval."
                )
            else:
                content = (
                    keys_response.content.decode() if keys_response.content else "N/A"
                )
                logger.error(
                    f"AgentRouter: Failed to list keys for Org ID. Status: {keys_response.status_code}, Body: {content}"
                )
                raise RuntimeError(
                    f"AgentRouter: API key list failed for Organization ID (status {keys_response.status_code})."
                )
        except RuntimeError:
            raise
        except Exception as e:
            logger.error(
                f"AgentRouter: Exception fetching Organization ID: {e}", exc_info=True
            )
            raise RuntimeError(f"AgentRouter: Exception fetching Organization ID: {e}")

    def _fetch_user_id_str(self) -> str:
        """Fetches and returns the user ID (as a string from UserAPIKey.user)
        associated with the API key. Raises RuntimeError if not found or user attribute is not an int.
        """
        try:
            logger.debug(
                "AgentRouter: Attempting to retrieve User ID by listing API keys..."
            )
            keys_response = key_list.sync_detailed(client=self.client)

            if (
                keys_response.status_code == 200
                and keys_response.parsed
                and keys_response.parsed.results
            ):
                current_token = self.client.token
                key_results: list[UserAPIKey] = keys_response.parsed.results
                for key_obj in key_results:
                    if current_token.startswith(key_obj.prefix):
                        if hasattr(key_obj, "user") and isinstance(key_obj.user, int):
                            user_id_as_str = str(key_obj.user)
                            logger.info(
                                f"AgentRouter: Successfully determined User ID (str): {user_id_as_str} from key prefix '{key_obj.prefix}'."
                            )
                            return user_id_as_str
                        else:
                            user_type = (
                                type(key_obj.user).__name__
                                if hasattr(key_obj, "user")
                                else "Missing"
                            )
                            logger.warning(
                                f"AgentRouter: Key prefix '{key_obj.prefix}' matched, but 'user' is not int (type: {user_type}). Skipping for User ID."
                            )
                logger.error(
                    f"AgentRouter: No API key found with a valid User (int) for token prefix '{current_token[:8]}...'."
                )
                raise RuntimeError(
                    "AgentRouter: Could not determine User ID (int) from API keys."
                )
            elif keys_response.parsed and not keys_response.parsed.results:
                logger.error("AgentRouter: API key list empty. Cannot find User ID.")
                raise RuntimeError(
                    "AgentRouter: API key list empty for User ID retrieval."
                )
            else:
                content = (
                    keys_response.content.decode() if keys_response.content else "N/A"
                )
                logger.error(
                    f"AgentRouter: Failed to list keys for User ID. Status: {keys_response.status_code}, Body: {content}"
                )
                raise RuntimeError(
                    f"AgentRouter: API key list failed for User ID (status {keys_response.status_code})."
                )
        except RuntimeError:
            raise
        except Exception as e:
            logger.error(f"AgentRouter: Exception fetching User ID: {e}", exc_info=True)
            raise RuntimeError(f"AgentRouter: Exception fetching User ID: {e}")

    def __init__(
        self,
        client: AuthenticatedClient,
        name: str,
        agent_type: AgentTypeEnum,
        endpoint: str,
        metadata: Optional[Dict[str, Any]] = None,
        adapter_operational_config: Optional[Dict[str, Any]] = None,
        overwrite_metadata: bool = True,  # Controls if backend agent metadata is updated if agent exists
    ):
        """
        Initializes the AgentRouter and registers a single agent.

        Ensures the specified agent exists in the backend (creating or updating as needed),
        then instantiates and stores its adapter in the router's registry.

        Args:
            client: Authenticated client for backend API interaction.
            name: Name for the agent in the backend.
            agent_type: The AgentTypeEnum for the agent (e.g., AgentTypeEnum.GOOGLE_ADK).
            endpoint: API endpoint URL for the agent service itself (used for backend registration
                      and potentially by the adapter if not overridden by backend_agent.endpoint).
            metadata: Metadata for the backend agent record.
                                  For ADK, adk_app_name is no longer explicitly managed here if it's same as agent name.
                                  For LiteLLM, SHOULD include {'name': 'model_name',
                                                             'endpoint': 'endpoint',
                                                             'api_key': 'optional_env_var_for_api_key', ...}
            adapter_operational_config: Runtime config for the adapter instance.
                                        Overrides or augments values from backend_agent.metadata.
                                        For ADK, may include {'user_id': ..., 'session_id': ...}.
                                        For LiteLLM, MUST provide 'name' (model string) if not in backend metadata.
            overwrite_metadata: If True, and an agent exists, its backend metadata is updated.

        Raises:
            ValueError: If agent_type is unsupported or adapter instantiation fails.
            RuntimeError: If backend communication or agent processing fails.
        """
        self.client = client
        self._agent_registry: Dict[str, Agent] = {}

        self.organization_id = self._fetch_organization_id()
        self.user_id_str = self._fetch_user_id_str()
        logger.info(
            f"AgentRouter initialized with Organization ID (UUID): {self.organization_id} and User ID (str): {self.user_id_str}"
        )

        if agent_type not in AGENT_TYPE_TO_ADAPTER_MAP:
            raise ValueError(
                f"Unsupported agent type: {agent_type}. "
                f"Supported types: {list(AGENT_TYPE_TO_ADAPTER_MAP.keys())}"
            )

        actual_metadata = metadata.copy() if metadata is not None else {}

        # adapter_operational_config is passed in, merge with any defaults we set here
        current_adapter_op_config = (
            adapter_operational_config.copy() if adapter_operational_config else {}
        )

        if agent_type == AgentTypeEnum.GOOGLE_ADK:
            # Ensure user_id is in the op_config for ADK, using the one fetched from API key
            if "user_id" not in current_adapter_op_config:
                current_adapter_op_config["user_id"] = self.user_id_str
                logger.info(
                    f"ADK Agent: Using fetched User ID '{self.user_id_str}' for adapter operational config."
                )
            else:
                logger.warning(
                    f"ADK Agent: 'user_id' was already present in adapter_operational_config ('{current_adapter_op_config['user_id']}'). Using that value instead of fetched one."
                )
            # session_id will be handled later, as it depends on run_id

        self.backend_agent = self.ensure_agent_in_backend(
            name=name,
            agent_type=agent_type,
            endpoint_for_backend=endpoint,
            metadata_for_backend=actual_metadata,
            update_metadata_if_exists=overwrite_metadata,
        )

        registration_key = str(self.backend_agent.id)

        self._configure_and_instantiate_adapter(
            name=name,
            agent_type=agent_type,
            registration_key=registration_key,
            adapter_operational_config=current_adapter_op_config,
        )

    def _configure_and_instantiate_adapter(
        self,
        name: str,
        agent_type: AgentTypeEnum,
        registration_key: str,
        adapter_operational_config: Optional[Dict[str, Any]],
    ) -> None:
        """
        Configures and instantiates the appropriate agent adapter based on agent_type
        and stores it in the router's registry.
        """
        adapter_class = AGENT_TYPE_TO_ADAPTER_MAP[
            agent_type
        ]  # agent_type already validated in __init__

        # Start with the operational config passed in
        adapter_instance_config = (
            adapter_operational_config.copy() if adapter_operational_config else {}
        )

        # Type-specific adapter configuration
        if agent_type == AgentTypeEnum.GOOGLE_ADK:
            adapter_instance_config["name"] = self.backend_agent.name
            adapter_instance_config["endpoint"] = self.backend_agent.endpoint
            if "user_id" not in adapter_instance_config:
                logger.error(
                    f"CRITICAL: user_id not found in adapter_instance_config for ADK agent '{self.backend_agent.name}' just before adapter instantiation. This should have been set in __init__."
                )
                # Fallback, though this indicates a logic flaw if reached.
                adapter_instance_config["user_id"] = self.user_id_str

        elif agent_type == AgentTypeEnum.LITELMM:
            if (
                "name" not in adapter_instance_config
            ):  # 'name' is the model string for LiteLLM
                if (
                    isinstance(self.backend_agent.metadata, dict)
                    and "name" in self.backend_agent.metadata
                ):
                    adapter_instance_config["name"] = self.backend_agent.metadata[
                        "name"
                    ]
                else:
                    raise ValueError(
                        f"LiteLLM agent '{name}' (ID: {registration_key}) missing "
                        f"'name' (model string) in adapter_operational_config or backend metadata. "
                        f"Cannot configure LiteLLMAgentAdapter."
                    )

            # Copy other relevant LiteLLM settings from backend_agent.metadata if not already in adapter_instance_config
            optional_litellm_keys = [
                "endpoint",
                "api_key",
                "max_new_tokens",
                "temperature",
                "top_p",
            ]
            if isinstance(self.backend_agent.metadata, dict):
                for key in optional_litellm_keys:
                    if (
                        key not in adapter_instance_config
                        and key in self.backend_agent.metadata
                    ):
                        adapter_instance_config[key] = self.backend_agent.metadata[key]

        # Instantiate and register the adapter
        try:
            adapter_instance = adapter_class(
                id=registration_key, config=adapter_instance_config
            )
            self._agent_registry[registration_key] = adapter_instance
            logger.info(
                f"Agent '{name}' (Backend ID: {registration_key}, Type: {agent_type.value}) "
                f"successfully initialized and registered with adapter {adapter_class.__name__}. "
                f"Adapter config keys: {list(adapter_instance_config.keys())}"  # Log keys for debug
            )
        except Exception as e:
            logger.error(
                f"Failed to instantiate adapter for agent '{name}' "
                f"(Backend ID: {registration_key}): {e}",
                exc_info=True,
            )
            raise ValueError(
                f"Failed to instantiate adapter {adapter_class.__name__}: {e}"
            ) from e

    def _find_existing_agent(
        self,
        name: str,
        agent_type: AgentTypeEnum,
    ) -> Optional[BackendAgentModel]:
        """
        Finds an existing agent by name, type, and organization in the backend.
        Uses self.organization_id (UUID) for matching.
        """
        logger.debug(
            f"SYNC_DEBUG: Entered _find_existing_agent for Name='{name}', Type='{agent_type.value}', OrgID='{self.organization_id}' (UUID)"
        )

        current_page: Union[Unset, int] = UNSET
        agents_processed_count = 0

        while True:
            list_response = None
            try:
                list_response = agent_list.sync_detailed(
                    client=self.client, page=current_page
                )
                logger.debug(
                    f"SYNC_DEBUG: Fetched page of agents. Status: {list_response.status_code if list_response else 'N/A'}"
                )

            except Exception as e:
                logger.error(
                    f"SYNC_DEBUG: An unexpected error occurred during 'agents_list.sync_detailed' while fetching page {current_page if not isinstance(current_page, Unset) else 'initial'}: {e}",
                    exc_info=True,
                )
                return None  # Or handle error more gracefully

            if (
                list_response
                and list_response.status_code == 200
                and list_response.parsed
            ):
                paginated_result = list_response.parsed
                results_list = getattr(paginated_result, "results", [])
                logger.debug(
                    f"SYNC_DEBUG: Page {current_page if not isinstance(current_page, Unset) else 'initial'} - Number of results on page: {len(results_list) if results_list else 0}"
                )
                if not isinstance(results_list, list):
                    logger.warning(
                        f"SYNC_DEBUG: Expected 'results' to be a list, but got {type(results_list)}. Full parsed response: {paginated_result}"
                    )
                    results_list = []

                for agent_model in results_list:
                    agents_processed_count += 1
                    logger.debug(
                        f"SYNC_DEBUG: Checking agent: ID={agent_model.id}, Name={getattr(agent_model, 'name', 'N/A')}, Type={getattr(agent_model, 'agent_type', getattr(agent_model, 'type', 'N/A'))}, Org={getattr(agent_model, 'organization', 'N/A')}"
                    )

                    name_matches = (
                        hasattr(agent_model, "name") and agent_model.name == name
                    )

                    org_matches = False
                    # agent_model.organization is UUID as per hackagent.models.Agent
                    if hasattr(agent_model, "organization") and isinstance(
                        agent_model.organization, UUID
                    ):
                        if agent_model.organization == self.organization_id:
                            org_matches = True
                        # else: # No need for else here, org_matches remains false
                        #    logger.debug(f"SYNC_DEBUG: OrgID (UUID) mismatch: agent_model.organization ('{agent_model.organization}') != expected self.organization_id ('{self.organization_id}') for agent '{agent_model.name}'")
                    # Check organization_detail.id as a fallback, though agent_model.organization should be primary
                    elif (
                        hasattr(agent_model, "organization_detail")
                        and hasattr(agent_model.organization_detail, "id")
                        and isinstance(agent_model.organization_detail.id, UUID)
                    ):
                        if agent_model.organization_detail.id == self.organization_id:
                            org_matches = True
                            logger.debug(
                                f"SYNC_DEBUG: Matched OrgID via organization_detail.id for agent '{agent_model.name}'"
                            )
                        # else:
                        #    logger.debug(f"SYNC_DEBUG: OrgID (UUID) mismatch via organization_detail.id: ('{agent_model.organization_detail.id}') != expected self.organization_id ('{self.organization_id}') for agent '{agent_model.name}'")
                    # The case where agent_model.organization is an int should not happen if model is correct, but good to log if it does.
                    elif hasattr(agent_model, "organization") and isinstance(
                        agent_model.organization, int
                    ):
                        logger.warning(
                            f"SYNC_DEBUG: agent_model.organization is an int ('{agent_model.organization}') for agent '{agent_model.name}'. Schema mismatch with expected UUID ('{self.organization_id}')."
                        )
                    # else: # Log if no organization attribute could be reliably checked
                    #    logger.debug(f"SYNC_DEBUG: Could not determine organization ID for comparison for agent '{agent_model.name}'. Expected UUID: {self.organization_id}")

                    type_matches = False
                    # The `agent_model` from the list might have `agent_type` (as per model def) or just `type`.
                    # The `type` attribute from `BackendAgentModel` (aliased as `Agent`) is `agent_type` in its definition.
                    # `AgentTypeEnum` is what `agent_type` (parameter) is.
                    current_agent_type_val = None
                    if (
                        hasattr(agent_model, "agent_type")
                        and agent_model.agent_type is not None
                        and not isinstance(agent_model.agent_type, Unset)
                    ):
                        if isinstance(agent_model.agent_type, AgentTypeEnum):
                            current_agent_type_val = agent_model.agent_type.value
                        elif isinstance(
                            agent_model.agent_type, str
                        ):  # If it's already a string
                            current_agent_type_val = agent_model.agent_type
                    elif (
                        hasattr(agent_model, "type") and agent_model.type is not None
                    ):  # Fallback for older/different field name
                        if isinstance(agent_model.type, AgentTypeEnum):
                            current_agent_type_val = agent_model.type.value
                        elif isinstance(agent_model.type, str):
                            current_agent_type_val = agent_model.type

                    if (
                        current_agent_type_val is not None
                        and current_agent_type_val == agent_type.value
                    ):
                        type_matches = True

                    if not name_matches:
                        logger.debug(
                            f"SYNC_DEBUG: Agent ID '{agent_model.id}' ('{agent_model.name}') failed name match (expected '{name}')"
                        )
                    elif not org_matches:
                        logger.debug(
                            f"SYNC_DEBUG: Agent ID '{agent_model.id}' ('{agent_model.name}') failed organization match (expected UUID '{self.organization_id}', found '{getattr(agent_model, 'organization', 'N/A')}' or detail '{getattr(agent_model.organization_detail, 'id', 'N/A') if hasattr(agent_model, 'organization_detail') else 'N/A'}')"
                        )
                    elif not type_matches:
                        logger.debug(
                            f"SYNC_DEBUG: Agent ID '{agent_model.id}' ('{agent_model.name}') failed type match (expected '{agent_type.value}', found '{current_agent_type_val}')"
                        )

                    if name_matches and org_matches and type_matches:
                        logger.info(
                            f"SYNC_DEBUG: Found existing backend agent '{name}' (Type: {agent_type.value}, OrgID: {self.organization_id}) "
                            f"with ID {agent_model.id} on page {current_page if not isinstance(current_page, Unset) else 'initial'}. Processed {agents_processed_count} agents total so far."
                        )
                        return agent_model

                if (
                    hasattr(paginated_result, "next_")
                    and paginated_result.next_
                    and not isinstance(paginated_result.next_, Unset)
                ):
                    next_page_url = paginated_result.next_
                    # Extract page number if it's a full URL. This is a bit simplistic.
                    # A more robust way would be to parse URL params if the API returns full URLs for next.
                    # If the API just returns the next page number, this is simpler.
                    # Assuming API might return simple page numbers or full URLs with ?page=NUMBER
                    try:
                        if isinstance(next_page_url, str) and "page=" in next_page_url:
                            current_page = int(
                                next_page_url.split("page=")[-1].split("&")[0]
                            )
                        elif isinstance(
                            next_page_url, int
                        ):  # If API directly gives next page number
                            current_page = next_page_url
                        else:  # Fallback for simple increment if only a URL string is given without obvious page number
                            current_page = (
                                current_page if isinstance(current_page, int) else 1
                            ) + 1
                    except ValueError:
                        logger.warning(
                            f"Could not parse next page number from URL: {next_page_url}. Using simple increment."
                        )
                        current_page = (
                            current_page if isinstance(current_page, int) else 1
                        ) + 1

                    logger.debug(
                        f"SYNC_DEBUG: Moving to next page of agents: {current_page}"
                    )
                else:
                    logger.debug(
                        f"SYNC_DEBUG: No more pages of agents to fetch. Processed {agents_processed_count} agents in total."
                    )
                    break

            elif list_response and list_response.status_code != 200:
                logger.error(
                    f"SYNC_DEBUG: Failed to list agents on page {current_page if not isinstance(current_page, Unset) else 'initial'}. Status: {list_response.status_code}, "
                    f"Body: {list_response.content.decode() if list_response.content else 'N/A'}"
                )
                return None
            elif not list_response:
                logger.error(
                    f"SYNC_DEBUG: Failed to get any response from agents_list API for page {current_page if not isinstance(current_page, Unset) else 'initial'}."
                )
                return None
            else:
                logger.warning(
                    f"SYNC_DEBUG: Unexpected state after trying to fetch page {current_page if not isinstance(current_page, Unset) else 'initial'}. list_response: {list_response}"
                )
                return None

        logger.debug(
            f"SYNC_DEBUG: No existing backend agent found matching Name='{name}', Type='{agent_type.value}', OrgID='{self.organization_id}' after searching all pages."
        )
        return None

    def _update_agent_metadata(
        self, agent_id: UUID, metadata_to_update: Dict[str, Any]
    ) -> BackendAgentModel:
        """Updates the metadata of an existing backend agent."""
        logger.info(f"Attempting to update metadata for backend agent ID: {agent_id}")
        patch_body = PatchedAgentRequest(metadata=metadata_to_update)
        try:
            update_response = agent_partial_update.sync_detailed(
                client=self.client, id=agent_id, body=patch_body
            )
            if update_response.status_code == 200 and update_response.parsed:
                logger.info(
                    f"Successfully updated metadata for backend agent ID: {agent_id}."
                )
                return update_response.parsed
            else:
                err_msg = (
                    f"Failed to update metadata for backend agent ID: {agent_id}. "
                    f"Status: {update_response.status_code}, "
                    f"Body: {update_response.content.decode() if update_response.content else 'N/A'}"
                )
                logger.error(err_msg)
                raise RuntimeError(err_msg)
        except Exception as e:
            err_msg_ex = f"Exception during backend agent metadata update for ID: {agent_id}: {e}"
            logger.error(err_msg_ex, exc_info=True)
            raise RuntimeError(err_msg_ex) from e

    def _create_new_agent(
        self,
        name: str,
        agent_type: AgentTypeEnum,
        endpoint: str,
        metadata: Dict[str, Any],
        description: str,
    ) -> BackendAgentModel:
        """Creates a new agent in the backend."""
        logger.info(
            f"Creating new backend agent: Name='{name}', Type='{agent_type.value}', OrgID='{self.organization_id}' (UUID)"
        )

        # IMPORTANT: AgentRequest.organization might expect an int or string representation of UUID.
        # If AgentRequest model expects an int, str(self.organization_id) or another conversion will be needed,
        # or the AgentRequest model itself needs to be updated to accept UUID.
        # For now, passing the UUID directly. This might require AgentRequest model adjustment.
        agent_req_body = AgentRequest(
            name=name,
            endpoint=endpoint,
            agent_type=agent_type,
            metadata=metadata,
            description=description,
            organization=self.organization_id,  # Passing UUID here.
        )

        try:
            create_response = agent_create.sync_detailed(
                client=self.client, body=agent_req_body
            )
            if create_response.status_code == 201 and create_response.parsed:
                logger.info(
                    f"Created backend agent '{name}' (Type: {agent_type.value}) "
                    f"with ID {create_response.parsed.id}."
                )
                return create_response.parsed
            else:
                body_content = (
                    create_response.content.decode()
                    if create_response.content
                    else "N/A"
                )
                err_msg = (
                    f"Failed to create backend agent '{name}'. Status: {create_response.status_code}, "
                    f"Body: {body_content}"
                )
                logger.error(err_msg)
                raise RuntimeError(err_msg)
        except Exception as e:
            err_msg_ex = (
                f"Exception during backend agent creation for Name='{name}': {e}"
            )
            logger.error(err_msg_ex, exc_info=True)
            raise RuntimeError(err_msg_ex) from e

    def ensure_agent_in_backend(
        self,
        name: str,
        agent_type: AgentTypeEnum,
        endpoint_for_backend: str,
        metadata_for_backend: Dict[str, Any],
        description_prefix: str = "Agent managed by router",
        update_metadata_if_exists: bool = True,
    ) -> BackendAgentModel:
        """
        Ensures an agent with the given specifications exists in the backend.
        Uses self.organization_id (UUID) from the router instance.
        """
        logger.info(
            f"Ensuring backend agent presence: Name='{name}', Type='{agent_type.value}', OrgID='{self.organization_id}' (UUID)"
        )

        existing_agent = self._find_existing_agent(name=name, agent_type=agent_type)

        if existing_agent:
            needs_metadata_update = False
            current_metadata = (
                existing_agent.metadata
                if isinstance(existing_agent.metadata, dict)
                else {}
            )
            metadata_to_patch = {}

            for key, value_for_backend in metadata_for_backend.items():
                if current_metadata.get(key) != value_for_backend:
                    metadata_to_patch[key] = value_for_backend
                    needs_metadata_update = True

            if needs_metadata_update and update_metadata_if_exists:
                logger.info(
                    f"Backend agent '{name}' exists and metadata needs update. Proceeding with update."
                )
                final_patch_payload = current_metadata.copy()
                final_patch_payload.update(metadata_to_patch)

                return self._update_agent_metadata(
                    agent_id=existing_agent.id, metadata_to_update=final_patch_payload
                )
            else:
                if needs_metadata_update and not update_metadata_if_exists:
                    logger.info(
                        f"Backend agent '{name}' exists and metadata differs, but update_metadata_if_exists is False. Skipping update."
                    )
                else:
                    logger.info(
                        f"Backend agent '{name}' exists and metadata is current or update is skipped."
                    )
                return existing_agent

        description = f"{description_prefix}: {name}"
        return self._create_new_agent(
            name=name,
            agent_type=agent_type,
            endpoint=endpoint_for_backend,
            metadata=metadata_for_backend,
            description=description,
        )

    def get_agent_instance(self, registration_key: str) -> Agent | None:
        """
        Retrieves an instantiated agent adapter from the router's registry.

        Args:
            registration_key: The backend agent's UUID string.

        Returns:
            An instance of the agent adapter, or None if not found.
        """
        instance = self._agent_registry.get(registration_key)
        if not instance:
            logger.warning(
                f"No agent adapter found in router registry for key: {registration_key}"
            )
        return instance

    async def route_request(
        self, registration_key: str, request_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Routes a request to the specified agent and returns its standardized response.

        Args:
            registration_key: The backend agent's UUID string.
            request_data: Data for the agent's handle_request method.

        Returns:
            Agent's response or an error dictionary.
        """
        logger.info(
            f"Routing request for agent with registration key: {registration_key}"
        )
        agent_instance = self.get_agent_instance(registration_key)

        if not agent_instance:
            logger.error(f"Could not find agent adapter for key: {registration_key}")
            return {
                "error": "AgentNotRegisteredInRouter",
                "message": f"Agent key '{registration_key}' not in router instances.",
                "registration_key": registration_key,
                "status_code": 404,
            }

        try:
            response = await agent_instance.handle_request(request_data)
            logger.info(
                f"Successfully processed request for agent key '{registration_key}'"
            )
            return response
        except Exception as e:
            logger.error(
                f"Error during request handling by adapter for agent key '{registration_key}': {e}",
                exc_info=True,
            )
            return {
                "error": "RequestHandlingError",
                "message": (
                    f"Adapter error for agent key '{registration_key}': {str(e)}"
                ),
                "registration_key": registration_key,
                "status_code": 500,
            }
