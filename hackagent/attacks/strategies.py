import logging
import abc
import json  # For ManagedAttackStrategy
import pandas as pd  # For AdvPrefix
import os  # Added for path joining
import httpx  # Added for manual HTTP call in AdvPrefix
from http import HTTPStatus  # Added for checking 201 status
from typing import Any, Optional, List, Dict, Tuple, TYPE_CHECKING
from uuid import UUID  # Added import

# Imports for specific strategies, moved from agent.py or direct_test_executor.py
from hackagent import errors  # Import the errors module
from hackagent.api.run import run_run_tests_create
from hackagent.api.attack.attack_create import (
    sync_detailed as attacks_create_sync_detailed,
)
from hackagent.models import Run
from hackagent.models.run_request import RunRequest
from hackagent.models.attack_request import (
    AttackRequest,
)  # For creating attacks via attacks_create API
from hackagent.errors import HackAgentError
from hackagent.attacks.advprefix import (
    AdvPrefixAttack,
)  # Used by LocalPrefix

if TYPE_CHECKING:
    from hackagent.agent import HackAgent

logger = logging.getLogger(__name__)

# --- Strategy Pattern for Attacks ---


class AttackStrategy(abc.ABC):
    """Abstract base class for an attack strategy."""

    def __init__(self, hack_agent: "HackAgent"):
        self.hack_agent = hack_agent
        self.client = hack_agent.client

    @abc.abstractmethod
    def execute(
        self,
        attack_config: Dict[str, Any],
        run_config_override: Optional[Dict[str, Any]],
        fail_on_run_error: bool,
        max_wait_time_seconds: Optional[int] = None,
        poll_interval_seconds: Optional[int] = None,
    ) -> Any:
        """Executes the attack strategy."""
        pass

    def _decode_response_content(self, response: httpx.Response) -> str:
        """
        Decodes the HTTP response content to a string.

        Args:
            response: The httpx.Response object.

        Returns:
            The decoded content as a string, or 'N/A' if content is None.
        """
        return (
            response.content.decode("utf-8", errors="replace")
            if response.content
            else "N/A"
        )

    def _parse_json_from_response_data(
        self,
        response: httpx.Response,
        decoded_content: str,
        attack_type_for_error_msg: str,
    ) -> Optional[Dict[str, Any]]:
        """
        Tries to parse JSON data from various parts of an httpx.Response.
        Handles direct content parsing and checks for pre-parsed attributes.

        Args:
            response: The httpx.Response object.
            decoded_content: The already decoded string content of the response.
            attack_type_for_error_msg: A string describing the attack type, for error messages.

        Returns:
            A dictionary if JSON parsing is successful, None otherwise.

        Raises:
            HackAgentError: If response status is 201 but JSON parsing fails critically.
        """
        parsed_data_dict: Optional[Dict[str, Any]] = None
        if response.content:
            try:
                parsed_data_dict = json.loads(decoded_content)
            except json.JSONDecodeError as jde:
                if (
                    response.status_code == 201
                ):  # Critical for 201 if body exists but is bad JSON
                    logger.error(
                        f"Failed to parse JSON for {attack_type_for_error_msg} (201 response with content): {jde}. Content: {decoded_content}"
                    )
                    raise HackAgentError(
                        f"Failed to parse 201 response JSON for {attack_type_for_error_msg} (content present): {jde}"
                    ) from jde
                logger.warning(
                    f"Could not parse JSON from response body for {attack_type_for_error_msg} (status {response.status_code}). Content: {decoded_content}",
                    exc_info=False,
                )  # exc_info=False to avoid verbose log for non-critical parse fail
                # Do not return None yet, try pre-parsed attributes next

        # Try pre-parsed attributes, especially if content parsing failed or content was empty
        if not parsed_data_dict and hasattr(response, "parsed") and response.parsed:
            logger.debug(
                f"Attempting to use pre-parsed attribute for {attack_type_for_error_msg}"
            )
            if hasattr(response.parsed, "additional_properties") and isinstance(
                response.parsed.additional_properties, dict
            ):
                parsed_data_dict = response.parsed.additional_properties
            elif isinstance(response.parsed, dict):
                parsed_data_dict = response.parsed
            else:
                logger.warning(
                    f"Response has 'parsed' attribute but it's not a usable dict for {attack_type_for_error_msg}. Type: {type(response.parsed)}"
                )

        return parsed_data_dict

    def _get_parsed_data_from_initiate_response(
        self,
        response: httpx.Response,
        decoded_content: str,
        attack_type_for_error_msg: str,
    ) -> Dict[str, Any]:
        """
        Handles different HTTP status codes to retrieve a parsed data dictionary
        from an attack initiation response.

        Args:
            response: The httpx.Response object.
            decoded_content: Decoded string content of the response.
            attack_type_for_error_msg: String describing attack type for errors.

        Returns:
            The parsed data dictionary.

        Raises:
            HackAgentError: If the response indicates failure or data cannot be parsed appropriately for critical statuses.
        """
        parsed_data_dict = self._parse_json_from_response_data(
            response, decoded_content, attack_type_for_error_msg
        )

        if response.status_code == 201:
            if not parsed_data_dict:
                # This case implies that _parse_json_from_response_data returned None for a 201, which means
                # either no content, or content that wasn't JSON, or pre-parsed attributes also failed.
                # If content was present but bad JSON, _parse_json_from_response_data would have raised.
                logger.error(
                    f"201 for {attack_type_for_error_msg} but no parsable dictionary body was found. Decoded content: '{decoded_content}', Pre-parsed type: {type(response.parsed if hasattr(response, 'parsed') else None)}"
                )
                raise HackAgentError(
                    f"201 for {attack_type_for_error_msg} but no parsable dictionary body was found."
                )

        elif response.status_code >= 300:
            err_text = f"Failed to initiate {attack_type_for_error_msg}. Status: {response.status_code}, Body: {decoded_content}"
            logger.error(err_text)
            raise HackAgentError(err_text)

        else:  # Unexpected success status codes (e.g., 200 OK instead of 201 Created, or other 2xx)
            logger.warning(
                f"Unexpected success status {response.status_code} from initiate_{attack_type_for_error_msg}. Content: {decoded_content}"
            )
            if (
                not parsed_data_dict
            ):  # If still no data after trying for an unexpected success status
                err_text = (
                    f"Could not obtain parsable data from initiate_{attack_type_for_error_msg} response with unexpected status {response.status_code}. "
                    f"Content: {decoded_content}"
                )
                logger.error(err_text)
                raise HackAgentError(err_text)

        if (
            not parsed_data_dict
        ):  # Should be caught by earlier checks, but as a final safeguard
            logger.error(
                f"Internal logic error: Parsed data dictionary is None for {attack_type_for_error_msg} status {response.status_code} without raising earlier. Content: {decoded_content}"
            )
            raise HackAgentError(
                f"Failed to obtain parsed data for {attack_type_for_error_msg} (status {response.status_code}). Check logs for parsing attempts."
            )
        return parsed_data_dict

    def _extract_ids_from_data_dict(
        self,
        parsed_data_dict: Dict[str, Any],
        attack_type_for_error_msg: str,
        original_content: str,
    ) -> Tuple[str, Optional[str]]:
        """
        Extracts 'id' (attack_id) and optionally 'associated_run_id' from a parsed data dictionary.
        Attack_id is considered mandatory.
        """
        raw_attack_id = parsed_data_dict.get("id")
        attack_id_str = str(raw_attack_id) if raw_attack_id is not None else None

        if attack_id_str is None:
            err_detail = (
                f"Could not extract mandatory attack_id ('{attack_id_str}') "
                f"from initiate_{attack_type_for_error_msg} response. "
                f"Source dict: {parsed_data_dict}, Original Decoded Content: '{original_content}'"
            )
            logger.error(err_detail)
            raise HackAgentError(err_detail)

        raw_run_id = parsed_data_dict.get("associated_run_id")
        run_id_str = str(raw_run_id) if raw_run_id is not None else None

        logger.info(
            f"Extracted Attack ID: {attack_id_str} and optional server-associated Run ID: {run_id_str if run_id_str else 'Not Provided'} for {attack_type_for_error_msg}."
        )
        return attack_id_str, run_id_str

    def extract_attack_and_run_ids_from_initiate_response(
        self, response: httpx.Response, attack_type_for_error_msg: str = "attack"
    ) -> Tuple[str, Optional[str]]:
        """Orchestrates the extraction of attack_id and optionally associated_run_id from an Attack creation response."""
        logger.debug(
            f"Attempting to extract Attack/Run IDs for '{attack_type_for_error_msg}' from response (status: {response.status_code})"
        )
        decoded_content = self._decode_response_content(response)
        parsed_data_dict = self._get_parsed_data_from_initiate_response(
            response, decoded_content, attack_type_for_error_msg
        )
        return self._extract_ids_from_data_dict(
            parsed_data_dict, attack_type_for_error_msg, decoded_content
        )


class AdvPrefix(AttackStrategy):
    """Strategy for 'advprefix' attacks."""

    def _prepare_and_validate_attack_params(
        self,
        attack_config: Dict[str, Any],
    ) -> List[Any]:
        """Validates and extracts necessary parameters from attack_config."""

        goals = attack_config.get("goals")
        if not isinstance(goals, list):
            raise ValueError(
                "'attack_config' must contain 'goals' list for AdvPrefixAttack."
            )

        return goals

    def _create_server_attack_record(
        self,
        victim_agent_id: UUID,
        organization_id: UUID,
        attack_config: Dict[str, Any],  # Used for summary
    ) -> str:
        """Creates the Attack record on the server and returns the attack_id."""
        logger.info("Creating Attack record on the server.")
        attack_type = "advprefix"

        payload = {
            "type": attack_type,
            "agent": str(victim_agent_id),  # Convert UUID to string
            "organization": str(organization_id),  # Convert UUID to string
            "configuration": attack_config,
        }
        try:
            attack_req_obj = AttackRequest.from_dict(payload)
            logger.debug(
                f"Attempting to create Attack record with payload: {attack_req_obj.to_dict()}"
            )
            response = attacks_create_sync_detailed(
                client=self.client, body=attack_req_obj
            )
        except Exception as e:
            logger.error(
                f"Failed to construct/send AttackRequest for {attack_type} record: {e}",
                exc_info=True,
            )
            raise HackAgentError(
                f"Failed to send AttackRequest for {attack_type} record: {e}"
            ) from e

        attack_id, _ = self.extract_attack_and_run_ids_from_initiate_response(
            response=response, attack_type_for_error_msg=attack_type
        )
        logger.info(f"Attack record created on server. Attack ID: {attack_id}.")
        return attack_id

    def _create_server_run_record(
        self,
        attack_id: str,
        victim_agent_id: str,
        run_config_override: Optional[Dict[str, Any]],
    ) -> str:
        """Explicitly creates a Run record on the server and returns the run_id."""
        logger.info(
            f"Attempting to explicitly create a Run record for Attack ID: {attack_id}"
        )
        payload = RunRequest(
            attack=attack_id,
            agent=victim_agent_id,
            run_config=run_config_override if run_config_override else {},
        )
        try:
            # response_obj is the custom hackagent.types.Response[Run]
            response_obj = run_run_tests_create.sync_detailed(
                client=self.client, body=payload
            )

            created_run: Optional[Run] = response_obj.parsed

            # If the auto-generated client didn't parse for 201, but it's a success, try manual parsing.
            if created_run is None and response_obj.status_code == HTTPStatus.CREATED:
                logger.info(
                    f"Run creation returned 201 (CREATED), attempting to manually parse response content for Attack ID: {attack_id}"
                )
                if response_obj.content:
                    try:
                        created_run_data = json.loads(
                            response_obj.content.decode("utf-8")
                        )
                        created_run = Run.from_dict(
                            created_run_data
                        )  # Use the Run model's from_dict
                        logger.info(
                            f"Manually parsed Run object from 201 response for Attack ID {attack_id}. Run ID: {created_run.id if created_run and hasattr(created_run, 'id') else 'Parse_Failed_Or_No_ID'}"
                        )
                    except json.JSONDecodeError as jde:
                        logger.error(
                            f"Failed to manually parse JSON from 201 response content for Attack ID {attack_id}: {jde}. Content: {response_obj.content.decode('utf-8', errors='replace')}",
                            exc_info=True,
                        )
                        # created_run remains None, will be caught by the check below
                    except Exception as e:
                        logger.error(
                            f"Unexpected error manually parsing 201 response content for Attack ID {attack_id}: {e}",
                            exc_info=True,
                        )
                        # created_run remains None, will be caught by the check below
                else:
                    logger.warning(
                        f"Run creation returned 201 (CREATED) but response content was empty for Attack ID: {attack_id}. Cannot manually parse."
                    )

            if not created_run or not hasattr(created_run, "id") or not created_run.id:
                status_code_val = (
                    response_obj.status_code
                    if hasattr(response_obj, "status_code")
                    else "Unknown Status"
                )
                content_val = (
                    response_obj.content.decode("utf-8", errors="replace")
                    if hasattr(response_obj, "content") and response_obj.content
                    else "No content"
                )

                logger.error(
                    f"Failed to get valid Run ID from run creation for Attack {attack_id}. "
                    f"Status: {status_code_val}, Parsed: {created_run}, Content: {content_val}"
                )
                raise HackAgentError(
                    f"Server API for Run creation returned status {status_code_val} "
                    f"but response parsing failed, lacked Run ID, or an error occurred. Content: {content_val}"
                )

            run_id = str(created_run.id)
            logger.info(
                f"Successfully created Run ID: {run_id} for Attack ID: {attack_id}"
            )
            return run_id

        except errors.UnexpectedStatus as use:
            # This is caught if client.raise_on_unexpected_status is True and server returns non-200
            error_content = (
                use.content.decode("utf-8", errors="replace")
                if use.content
                else "No content"
            )
            logger.error(
                f"API error (UnexpectedStatus {use.status_code}) creating Run for Attack {attack_id}: {error_content}",
                exc_info=True,
            )
            raise HackAgentError(
                f"Failed to create Run for Attack {attack_id} (API status {use.status_code}): {error_content}"
            ) from use
        except Exception as e:
            logger.error(
                f"Error creating Run for Attack {attack_id}: {e}", exc_info=True
            )
            raise HackAgentError(
                f"Failed to create Run for Attack {attack_id}: {e}"
            ) from e

    def _prepare_attack_config(
        self,
        attack_config: Dict[str, Any],
        run_id: str,
        attack_id: str,
    ) -> Dict[str, Any]:
        """Prepares the configuration for the local AdvPrefixAttack."""
        logger.debug(f"Preparing local attack config for Run ID: {run_id}")
        # Deep copy the user-provided attack_config to avoid modifying it directly.
        prepared_config = json.loads(json.dumps(attack_config))

        # Explicitly set/override 'run_id' with the server-generated run_id.
        # This 'run_id' will be used by AdvPrefixAttack to initialize its self.run_id.
        original_config_run_id = prepared_config.get("run_id")
        prepared_config["run_id"] = run_id
        if original_config_run_id and original_config_run_id != run_id:
            logger.info(
                f"Overriding 'run_id' in attack_config from '{original_config_run_id}' to server Run ID '{run_id}' for AdvPrefixAttack."
            )
        elif not original_config_run_id:
            logger.info(
                f"Set 'run_id' in attack_config to server Run ID '{run_id}' for AdvPrefixAttack."
            )

        # Update with other necessary parameters for AdvPrefixAttack
        prepared_config.update(
            {
                "hackagent_client": self.client,
                "agent_router": self.hack_agent.router,
                # "initial_run_id": run_id, # This is no longer needed as AdvPrefixAttack.run will use self.run_id
                "attack_id": attack_id,
            }
        )

        # Ensure 'output_dir' is present, defaulting if necessary.
        # AdvPrefixAttack uses this with its self.run_id to create self.run_dir.
        if "output_dir" not in prepared_config:
            # Defaulting output_dir based on attack_id if not provided.
            # Note: AdvPrefixAttack's __init__ also has a similar output_dir join with its self.run_id.
            # This path is more of a base for where AdvPrefixAttack will create its specific run_id subdir.
            prepared_config["output_dir"] = f"./logs/runs/{attack_id}"
            logger.warning(
                f"'output_dir' not in attack_config for AdvPrefixAttack, defaulting to {prepared_config['output_dir']}"
            )

        return prepared_config

    def _execute_local_prefix_attack(
        self,
        attack_config: Dict[str, Any],
        goals: List[Any],
        run_id: str,  # Server run_id
        attack_id: str,
    ) -> Optional[pd.DataFrame]:
        """Executes the AdvPrefixAttack locally."""
        logger.info(
            f"Executing local prefix attack for Attack ID {attack_id}, Server Run ID {run_id}."
        )
        try:
            # runner_config from _prepare_attack_config is a flat dictionary
            # containing pipeline params, client object, and router object.
            flat_prepared_config = self._prepare_attack_config(
                attack_config, run_id, attack_id
            )

            # Extract the client and router objects that AdvPrefixAttack expects as direct arguments.
            # The key for the client object in flat_prepared_config is "hackagent_client".
            adv_prefix_client = flat_prepared_config.pop("hackagent_client")
            adv_prefix_router = flat_prepared_config.pop("agent_router")

            # Remove other keys that are not part of AdvPrefixAttack's 'config' dictionary
            # or were passed for strategy-level logic but not for AdvPrefixAttack.__init__.
            flat_prepared_config.pop(
                "attack_type", None
            )  # Already handled if in original attack_config
            flat_prepared_config.pop(
                "goals", None
            )  # Already handled if in original attack_config

            # The remaining flat_prepared_config is now the dictionary
            # that AdvPrefixAttack expects for its 'config' parameter.
            # This dictionary includes user's settings, run_id, attack_id, output_dir etc.

            runner = AdvPrefixAttack(
                config=flat_prepared_config,
                client=adv_prefix_client,
                agent_router=adv_prefix_router,
            )

            # AdvPrefixAttack.run will use its self.run_id, which is initialized from runner_config["run_id"].
            results_df = runner.run(goals=goals)  # No longer pass initial_run_id
            logger.info(
                f"Local prefix attack completed for Attack ID {attack_id}, Server Run ID {run_id}."
            )
            return results_df
        except Exception as e:
            logger.error(
                f"Error during local prefix attack execution for Attack ID {attack_id}, Server Run ID {run_id}: {e}",
                exc_info=True,
            )
            return None  # Or re-raise if appropriate for the calling context

    def _log_local_run_persistence_info(
        self,
        attack_config: Dict[str, Any],
        attack_id: str,
        run_id: str,
        fail_on_run_error: bool,  # To decide if error during this info step is critical
    ):
        """Logs information about where local run data (like CSVs for Step10) would be."""
        # This method currently only logs. If actual operations were done, error handling would be more critical.
        try:
            base_output_dir = attack_config.get(
                "output_dir", f"./hackagent_local_runs/{attack_id}"
            )
            actual_run_output_dir = os.path.join(base_output_dir, f"run_{run_id}")
            input_csv_hint = attack_config.get(
                "input_csv_for_model_persistence", "step9_output.csv"
            )
            logger.info(
                f"Local run data (for potential Pydantic model persistence/Step10): Dir='{actual_run_output_dir}', CSV hint='{input_csv_hint}'."
            )
        except Exception as e:
            logger.error(
                f"Error preparing local run persistence info for Attack {attack_id}: {e}",
                exc_info=True,
            )
            if fail_on_run_error:
                # This is just logging info, so might not be fatal unless other operations depend on it.
                # For now, just log and continue, but could raise if this setup was critical.
                pass

    def execute(
        self,
        attack_config: Dict[str, Any],
        run_config_override: Optional[Dict[str, Any]],
        fail_on_run_error: bool,
    ) -> Any:
        """
        Executes the AdvPrefix attack.
        This involves:
        1. Creating an Attack record on the server.
        2. Creating a Run record on the server associated with the Attack.
        3. Executing the local AdvPrefix logic (e.g., notebook steps).
        4. Potentially updating the server Run/Attack with results or status.
        """
        victim_agent_id: UUID = self.hack_agent.router.backend_agent.id
        organization_id: UUID = self.hack_agent.router.organization_id

        if not victim_agent_id or not organization_id:
            raise HackAgentError(
                "Victim agent ID or Organization ID is not available. Ensure agent is initialized."
            )

        # 1. Create Attack record on the server
        attack_id = self._create_server_attack_record(
            victim_agent_id=victim_agent_id,
            organization_id=organization_id,
            attack_config=attack_config,  # Pass for summary or details
        )
        logger.info(f"AdvPrefix server Attack record created with ID: {attack_id}")

        # 2. Create Run record on the server
        run_id = self._create_server_run_record(
            attack_id=attack_id,
            victim_agent_id=victim_agent_id,
            run_config_override=run_config_override,
        )
        logger.info(
            f"AdvPrefix server Run record created with ID: {run_id} for Attack ID: {attack_id}"
        )

        # 3. Execute the local AdvPrefix attack logic
        goals = attack_config.get("goals")
        if not goals:
            raise ValueError("AdvPrefix attack requires 'goals' in attack_config.")

        # Assuming _execute_local_prefix_attack is now synchronous
        local_results_df = self._execute_local_prefix_attack(
            attack_config=attack_config, goals=goals, run_id=run_id, attack_id=attack_id
        )

        # 4. Log persistence info (which internally might update server records)
        # This step might be expanded to explicitly update server records if needed.
        self._log_local_run_persistence_info(
            attack_config, attack_id, run_id, fail_on_run_error
        )

        if local_results_df is None and fail_on_run_error:
            raise HackAgentError(
                f"AdvPrefix local execution failed for Attack ID {attack_id} and Run ID {run_id}."
            )

        logger.info(f"AdvPrefix attack execution completed for Attack ID {attack_id}.")
        # Return the DataFrame from the local execution as the primary result for now.
        # Future: Might return a more comprehensive result object or the server Run object.
        return local_results_df


# --- End Strategy Pattern ---
