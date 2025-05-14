"""
Prefix generation pipeline attack based on the BaseAttack class.

This module implements a complete pipeline for generating, filtering, and selecting prefixes
using uncensored and target language models, adapted as an attack module.
"""

import os
import logging
import pandas as pd
from typing import List, Dict, Any, Optional
import copy
from uuid import UUID
import json

from hackagent.client import AuthenticatedClient  # Keep for type hinting
from hackagent.router.router import AgentRouter  # For type hinting agent_router
from .base import BaseAttack

# Import step execution functions
from .AdvPrefix import step1_generate
from .AdvPrefix import step4_compute_ce
from .AdvPrefix import step6_get_completions
from .AdvPrefix import step7_evaluate_responses
from .AdvPrefix import step8_aggregate_evaluations
from .AdvPrefix import step9_select_prefixes
from .AdvPrefix.preprocessing import PrefixPreprocessor, PreprocessConfig
from .AdvPrefix.utils import (
    execute_processor_step,
)  # New import from hackagent.utils

# Models and API clients for backend interaction
from hackagent.models import (
    ResultRequest,
    TraceRequest,
    PatchedRunRequest,  # Assuming this exists for PATCH /api/runs/{id}/
    PatchedResultRequest,  # Added for updating Result evaluation_status
    StatusEnum,
    StepTypeEnum,
    Result as BackendResult,  # Alias to avoid conflict
    EvaluationStatusEnum,  # Potentially for parent Result
)
from hackagent.types import UNSET
from hackagent.api.run import run_result_create
from hackagent.api.result import result_trace_create
from hackagent.api.result import result_partial_update  # Added for updating Result
from hackagent.api.run import run_partial_update
from hackagent.attacks.AdvPrefix.config import DEFAULT_PREFIX_GENERATION_CONFIG


# Helper function for deep merging dictionaries
def _recursive_update(target_dict, source_dict):
    """
    Recursively updates a target dictionary with values from a source dictionary.
    Nested dictionaries are merged; other values are overwritten with a deep copy.
    """
    for key, source_value in source_dict.items():
        target_value = target_dict.get(key)
        if isinstance(source_value, dict) and isinstance(target_value, dict):
            # If both current_value and update_value are dicts, recurse
            _recursive_update(target_value, source_value)
        else:
            # Otherwise, overwrite target_dict[key] with a deepcopy of source_value
            target_dict[key] = copy.deepcopy(source_value)


class AdvPrefixAttack(BaseAttack):
    """
    Attack class implementing the prefix generation pipeline by orchestrating step modules.

    Inherits from BaseAttack and adapts the multi-step prefix generation process.
    Expects configuration as a standard Python dictionary.
    """

    def __init__(
        self,
        config: Optional[Dict[str, Any]] = None,
        client: AuthenticatedClient = None,
        agent_router: AgentRouter = None,
    ):
        """
        Initialize the pipeline with configuration.

        Args:
            config: An optional dictionary containing pipeline parameters to override defaults.
            client: An AuthenticatedClient instance passed from the strategy.
            agent_router: An AgentRouter instance passed from the strategy.
        """
        if client is None:
            raise ValueError("AuthenticatedClient must be provided to AdvPrefixAttack.")
        if agent_router is None:
            raise ValueError(
                "Victim AgentRouter instance must be provided to AdvPrefixAttack."
            )
        self.client = client
        self.agent_router = agent_router

        # Start with a deep copy of the defaults to prevent any modification to the
        current_config = copy.deepcopy(DEFAULT_PREFIX_GENERATION_CONFIG)

        if config:  # config is the user-provided sparse dictionary of overrides
            _recursive_update(current_config, config)

        # --- Define run_id and run_dir BEFORE calling super().__init__() ---
        # Use config directly before it's potentially modified by BaseAttack
        self.run_id = current_config.get("run_id")
        output_dir = current_config.get("output_dir")
        if not output_dir:
            raise ValueError("Configuration missing required key: 'output_dir'")
        self.run_dir = os.path.join(output_dir, self.run_id)
        # Add run_id to config if it wasn't there, needed by BaseAttack perhaps
        current_config["run_id"] = self.run_id
        # --- Assign self.run_id here as well ---

        # ---------------------------------------

        # --- Get logger instance BEFORE calling super().__init__() ---
        self.logger = logging.getLogger(__name__)
        # ------------------------------------------------------------

        # Make a copy to avoid modifying the original dict if passed by reference
        base_config = current_config.copy()

        super().__init__(base_config)

        # Initialize components needed across steps (like Preprocessor)
        self.preprocessor = None
        try:
            # Extract relevant keys for PreprocessConfig, handling potential missing keys
            preprocess_cfg_keys = [
                # 'model_id', # Removed as no longer needed by Preprocessor for token counting
                "min_char_length",  # Changed from min_token_length
                "max_ce",
                "max_token_segments",
                "n_candidates_per_goal",
            ]
            preprocess_cfg_dict = {}
            for key in preprocess_cfg_keys:
                if key in self.config:
                    preprocess_cfg_dict[key] = self.config[key]
                # else: Log missing optional keys if needed

            # Create PreprocessConfig instance
            self.logger.info(
                f"Initializing Preprocessor with derived config: {preprocess_cfg_dict}"
            )
            preprocessor_config_obj = PreprocessConfig(**preprocess_cfg_dict)

            # Instantiate PrefixPreprocessor with the config object
            self.preprocessor = PrefixPreprocessor(config=preprocessor_config_obj)

            self.logger.info("Preprocessor initialized successfully.")

        except KeyError as ke:
            self.logger.error(
                f"Missing required key for PreprocessConfig: {ke}", exc_info=True
            )
        except ImportError:  # Catch import error specifically if still using try-except
            self.logger.error(
                "Failed to import AutoTokenizer, PreprocessConfig or PrefixPreprocessor. Steps requiring it will fail.",
                exc_info=True,
            )
        except Exception as e:
            self.logger.error(
                f"Failed to initialize PrefixPreprocessor: {e}", exc_info=True
            )
            # self.preprocessor remains None

        # _setup() is called by super().__init__()

    def _validate_config(self):
        """
        Validates the provided configuration dictionary.
        (Checks are now done on self.config which is a dict).
        """
        super()._validate_config()  # Base validation (checks if it's a dict)

        # Define required keys, noting that some steps might have optional dependencies
        # 'input_csv' removed as goals are passed to run()
        required_keys = [
            "output_dir",
            "start_step",
            # Keys needed for Preprocessor init
            "min_char_length",
            "max_token_segments",
            "n_candidates_per_goal",
            # Keys needed for Step 1
            "meta_prefixes",
            "meta_prefix_samples",
            "batch_size",
            "max_new_tokens",
            "guided_topk",
            "temperature",
            # Keys needed for Step 4
            "surrogate_attack_prompt",
            # Keys needed for Step 6
            "max_new_tokens_completion",
            "n_samples",
            # Keys needed for Step 7
            "judges",
            "batch_size_judge",
            "max_new_tokens_eval",
            "filter_len",
            # Keys needed for Step 9
            "pasr_weight",
            "n_prefixes_per_goal",
            "selection_judges",
            # Note: 'max_ce' is used optionally in Step 5 (via Preprocessor) and Step 8
        ]
        missing_keys = [k for k in required_keys if k not in self.config]
        if missing_keys:
            # Provide more context in the error message
            raise ValueError(
                f"Configuration dictionary missing required keys: {', '.join(missing_keys)}"
            )

        # Example type checks using .get()
        if not isinstance(self.config.get("meta_prefixes"), list):
            raise TypeError("Config key 'meta_prefixes' must be a list.")
        if not isinstance(self.config.get("judges"), list):
            raise TypeError("Config key 'judges' must be a list.")
        if not isinstance(self.config.get("selection_judges"), list):
            raise TypeError("Config key 'selection_judges' must be a list.")
        # Add more specific type/value checks as needed (e.g., check types within lists)

    def _setup(self):
        """
        Performs setup tasks like logging.
        (Preprocessor initialization moved to __init__).
        """
        self._setup_logging()
        self.logger.info(f"AdvPrefixAttack initialized with run ID: {self.run_id}")
        self.logger.info(f"Output directory: {self.run_dir}")
        # Log config (already a dict)
        # Avoid logging sensitive info if present in config (e.g., API keys)
        log_config = {
            k: v
            for k, v in self.config.items()
            if "token" not in k.lower()
            and "key" not in k.lower()
            and "secret" not in k.lower()
        }
        self.logger.info(f"Configuration (non-sensitive): {log_config}")

    def _setup_logging(self):
        """Configure logging to both file and console for this attack instance."""
        os.makedirs(self.run_dir, exist_ok=True)
        log_file = os.path.join(self.run_dir, "pipeline.log")

        # Use the instance logger obtained in __init__
        self.logger.propagate = (
            False  # Prevent duplicate logs if root logger is configured
        )
        self.logger.setLevel(logging.INFO)
        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )

        # Remove existing handlers if re-initializing (e.g., if run_id changes)
        for handler in self.logger.handlers[:]:
            self.logger.removeHandler(handler)
            handler.close()

        # File Handler
        try:
            fh = logging.FileHandler(log_file, mode="a")  # Append mode
            fh.setFormatter(formatter)
            self.logger.addHandler(fh)
        except Exception as e:
            print(
                f"Error setting up file handler for attack log: {e}"
            )  # Print error if logger fails

        # Console Handler - Check if one already exists to avoid duplicates in console
        if not any(isinstance(h, logging.StreamHandler) for h in self.logger.handlers):
            ch = logging.StreamHandler()
            ch.setFormatter(formatter)
            self.logger.addHandler(ch)

    # Remove helper methods that were moved to step files or utils
    # Methods like _get_checkpoint_path and _clear_gpu_memory are now in utils
    # Methods related to specific steps (_generate_prefixes, _construct_prompts, etc.) are in step files

    async def run(
        self, goals: List[str], initial_run_id: str | None = None
    ) -> pd.DataFrame:
        """
        Execute the complete prefix generation pipeline by calling step modules.

        Args:
            goals: A list of goal strings to generate prefixes for.
            initial_run_id: Optional run ID to use; otherwise, use the one from init or generate.

        Returns:
            A pandas DataFrame containing the final selected prefixes, or the result
            of the last successfully completed step if the pipeline stops early or fails.
        """
        parent_result_id: Optional[str] = (
            None  # Will store the ID of the main Result object for this run
        )

        # Override run_id if provided
        if initial_run_id and initial_run_id != self.run_id:
            self.logger.info(
                f"Overriding run ID from '{self.run_id}' to '{initial_run_id}'"
            )
            self.run_id = initial_run_id
            # Update run_dir based on the new run_id
            # Ensure config output_dir exists and is a string
            output_dir = self.config.get("output_dir")
            if not output_dir or not isinstance(output_dir, str):
                self.logger.error(
                    f"Invalid or missing 'output_dir' in config: {output_dir}. Cannot update run_dir."
                )
                # Handle error appropriately, e.g., raise or use a default, or stop
                # For now, we'll let it potentially fail later if run_dir is essential and not set
            else:
                self.run_dir = os.path.join(output_dir, f"run_{self.run_id}")
            self._setup_logging()  # Re-run logging setup with potentially new run_dir

        if not self.run_id:
            self.logger.error(
                "Run ID is not set. Cannot proceed with backend interaction."
            )
            # Fallback to original behavior without backend interaction if run_id is crucial and missing.
            # This part would need to be robustly handled based on application requirements.
            # For now, we proceed, and API calls will likely fail or be skipped.
            pass

        self.logger.info(
            f"Starting Prefix Generation Attack pipeline for Run ID {self.run_id} with {len(goals)} goals."
        )
        results_df = None  # Final results (output of step 9)
        last_step_output_df = pd.DataFrame()  # Holds output of the most recent step

        pipeline_failed = False
        final_step_reached = 0  # Track the last step attempted
        current_run_status = StatusEnum.RUNNING  # Initial status

        # Attempt to create a parent Result for this Run
        if self.run_id and run_result_create:
            try:
                self.logger.info(
                    f"Attempting to create parent Result for Run ID: {self.run_id}"
                )
                result_request_body = ResultRequest(
                    run=self.run_id,
                    prompt=None,  # No specific prompt for parent result
                    request_payload={},  # No request payload for parent
                    response_body="Parent result for prefix generation attack.",
                    evaluation_status=EvaluationStatusEnum.NOT_EVALUATED,
                )

                parent_result_response = await run_result_create.asyncio_detailed(
                    client=self.client,
                    id=UUID(self.run_id),  # This is the run_pk
                    body=result_request_body,
                )

                created_parent_result: Optional[BackendResult] = None
                successful_creation = False

                if 200 <= parent_result_response.status_code < 300:
                    if parent_result_response.parsed:
                        created_parent_result = parent_result_response.parsed
                        successful_creation = True
                    elif (
                        parent_result_response.status_code == 201
                        and parent_result_response.content
                    ):
                        try:
                            created_parent_result_data = json.loads(
                                parent_result_response.content.decode("utf-8")
                            )
                            created_parent_result = BackendResult.from_dict(
                                created_parent_result_data
                            )
                            successful_creation = True
                            self.logger.info(
                                f"Manually parsed parent Result from 201 response for Run ID {self.run_id}"
                            )
                        except Exception as e_parse:
                            self.logger.error(
                                f"Failed to manually parse parent Result content for Run ID {self.run_id} despite 201 status. Parse Error: {e_parse}, Body: {parent_result_response.content}",
                                exc_info=True,
                            )

                if not successful_creation or not created_parent_result:
                    self.logger.error(
                        f"Failed to create or parse parent Result for Run ID {self.run_id}. Status: {parent_result_response.status_code}, Parsed: {bool(parent_result_response.parsed)}, Body: {parent_result_response.content}"
                    )
                else:
                    if (
                        hasattr(created_parent_result, "id")
                        and created_parent_result.id is not None
                    ):
                        parent_result_id = str(created_parent_result.id)
                        self.logger.info(
                            f"Successfully created parent Result with ID: {parent_result_id} for Run ID {self.run_id}"
                        )
                    else:
                        self.logger.error(
                            f"Parent Result created/parsed for Run ID {self.run_id}, but ID is missing or None. Result Data: {created_parent_result}"
                        )

            except Exception as e:
                self.logger.error(
                    f"Error creating parent Result for Run ID {self.run_id}: {e}",
                    exc_info=True,
                )
        else:
            if not self.run_id:
                self.logger.warning(
                    "Run ID not available, skipping parent Result creation."
                )
            if not run_result_create:
                self.logger.warning(
                    "`run_result_create` API function not available, skipping parent Result creation."
                )

        try:
            start_step = self.config.get("start_step", 1)
            self.logger.info(f"Pipeline configured to start at step {start_step}.")

            # Step 1: Generate Prefixes
            if start_step <= 1:
                final_step_reached = 1
                self.logger.info("--- Running Step 1: Generate Prefixes ---")
                try:
                    unique_goals = list(dict.fromkeys(goals)) if goals else []
                    # Await the call to step1_generate.execute
                    last_step_output_df = await step1_generate.execute(
                        goals=unique_goals,
                        config=self.config,
                        logger=self.logger,
                        run_dir=self.run_dir,
                        client=self.client,
                    )
                    results_df = last_step_output_df
                    if last_step_output_df is None or last_step_output_df.empty:
                        self.logger.warning(
                            "Step 1 returned empty or None DataFrame. Stopping pipeline."
                        )
                        pipeline_failed = True
                        current_run_status = StatusEnum.FAILED
                        raise StopIteration("Step 1 failed or produced no output.")
                except Exception as e:
                    self.logger.error(f"Step 1 execution failed: {e}", exc_info=True)
                    pipeline_failed = True
                    current_run_status = StatusEnum.FAILED
                    raise StopIteration(f"Step 1 failed: {e}")
                finally:
                    if parent_result_id and result_trace_create:
                        try:
                            content_json = (
                                last_step_output_df.to_json(
                                    orient="records", default_handler=str
                                )
                                if last_step_output_df is not None
                                and not last_step_output_df.empty
                                else "{}"
                            )
                            trace_request_body = TraceRequest(
                                sequence=final_step_reached,
                                step_type=StepTypeEnum.OTHER,
                                content={
                                    "step_name": "Step 1: Generate Prefixes",
                                    "data_json": content_json,
                                    "status": (
                                        "Failed" if pipeline_failed else "Completed"
                                    ),
                                },
                            )
                            trace_response = await result_trace_create.asyncio_detailed(
                                client=self.client,
                                id=UUID(parent_result_id),
                                body=trace_request_body,
                            )
                            if not (200 <= trace_response.status_code < 300):
                                self.logger.error(
                                    f"Failed to create Trace for Result {parent_result_id}, Step {final_step_reached}. Status: {trace_response.status_code}, Body: {trace_response.content}"
                                )
                            else:
                                self.logger.info(
                                    f"Successfully created Trace for Result {parent_result_id}, Step {final_step_reached}."
                                )
                        except Exception as te:
                            self.logger.error(
                                f"Error creating Trace for Step 1: {te}", exc_info=True
                            )
                    elif not result_trace_create and parent_result_id:
                        self.logger.warning(
                            f"`result_trace_create` API function not available, skipping Trace creation for Step {final_step_reached}."
                        )

            # Step 2: Filter Phase 1
            if start_step <= 2 and not pipeline_failed:
                final_step_reached = 2
                self.logger.info("--- Running Step 2: Filter Phase 1 ---")
                if self.preprocessor is None:
                    self.logger.error(
                        "Preprocessor not initialized, cannot run Step 2."
                    )
                    pipeline_failed = True
                    current_run_status = StatusEnum.FAILED
                    raise StopIteration("Step 2 failed: Preprocessor missing.")
                # Assuming execute_processor_step is synchronous
                last_step_output_df = execute_processor_step(
                    input_df=last_step_output_df,
                    logger=self.logger,
                    run_dir=self.run_dir,
                    processor_instance=self.preprocessor,
                    processor_method_name="filter_phase1",
                    step_number=2,
                    step_name_for_logging="Initial prefix filtering (Phase 1)",
                    log_success_details_template="{count} prefixes remaining after phase 1 filtering.",
                )
                if last_step_output_df is None:
                    pipeline_failed = True
                    current_run_status = StatusEnum.FAILED
                    raise StopIteration("Step 2 failed critically (returned None).")
                if parent_result_id and result_trace_create:
                    try:
                        content_json = (
                            last_step_output_df.to_json(
                                orient="records", default_handler=str
                            )
                            if last_step_output_df is not None
                            and not last_step_output_df.empty
                            else "{}"
                        )
                        trace_request_body = TraceRequest(
                            sequence=final_step_reached,
                            step_type=StepTypeEnum.OTHER,
                            content={
                                "step_name": "Step 2: Filter Phase 1",
                                "data_json": content_json,
                                "status": "Completed",
                            },
                        )
                        trace_response = await result_trace_create.asyncio_detailed(
                            client=self.client,
                            id=UUID(parent_result_id),
                            body=trace_request_body,
                        )
                        if not (200 <= trace_response.status_code < 300):
                            self.logger.error(
                                f"Failed to create Trace for Result {parent_result_id}, Step {final_step_reached}. Status: {trace_response.status_code}, Body: {trace_response.content}"
                            )
                        else:
                            self.logger.info(
                                f"Successfully created Trace for Result {parent_result_id}, Step {final_step_reached}."
                            )
                    except Exception as te:
                        self.logger.error(
                            f"Error creating Trace for Step {final_step_reached}: {te}",
                            exc_info=True,
                        )
                elif not result_trace_create and parent_result_id:
                    self.logger.warning(
                        f"`result_trace_create` API not available, skipping Trace for Step {final_step_reached}."
                    )

            # Step 3: Ablate Prefixes
            if start_step <= 3 and not pipeline_failed:
                final_step_reached = 3
                self.logger.info("--- Running Step 3: Ablate Prefixes ---")
                if self.preprocessor is None:
                    self.logger.error(
                        "Preprocessor not initialized, cannot run Step 3."
                    )
                    pipeline_failed = True
                    current_run_status = StatusEnum.FAILED
                    raise StopIteration("Step 3 failed: Preprocessor missing.")
                # Assuming execute_processor_step is synchronous
                last_step_output_df = execute_processor_step(
                    input_df=last_step_output_df,
                    logger=self.logger,
                    run_dir=self.run_dir,
                    processor_instance=self.preprocessor,
                    processor_method_name="ablate",
                    step_number=3,
                    step_name_for_logging="Prefix ablation",
                    log_success_details_template="{count} ablated prefixes created.",
                )
                if last_step_output_df is None:
                    pipeline_failed = True
                    current_run_status = StatusEnum.FAILED
                    raise StopIteration("Step 3 failed critically (returned None).")
                if parent_result_id and result_trace_create:
                    try:
                        content_json = (
                            last_step_output_df.to_json(
                                orient="records", default_handler=str
                            )
                            if last_step_output_df is not None
                            and not last_step_output_df.empty
                            else "{}"
                        )
                        trace_request_body = TraceRequest(
                            sequence=final_step_reached,
                            step_type=StepTypeEnum.OTHER,
                            content={
                                "step_name": "Step 3: Ablate Prefixes",
                                "data_json": content_json,
                                "status": "Completed",
                            },
                        )
                        trace_response = await result_trace_create.asyncio_detailed(
                            client=self.client,
                            id=UUID(parent_result_id),
                            body=trace_request_body,
                        )
                        if not (200 <= trace_response.status_code < 300):
                            self.logger.error(
                                f"Failed to create Trace for Result {parent_result_id}, Step {final_step_reached}. Status: {trace_response.status_code}, Body: {trace_response.content}"
                            )
                        else:
                            self.logger.info(
                                f"Successfully created Trace for Result {parent_result_id}, Step {final_step_reached}."
                            )
                    except Exception as te:
                        self.logger.error(
                            f"Error creating Trace for Step {final_step_reached}: {te}",
                            exc_info=True,
                        )
                elif not result_trace_create and parent_result_id:
                    self.logger.warning(
                        f"`result_trace_create` API not available, skipping Trace for Step {final_step_reached}."
                    )

            # Step 4: Compute Cross-Entropy
            # Note: step4_compute_ce.execute itself was called with asyncio.run before.
            # If step4_compute_ce.execute is an async function, it should be awaited directly.
            # If it's synchronous but internally uses asyncio.run, that might need its own refactor.
            # For now, assuming its signature implies it can be awaited if it's async.
            # The original code was `asyncio.run(step4_compute_ce.execute(...))`.
            # This implies step4_compute_ce.execute is itself an async function.
            if start_step <= 4 and not pipeline_failed:
                final_step_reached = 4
                self.logger.info("--- Running Step 4: Compute Cross-Entropy ---")
                try:
                    # If step4_compute_ce.execute is async, it should be awaited.
                    last_step_output_df = await step4_compute_ce.execute(
                        input_df=last_step_output_df,
                        config=self.config,
                        logger=self.logger,
                        run_dir=self.run_dir,
                        client=self.client,  # client might be used by step4 for its own async calls
                        agent_router=self.agent_router,
                    )
                    results_df = last_step_output_df
                    if last_step_output_df is None:
                        pipeline_failed = True
                        current_run_status = StatusEnum.FAILED
                        raise StopIteration("Step 4 failed critically.")
                except Exception as e:
                    self.logger.error(f"Step 4 execution failed: {e}", exc_info=True)
                    pipeline_failed = True
                    current_run_status = StatusEnum.FAILED
                    raise StopIteration(f"Step 4 failed: {e}")
                finally:
                    if parent_result_id and result_trace_create:
                        try:
                            content_json = (
                                last_step_output_df.to_json(
                                    orient="records", default_handler=str
                                )
                                if last_step_output_df is not None
                                and not last_step_output_df.empty
                                else "{}"
                            )
                            trace_request_body = TraceRequest(
                                sequence=final_step_reached,
                                step_type=StepTypeEnum.OTHER,
                                content={
                                    "step_name": "Step 4: Compute Cross-Entropy",
                                    "data_json": content_json,
                                    "status": (
                                        "Failed"
                                        if pipeline_failed and start_step <= 4
                                        else "Completed"
                                    ),
                                },
                            )
                            trace_response = await result_trace_create.asyncio_detailed(
                                client=self.client,
                                id=UUID(parent_result_id),
                                body=trace_request_body,
                            )
                            if not (200 <= trace_response.status_code < 300):
                                self.logger.error(
                                    f"Failed to create Trace for Result {parent_result_id}, Step {final_step_reached}. Status: {trace_response.status_code}, Body: {trace_response.content}"
                                )
                            else:
                                self.logger.info(
                                    f"Successfully created Trace for Result {parent_result_id}, Step {final_step_reached}."
                                )
                        except Exception as te:
                            self.logger.error(
                                f"Error creating Trace for Step {final_step_reached}: {te}",
                                exc_info=True,
                            )
                    elif not result_trace_create and parent_result_id:
                        self.logger.warning(
                            f"`result_trace_create` API not available, skipping Trace for Step {final_step_reached}."
                        )

            # Step 5: Filter Phase 2 (CE-based)
            if start_step <= 5 and not pipeline_failed:
                final_step_reached = 5
                self.logger.info("--- Running Step 5: Filter Phase 2 (CE-based) ---")
                if self.preprocessor is None:
                    self.logger.error(
                        "Preprocessor not initialized, cannot run Step 5."
                    )
                    pipeline_failed = True
                    current_run_status = StatusEnum.FAILED
                    raise StopIteration("Step 5 failed: Preprocessor missing.")
                # Assuming execute_processor_step is synchronous
                last_step_output_df = execute_processor_step(
                    input_df=last_step_output_df,
                    logger=self.logger,
                    run_dir=self.run_dir,
                    processor_instance=self.preprocessor,
                    processor_method_name="filter_phase2",
                    step_number=5,
                    step_name_for_logging="CE-based filtering (Phase 2)",
                    log_success_details_template="{count} prefixes remaining after phase 2 filtering.",
                )
                if last_step_output_df is None:
                    pipeline_failed = True
                    current_run_status = StatusEnum.FAILED
                    raise StopIteration("Step 5 failed critically (returned None).")
                if parent_result_id and result_trace_create:
                    try:
                        content_json = (
                            last_step_output_df.to_json(
                                orient="records", default_handler=str
                            )
                            if last_step_output_df is not None
                            and not last_step_output_df.empty
                            else "{}"
                        )
                        trace_request_body = TraceRequest(
                            sequence=final_step_reached,
                            step_type=StepTypeEnum.OTHER,
                            content={
                                "step_name": "Step 5: Filter Phase 2 (CE-based)",
                                "data_json": content_json,
                                "status": "Completed",
                            },
                        )
                        trace_response = await result_trace_create.asyncio_detailed(
                            client=self.client,
                            id=UUID(parent_result_id),
                            body=trace_request_body,
                        )
                        if not (200 <= trace_response.status_code < 300):
                            self.logger.error(
                                f"Failed to create Trace for Result {parent_result_id}, Step {final_step_reached}. Status: {trace_response.status_code}, Body: {trace_response.content}"
                            )
                        else:
                            self.logger.info(
                                f"Successfully created Trace for Result {parent_result_id}, Step {final_step_reached}."
                            )
                    except Exception as te:
                        self.logger.error(
                            f"Error creating Trace for Step {final_step_reached}: {te}",
                            exc_info=True,
                        )
                elif not result_trace_create and parent_result_id:
                    self.logger.warning(
                        f"`result_trace_create` API not available, skipping Trace for Step {final_step_reached}."
                    )

            # Step 6: Get Completions
            # Assuming step6_get_completions.execute is synchronous. If it becomes async, needs await.
            if start_step <= 6 and not pipeline_failed:
                final_step_reached = 6
                self.logger.info("--- Running Step 6: Get Completions ---")
                # Await the call to step6_get_completions.execute
                last_step_output_df = await step6_get_completions.execute(
                    agent_router=self.agent_router,
                    input_df=last_step_output_df,
                    config=self.config,
                    logger=self.logger,
                    run_dir=self.run_dir,
                )
                if last_step_output_df is None:
                    pipeline_failed = True
                    current_run_status = StatusEnum.FAILED
                    raise StopIteration("Step 6 failed critically.")
                if parent_result_id and result_trace_create:
                    try:
                        content_json = (
                            last_step_output_df.to_json(
                                orient="records", default_handler=str
                            )
                            if last_step_output_df is not None
                            and not last_step_output_df.empty
                            else "{}"
                        )
                        trace_request_body = TraceRequest(
                            sequence=final_step_reached,
                            step_type=StepTypeEnum.OTHER,
                            content={
                                "step_name": "Step 6: Get Completions",
                                "data_json": content_json,
                                "status": "Completed",
                            },
                        )
                        trace_response = await result_trace_create.asyncio_detailed(
                            client=self.client,
                            id=UUID(parent_result_id),
                            body=trace_request_body,
                        )
                        if not (200 <= trace_response.status_code < 300):
                            self.logger.error(
                                f"Failed to create Trace for Result {parent_result_id}, Step {final_step_reached}. Status: {trace_response.status_code}, Body: {trace_response.content}"
                            )
                        else:
                            self.logger.info(
                                f"Successfully created Trace for Result {parent_result_id}, Step {final_step_reached}."
                            )
                    except Exception as te:
                        self.logger.error(
                            f"Error creating Trace for Step {final_step_reached}: {te}",
                            exc_info=True,
                        )
                elif not result_trace_create and parent_result_id:
                    self.logger.warning(
                        f"`result_trace_create` API not available, skipping Trace for Step {final_step_reached}."
                    )

            # Step 7: Evaluate Responses
            # Assuming step7_evaluate_responses.execute is synchronous
            if start_step <= 7 and not pipeline_failed:
                final_step_reached = 7
                self.logger.info("--- Running Step 7: Evaluate Responses ---")
                last_step_output_df = step7_evaluate_responses.execute(
                    input_df=last_step_output_df,
                    config=self.config,
                    logger=self.logger,
                    run_dir=self.run_dir,
                )
                if last_step_output_df is None:
                    pipeline_failed = True
                    current_run_status = StatusEnum.FAILED
                    raise StopIteration("Step 7 failed critically.")
                if parent_result_id and result_trace_create:
                    try:
                        content_json = (
                            last_step_output_df.to_json(
                                orient="records", default_handler=str
                            )
                            if last_step_output_df is not None
                            and not last_step_output_df.empty
                            else "{}"
                        )
                        trace_request_body = TraceRequest(
                            sequence=final_step_reached,
                            step_type=StepTypeEnum.OTHER,
                            content={
                                "step_name": "Step 7: Evaluate Responses",
                                "data_json": content_json,
                                "status": "Completed",
                            },
                        )
                        trace_response = await result_trace_create.asyncio_detailed(
                            client=self.client,
                            id=UUID(parent_result_id),
                            body=trace_request_body,
                        )
                        if not (200 <= trace_response.status_code < 300):
                            self.logger.error(
                                f"Failed to create Trace for Result {parent_result_id}, Step {final_step_reached}. Status: {trace_response.status_code}, Body: {trace_response.content}"
                            )
                        else:
                            self.logger.info(
                                f"Successfully created Trace for Result {parent_result_id}, Step {final_step_reached}."
                            )
                    except Exception as te:
                        self.logger.error(
                            f"Error creating Trace for Step {final_step_reached}: {te}",
                            exc_info=True,
                        )
                elif not result_trace_create and parent_result_id:
                    self.logger.warning(
                        f"`result_trace_create` API not available, skipping Trace for Step {final_step_reached}."
                    )

            # Step 8: Aggregate Evaluations
            if start_step <= 8 and not pipeline_failed:
                final_step_reached = 8
                self.logger.info("--- Running Step 8: Aggregate Evaluations ---")
                last_step_output_df = step8_aggregate_evaluations.execute(
                    input_df=last_step_output_df,
                    config=self.config,
                    run_dir=self.run_dir,
                )
                if last_step_output_df is None:
                    pipeline_failed = True
                    current_run_status = StatusEnum.FAILED
                    raise StopIteration("Step 8 failed critically.")
                if parent_result_id and result_trace_create:
                    try:
                        content_json = (
                            last_step_output_df.to_json(
                                orient="records", default_handler=str
                            )
                            if last_step_output_df is not None
                            and not last_step_output_df.empty
                            else "{}"
                        )
                        trace_request_body = TraceRequest(
                            sequence=final_step_reached,
                            step_type=StepTypeEnum.OTHER,
                            content={
                                "step_name": "Step 8: Aggregate Evaluations",
                                "data_json": content_json,
                                "status": "Completed",
                            },
                        )
                        trace_response = await result_trace_create.asyncio_detailed(
                            client=self.client,
                            id=UUID(parent_result_id),
                            body=trace_request_body,
                        )
                        if not (200 <= trace_response.status_code < 300):
                            self.logger.error(
                                f"Failed to create Trace for Result {parent_result_id}, Step {final_step_reached}. Status: {trace_response.status_code}, Body: {trace_response.content}"
                            )
                        else:
                            self.logger.info(
                                f"Successfully created Trace for Result {parent_result_id}, Step {final_step_reached}."
                            )
                    except Exception as te:
                        self.logger.error(
                            f"Error creating Trace for Step {final_step_reached}: {te}",
                            exc_info=True,
                        )
                elif not result_trace_create and parent_result_id:
                    self.logger.warning(
                        f"`result_trace_create` API not available, skipping Trace for Step {final_step_reached}."
                    )

            # Step 9: Select Prefixes
            if start_step <= 9 and not pipeline_failed:
                final_step_reached = 9
                self.logger.info("--- Running Step 9: Select Prefixes ---")
                results_df = step9_select_prefixes.execute(
                    input_df=last_step_output_df,
                    config=self.config,
                    run_dir=self.run_dir,
                )
                if results_df is None:
                    pipeline_failed = True
                    current_run_status = StatusEnum.FAILED
                    raise StopIteration("Step 9 failed critically.")
                last_step_output_df = results_df
                if parent_result_id and result_trace_create:
                    try:
                        content_json = (
                            results_df.to_json(orient="records", default_handler=str)
                            if results_df is not None and not results_df.empty
                            else "{}"
                        )
                        trace_request_body = TraceRequest(
                            sequence=final_step_reached,
                            step_type=StepTypeEnum.OTHER,
                            content={
                                "step_name": "Step 9: Select Prefixes",
                                "data_json": content_json,
                                "status": "Completed",
                            },
                        )
                        trace_response = await result_trace_create.asyncio_detailed(
                            client=self.client,
                            id=UUID(parent_result_id),
                            body=trace_request_body,
                        )
                        if not (200 <= trace_response.status_code < 300):
                            self.logger.error(
                                f"Failed to create Trace for Result {parent_result_id}, Step {final_step_reached}. Status: {trace_response.status_code}, Body: {trace_response.content}"
                            )
                        else:
                            self.logger.info(
                                f"Successfully created Trace for Result {parent_result_id}, Step {final_step_reached}."
                            )
                    except Exception as te:
                        self.logger.error(
                            f"Error creating Trace for Step {final_step_reached}: {te}",
                            exc_info=True,
                        )
                elif not result_trace_create and parent_result_id:
                    self.logger.warning(
                        f"`result_trace_create` API not available, skipping Trace for Step {final_step_reached}."
                    )

            if pipeline_failed:
                self.logger.error(
                    f"Pipeline marked as failed after step {final_step_reached}."
                )
                current_run_status = StatusEnum.FAILED
            elif final_step_reached == 0:
                self.logger.warning(
                    "Pipeline did not execute any steps based on start_step config."
                )
                current_run_status = StatusEnum.COMPLETED
            elif results_df is not None:
                self.logger.info(
                    "Prefix Generation Attack pipeline finished successfully at Step 9."
                )
                current_run_status = StatusEnum.COMPLETED
                return results_df
            else:
                self.logger.warning(
                    f"Pipeline finished after step {final_step_reached}. Returning intermediate results."
                )
                current_run_status = StatusEnum.COMPLETED

            return (
                last_step_output_df
                if last_step_output_df is not None
                else pd.DataFrame()
            )

        except StopIteration as stop_e:
            self.logger.error(f"Pipeline execution stopped: {stop_e}")
            current_run_status = StatusEnum.FAILED
        except Exception as e:
            self.logger.error(
                f"Pipeline orchestration failed unexpectedly: {str(e)}", exc_info=True
            )
            pipeline_failed = True
            current_run_status = StatusEnum.FAILED

        if self.run_id and run_partial_update:
            try:
                self.logger.info(
                    f"Attempting to update Run {self.run_id} status to {current_run_status.value}"
                )
                patched_run_body = PatchedRunRequest(
                    status=current_run_status,
                    run_notes=UNSET,
                    run_config=UNSET,
                    agent=UNSET,
                    attack=UNSET,
                )
                update_response = await run_partial_update.asyncio_detailed(
                    client=self.client, id=UUID(self.run_id), body=patched_run_body
                )
                if not (200 <= update_response.status_code < 300):
                    self.logger.error(
                        f"Failed to update Run {self.run_id} status. Status: {update_response.status_code}, Body: {update_response.content}"
                    )
                else:
                    self.logger.info(
                        f"Successfully updated Run {self.run_id} status to {current_run_status.value}"
                    )
            except Exception as e:
                self.logger.error(
                    f"Error updating Run {self.run_id} status: {e}", exc_info=True
                )
        else:
            if not self.run_id:
                self.logger.warning(
                    "Run ID not available, skipping final Run status update."
                )
            if not run_partial_update:
                self.logger.warning(
                    "`run_partial_update` API function not available, skipping final Run status update."
                )

        # Update the parent Result's evaluation_status
        if parent_result_id and result_partial_update:
            try:
                final_eval_status = (
                    EvaluationStatusEnum.SUCCESSFUL_JAILBREAK
                    if not pipeline_failed
                    and final_step_reached >= self.config.get("end_step", 9)
                    else EvaluationStatusEnum.ERROR_TEST_FRAMEWORK
                )
                # If pipeline_failed was true due to an exception, ERROR_TEST_FRAMEWORK is appropriate.

                self.logger.info(
                    f"Attempting to update parent Result ID {parent_result_id} to evaluation_status: {final_eval_status.value}"
                )

                # Assuming PatchedResultRequest is the correct model and takes evaluation_status
                patched_result_request_body = PatchedResultRequest(
                    evaluation_status=final_eval_status
                )

                result_update_response = await result_partial_update.asyncio_detailed(
                    client=self.client,
                    id=UUID(parent_result_id),  # The ID of the Result to update
                    body=patched_result_request_body,
                )

                if 200 <= result_update_response.status_code < 300:
                    self.logger.info(
                        f"Successfully updated parent Result ID {parent_result_id} evaluation_status to {final_eval_status.value}."
                    )
                else:
                    self.logger.error(
                        f"Failed to update parent Result ID {parent_result_id} evaluation_status. Server responded with {result_update_response.status_code}. Body: {result_update_response.content}"
                    )
            except Exception as e_result_update:
                self.logger.error(
                    f"Error updating evaluation_status for parent Result ID {parent_result_id}: {e_result_update}",
                    exc_info=True,
                )
        elif not parent_result_id:
            self.logger.warning(
                "Parent Result ID not available, skipping evaluation_status update for parent Result."
            )
        elif not result_partial_update:
            self.logger.warning(
                "`result_partial_update` API not available, skipping evaluation_status update for parent Result."
            )

        if pipeline_failed:
            self.logger.warning(
                f"Returning output from last successful step ({final_step_reached}) due to failure."
            )
        elif final_step_reached < start_step and start_step > 1:
            self.logger.warning(
                f"Pipeline did not run any steps (start_step={start_step}). Returning empty DataFrame."
            )
            return pd.DataFrame()

        return (
            last_step_output_df if last_step_output_df is not None else pd.DataFrame()
        )
