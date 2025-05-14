import os
import pandas as pd
import logging
import litellm
from typing import List, Dict, Optional, Tuple, Any

# Imports needed for execute_processor_step
from typing import Callable  # Keep for execute_processor_step


logger = logging.getLogger(__name__)  # Use a logger specific to utils


def get_checkpoint_path(run_dir: str, step_num: int) -> str:
    """Generate the standard path for a step's output checkpoint."""
    filename = f"step{step_num}_output.csv"
    return os.path.join(run_dir, filename)


def call_litellm_completion(
    model_id: str,
    messages: List[Dict[str, str]],
    endpoint: Optional[str],
    api_key: Optional[str],
    timeout: int,
    temperature: float,
    max_tokens: int,
    logprobs: bool = False,
    top_p: float = 1.0,
    logger: Optional[logging.Logger] = None,
) -> Tuple[Optional[str], Optional[Any], Optional[Exception]]:
    """
    Wrapper function to call litellm.completion and handle common exceptions.

    Args:
        model_id: The model identifier string for LiteLLM.
        messages: The list of messages for the chat completion.
        endpoint: The API base URL.
        api_key: The API key.
        timeout: Request timeout in seconds.
        temperature: Sampling temperature.
        max_tokens: Maximum new tokens to generate.
        logprobs: Whether to request logprobs (default: False).
        top_p: Top-p sampling parameter (default: 1.0).
        logger: Optional logger instance.

    Returns:
        A tuple: (content_string, logprobs_object, error_object).
        - If successful: returns (content, logprobs_data, None). content and logprobs_data may be None depending on the API response.
        - If error occurs: returns (None, None, error_object).
    """
    litellm.drop_params = True  # Drop unsupported parameters

    if logger is None:
        logger = logging.getLogger(__name__)  # Fallback logger

    content_string: Optional[str] = None
    logprobs_data: Optional[Any] = None
    error_object: Optional[Exception] = None

    try:
        litellm_kwargs = {
            "model": model_id,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "top_p": top_p,
            "endpoint": endpoint,
            "api_key": api_key,
            "timeout": timeout,
            "logprobs": logprobs,
        }
        # Filter out None values, especially api_key and endpoint if not provided
        litellm_kwargs = {k: v for k, v in litellm_kwargs.items() if v is not None}

        logger.debug(
            f"Calling litellm.completion with kwargs: { {k: v for k, v in litellm_kwargs.items() if k != 'api_key'} }"
        )  # Don't log key
        response = litellm.completion(**litellm_kwargs)

        # Extract content
        if (
            response
            and response.choices
            and response.choices[0].message
            and response.choices[0].message.content
        ):
            content_string = response.choices[0].message.content
        else:
            logger.warning(
                f"LiteLLM call successful but no content found in response structure. Model: {model_id}"
            )

        # Extract logprobs if requested and available
        if logprobs:
            if response and response.choices and response.choices[0].logprobs:
                logprobs_data = response.choices[0].logprobs
            else:
                logger.warning(
                    f"Logprobs were requested but not found in response structure. Model: {model_id}"
                )

    except litellm.exceptions.BadRequestError as llm_bad_req_err:
        logger.error(f"LiteLLM Bad Request Error: {llm_bad_req_err}")
        error_object = llm_bad_req_err
    except litellm.exceptions.APIError as llm_api_err:
        logger.error(f"LiteLLM API Error: {llm_api_err}")
        error_object = llm_api_err
    except Exception as e:
        logger.error(f"Generic error during LiteLLM call: {e}", exc_info=True)
        error_object = e

    return content_string, logprobs_data, error_object


def execute_processor_step(
    input_df: pd.DataFrame,
    logger: logging.Logger,  # This logger is passed in, not using the module-level logger directly
    run_dir: str,
    processor_instance: Any,
    processor_method_name: str,
    step_number: int,
    step_name_for_logging: str,
    log_success_details_template: str,
    **processor_method_kwargs: Any,
) -> pd.DataFrame:
    """
    Executes a generic step in the pipeline that involves calling a method on a processor object.
    Now resides in the main utils.py.

    Args:
        input_df: The input DataFrame.
        logger: The logger instance passed from the calling context (e.g., AdvPrefixAttack).
        run_dir: The directory for the current run, for saving checkpoints.
        processor_instance: The instance of the processor (e.g., PrefixPreprocessor).
        processor_method_name: The name of the method to call on the processor_instance.
        step_number: The current step number in the pipeline.
        step_name_for_logging: A descriptive name for this step for logging purposes.
        log_success_details_template: A string template for the success log message.
                                      It should accept a 'count' keyword for len(processed_df).
        **processor_method_kwargs: Additional keyword arguments to pass to the processor method.

    Returns:
        The processed DataFrame, or the original DataFrame if an error occurs or input is empty.
    """
    logger.info(f"--- Running Step {step_number}: {step_name_for_logging} ---")

    if input_df.empty:
        logger.warning(
            f"Step {step_number} ({step_name_for_logging}) received an empty DataFrame. Skipping."
        )
        return input_df

    processed_df: pd.DataFrame
    try:
        method_to_call: Callable[..., pd.DataFrame] = getattr(
            processor_instance, processor_method_name
        )
        if processor_method_kwargs:
            processed_df = method_to_call(input_df, **processor_method_kwargs)
        else:
            processed_df = method_to_call(input_df)

    except AttributeError:
        logger.error(
            f"Processor object does not have method '{processor_method_name}'. Step {step_number} ({step_name_for_logging}) failed.",
            exc_info=True,
        )
        return input_df
    except Exception as e:
        logger.error(
            f"Error during {processor_instance.__class__.__name__}.{processor_method_name} for Step {step_number} ({step_name_for_logging}): {e}",
            exc_info=True,
        )
        return input_df

    # This now calls get_checkpoint_path defined in this same utils.py file
    output_path = get_checkpoint_path(run_dir, step_number)
    try:
        processed_df.to_csv(output_path, index=False)
        count = len(processed_df)
        success_details = log_success_details_template.format(count=count)
        logger.info(
            f"Step {step_number} ({step_name_for_logging}) complete. {success_details}"
        )
        logger.info(f"Checkpoint saved to {output_path}")
    except Exception as e:
        logger.error(
            f"Failed to save checkpoint for Step {step_number} ({step_name_for_logging}) to {output_path}: {e}"
        )

    return processed_df
