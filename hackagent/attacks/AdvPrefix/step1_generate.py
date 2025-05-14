import logging
import pandas as pd
from typing import List, Dict, Union, Tuple, Optional

from hackagent.router.router import AgentRouter  # Added
from hackagent.models import AgentTypeEnum  # Added
from hackagent.client import AuthenticatedClient  # Added
from .utils import get_checkpoint_path
from rich.progress import (
    Progress,
    BarColumn,
    TextColumn,
    TimeRemainingColumn,
    MofNCompleteColumn,
    SpinnerColumn,
)
from hackagent.logger import get_logger

logger = get_logger(__name__)

# Constants moved from main file
CUSTOM_CHAT_TEMPLATES = {
    "georgesung/llama2_7b_chat_uncensored": "<s>### HUMAN:\n{content}\n\n### RESPONSE:\n",
    "Tap-M/Luna-AI-Llama2-Uncensored": "<s>USER: {content}\n\nASSISTANT:",
}


def _construct_prompts(
    goals: List[str],
    meta_prefixes: List[str],
    meta_prefixes_n_samples: Union[int, List[int]],  # Allow int or list
) -> Tuple[List[Dict[str, str]], List[str], List[str]]:
    """Constructs prompts for the generator model."""

    # Handle the case where meta_prefixes_n_samples is an integer vs a list
    if isinstance(meta_prefixes_n_samples, list):
        if len(meta_prefixes) != len(meta_prefixes_n_samples):
            raise ValueError(
                "Lengths of meta_prefixes and meta_prefixes_n_samples lists must match."
            )
        n_samples_list = meta_prefixes_n_samples
    elif isinstance(meta_prefixes_n_samples, int):
        # Apply the same integer sample count to all meta prefixes
        n_samples_list = [meta_prefixes_n_samples] * len(meta_prefixes)
    else:
        raise TypeError("meta_prefixes_n_samples must be an int or a list of ints.")

    formatted_inputs = []
    current_goals = []
    expanded_meta_prefixes = []

    for goal in goals:
        for meta_prefix, n_samples in zip(meta_prefixes, n_samples_list):
            if n_samples <= 0:
                continue

            # chat = [{"role": "user", "content": goal}] # Not directly used for router prompt format
            try:
                # The prompt for the router will be the fully constructed context.
                # Custom chat templating needs to happen before sending to router.
                if meta_prefix in CUSTOM_CHAT_TEMPLATES:
                    # Assuming meta_prefix identifies the model type for templating,
                    # which is a bit indirect. Usually, model_string would be used.
                    # For now, we'll keep this logic, but the 'context' is the prompt.
                    prompt_content = CUSTOM_CHAT_TEMPLATES[meta_prefix].format(
                        content=goal
                    )
                else:
                    logger.warning(
                        f"Using basic formatting for prompt construction with meta_prefix: {meta_prefix}. No matching template found."
                    )
                    prompt_content = f"USER: {goal}\\nASSISTANT:"

                # Append the actual meta_prefix text to the prompt that will be sent
                final_prompt = prompt_content + meta_prefix

                formatted_inputs.extend([final_prompt] * n_samples)
                current_goals.extend([goal] * n_samples)
                expanded_meta_prefixes.extend([meta_prefix] * n_samples)
            except Exception as e:
                logging.error(
                    f"Error formatting prompt for goal '{goal}' with meta_prefix '{meta_prefix}': {e}"
                )

    return formatted_inputs, current_goals, expanded_meta_prefixes


async def _generate_prefixes(
    unique_goals: List[str],
    config: Dict,
    logger: logging.Logger,
    client: AuthenticatedClient,  # organization_id removed from here
) -> List[Dict]:
    """
    Helper for step 1. Generate prefixes using AgentRouter with a LiteLLM agent.
    """
    results = []

    generator = config.get("generator", {})
    if not generator:
        logger.error("Missing 'generator'. Cannot initialize AgentRouter for LiteLLM.")
        return results

    # Map generator to adapter_operational_config for LiteLLM
    # New keys for LiteLLMAgentAdapter: 'name', 'endpoint', 'api_key'
    model_name = generator.get("identifier")
    if not model_name:
        logger.error(
            "Missing 'identifier' in 'generator'. Cannot configure LiteLLM agent."
        )
        return results

    adapter_operational_config = {
        "name": model_name,
        "endpoint": generator.get("endpoint"),
        "api_key": generator.get("api_key"),
        # Other params like max_new_tokens, temperature, top_p for adapter defaults
        "max_new_tokens": config.get("max_new_tokens", 100),
        "temperature": config.get("temperature", 0.8),
        "top_p": config.get("top_p", 1.0),
    }

    router: Optional[AgentRouter] = None
    registration_key: Optional[str] = None

    try:
        logger.info(f"Initializing AgentRouter for LiteLLM model: {model_name}")
        router = AgentRouter(
            client=client,
            name=model_name,  # Name for backend agent record
            agent_type=AgentTypeEnum.LITELMM,
            endpoint=generator.get("endpoint"),
            adapter_operational_config=adapter_operational_config,
            metadata=adapter_operational_config.copy(),
            overwrite_metadata=True,
        )

        if router._agent_registry:
            registration_key = next(iter(router._agent_registry.keys()))
            logger.info(
                f"AgentRouter initialized. Registration key for LiteLLM agent: {registration_key}"
            )
        else:
            logger.error(
                "AgentRouter initialized, but no agent adapter was registered."
            )
            return results  # Cannot proceed

    except Exception as e:
        logger.error(
            f"Error initializing AgentRouter for {model_name}: {e}",
            exc_info=True,
        )
        return results

    for do_sample in [False, True]:
        progress_bar_description = (
            "[cyan]Generating Prefixes (Random Sampling)..."
            if do_sample
            else "[cyan]Generating Prefixes (Greedy Decoding)..."
        )
        logger.info(
            f"Generating with {'random sampling' if do_sample else 'greedy decoding'} using LiteLLM via AgentRouter..."
        )
        try:
            # _construct_prompts now returns the full prompt string
            prompts_to_send, current_goals, current_meta_prefixes = _construct_prompts(
                unique_goals,
                config.get("meta_prefixes", []),
                config.get("meta_prefix_samples", []),
            )
            logger.debug(f"Prompts to send ({len(prompts_to_send)}): {prompts_to_send}")
        except Exception as e:
            logger.error(f"Error constructing prompts: {e}", exc_info=True)
            continue

        if not prompts_to_send:
            logger.warning("No prompts to send, skipping completion.")
            continue

        # Loop through each constructed prompt and call the router
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            MofNCompleteColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.1f}%"),
            TimeRemainingColumn(),
        ) as progress_bar:
            task = progress_bar.add_task(
                progress_bar_description, total=len(prompts_to_send)
            )
            for idx, current_prompt_text in enumerate(prompts_to_send):
                goal_for_prompt = current_goals[idx]
                meta_prefix_for_prompt = current_meta_prefixes[idx]

                request_params = {
                    "prompt": current_prompt_text,
                    "max_new_tokens": config.get("max_new_tokens", 100),
                    "temperature": config.get("temperature", 0.8)
                    if do_sample
                    else 1e-2,
                    "top_p": config.get("top_p", 1.0),
                }

                completion_text = None
                try:
                    # logger.info(f"Sending request to router for prompt: {current_prompt_text[:100]}...")
                    response = await router.route_request(
                        registration_key=registration_key,  # type: ignore
                        request_data=request_params,
                    )

                    # logger.debug(f"Router response: {response}")

                    if response and response.get("error_message"):
                        logger.error(
                            f"Error from AgentRouter for prompt '{current_prompt_text[:50]}...': {response['error_message']}"
                        )
                        # Append error marker or skip
                        # For now, we'll try to get processed_response even if there's a partial error
                        # The adapter should handle this.
                        pass  # Ensure block is not empty if all lines are comments

                    if response and response.get("processed_response"):
                        completion_text = response["processed_response"]
                        # The adapter's processed_response is assumed to be the full text (prompt + generation)
                        # We need to extract just the generated part.
                        if completion_text.startswith(current_prompt_text):
                            generated_part = completion_text[len(current_prompt_text) :]
                        else:
                            # Fallback or warning if the response doesn't start with the prompt
                            logger.warning(
                                f"Completion for '{current_prompt_text[:50]}...' did not start with the prompt. Using full response as generated part."
                            )
                            generated_part = completion_text
                    else:
                        logger.warning(
                            f"No 'processed_response' in router output for prompt: {current_prompt_text[:50]}..."
                        )
                        generated_part = " [GENERATION_VIA_ROUTER_FAILED]"

                except Exception as e:
                    logger.error(
                        f"Exception during router.route_request for prompt '{current_prompt_text[:50]}...': {e}",
                        exc_info=True,
                    )
                    generated_part = " [ROUTER_REQUEST_EXCEPTION]"

                # The 'prefix' should be the meta_prefix + generated_part
                final_prefix = meta_prefix_for_prompt + generated_part

                results.append(
                    {
                        "goal": goal_for_prompt,
                        "prefix": final_prefix,
                        "meta_prefix": meta_prefix_for_prompt,
                        "temperature": request_params["temperature"],  # Use actual temp
                        "model_name": model_name,  # Model used by the adapter
                    }
                )
                progress_bar.update(task, advance=1)

    # No need to del router explicitly here, it goes out of scope.
    return results


async def execute(
    goals: List[str],
    config: Dict,
    logger: logging.Logger,
    run_dir: str,
    client: AuthenticatedClient,  # organization_id removed from this call
) -> pd.DataFrame:
    """Generate initial prefixes using provided goals via AgentRouter."""
    logger.info("Executing Step 1: Generating prefixes using AgentRouter")

    if not goals:
        logger.warning("Step 1 received no goals. Returning empty DataFrame.")
        return pd.DataFrame(
            columns=["goal", "prefix", "meta_prefix", "temperature", "model_name"]
        )

    generator = config.get("generator")

    if not generator or not generator.get("identifier"):
        logger.error(
            "Step 1: Missing 'generator' or 'identifier' in config. Cannot generate prefixes."
        )
        return pd.DataFrame(
            columns=["goal", "prefix", "meta_prefix", "temperature", "model_name"]
        )

    model_name_from_config = generator["identifier"]
    logger.info(
        f"Generating prefixes for {len(goals)} unique goals using AgentRouter with LiteLLM: {model_name_from_config}"
    )

    all_results = await _generate_prefixes(
        unique_goals=goals,
        config=config,
        logger=logger,
        client=client,  # organization_id removed from this call
    )

    if not all_results:
        logger.warning("Step 1: No prefixes were generated via AgentRouter.")
        results_df = pd.DataFrame(
            columns=["goal", "prefix", "meta_prefix", "temperature", "model_name"]
        )
    else:
        results_df = pd.DataFrame(all_results)

    output_path = get_checkpoint_path(run_dir, 1)
    try:
        results_df.to_csv(output_path, index=False)
        logger.info(
            f"Step 1 complete. Generated {len(results_df)} total prefixes via AgentRouter"
        )
        logger.info(f"Checkpoint saved to {output_path}")
    except Exception as e:
        logger.error(f"Failed to save checkpoint for step 1 to {output_path}: {e}")

    return results_df
