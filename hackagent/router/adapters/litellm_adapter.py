from hackagent.router.base import Agent
from typing import Any, Dict, Optional, List
import logging
import asyncio  # Required for async handle_request
import litellm

import os
# from rich.progress import Progress # Removed Progress import


# --- Custom Exceptions ---
class LiteLLMConfigurationError(Exception):
    """Custom exception for LiteLLM adapter configuration issues."""

    pass


logger = logging.getLogger(__name__)  # Module-level logger


class LiteLLMAgentAdapter(Agent):
    """
    Adapter for interacting with LLMs via the LiteLLM library.
    """

    def __init__(self, id: str, config: Dict[str, Any]):
        """
        Initializes the LiteLLMAgentAdapter.

        Args:
            id: The unique identifier for this LiteLLM agent instance.
            config: Configuration dictionary for the LiteLLM agent.
                          Expected keys:
                          - 'name': Model string for LiteLLM (e.g., "ollama/llama3").
                          - 'endpoint' (optional): Base URL for the API.
                          - 'api_key' (optional): Name of the environment variable holding the API key.
                          - 'max_new_tokens' (optional): Default max tokens for generation (defaults to 100).
                          - 'temperature' (optional): Default temperature (defaults to 0.8).
                          - 'top_p' (optional): Default top_p (defaults to 0.95).
        """
        super().__init__(id, config)
        self.logger = logging.getLogger(
            f"{__name__}.{self.id}"
        )  # Instance-specific logger

        if "name" not in self.config:
            msg = (
                f"Missing required configuration key 'name' (for model string) for "
                f"LiteLLMAgentAdapter: {self.id}"
            )
            self.logger.error(msg)
            raise LiteLLMConfigurationError(msg)

        self.model_name: str = self.config["name"]
        self.api_base_url: Optional[str] = self.config.get("endpoint")

        api_key: Optional[str] = self.config.get("api_key")
        self.actual_api_key: Optional[str] = (
            os.environ.get(api_key) if api_key else None
        )

        self.logger.info(
            f"LiteLLMAgentAdapter '{self.id}' initialized for model: '{self.model_name}'"
            + (f" API Base: '{self.api_base_url}'" if self.api_base_url else "")
            + (f" API Key: '{api_key}'" if api_key else "")
        )

        # Store default generation parameters
        self.default_max_new_tokens = self.config.get("max_new_tokens", 100)
        self.default_temperature = self.config.get("temperature", 0.8)
        self.default_top_p = self.config.get("top_p", 0.95)

    async def _execute_litellm_completion(
        self,
        texts: List[str],
        max_new_tokens: int,
        temperature: float,
        top_p: float,
        **kwargs,
    ) -> List[str]:
        """
        Internal method to generate completions using litellm.completion.
        """
        if not texts:
            return []

        completions = []
        self.logger.info(
            f"Sending {len(texts)} requests via LiteLLM to model '{self.model_name}'..."
        )

        # Removed Progress wrapper as it can conflict with outer progress bars
        for text_prompt in texts:
            messages = [{"role": "user", "content": text_prompt}]

            try:
                litellm_params = {
                    "model": self.model_name,
                    "messages": messages,
                    "max_tokens": max_new_tokens,
                    "temperature": temperature,
                    "top_p": top_p,
                    "api_base": self.api_base_url,
                    "api_key": self.actual_api_key,
                }
                # Merge any additional kwargs passed directly for litellm.completion
                litellm_params.update(kwargs)

                # Filter out None values from litellm_params as litellm might not like them for all keys
                # Specifically, api_base and api_key can be None if not provided.
                # LiteLLM handles None for api_base and api_key appropriately.
                # litellm_params = {k: v for k, v in litellm_params.items() if v is not None}

                max_retries = 3
                retry_delay = 2  # seconds
                for attempt in range(max_retries):
                    try:
                        response = await asyncio.to_thread(
                            litellm.completion, **litellm_params
                        )
                        # response = litellm.completion(**litellm_params) # original sync call

                        if (
                            response
                            and response.choices
                            and response.choices[0].message
                            and response.choices[0].message.content
                        ):
                            completion_text = response.choices[0].message.content
                        else:
                            self.logger.warning(
                                f"LiteLLM received unexpected response structure for model '{self.model_name}'. Response: {response}"
                            )
                            completion_text = " [GENERATION_ERROR: UNEXPECTED_RESPONSE]"

                        full_text = text_prompt + completion_text
                        completions.append(full_text)
                        break  # Success, exit retry loop
                    except Exception as e:
                        self.logger.warning(
                            f"LiteLLM attempt {attempt + 1}/{max_retries} failed for model '{self.model_name}': {e}"
                        )
                        if attempt + 1 == max_retries:
                            self.logger.error(
                                f"LiteLLM completion failed after {max_retries} attempts for model '{self.model_name}'.",
                                exc_info=True,
                            )
                            completions.append(
                                text_prompt + " [GENERATION_ERROR: MAX_RETRIES]"
                            )
                        else:
                            # time.sleep(retry_delay) # Can't use time.sleep in async directly
                            await asyncio.sleep(retry_delay)  # Use asyncio.sleep
            except Exception as outer_e:
                self.logger.error(
                    f"Critical error during LiteLLM request preparation or retry logic: {outer_e}",
                    exc_info=True,
                )
                completions.append(text_prompt + " [GENERATION_ERROR: SETUP_FAILURE]")

        self.logger.info(
            f"Finished LiteLLM requests for model '{self.model_name}'. Generated {len(completions)} responses."
        )
        return completions

    async def handle_request(self, request_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Handles an incoming request by processing it through LiteLLM.

        Args:
            request_data: A dictionary containing the request data.
                          Expected keys:
                          - 'prompt': The text prompt to send to the LLM.
                          - 'max_new_tokens' (optional): Override default max tokens.
                          - 'temperature' (optional): Override default temperature.
                          - 'top_p' (optional): Override default top_p.
                          - Any other kwargs to pass to litellm.completion.
        Returns:
            A dictionary representing the agent's response or an error.
        """
        prompt_text = request_data.get("prompt")
        if not prompt_text:
            self.logger.warning("No 'prompt' found in request_data.")
            return self._build_error_response(
                error_message="Request data must include a 'prompt' field.",
                status_code=400,
                raw_request=request_data,
            )

        self.logger.info(
            f"Handling request for LiteLLM adapter {self.id} with prompt: '{prompt_text[:75]}...'"
        )

        max_new_tokens = request_data.get("max_new_tokens", self.default_max_new_tokens)
        temperature = request_data.get("temperature", self.default_temperature)
        top_p = request_data.get("top_p", self.default_top_p)

        excluded_keys = {"prompt", "max_new_tokens", "temperature", "top_p"}
        additional_kwargs = {
            k: v for k, v in request_data.items() if k not in excluded_keys
        }

        try:
            # The _execute_litellm_completion method now handles asyncio.to_thread internally for litellm.completion
            # and also the retry loop with asyncio.sleep
            completions = await self._execute_litellm_completion(
                texts=[prompt_text],
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                **additional_kwargs,
            )

            if completions and isinstance(completions, list) and len(completions) > 0:
                full_response_text = completions[0]
                processed_response = full_response_text

                if "[GENERATION_ERROR:" in processed_response:
                    self.logger.warning(
                        f"LiteLLM generation indicated an error for adapter {self.id}: {processed_response}"
                    )
                    error_detail = processed_response[
                        len(prompt_text) :
                    ].strip()  # Get the error part
                    return self._build_error_response(
                        error_message=f"LiteLLM generation error: {error_detail}",
                        status_code=500,
                        raw_request=request_data,
                    )

                self.logger.info(
                    f"Successfully processed request for LiteLLM adapter {self.id}."
                )
                return {
                    "raw_request": request_data,
                    "processed_response": processed_response,
                    "status_code": 200,
                    "raw_response_headers": None,
                    "raw_response_body": None,
                    "agent_specific_data": {
                        "model_name": self.model_name,  # Updated key
                        "invoked_parameters": {
                            "max_new_tokens": max_new_tokens,
                            "temperature": temperature,
                            "top_p": top_p,
                            **additional_kwargs,
                        },
                    },
                    "error_message": None,
                    "agent_id": self.id,
                    "adapter_type": "LiteLLMAgentAdapter",
                }
            else:
                self.logger.error(
                    f"LiteLLM returned empty or invalid completions for adapter {self.id}."
                )
                return self._build_error_response(
                    error_message="LiteLLM returned empty or invalid result.",
                    status_code=500,
                    raw_request=request_data,
                )
        except Exception as e:
            self.logger.exception(
                f"Unexpected error in LiteLLMAgentAdapter handle_request for agent {self.id}: {e}"
            )
            return self._build_error_response(
                error_message=f"Unexpected adapter error: {type(e).__name__} - {str(e)}",
                status_code=500,
                raw_request=request_data,
            )

    def _build_error_response(
        self,
        error_message: str,
        status_code: Optional[int],
        raw_request: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        return {
            "raw_request": raw_request,
            "processed_response": None,
            "status_code": status_code if status_code is not None else 500,
            "raw_response_headers": None,
            "raw_response_body": None,
            "agent_specific_data": {
                "model_name": self.model_name if hasattr(self, "model_name") else "N/A"
            },  # Updated key
            "error_message": error_message,
            "agent_id": self.id,
            "adapter_type": "LiteLLMAgentAdapter",
        }
