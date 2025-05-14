import logging

from .client import AuthenticatedClient
from .api.prompt import prompt_list, prompt_create
from .models.prompt import Prompt
from .models.prompt_request import PromptRequest

logger = logging.getLogger(__name__)


def get_or_create_prompt(
    client: AuthenticatedClient,
    name: str,
    text: str,
    category: str,
    organization_id: int,
    evaluation_criteria: str = "Evaluate for expected behavior and safety.",
    extra_tags: list = None,
) -> Prompt:
    """Gets an existing prompt by name or creates a new one, returning the Prompt model."""
    logger.info(f"Checking for prompt: {name}")
    response = prompt_list.sync_detailed(client=client)

    if response.status_code == 200 and response.parsed:
        for prompt_model in response.parsed.results:
            if prompt_model.name == name:
                log_msg = f"Found existing prompt '{name}' with ID {prompt_model.id}."
                logger.info(log_msg)
                return prompt_model

    log_msg = f"Prompt '{name}' not found or no exact match, creating new one..."
    logger.info(log_msg)

    tags_data = ["utility_created"]
    if extra_tags:
        tags_data.extend(extra_tags)

    prompt_req_body = PromptRequest(
        name=name,
        prompt_text=text,
        category=category,
        evaluation_criteria=evaluation_criteria,
        tags=tags_data,
        organization=organization_id,
    )
    create_response = prompt_create.sync_detailed(client=client, body=prompt_req_body)

    if create_response.status_code == 201 and create_response.parsed:
        log_msg = f"Created prompt '{name}' with ID {create_response.parsed.id}."
        logger.info(log_msg)
        return create_response.parsed
    else:
        body_content = (
            create_response.content.decode() if create_response.content else "N/A"
        )
        err_msg = (
            f"Failed to create prompt. Status: {create_response.status_code}, "
            f"Body: {body_content}"
        )
        logger.error(err_msg)
        raise RuntimeError(err_msg)
