import os
import logging
from dataclasses import dataclass
from typing import Optional
import pandas as pd
from rich.progress import (
    Progress,
    BarColumn,
    TextColumn,
    TimeRemainingColumn,
    MofNCompleteColumn,
    SpinnerColumn,
)
from .utils import call_litellm_completion  # Import the utility

# Configure LiteLLM logging (optional)
# litellm.set_verbose = True

logger = logging.getLogger(__name__)

# --- Configuration Classes ---


@dataclass
class ScorerConfig:
    """Base configuration for scorers"""

    model_id: str  # Identifier for the model used for scoring (litellm string)
    batch_size: int = 1  # Default to 1 for LiteLLM scorer processing
    # surrogate_attack_prompt: str = "" # Maybe needed if formatting prompts


@dataclass
class LiteLLMAPIScoreConfig(ScorerConfig):
    """Configuration specific to LiteLLM API-based scoring"""

    endpoint: Optional[str] = None
    api_key: Optional[str] = None
    surrogate_attack_prompt: str = ""  # Prompt template if needed for API call
    logprob_token_buffer: int = (
        5  # How many extra tokens to request beyond estimated prefix length
    )
    request_timeout: int = 120  # Timeout for API calls


# --- Base Scorer Class (Optional but good practice) ---
class BaseScorer:
    """Abstract base class for scorers."""

    def __init__(self, config: ScorerConfig):
        self.config = config
        self.logger = logging.getLogger(self.__class__.__name__)

    def calculate_score(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculates scores (e.g., NLL) for prefixes."""
        raise NotImplementedError

    def __del__(self):
        """Cleanup resources."""
        pass


class LiteLLMAPIScorer(BaseScorer):
    """
    Calculate an approximate NLL score for prefixes using LiteLLM APIs that support logprobs.
    Note: This is NOT equivalent to precise CE calculation.
    """

    def __init__(self, config: LiteLLMAPIScoreConfig):
        super().__init__(config)
        self.config: LiteLLMAPIScoreConfig  # Type hint
        self.api_key = None
        if self.config.api_key:
            self.api_key = os.environ.get(self.config.api_key)
            if not self.api_key:
                self.logger.warning(
                    f"Environment variable {self.config.api_key} not set for API key."
                )

        self.logger.info(
            f"LiteLLMAPIScorer initialized for model {self.config.model_id}. Token counts will be estimated based on character length."
        )

    def _estimate_token_count(self, text: str) -> int:
        """Estimate token count using character count."""
        # Only use character-based estimation
        # Rough estimate: 4 chars per token (adjust if needed)
        count = (len(text) // 4) + 1
        # self.logger.debug(f"Estimated token count for text length {len(text)}: {count}")
        return count

    def calculate_score(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate approximate NLL score using litellm.completion logprobs via utility.

        Args:
            df: DataFrame with 'goal' and 'prefix' columns.

        Returns:
            DataFrame with additional 'prefix_nll' column (approximate score).
        """
        self.logger.info(
            f"Calculating approximate NLL (LiteLLM) for {len(df)} prefixes using {self.config.model_id}"
        )

        result_df = df.copy()
        # Ensure column exists, default to infinity (bad score)
        if "prefix_nll" not in result_df.columns:
            result_df["prefix_nll"] = float("inf")
        else:
            result_df["prefix_nll"] = result_df["prefix_nll"].fillna(float("inf"))

        approx_nlls = []
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            MofNCompleteColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.1f}%"),
            TimeRemainingColumn(),
        ) as progress_bar:
            task = progress_bar.add_task(
                "[cyan]Calculating Approx NLL (LiteLLM)...", total=len(df)
            )
            for index, row in df.iterrows():
                goal = row["goal"]
                prefix = row["prefix"]
                current_nll = float("inf")  # Default to bad score

                try:
                    # 1. Format the goal as the prompt
                    formatted_goal = goal
                    if self.config.surrogate_attack_prompt:
                        try:
                            formatted_goal += (
                                self.config.surrogate_attack_prompt.format(prefix="")
                            )  # Still formatting here
                        except Exception as fmt_e:
                            self.logger.warning(
                                f"Failed to format surrogate prompt for goal scoring, using original. Error: {fmt_e}"
                            )

                    messages = [{"role": "user", "content": formatted_goal}]

                    # 2. Estimate prefix length and set max_tokens
                    num_prefix_tokens = self._estimate_token_count(prefix.lstrip())
                    max_tokens_to_request = (
                        num_prefix_tokens + self.config.logprob_token_buffer
                    )

                    # 3. Call the utility function requesting logprobs
                    _content, logprob_info, error = call_litellm_completion(
                        model_id=self.config.model_id,
                        messages=messages,
                        endpoint=self.config.endpoint,
                        api_key=self.api_key,
                        timeout=self.config.request_timeout,
                        temperature=0.0,  # Deterministic for scoring
                        max_tokens=max_tokens_to_request,
                        logprobs=True,  # Request logprobs
                        logger=self.logger,
                    )

                    # 4. Handle error or process logprobs
                    if error:
                        self.logger.error(
                            f"LiteLLM call failed during NLL calculation for item {index}: {error}"
                        )
                        # Keep current_nll as inf
                    elif logprob_info:
                        # Process logprobs (logic remains similar)
                        content_from_logprobs = getattr(logprob_info, "content", None)
                        if content_from_logprobs and isinstance(
                            content_from_logprobs, list
                        ):
                            returned_tokens = [
                                item.get("token") for item in content_from_logprobs
                            ]
                            returned_logprobs = [
                                item.get("logprob") for item in content_from_logprobs
                            ]
                            generated_text = "".join(
                                t for t in returned_tokens if t is not None
                            )
                            target_prefix_stripped = prefix.lstrip()

                            if generated_text.startswith(target_prefix_stripped):
                                count = 0
                                summed_logprob = 0.0
                                for tok, lp in zip(returned_tokens, returned_logprobs):
                                    if lp is None:
                                        continue
                                    summed_logprob += lp
                                    count += 1
                                    if count >= num_prefix_tokens:
                                        break
                                if count > 0:
                                    current_nll = -summed_logprob
                                else:
                                    self.logger.warning(
                                        f"Logprob alignment for item {index}: No valid logprobs found."
                                    )
                            elif (
                                target_prefix_stripped.startswith(generated_text)
                                and len(generated_text) > 0
                            ):
                                self.logger.warning(
                                    f"Logprob alignment for item {index}: API generated only '{generated_text[:50]}...' which is a prefix of target."
                                )
                                summed_logprob = sum(
                                    lp for lp in returned_logprobs if lp is not None
                                )
                                if returned_logprobs and any(
                                    lp is not None for lp in returned_logprobs
                                ):
                                    current_nll = -summed_logprob
                            else:
                                self.logger.warning(
                                    f"Logprob alignment for item {index}: Generated text '{generated_text[:50]}...' does not match target prefix '{target_prefix_stripped[:50]}...'."
                                )
                        else:
                            self.logger.warning(
                                f"Logprobs returned for item {index}, but in unexpected format: {logprob_info}"
                            )
                    else:
                        # Utility function already logs warning if logprobs requested but not found
                        pass  # Keep current_nll as inf

                except Exception as e:  # Catch errors in the calling code (e.g., formatting, estimate_token_count)
                    self.logger.error(
                        f"Error calculating approx NLL for item {index} outside LiteLLM call: {e}",
                        exc_info=True,
                    )
                    # Keep current_nll as inf

                approx_nlls.append(current_nll)
                progress_bar.update(task, advance=1)

        # Update DataFrame
        if len(approx_nlls) == len(result_df):
            result_df["prefix_nll"] = approx_nlls
        else:
            self.logger.error(
                f"Mismatch between calculated approx NLLs ({len(approx_nlls)}) and DataFrame rows ({len(result_df)}). NLL column may be incorrect."
            )
            # Pad/truncate as fallback
            if len(approx_nlls) < len(result_df):
                approx_nlls.extend([float("inf")] * (len(result_df) - len(approx_nlls)))
            result_df["prefix_nll"] = approx_nlls[: len(result_df)]

        self.logger.info(
            f"Finished calculating approximate NLL (LiteLLM) for {len(result_df)} prefixes."
        )
        return result_df

    def __del__(self):
        self.logger.info(
            "LiteLLMAPIScorer resources released (no explicit cleanup needed)."
        )
