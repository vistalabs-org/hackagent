import logging
import pandas as pd
from typing import Dict, Any

from hackagent.attacks.AdvPrefix.selector import (
    PrefixSelectorConfig,
    PrefixSelector,
)

from .utils import get_checkpoint_path

logger = logging.getLogger(__name__)


def execute(
    input_df: pd.DataFrame, config: Dict[str, Any], run_dir: str
) -> pd.DataFrame:
    """Select final prefixes based on specified judges and selection criteria using input DataFrame."""
    logger.info("Executing Step 9: Selecting final prefixes")

    if input_df.empty:
        logger.warning("Step 9 received an empty DataFrame. Skipping selection.")
        return input_df

    selector = None  # Ensure cleanup
    selected_df = input_df  # Default to input if selection fails

    try:
        # Initialize selector here
        selector_config = PrefixSelectorConfig(
            pasr_weight=config.get("pasr_weight", 0.5),
            n_prefixes_per_goal=config.get("n_prefixes_per_goal", 3),
            judges=config.get("selection_judges", []),
        )
        selector = PrefixSelector(selector_config)

        # Select prefixes
        selected_df = selector.select_prefixes(input_df)
        logger.info(f"Selection complete. Selected {len(selected_df)} prefixes.")

    except Exception as e:
        logger.error(f"Error during prefix selection: {e}", exc_info=True)
        logger.warning("Returning unselected prefixes due to selection error.")
        selected_df = input_df  # Fallback to returning the input df

    finally:
        del selector
        # No GPU cleanup needed typically for selection

    # Save results checkpoint (final step)
    output_path = get_checkpoint_path(run_dir, 9)
    try:
        selected_df.to_csv(output_path, index=False)
        logger.info("Step 9 complete.")
        logger.info(f"Final selected prefixes checkpoint saved to {output_path}")
    except Exception as e:
        logger.error(f"Failed to save checkpoint for step 9 to {output_path}: {e}")

    return selected_df
