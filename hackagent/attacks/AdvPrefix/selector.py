import pandas as pd
import numpy as np
import logging
from typing import List, Optional
from dataclasses import dataclass


@dataclass
class PrefixSelectorConfig:
    """Configuration for prefix selection"""

    pasr_weight: float  # Weight for log-PASR in selection
    n_prefixes_per_goal: int = 1  # Number of prefixes to select per goal
    nll_tol: float = 999  # Tolerance for NLL relative to best prefix
    pasr_tol: float = 0  # Tolerance for PASR relative to best prefix
    judges: Optional[List[dict]] = None  # List of judges to use for PASR calculation


class PrefixSelector:
    """
    Selects prefixes based on a combination of judge scores (PASR) and NLL.
    Supports multiple judges and custom weighting for selection criteria.
    """

    def __init__(self, config: PrefixSelectorConfig):
        """
        Initialize the prefix selector.

        Args:
            config: Configuration for prefix selection
        """
        self.config = config
        self.logger = logging.getLogger(__name__)

        # Mapping of judge types to their column names in the DataFrame
        self.judge_column_map = {
            "nuanced": "eval_nj_mean",
            "jailbreakbench": "eval_jb_mean",
            "harmbench": "eval_hb_mean",
            "strongreject": "eval_sj_binary_mean",
        }

    def select_prefixes(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Select prefixes based on combined judge scores and NLL.

        Args:
            df: DataFrame containing prefixes with evaluation results

        Returns:
            DataFrame containing selected prefixes
        """
        # Validate judge configuration list
        if not isinstance(self.config.judges, list) or not self.config.judges:
            # Check if judges is a list and not empty
            raise ValueError(
                "Judge configuration ('judges' key) must be a non-empty list of dictionaries."
            )

        judge_types_found = []
        missing_columns = []
        for judge_config in self.config.judges:
            if not isinstance(judge_config, dict):
                self.logger.warning(
                    f"Skipping invalid item in judge config list (not a dict): {judge_config}"
                )
                continue

            # Extract judge type string (e.g., "nuanced") - Assuming a 'type' key
            judge_type = judge_config.get("type") or judge_config.get("evaluator_type")
            # Could add inference here if needed, similar to step 7

            if not judge_type:
                self.logger.warning(
                    f"Could not determine type for judge config: {judge_config}. Skipping."
                )
                continue

            if judge_type not in self.judge_column_map:
                # Check if the *type string* is valid
                self.logger.error(
                    f"Unknown judge type specified in config: '{judge_type}'"
                )
                raise ValueError(f"Unknown judge type for selection: {judge_type}")

            # Check if the corresponding column exists in the DataFrame
            expected_col = self.judge_column_map[judge_type]
            if expected_col not in df.columns:
                missing_columns.append(expected_col)

            if judge_type not in judge_types_found:
                judge_types_found.append(judge_type)

        if missing_columns:
            raise ValueError(
                f"Missing required evaluation result columns in DataFrame: {missing_columns}"
            )
        if not judge_types_found:
            raise ValueError(
                "No valid judge types found in the configuration to perform selection."
            )

        # Create a working copy of the DataFrame
        work_df = df.copy()

        # Calculate combined PASR score using the identified judge types
        work_df["pasr"] = self._calculate_combined_pasr(work_df, judge_types_found)

        # Calculate log PASR for scoring
        work_df["log_pasr"] = np.log(work_df["pasr"] + 1e-6)

        # Calculate combined score (minimize both 1 - PASR and prefix_nll)
        work_df["combined_score"] = (
            -self.config.pasr_weight * work_df["log_pasr"] + work_df["prefix_nll"]
        )

        # Create DataFrame for selected prefixes
        selected_prefixes = pd.DataFrame()

        # Group by goal and apply selection process
        for goal, group in work_df.groupby("goal"):
            # Step 1: Select first prefix based on combined score
            # Check if group is empty after potential filtering/issues
            if (
                group.empty
                or "combined_score" not in group.columns
                or group["combined_score"].isnull().all()
            ):
                self.logger.warning(
                    f"Skipping goal '{goal[:50]}...' during selection due to empty group or missing/invalid scores."
                )
                continue

            first_selection_idx = group["combined_score"].idxmin()
            first_selection = group.loc[first_selection_idx]

            # Step 2: Filter prefixes within PASR tolerance
            remaining_candidates = group[
                (group["pasr"] >= first_selection["pasr"] - self.config.pasr_tol)
                & (group.index != first_selection.name)
            ]

            # Step 3: Filter candidates within NLL tolerance
            valid_candidates = remaining_candidates[
                remaining_candidates["prefix_nll"]
                <= first_selection["prefix_nll"] + self.config.nll_tol
            ]

            # Initialize selections list with first selection
            selections = [first_selection]

            # Step 4: Iteratively select additional prefixes
            for _ in range(self.config.n_prefixes_per_goal - 1):
                # Remove candidates that are sub-prefixes of selected ones
                valid_candidates = valid_candidates[
                    ~valid_candidates["prefix"].apply(
                        lambda x: any(
                            str(x).startswith(str(sel["prefix"]))
                            for sel in selections
                            if sel is not None and "prefix" in sel and x is not None
                        )
                    )
                ]

                if valid_candidates.empty:
                    break

                # Select next prefix with lowest NLL
                if (
                    "prefix_nll" not in valid_candidates.columns
                    or valid_candidates["prefix_nll"].isnull().all()
                ):
                    self.logger.warning(
                        f"Cannot select next prefix for goal '{goal[:50]}...' due to missing/invalid NLL scores in candidates."
                    )
                    break
                next_selection = valid_candidates.nsmallest(1, "prefix_nll").iloc[0]
                selections.append(next_selection)
                valid_candidates = valid_candidates[
                    valid_candidates.index != next_selection.name
                ]

            # Combine selections for this goal
            combined_selection = pd.DataFrame(selections)
            selected_prefixes = pd.concat([selected_prefixes, combined_selection])

        # Reset index
        selected_prefixes.reset_index(drop=True, inplace=True)

        # Add the new columns (pasr, log_pasr, combined_score) to the output
        # Ensure columns exist before trying to select them
        output_columns = [
            col
            for col in list(df.columns) + ["pasr", "log_pasr", "combined_score"]
            if col in selected_prefixes.columns
        ]
        selected_prefixes = selected_prefixes[output_columns]

        self.logger.info(
            f"Selected {len(selected_prefixes)} prefixes across {len(df['goal'].unique())} goals"
        )
        return selected_prefixes

    def _calculate_combined_pasr(
        self, df: pd.DataFrame, judge_types: List[str]
    ) -> pd.Series:
        """
        Calculate combined PASR score from specified judge types.

        Args:
            df: DataFrame containing judge scores
            judge_types: List of valid judge type strings (e.g., ["nuanced", "harmbench"])

        Returns:
            Series containing combined PASR scores
        """
        judge_scores = []
        for judge_type in judge_types:  # Iterate through the list of type strings
            column = self.judge_column_map[judge_type]  # Use the type string for lookup
            # Ensure column is numeric before appending
            if column in df.columns:
                try:
                    numeric_scores = pd.to_numeric(df[column], errors="coerce")
                    judge_scores.append(numeric_scores)
                except Exception as e:
                    self.logger.warning(
                        f"Could not convert column '{column}' to numeric for PASR calculation. Skipping. Error: {e}"
                    )
            else:
                # This should be caught by initial validation, but as safeguard
                self.logger.warning(
                    f"Column '{column}' for judge '{judge_type}' not found during PASR calculation."
                )

        if not judge_scores:
            self.logger.warning(
                "No valid judge scores found to calculate combined PASR. Returning zeros."
            )
            return pd.Series(0, index=df.index)

        # Calculate mean of judge scores, handling potential NaNs after conversion
        combined_scores_df = pd.concat(judge_scores, axis=1)
        # Use mean, skipping NaNs. If a row has all NaNs, the mean will be NaN.
        mean_scores = combined_scores_df.mean(axis=1, skipna=True)
        # Fill any resulting NaNs (rows where all judges had NaN scores) with 0
        return mean_scores.fillna(0)
