import pandas as pd
from typing import Dict, Any

from .utils import get_checkpoint_path

# Map judge type to expected column prefix/name used for aggregation stats
JUDGE_AGG_COLUMN_MAP = {
    "nuanced": "eval_nj",
    "jailbreakbench": "eval_jb",
    "harmbench": "eval_hb",
}

GROUP_KEYS = ["goal", "prefix"]


def execute(
    input_df: pd.DataFrame, config: Dict[str, Any], run_dir: str
) -> pd.DataFrame:
    """
    Aggregate evaluation results from different judges using the input DataFrame.
    Combines results from multiple evaluation samples and judges into single scores per prefix.
    """
    print("Executing Step 8: Aggregating evaluation results")

    if input_df.empty:
        print("WARNING: Step 8 received an empty DataFrame. Skipping aggregation.")
        # Define expected aggregated columns if returning empty
        cols = GROUP_KEYS + [
            "prefix_nll",
            "model_name",
            "meta_prefix",
            "temperature",
            "n_eval_samples",
        ]
        for _, col_base in JUDGE_AGG_COLUMN_MAP.items():
            cols.extend([f"{col_base}_mean", f"{col_base}_count"])
        return pd.DataFrame(columns=cols)

    analysis = input_df.copy()

    # Optionally filter based on cross-entropy / NLL score
    if "prefix_nll" in analysis.columns and config.get("max_ce") is not None:
        try:
            max_ce_threshold = float(config.get("max_ce"))
            initial_count = len(analysis)
            # Use dictionary access for config
            analysis = analysis[analysis["prefix_nll"] < max_ce_threshold]
            filtered_count = len(analysis)
            print(
                f"Filtered {initial_count - filtered_count} rows based on prefix_nll >= {max_ce_threshold}"
            )
        except KeyError:
            print("WARNING: 'max_ce' key not found in config, skipping NLL filtering.")
        except Exception as e:
            print(f"ERROR: Error during NLL filtering in aggregation: {e}")
            # Continue without NLL filtering if error occurs
    elif "prefix_nll" not in analysis.columns:
        print(
            "WARNING: Column 'prefix_nll' not found. Skipping NLL filtering in aggregation step."
        )

    # Detect available judges based on column names for aggregation
    available_judges_agg_cols = {}
    judges_in_config = config.get("judges", [])  # Judges that were supposed to run
    for judge_type, col_name in JUDGE_AGG_COLUMN_MAP.items():
        if col_name in analysis.columns:
            available_judges_agg_cols[judge_type] = col_name
        else:
            # Log if any expected judge column is missing
            if judge_type in judges_in_config:
                print(
                    f"WARNING: Expected aggregation column '{col_name}' for judge '{judge_type}' not found in the dataframe for Step 8."
                )

    if not available_judges_agg_cols:
        print(
            "ERROR: No recognized evaluation result columns found for aggregation. Check step 7 output."
        )
        output_path = get_checkpoint_path(run_dir, 8)
        try:
            analysis.to_csv(output_path, index=False)
            print(
                f"WARNING: Step 8 saving unaggregated data to {output_path} due to missing judge columns."
            )
        except Exception as e:
            print(
                f"ERROR: Failed to save unaggregated data checkpoint for step 8 to {output_path}: {e}"
            )
        return analysis  # Return unaggregated data

    print(
        f"Found aggregation columns for judges: {list(available_judges_agg_cols.keys())}"
    )

    # Ensure group keys exist
    if not all(key in analysis.columns for key in GROUP_KEYS):
        missing_keys = [key for key in GROUP_KEYS if key not in analysis.columns]
        print(
            f"ERROR: Missing required grouping keys for aggregation: {missing_keys}. Cannot aggregate."
        )
        output_path = get_checkpoint_path(run_dir, 8)
        try:
            analysis.to_csv(output_path, index=False)
            print(
                f"WARNING: Step 8 saving unaggregated data to {output_path} due to missing group keys."
            )
        except Exception as e:
            print(
                f"ERROR: Failed to save unaggregated data checkpoint for step 8 to {output_path}: {e}"
            )
        return analysis

    # Define aggregations
    agg_funcs = {
        # Use pd.NamedAgg for clarity and future compatibility
        "prefix_nll": pd.NamedAgg(column="prefix_nll", aggfunc="first"),
        "model_name": pd.NamedAgg(column="model_name", aggfunc="first"),
        "meta_prefix": pd.NamedAgg(column="meta_prefix", aggfunc="first"),
        "temperature": pd.NamedAgg(column="temperature", aggfunc="first"),
        # Count samples - use one of the group keys or index if reset
        "n_eval_samples": pd.NamedAgg(column=GROUP_KEYS[0], aggfunc="size"),
    }

    # Add judge-specific aggregations
    for judge_type, col_name in available_judges_agg_cols.items():
        # Ensure the column is numeric before calculating mean
        try:
            analysis[col_name] = pd.to_numeric(analysis[col_name], errors="coerce")
            agg_funcs[f"{col_name}_mean"] = pd.NamedAgg(column=col_name, aggfunc="mean")
            agg_funcs[f"{col_name}_count"] = pd.NamedAgg(
                column=col_name, aggfunc="count"
            )  # Count non-NA numeric values
            print(
                f"DEBUG: Added mean/count aggregation for numeric column '{col_name}'"
            )
        except KeyError:
            print(
                f"WARNING: Column '{col_name}' unexpectedly missing during aggregation setup. Skipping mean/count."
            )
        except Exception as e:
            print(
                f"ERROR: Could not convert column '{col_name}' to numeric for aggregation. Skipping mean/count. Error: {e}"
            )
            # Optionally add just size aggregation if mean fails?
            agg_funcs[f"{col_name}_size"] = pd.NamedAgg(column=col_name, aggfunc="size")

    # Perform aggregation
    try:
        grouped = analysis.groupby(GROUP_KEYS, observed=False, dropna=False)
        aggregated = grouped.agg(**agg_funcs)
        aggregated = aggregated.reset_index()
    except Exception as e:
        print(
            f"ERROR: Error during aggregation: {e}. Check aggregation functions and column types."
        )
        output_path = get_checkpoint_path(run_dir, 8)
        try:
            analysis.to_csv(output_path, index=False)
            print(
                f"WARNING: Step 8 saving unaggregated data to {output_path} due to aggregation error."
            )
        except Exception as e_save:
            print(
                f"ERROR: Failed to save unaggregated data checkpoint for step 8 to {output_path}: {e_save}"
            )
        return analysis  # Return unaggregated on error

    # Save results checkpoint
    output_path = get_checkpoint_path(run_dir, 8)
    try:
        aggregated.to_csv(output_path, index=False)
        print(f"Step 8 complete. Aggregated {len(aggregated)} prefix results.")
        print(f"Checkpoint saved to {output_path}")
    except Exception as e:
        print(f"ERROR: Failed to save checkpoint for step 8 to {output_path}: {e}")

    return aggregated
