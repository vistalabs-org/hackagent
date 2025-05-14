import logging
import pandas as pd
import multiprocessing
import os
from typing import Dict
from dataclasses import fields  # Import fields to inspect dataclass

from hackagent.attacks.AdvPrefix.scorer_parser import (
    EvaluatorConfig,
    NuancedEvaluator,
    HarmBenchEvaluator,
    JailbreakBenchEvaluator,
)
from hackagent.attacks.AdvPrefix.utils import get_checkpoint_path

EVALUATOR_MAP = {
    "nuanced": NuancedEvaluator,
    "jailbreakbench": JailbreakBenchEvaluator,
    "harmbench": HarmBenchEvaluator,
}

MERGE_KEYS = ["goal", "prefix", "completion"]  # Standard merge keys

JUDGE_COLUMN_MAP = {
    "nuanced": ["eval_nj", "explanation_nj"],
    "jailbreakbench": ["eval_jb", "explanation_jb"],
    "harmbench": ["eval_hb", "explanation_hb"],
}


def _run_evaluator_process_wrapper(
    judge_type: str, config_dict_serializable: Dict, df: pd.DataFrame
):
    """Static method to run a specific evaluator, suitable for multiprocessing."""
    process_logger = logging.getLogger(__name__ + f".evaluator_process_{judge_type}")
    process_logger.info(f"Evaluator process started for judge: {judge_type}")

    evaluator_class = EVALUATOR_MAP.get(judge_type)
    if not evaluator_class:
        process_logger.warning(f"Unknown judge type: {judge_type}, skipping")
        return None

    evaluator = None  # Ensure cleanup
    try:
        # Filter the config dict to only include keys expected by EvaluatorConfig
        expected_fields = {f.name for f in fields(EvaluatorConfig)}
        filtered_config_dict = {
            k: v for k, v in config_dict_serializable.items() if k in expected_fields
        }

        # model_id is handled specially by EvaluatorConfig.with_default_model
        # or within the specific evaluator's __init__.
        # We should ensure model_id from the judge's config is passed if present.
        if (
            "model_id" in config_dict_serializable
            and config_dict_serializable["model_id"]
        ):
            filtered_config_dict["model_id"] = config_dict_serializable["model_id"]
        elif (
            "identifier" in config_dict_serializable
            and config_dict_serializable["identifier"]
        ):
            # Fallback to using 'identifier' if 'model_id' wasn't explicitly passed/overridden
            filtered_config_dict["model_id"] = config_dict_serializable["identifier"]

        process_logger.debug(
            f"Filtered config for {judge_type} evaluator: {filtered_config_dict}"
        )
        evaluator_config = EvaluatorConfig(**filtered_config_dict)

        # Instantiate the specific evaluator class
        evaluator = evaluator_class(evaluator_config)
        evaluated_df = evaluator.evaluate(df)

        process_logger.info(f"Evaluator process finished for judge: {judge_type}")
        # Return only the essential columns: merge keys + judge-specific columns
        eval_cols = JUDGE_COLUMN_MAP.get(judge_type, [])
        # Ensure merge keys are present in the returned df
        if not all(key in evaluated_df.columns for key in MERGE_KEYS):
            process_logger.error(
                f"Evaluation result for {judge_type} is missing merge keys {MERGE_KEYS}. Available: {evaluated_df.columns}. Returning None."
            )
            return None

        cols_to_return = MERGE_KEYS + [
            col for col in eval_cols if col in evaluated_df.columns
        ]
        return evaluated_df[cols_to_return]

    except Exception as e:
        process_logger.error(
            f"Error occurred while running {judge_type} evaluator: {str(e)}",
            exc_info=True,
        )
        return None  # Indicate failure
    finally:
        # Cleanup
        del evaluator
        process_logger.info(
            f"Evaluator process cleanup finished for judge: {judge_type}"
        )


def execute(
    input_df: pd.DataFrame, config: Dict, logger: logging.Logger, run_dir: str
) -> pd.DataFrame:
    """Evaluate completions using specified judges."""
    logger.info("Executing Step 7: Evaluating responses")
    original_df = input_df

    if original_df.empty:
        logger.warning("Step 7 received an empty DataFrame. Skipping evaluation.")
        return original_df

    # Config key 'judges' should be a list of dictionaries
    judge_configs_list = config.get("judges")
    if not isinstance(judge_configs_list, list) or not judge_configs_list:
        logger.warning(
            "Step 7: 'judges' key in configuration is missing, not a list, or empty. Skipping evaluation."
        )
        return original_df

    # Base config for evaluators (extract non-judge-specific params)
    evaluator_base_config_dict = {
        "batch_size": config.get("batch_size_judge"),
        "max_new_tokens_eval": config.get("max_new_tokens_eval"),
        "filter_len": config.get("filter_len"),
        # General API settings (judges might override)
        "endpoint": config.get("judge_endpoint"),
        "api_key": config.get("judge_api_key"),
        "request_timeout": config.get("judge_request_timeout"),
    }

    judge_results_dfs = {}
    failed_judges = []
    async_results = []
    judges_to_run = []  # Store valid (type, config_dict) tuples

    # --- Prepare Judge Runs ---
    for judge_config_item in judge_configs_list:
        if not isinstance(judge_config_item, dict):
            logger.warning(
                f"Skipping invalid item in 'judges' list (not a dict): {judge_config_item}"
            )
            continue

        # Extract the judge type string (e.g., "nuanced", "harmbench")
        # Assuming the type is specified by a 'type' key in the judge dict.
        # Alternative: Infer based on 'identifier'?
        judge_type_str = judge_config_item.get(
            "evaluator_type"
        ) or judge_config_item.get("type")
        judge_identifier = judge_config_item.get("identifier")

        if not judge_type_str:
            # If type isn't explicit, try to infer (this part might need refinement)
            if "nuanced" in judge_identifier.lower():
                judge_type_str = "nuanced"
            elif "harmbench" in judge_identifier.lower():
                judge_type_str = "harmbench"
            elif "jailbreak" in judge_identifier.lower():
                judge_type_str = "jailbreakbench"
            else:
                logger.warning(
                    f"Could not determine evaluator type for judge config: {judge_config_item}. Skipping."
                )
                continue

        # Check if the extracted type string is valid
        if judge_type_str not in EVALUATOR_MAP:
            logger.warning(
                f"Skipping unknown judge type '{judge_type_str}' specified in config: {judge_config_item}"
            )
            continue

        # Prepare the specific config to pass to the subprocess
        # Start with base, then override with judge-specific settings
        subprocess_config = evaluator_base_config_dict.copy()
        subprocess_config.update(judge_config_item)  # Override base with specifics
        # Ensure model_id is set correctly (use 'identifier')
        if judge_identifier:
            subprocess_config["model_id"] = judge_identifier

        judges_to_run.append((judge_type_str, subprocess_config))

    if not judges_to_run:
        logger.warning(
            "Step 7: No valid judges found after processing configuration. Skipping evaluation."
        )
        return original_df

    # --- Setup Multiprocessing Pool ---
    try:
        current_start_method = multiprocessing.get_start_method(allow_none=True)
        if current_start_method != "spawn":
            multiprocessing.set_start_method("spawn", force=True)
            logger.info("Set multiprocessing start method to 'spawn' for Step 7.")
    except Exception as e:
        logger.warning(f"Could not set multiprocessing start method to spawn: {e}")

    num_judges = len(judges_to_run)
    num_workers = min(num_judges, os.cpu_count() or 1, 4)
    logger.info(
        f"Starting evaluation pool with {num_workers} workers for {num_judges} judges."
    )

    # --- Dispatch and Collect Results ---
    with multiprocessing.Pool(processes=num_workers) as pool:
        # Dispatch tasks using the prepared list
        for judge_type_str, subprocess_config in judges_to_run:
            logger.info(
                f"Dispatching evaluation with {judge_type_str} judge. Config: {subprocess_config}"
            )
            args = (judge_type_str, subprocess_config, original_df.copy())
            async_results.append(
                pool.apply_async(_run_evaluator_process_wrapper, args=args)
            )

        # Collect results using the order in judges_to_run
        for (judge_type_str, _), result in zip(judges_to_run, async_results):
            try:
                evaluated_df_subset = result.get()
                if evaluated_df_subset is not None:
                    judge_results_dfs[judge_type_str] = evaluated_df_subset
                    logger.info(
                        f"Successfully completed evaluation for judge: {judge_type_str}"
                    )
                else:
                    failed_judges.append(judge_type_str)
                    logger.error(
                        f"Evaluation failed for judge: {judge_type_str} (process returned None)"
                    )
            except Exception as e:
                failed_judges.append(judge_type_str)
                logger.error(
                    f"Evaluation task failed for judge {judge_type_str}: {e}",
                    exc_info=True,
                )

    # --- Merge Results ---
    final_df = original_df.copy()
    successful_judges = list(judge_results_dfs.keys())

    if not successful_judges:
        logger.warning(
            "Step 7: No judges completed successfully. Returning original DataFrame."
        )
    else:
        logger.info(f"Merging results from successful judges: {successful_judges}")
        for judge_type_str in successful_judges:
            judge_df_subset = judge_results_dfs[judge_type_str]
            eval_cols = JUDGE_COLUMN_MAP.get(judge_type_str, [])
            judge_cols_present = [
                col for col in eval_cols if col in judge_df_subset.columns
            ]

            if not judge_cols_present:
                logger.warning(
                    f"No specific evaluation columns found in result for judge {judge_type_str}. Skipping merge for this judge."
                )
                continue

            try:
                final_df = final_df.merge(
                    judge_df_subset,
                    on=MERGE_KEYS,
                    how="left",
                    suffixes=("", f"_{judge_type_str}_dup"),
                )
                logger.debug(f"Merged results from judge {judge_type_str}")
            except Exception as e:
                logger.error(f"Error merging results for judge {judge_type_str}: {e}")

    # Save final merged results checkpoint
    output_path = get_checkpoint_path(run_dir, 7)
    try:
        final_df.to_csv(output_path, index=False)
        logger.info(f"Step 7 complete. Evaluated {len(final_df)} responses.")
        if successful_judges:
            logger.info(
                f"Successfully completed judges: {', '.join(successful_judges)}"
            )
        if failed_judges:
            logger.warning(f"Failed judges: {', '.join(failed_judges)}")
        logger.info(f"Final evaluation results checkpoint saved to {output_path}")
    except Exception as e:
        logger.error(f"Failed to save checkpoint for step 7 to {output_path}: {e}")

    return final_df
