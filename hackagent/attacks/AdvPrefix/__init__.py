"""
AdvPrefix Attack Module

Suppress pandas warnings for cleaner attack execution output.
"""

import warnings

# Suppress pandas FutureWarnings specifically for groupby operations
# This addresses warnings from preprocessing operations in the AdvPrefix pipeline
warnings.filterwarnings(
    "ignore", category=FutureWarning, message=".*include_groups.*", module="pandas.*"
)
