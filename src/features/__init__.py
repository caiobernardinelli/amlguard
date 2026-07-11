"""Feature engineering package.

Re-exports the public feature API so callers can use
``from src.features import build_model_features`` regardless of module layout.
"""

from src.features.build_features import (
    MODEL_FEATURES,
    add_same_account_flag,
    add_sender_previous_tx_count,
    add_temporal_features,
    build_model_features,
)

__all__ = [
    "MODEL_FEATURES",
    "add_temporal_features",
    "add_same_account_flag",
    "add_sender_previous_tx_count",
    "build_model_features",
]
