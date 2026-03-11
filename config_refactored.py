"""Project-wide configuration for graph construction and DaRIM experiments.

This module centralizes dimension-related settings and provides lightweight
validation helpers so downstream code does not need to hardcode values.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class DimensionConfig:
    """Dimensions used by feature construction and hyperparameter models."""

    feature_dimension: int = 10
    theta_dimension: int = 10
    node_embedding_dimension: int = 5

    def validate(self) -> None:
        if self.theta_dimension != self.feature_dimension:
            raise ValueError(
                f"theta_dimension ({self.theta_dimension}) must equal "
                f"feature_dimension ({self.feature_dimension})"
            )
        if self.feature_dimension <= 0 or self.theta_dimension <= 0:
            raise ValueError("All dimensions must be positive integers.")
        if self.node_embedding_dimension <= 0:
            raise ValueError("node_embedding_dimension must be positive.")

    @property
    def is_concat_node_feature_layout(self) -> bool:
        return self.feature_dimension == 2 * self.node_embedding_dimension


DIMENSIONS = DimensionConfig()
DIMENSIONS.validate()

# -----------------------------------------------------------------------------
# Backward-compatible aliases
# -----------------------------------------------------------------------------
FEATURE_DIMENSION = DIMENSIONS.feature_dimension
THETA_DIMENSION = DIMENSIONS.theta_dimension
NODE_EMBEDDING_DIM = DIMENSIONS.node_embedding_dimension
SUPERSIZE = THETA_DIMENSION
DIMENSION = FEATURE_DIMENSION


def validate_dimensions() -> None:
    """Compatibility wrapper for older imports."""
    DIMENSIONS.validate()
