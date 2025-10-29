"""Project-wide configuration toggles for the EP-TPD unification pipeline."""

from __future__ import annotations

from enum import Enum


class FigureMode(str, Enum):
    """Valid rendering modes for figure generation."""
    LARGE = "large"
    SMALL = "small"


# Toggle between the two figure rendering modes above.
FIGURE_MODE: FigureMode = FigureMode.LARGE

# Force the ETL pipeline to re-run even if transformed data already exists.
FORCE_ETL: bool = True
