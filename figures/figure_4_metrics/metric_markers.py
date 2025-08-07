"""
metric_markers.py
Utility for putting ★ / ▲ markers onto 1-D curves.

Two use-cases
-------------
1.  **Left-column theory curves**
    • always put both markers at their κ-targets.

2.  **Right-hand experimental splitting curves**
    • put a single marker *only if* the curve’s **last** x-value coincides
      (within a small tolerance) with that κ-target.  Activate via
      ``only_endpoint=True``.
"""
from __future__ import annotations

from typing import Any

import numpy as np
import matplotlib.pyplot as plt
from settings import STYLE


def _pos(a: Any, idx: int):
    """Return element *idx* by **position** for ndarray *or* pandas Series."""
    return a.iloc[idx] if hasattr(a, "iloc") else a[idx]


def scatter_metric_markers(
        ax: plt.Axes,
        x_grid,                       # 1-D: ndarray or Series
        y_grid,                       # same length / type as x_grid
        phi: float,
        color: str,
        *,
        only_endpoint: bool = False,
        branch_marker: str = None,
) -> None:
    x_vals = np.asarray(x_grid)
    y_vals = np.asarray(y_grid)

    x_last = _pos(x_grid, -1)

    for marker, kmap, msize in (
            ("o", STYLE.star_kappa, STYLE.star_ms),
            ("^", STYLE.tri_kappa,  STYLE.tri_ms),
    ):

        if phi not in kmap:
            continue

        x_tgt = kmap[phi]

        idx = int(np.argmin(np.abs(x_vals - x_tgt)))
        x_val_plotted =             _pos(x_grid, idx),
        y_val_plotted =             _pos(y_grid, idx),
        # index of nearest point to x_tgt

        if only_endpoint:
            x_val_plotted = x_vals[-1] if not np.isclose(phi, np.pi/2) else x_vals[1]
            y_val_plotted = y_vals[-1] if not np.isclose(phi, np.pi/2) else y_vals[1]

        if marker == "o":
            # single hollow-like circle with color border
            ax.scatter(
                x_val_plotted,
                y_val_plotted,
                marker="o",
                s=msize,
                facecolor="white",      # hollow fill
                edgecolor=color,        # colored border
                linewidth=2.25,
                zorder=10,
            )
        else:
            # single hollow-like triangle with color border
            ax.scatter(
                x_val_plotted,
                y_val_plotted,
                marker="^",
                s=msize,
                facecolor="white",      # hollow fill
                edgecolor=color,        # colored border
                linewidth=2.25,
                zorder=10,
            )
