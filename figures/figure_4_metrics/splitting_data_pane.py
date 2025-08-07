"""
splitting_data_pane.py
----------------------

Plot a *single* dimension-less peak–splitting pane:

    •  x–axis  :  Δκ/J   (φ = 0, π/2)   or   Δf/J (φ = π)
    •  y–axis  :  (ν₊ – ν₋)/J                     (dimension-less splitting)
    •  points  :  experimental data  (error bars scaled)
    •  curve   :  theory splitting
    •  h-line  :  min-split from ``min_split_over_J``   (dashed)

All colours / marker sizes come from *settings.py* (imported as ``STYLE``).
"""

from __future__ import annotations

from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from analysis_pt.FIG_METRICS_ND.settings import FigMetricStyle
from analysis_pt.FIG_METRICS_ND.tpd_locations_nd import standard_tpd_locations
from settings import STYLE
from analysis_pt.FIG_METRICS_ND.metric_calculations import min_split_over_J
from analysis_pt.FIG_METRICS_ND.metric_markers import scatter_metric_markers

# --------------------------------------------------------------------------
ERROR_BAR_FACTOR = 5                     # same multiplier as big-pane
HLINE_ZORDER     = 5
DATA_ZORDER      = 6
THEORY_ZORDER    = 7


def _x_info(phi: float) -> Tuple[str, str]:
    """Return (csv-column-name, xlabel) for a given φ."""
    if np.isclose(phi, np.pi):      # anti-reciprocal
        return "Delta_f",  r"$\tilde \Delta_f$"
    # reciprocal or hybrid – we use Δκ
    if phi == 0:
        return "Delta_kappa", r"$\tilde \Delta_\kappa$"
    else:
        return "Delta_kappa", r"$\tilde \Delta_\kappa$ (Hyperbolic)"


def _split_data_from_csv(
        peaks: pd.DataFrame,
        x_key: str,
        j_scale: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Build (x, split, yerr) arrays from raw *dimensioned* peak CSV.

    • divide *x* and *ν* by J,
    • group by x-value to compute |ν₊ – ν₋|.
    """
    # dimension-less columns
    peaks = peaks.copy()
    peaks[x_key]      = peaks[x_key]      / j_scale
    peaks["peak_mean"] = peaks["peak_mean"] / j_scale
    peaks["err_low"]   = peaks["err_low"]   / j_scale * ERROR_BAR_FACTOR
    peaks["err_high"]  = peaks["err_high"]  / j_scale * ERROR_BAR_FACTOR

    # group measurements taken at identical x (two branches per Δκ or Δf)
    grouped = peaks.groupby(x_key, sort=True)

    x_vals, split_vals, err_low, err_high = [], [], [], []
    for x_val, grp in grouped:
        if len(grp) < 2:                                 # need both branches
            continue
        peak_vals = grp["peak_mean"].values
        order     = np.argsort(peak_vals)
        # splitting
        split = peak_vals[order[-1]] - peak_vals[order[0]]
        # combine the two asymmetric errors
        e_low  = grp["err_low" ].values[order[-1]] + grp["err_low" ].values[order[0]]
        e_high = grp["err_high"].values[order[-1]] + grp["err_high"].values[order[0]]

        x_vals   .append(x_val)
        split_vals.append(split)
        err_low  .append(e_low)
        err_high .append(e_high)

    x_vals     = np.asarray(x_vals)
    split_vals = np.asarray(split_vals)
    y_err      = np.vstack([err_low, err_high])          # shape (2, N)

    return x_vals, split_vals, y_err


def plot_splitting_pane(
        ax: plt.Axes,
        data_dir: Path,
        exp_id: str,
        phi: float,
        *,
        include_legend: bool = True,
        include_data_label: bool = True,
) -> None:
    """
    Draw the splitting pane for one experiment *exp_id* on the given Axes.
    CSV files must exist in ``data_dir``:

        • <id>_peaks.csv
        • <id>_theory.csv
        • <id>_params.csv
    """
    peaks   = pd.read_csv(data_dir / f"{exp_id}_peaks.csv")
    theory  = pd.read_csv(data_dir / f"{exp_id}_theory.csv")
    params  = pd.read_csv(data_dir / f"{exp_id}_params.csv").iloc[0]

    j_scale   = params["J"]
    kappa_c   = params["kappa_c"]             # in Hz
    delta_k   = params.get("Delta_kappa_std", np.nan)  # not needed here

    x_key, x_lab = _x_info(phi)

    # dimension-less theory split
    theory[x_key] = theory[x_key] / j_scale
    theory_split  = np.abs(theory["nu_plus"] - theory["nu_minus"]) / j_scale

    # ---- experimental splitting & errors ----
    x_exp, y_exp, y_err = _split_data_from_csv(peaks, x_key, j_scale)

    kappa_tilde_c = kappa_c / j_scale
    star_value = FigMetricStyle.star_kappa[phi]
    tri_value = FigMetricStyle.tri_kappa[phi]
    tpd = standard_tpd_locations(phi, kappa_tilde_c, left_tpd=True)

    if np.isclose(phi, np.pi):
        theory_label=r"Theory $\phi=\pi$"
    elif np.isclose(phi, np.pi/2.0):
        theory_label=r"Theory $\phi=\pi/2$"
    elif np.isclose(phi, 0):
        theory_label=r"Theory $\phi=0$"
    else:
        theory_label=r"Theory $\phi={phi}$"
    # What is closer to kapa_tilde_c, star or triangle?
    is_small = abs(star_value - kappa_tilde_c) < abs(tri_value - kappa_tilde_c)
    if is_small:
        marker_type = "o"                             # hollow circle
        marker_fill = "none"                          # hollow
        theory_line_style = '-'
        min_splitting_line_style = '--'
        theory_label = theory_label
        data_label = r"Small $\tilde \kappa_c$ (Data)"
        min_splitting_label = r"$\min(\tilde \Delta _\nu)$"
    else:
        marker_type = "^"                             # triangle
        marker_fill = "none"                # filled
        theory_line_style = '-'
        min_splitting_line_style = '--'
        theory_label = None
        min_splitting_label = None
        data_label = r"Large $\tilde \kappa_c$ (Data)"

    # ---- plot ----------------------------------------------------------
    theory_color = STYLE.theory_color_map(phi)
    ax.errorbar(
        x_exp, y_exp, yerr=y_err,
        fmt=marker_type,
        markersize=STYLE.data_ms,
        ecolor=theory_color, capsize=0,
        color=theory_color,
        markerfacecolor=marker_fill,
        markeredgewidth=1.25,
        linewidth=2,
        zorder=DATA_ZORDER,
        label=data_label,
    )
    ax.plot(
        theory[x_key], theory_split,
        color=theory_color, linewidth=STYLE.theory_lw,
        zorder=THEORY_ZORDER,
        label=theory_label if include_legend else None,
        linestyle=theory_line_style
    )

    # ---- minimum splitting dashed line --------------------------------
    # uncertainties already dimension-less (over-J) in composite driver
    j_scale_ghz = j_scale / 1e9
    df_unc = 1e-5 / j_scale_ghz
    dk_unc = 1e-5 / j_scale_ghz

    min_split_tilde = min_split_over_J(phi, kappa_tilde_c, df_unc, dk_unc)
    # More complicated horizontal line plotting
    x_vals = theory[x_key].values
    x_min, x_max = np.min(x_vals), np.max(x_vals)
    x_range = x_max - x_min
    radius = 0.2 * x_range   # or some fixed width like 0.3
    x_tpd = tpd.Delta_tilde_kappa if not np.isclose(phi, np.pi) else tpd.Delta_tilde_f
    xmin_hline = max(x_min, x_tpd - radius)
    xmax_hline = min(x_max, x_tpd + radius)

    ax.hlines(
        y=min_split_tilde,
        xmin=xmin_hline,
        xmax=xmax_hline,
        color=STYLE.min_split_color,
        linestyle=min_splitting_line_style,
        linewidth=STYLE.min_split_lw,
        zorder=HLINE_ZORDER,
        label=min_splitting_label if include_legend else None,
    )
    # optional star / marker at chosen κ̃
    # if phi in FigMetricStyle.star_kappa:
    #     scatter_metric_markers(
    #         ax, theory[x_key], theory_split,
    #         phi, theory_color, only_endpoint=True,
    #         branch_marker=marker_type
    #     )

    ax.set_xlabel(x_lab, fontsize=STYLE.label_font)
    ax.set_ylabel(r"$\tilde \Delta_\nu$", fontsize=STYLE.label_font)
    ax.tick_params(labelsize=STYLE.tick_font)

    # If phi val is close to np.pi/2, set the ylim equal to a bit higher than it is
    if np.isclose(phi, np.pi/2.0):
        # get current y lim, set the upper a bit higher
        y_min, y_max = ax.get_ylim()
        y_max = y_max + .2 * y_max
        ax.set_ylim(y_min, y_max)

    if include_legend:
        ax.legend(fontsize=STYLE.legend_font, framealpha=1.0, loc='upper left',  borderpad=.1, bbox_to_anchor=(-0.01, 1.02))

    # let caller set limits if desired
