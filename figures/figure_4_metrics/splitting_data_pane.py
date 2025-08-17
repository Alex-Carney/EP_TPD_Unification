"""
splitting_data_pane.py

Plot a single dimensionless peak-splitting pane:

    x-axis  :  Delta_kappa/J (phi = 0, pi/2) or Delta_f/J (phi = pi)
    y-axis  :  (nu_plus - nu_minus)/J
    points  :  experimental data (error bars scaled)
    curve   :  theory splitting from TheoryDataPoint
    h-line  :  min-split from min_split_over_J

All colors and marker sizes come from settings.STYLE.
"""

from __future__ import annotations

from typing import Tuple
import numpy as np
import matplotlib.pyplot as plt

from figures.figure_3_experiment.experiment_tpds import standard_tpd_locations
from settings import FigMetricStyle
from settings import STYLE
from metric_calculations import min_split_over_J
from metric_markers import scatter_metric_markers

from models.analysis import AnalyzedExperiment, TheoryDataPoint, AnalyzedAggregateTrace

# --------------------------------------------------------------------------
ERROR_BAR_FACTOR = 5
HLINE_ZORDER     = 5
DATA_ZORDER      = 6
THEORY_ZORDER    = 7

# Very small threshold to suppress only truly coalesced branches
# Keep this tiny so we do not hide valid early split data
SPLIT_MIN_TILDE = 1e-4


def _x_info(phi: float) -> Tuple[str, str]:
    """Return (key, xlabel) for a given phi."""
    if np.isclose(phi, np.pi):
        return "Delta_f",  r"$\tilde \Delta_f$"
    if np.isclose(phi, 0.0):
        return "Delta_kappa", r"$\tilde \Delta_\kappa$"
    return "Delta_kappa", r"$\tilde \Delta_\kappa$ (Hyperbolic)"


def _split_data_from_models(
    peaks: list[AnalyzedAggregateTrace],
    x_key: str,
    j_scale: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Build (x, split, yerr) from ORM rows.

    Mirrors CSV behavior, but each AnalyzedAggregateTrace already contains
    both branches. So there is no need to group by x to find a pair.
    We compute splitting at each trace and combine branch errors.
    """
    x_vals = []
    split_vals = []
    err_low = []
    err_high = []

    for p in peaks:
        vals = [
            p.nu_plus_mean_data_Hz, p.nu_minus_mean_data_Hz,
            p.nu_plus_err_low_data_Hz, p.nu_minus_err_low_data_Hz,
            p.nu_plus_err_high_data_Hz, p.nu_minus_err_high_data_Hz,
            p.Delta_f_Hz, p.Delta_kappa_Hz,
        ]
        if any(v is None for v in vals):
            continue
        # coerce to float and check finiteness
        try:
            nums = [float(v) for v in vals]
        except Exception:
            continue
        if any(not np.isfinite(v) for v in nums):
            continue

        x_dimless = (p.Delta_f_Hz if x_key == "Delta_f" else p.Delta_kappa_Hz) / j_scale

        # dimensionless means
        nu_p = p.nu_plus_mean_data_Hz / j_scale
        nu_m = p.nu_minus_mean_data_Hz / j_scale
        split = abs(nu_p - nu_m)

        # skip only truly coalesced branches
        if split <= SPLIT_MIN_TILDE:
            continue

        # asymmetric errors are distances from the mean for each branch
        # then combined across the two branches, like the CSV did
        e_low_plus  = abs((p.nu_plus_mean_data_Hz  - p.nu_plus_err_low_data_Hz)  / j_scale)
        e_low_minus = abs((p.nu_minus_mean_data_Hz - p.nu_minus_err_low_data_Hz) / j_scale)
        e_high_plus  = abs((p.nu_plus_err_high_data_Hz  - p.nu_plus_mean_data_Hz)  / j_scale)
        e_high_minus = abs((p.nu_minus_err_high_data_Hz - p.nu_minus_mean_data_Hz) / j_scale)

        e_low_combined  = (e_low_plus  + e_low_minus)  * ERROR_BAR_FACTOR
        e_high_combined = (e_high_plus + e_high_minus) * ERROR_BAR_FACTOR

        x_vals.append(x_dimless)
        split_vals.append(split)
        err_low.append(e_low_combined)
        err_high.append(e_high_combined)

    if not x_vals:
        return np.array([]), np.array([]), np.zeros((2, 0))

    # Preserve original order. Do not sort. Your axis logic controls ranges.
    return (
        np.asarray(x_vals),
        np.asarray(split_vals),
        np.vstack([np.asarray(err_low), np.asarray(err_high)]),
    )


def plot_splitting_pane(
        ax: plt.Axes,
        analyzed_experiment: AnalyzedExperiment,
        phi: float,
        *,
        include_legend: bool = True,
        include_data_label: bool = True,
) -> None:
    """
    Draw the splitting pane for one experiment from ORM data.
    """
    j_scale = analyzed_experiment.J_avg
    kappa_c = analyzed_experiment.kappa_c_avg

    x_key, x_lab = _x_info(phi)

    # theory arrays (dimensionless). Keep DB order to match your previous axis behavior.
    theory: list[TheoryDataPoint] = analyzed_experiment.theory_data_points
    t_x = np.asarray([(tp.Delta_f if x_key == "Delta_f" else tp.Delta_kappa) / j_scale for tp in theory])
    t_split = np.asarray([abs(tp.nu_plus - tp.nu_minus) / j_scale for tp in theory])

    # filter only non-finite values in theory, do not sort, do not threshold away real data
    if t_x.size:
        mask_t = np.isfinite(t_x) & np.isfinite(t_split)
        t_x = t_x[mask_t]
        t_split = t_split[mask_t]

    # experimental arrays from DB (dimensionless)
    peaks: list[AnalyzedAggregateTrace] = analyzed_experiment.analyzed_aggregate_traces
    x_exp, y_exp, y_err = _split_data_from_models(peaks, x_key, j_scale)

    # decide labels and styles based on kappa_tilde_c proximity to STYLE target
    kappa_tilde_c = kappa_c / j_scale
    star_value = FigMetricStyle.star_kappa[phi]
    tri_value = FigMetricStyle.tri_kappa[phi]
    is_small = abs(star_value - kappa_tilde_c) < abs(tri_value - kappa_tilde_c)
    if is_small:
        marker_type = "o"
        marker_fill = "none"
        theory_line_style = "-"
        min_splitting_line_style = "--"
        if np.isclose(phi, np.pi):
            theory_label = r"Theory $\phi=\pi$"
        elif np.isclose(phi, np.pi/2.0):
            theory_label = r"Theory $\phi=\pi/2$"
        elif np.isclose(phi, 0.0):
            theory_label = r"Theory $\phi=0$"
        else:
            theory_label = r"Theory"
        data_label = r"Small $\tilde \kappa_c$ (Data)"
        min_splitting_label = r"$\min(\tilde \Delta _\nu)$"
    else:
        marker_type = "^"
        marker_fill = "none"
        theory_line_style = "-"
        min_splitting_line_style = "--"
        theory_label = None
        min_splitting_label = None
        data_label = r"Large $\tilde \kappa_c$ (Data)"

    theory_color = STYLE.theory_color_map(phi)

    # plot experimental points
    if x_exp.size:
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
            label=data_label if include_data_label else None,
        )

    # plot theory split
    if t_x.size:
        ax.plot(
            t_x, t_split,
            color=theory_color, linewidth=STYLE.theory_lw,
            zorder=THEORY_ZORDER,
            label=theory_label if include_legend else None,
            linestyle=theory_line_style
        )

    # horizontal min-split near the TPD x-location
    # uncertainties are already dimensionless in the composite driver, but we match your prior approach
    j_scale_ghz = j_scale / 1e9
    df_unc = 1e-5 / j_scale_ghz
    dk_unc = 1e-5 / j_scale_ghz
    min_split_tilde = min_split_over_J(phi, kappa_tilde_c, df_unc, dk_unc)

    # choose a span centered on the TPD similar to your old logic
    tpd = standard_tpd_locations(phi, kappa_tilde_c, left_tpd=True)
    x_vals_for_span = t_x if t_x.size else x_exp
    if x_vals_for_span.size:
        x_min, x_max = float(np.min(x_vals_for_span)), float(np.max(x_vals_for_span))
        x_range = x_max - x_min
        radius = 0.2 * x_range if x_range > 0 else 0.2
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

    ax.set_xlabel(x_lab, fontsize=STYLE.label_font)
    ax.set_ylabel(r"$\tilde \Delta_\nu$", fontsize=STYLE.label_font)
    ax.tick_params(labelsize=STYLE.tick_font)

    # match your old ylim tweak for phi=pi/2 only
    if np.isclose(phi, np.pi/2.0):
        y_min, y_max = ax.get_ylim()
        ax.set_ylim(y_min, y_max + 0.2 * y_max)

    if include_legend:
        ax.legend(fontsize=STYLE.legend_font, framealpha=1.0, loc="upper left",
                  borderpad=.1, bbox_to_anchor=(-0.01, 1.02))
