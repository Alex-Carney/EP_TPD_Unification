"""
big_pane.py
Draw one large dimensionless panel.
Every y value is divided by J_scale and then shifted by  −f_c/J_scale.
"""

from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import matplotlib.transforms as mtrans
from scipy.ndimage import gaussian_filter1d

from figures.figure_3_experiment.experiment_tpds import standard_ep_locations
from figures.figure_3_experiment.style_maps import phase_peak_theory_color_map
from models.analysis import AnalyzedExperiment, TheoryDataPoint, AnalyzedAggregateTrace

LEGEND_FONT_SIZE = 13

VERT_W                           = 2.0
BASE_LINEWIDTH          = 2.5

# ----------------------------------------------------------------------
def forgotten_special_lines(ax, x_vals, phi_val, kappa_tilde_c):
    # Just in case the instability transition isn't included in the CSV, calculate it here
    # Example, for phi = 0, instability happens at Delta_kappa/J = kappa_tilde_c
    if phi_val == 0:
        lbl = r"$\tilde \Delta_\kappa^{NL}$"
        if kappa_tilde_c ** 2 - 4 < 0:
            # Find where x_vals == kappa_tilde_c
            ax.axvline(kappa_tilde_c, color="lime", ls="-", lw=VERT_W, label=lbl)
        else:
            target_x = (kappa_tilde_c**2 + 4) / (2 * kappa_tilde_c)
            ax.axvline(target_x, color="lime", ls="-", lw=VERT_W,  label=lbl)

def draw_special_lines(ax, specials, x_key, phi_val):
    """Vertical reference lines (already scaled by J)."""
    done = set()
    for _, row in specials.iterrows():
        name = row["name"]
        xval = row["x"]
        if name.startswith("TPD"):
            lbl = r"$\tilde \Delta_\kappa^{TPD}$" if not np.isclose(phi_val, np.pi) else r"$\tilde \Delta_f^{TPD}$"
            col = "cyan"
        elif name.startswith("EP"):
            lbl = r"$\tilde \Delta_\kappa^{EP}$" if not np.isclose(phi_val, np.pi)  else r"$\tilde \Delta_f^{EP}$"
            col = "red"
        elif name.startswith("Instab"):
            if phi_val == 0:
                lbl = r"$\tilde \Delta_\kappa^{NL}$"
            elif np.isclose(phi_val, np.pi):
                lbl = r"$\tilde \Delta_f^{NL}$"
            else:
                lbl = None
            col = "lime"
        else:
            continue

        if name.endswith("_low") or name.endswith("_high"):
            base = name.rsplit("_", 1)[0]
            if base in done:
                continue
            low = specials.loc[specials["name"] == base + "_low", "x"].values[0]
            high = specials.loc[specials["name"] == base + "_high", "x"].values[0]
            # ax.axvline(low,  color=col, ls="--", lw=1.0)
            # ax.axvline(high, color=col, ls="--", lw=1.0)
            # ax.axvline(low,  color=col, ls="--", lw=1.0,
            #            label=r"$\sigma_{exp}(" + lbl[1:-1] + ")$" if lbl else None)
            done.add(base)
        else:
            ax.axvline(xval, color=col, ls="-", lw=VERT_W, label=lbl)

# ----------------------------------------------------------------------
def _fmt(ax):
    fmt = mtick.FuncFormatter(lambda v, pos: f"{v:.2f}")
    ax.xaxis.set_major_formatter(fmt)
    ax.yaxis.set_major_formatter(fmt)

# ----------------------------------------------------------------------
def plot_big_pane(
        ax: plt.Axes,
        data_dir: Path,
        analyzed_experiment: AnalyzedExperiment,
        *,
        x_key: Literal["Delta_f", "Delta_kappa"],
        xlab: str,
        J_scale: float,
        f_c: float,
        kappa_c: float,
        phi_val: float,
        draw_unstable: bool = True,
        xlims: tuple[float, float] = None,
        ylims_freq: tuple[float, float] = None,
        include_legend: bool = True,
):


    scale   = J_scale
    # TPD_DELTA_F = np.sqrt(4 * J_scale**2 + kappa_c**2) # Only for Delta_f
    # offset  = f_c / scale  if x_key=="Delta_kappa" else (f_c - TPD_DELTA_F/2) / scale

    # tpd_loc_tilde = standard_tpd_locations(phi_val, kappa_c / scale)
    # tpd_df = tpd_loc_tilde.Delta_tilde_f * scale
    # offset = (f_c - tpd_df/2) / scale
    ep_loc_tilde = standard_ep_locations(phi_val) if not np.isclose(phi_val, np.pi) else standard_ep_locations(phi_val, left_ep=False)
    ep_df = ep_loc_tilde.Delta_tilde_f * scale
    offset = (f_c - ep_df/2) / scale

    peaks    = pd.read_csv(data_dir / f"{exp_id}_peaks.csv")
    theory   = pd.read_csv(data_dir / f"{exp_id}_theory.csv")
    specials = pd.read_csv(data_dir / f"{exp_id}_specials.csv")

    theory: list[TheoryDataPoint] = analyzed_experiment.theory_data_points
    peaks: list[AnalyzedAggregateTrace] = analyzed_experiment.analyzed_aggregate_traces

    tpd_loc = analyzed_experiment.TPD_location / scale
    ep_loc  = analyzed_experiment.EP_location / scale
    instab_loc = analyzed_experiment.Instability_location / scale if analyzed_experiment.Instability_location is not None else None

    theory_nu_plus = np.asarray([theory_point.nu_plus / scale - offset for theory_point in theory])
    theory_nu_minus = np.asarray([theory_point.nu_minus / scale  - offset  for theory_point in theory])
    df_theory = np.asarray([theory_point.Delta_f / scale for theory_point in theory])
    dk_theory = np.asarray([theory_point.Delta_kappa / scale for theory_point in theory])

    df_peaks = np.asarray([peak.Delta_f_Hz / scale for peak in peaks])
    dk_peaks = np.asarray([peak.Delta_kappa_Hz / scale for peak in peaks])
    nu_minus_mean = np.asarray([peak.nu_minus_mean_data_Hz / scale  - offset  for peak in peaks])
    nu_plus_mean = np.asarray([peak.nu_plus_mean_data_Hz / scale  - offset  for peak in peaks])
    nu_minus_err_low = np.asarray([peak.nu_minus_err_low_data_Hz / scale  - offset  for peak in peaks])
    nu_minus_err_high = np.asarray([peak.nu_minus_err_high_data_Hz / scale  - offset  for peak in peaks])
    nu_plus_err_low = np.asarray([peak.nu_plus_err_low_data_Hz / scale  - offset  for peak in peaks])
    nu_plus_err_high = np.asarray([peak.nu_plus_err_high_data_Hz / scale  - offset  for peak in peaks])

    # divide x by J
    for df in (peaks, theory):
        df[x_key] = df[x_key] / scale

    # divide and shift y arrays
    def _shift(df, col):
        if col in df.columns:
            df[col] = df[col] / scale - offset

    for df in (peaks, theory):
        for col in ["peak_mean", "nu_plus", "nu_minus",
                    "nu_plus_mc_low", "nu_plus_mc_high",
                    "nu_minus_mc_low", "nu_minus_mc_high"]:
            _shift(df, col)

    # Divide all specials["x"] by sc ale
    specials["x"] = specials["x"] / scale

    # errors only divide (no shift)
    for col in ["err_low", "err_high"]:
        if col in peaks.columns:
            peaks[col] = peaks[col] / scale


    ERROR_BAR_FACTOR = 5
    err_low = peaks["err_low"]
    err_high = peaks["err_high"]
    yerr_scaled = np.array([
        np.asarray(err_low) * ERROR_BAR_FACTOR,
        np.asarray(err_high) * ERROR_BAR_FACTOR
    ])
    # scatter
    ax.errorbar(
        peaks[x_key], peaks["peak_mean"],
        yerr=yerr_scaled,
        fmt="o", markersize=5, ecolor="black", capsize=0,
        alpha=1, color="black",
        markerfacecolor='none',
        markeredgecolor='black',
        label=r"$\tilde \nu_\pm$ Data" if phi_val == 0 else None
    )

    print(f'For phi = {phi_val}, J is {J_scale / 1e6} MHz, the range of ${x_key} goes from {np.min(peaks[x_key])} to {np.max(peaks[x_key])}')

    theory_color = phase_peak_theory_color_map(phi_val)

    theory_var = df_theory if x_key == "Delta_f" else dk_theory
    ax.plot(theory_var, theory_nu_plus,
            color=theory_color, linewidth=2)
    ax.plot(theory_var, theory_nu_minus,
            color=theory_color, linewidth=2,
            label=r"$\tilde \nu_\pm$")


    draw_special_lines(ax, specials, x_key, phi_val)
    forgotten_special_lines(ax, theory_var, phi_val, kappa_c / scale)

    ax.set_xlabel(xlab)
    ax.set_ylabel(r"(Frequency - $f_{EP}$)/J")

    if include_legend and kappa_c / J_scale < 1.5:
        if phi_val == 0:
            ax.set_xlim(-2.2e6 / scale, None)
            ax.legend(fontsize=LEGEND_FONT_SIZE,
                      loc="center right", framealpha=1, borderpad=0.1,)
        else:
            ax.legend(fontsize=LEGEND_FONT_SIZE,
                      loc="lower left", framealpha=1, borderpad=0.1)

    _fmt(ax)

    # unstable label (Delta_f)
    if x_key == "Delta_f":
        inst = specials[specials["name"].str.startswith("Instab") &
                        ~specials["name"].str.endswith(("_low", "_high"))]
        if not inst.empty:
            x_inst = inst["x"].iloc[0]
            span   = ax.get_xlim()[1] - ax.get_xlim()[0]
            offs   = 0.125 * span
            trans  = mtrans.blended_transform_factory(ax.transData, ax.transAxes)
            if draw_unstable:
                ax.text(x_inst - offs, 0.85, "unstable",
                        transform=trans, ha="right", va="top",
                        fontsize=LEGEND_FONT_SIZE + 4, fontweight="bold",
                        rotation=90)

    # Finally set the axis limits
    ax.set_xlim(xlims[0], xlims[1])
    ax.set_ylim(ylims_freq[0], ylims_freq[1])
