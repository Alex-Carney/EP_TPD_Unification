#!/usr/bin/env python3
"""
FINAL FIGURE: 6 panels (2 rows × 3 columns)

Layout:
  Row 0 (a):  (a-i) Disc κ̃_c=1.5   (a-ii) Disc κ̃_c=2.0   (a-iii) Disc κ̃_c=2.5
  Row 1 (b):  (b-i) Theory κ̃_c=1.5  (b-ii) Theory κ̃_c=2.0  (b-iii) Theory κ̃_c=2.5

Each column shares X between disc and theory.
All disc panels share Y. All theory panels share Y.
X-axis is dimensionless (Δ̃κ − Δ̃κ_TPD).
Theory Y-axis is Frequency / J (dimensionless).

Colors: κ̃_c=1.5 = CRIMSON, κ̃_c=2.0 = ROYALBLUE, κ̃_c=2.5 = FORESTGREEN
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.lines import Line2D
import matplotlib.patches as mpatches

# =============================================================================
# OUTPUT CONFIGURATION
# =============================================================================
OUTPUT_DIR = Path("results")
OUTPUT_FILENAME = "FIG_5_6panel.png"
OUTPUT_FILENAME_SVG = "FIG_5_6panel.svg"
DPI = 400

# =============================================================================
# FIGURE LAYOUT
# =============================================================================
FIGURE_SIZE = (18, 6)

# Labels: (a) disc row, (b) theory row
PANEL_LABELS = {
    "disc_row": "(a)",
    "theory_row": "(b)",
}
PANEL_LABEL_FONTSIZE = 30
PANEL_LABEL_FONTWEIGHT = "bold"
PANEL_LABEL_X = -0.13
PANEL_LABEL_Y = 0.93

# Sub-labels inside panels
SUBLABEL_FONTSIZE = 21
SUBLABEL_FONTWEIGHT = "bold"
SUBLABEL_X = 0.03
SUBLABEL_Y = 0.985

# Height ratio: disc row vs theory row
HEIGHT_RATIO_DISC = 0.38
HEIGHT_RATIO_THEORY = .85

# =============================================================================
# STYLING - FONTS
# =============================================================================
FONTSIZE_AXIS_LABEL = 29
FONTSIZE_TICK = 22
FONTSIZE_LEGEND = 19
FONTSIZE_ANNOTATION = 21
FONTSIZE_DISC_AXIS = 26
FONTSIZE_DISC_TICK = 22
FONTSIZE_DISC_LEGEND = 19

# =============================================================================
# STYLING - LINES
# =============================================================================
LW_THEORY = 4
LW_TPD = 2.8
LW_DISC_CONTOUR = 2.6
LW_SIGMA_LINE = 2.4

# =============================================================================
# STYLING - COLORS
# =============================================================================
COLOR_TPD = "cyan"
COLOR_DISC = "cyan"
COLOR_SIGMA_LINES = "darkorange"
COLOR_DISC_BAND = "darkorange"
ALPHA_SPREAD = 0.25
ALPHA_DISC_BAND = 0.22

# Per-column colors (left, middle, right)
COLUMN_COLORS = ["crimson", "royalblue", "forestgreen"]

# =============================================================================
# PHYSICS CONFIGURATION
# =============================================================================
PHI = 0.0
F_C = 0.0

# Three κ̃_c values (left → right)
KAPPA_C_VALUES = [1.5, 2.0, 2.5]
KAPPA_C_SUFFIXES = ["", "", ""]   # no suffix needed now

THEORY_CONFIG = {
    # Sweep range in DIMENSIONLESS units (relative to TPD)
    "dk_range_from_tpd_dimless": (-4e-3, 16e-3),
    "n_dk_points": 2501,
    "enable_noise_spread": True,
    "n_noise_samples": 10_000,
    "additive_noise_std": 0.000,
    "parametric_noise_dk_std": 0.0000,
    "parametric_noise_df_std": 1e-3,
    # X-axis limits (dimensionless, relative to TPD)
    "xlim_dimless": (-3e-3, 14e-3),
    # Y-axis limits (dimensionless Frequency/J)
    "ylim_dimless": (-0.15, 0.15),
}

# =============================================================================
# DISC CONTOUR PANELS
# =============================================================================
SIGMA_DF_DIMLESS = 1e-3
DISC_GRID_N = 1201
DISC_Y_HALFSPAN_DIMLESS = 1.5e-3


# =============================================================================
# PHYSICS
# =============================================================================

def tpd_location_formula(kappa_c: float) -> float:
    if kappa_c >= math.sqrt(8):
        raise ValueError(f"kappa_c={kappa_c} >= sqrt(8)")
    return kappa_c / 2.0 - math.sqrt(8.0 - kappa_c**2) / 2.0


def _all_roots(delta_f: float, delta_kappa: float, kappa_c: float, phi: float) -> np.ndarray:
    cos_phi = float(np.cos(phi))
    sin_phi = float(np.sin(phi))
    kappa_bar = kappa_c - delta_kappa

    p = (-(delta_f / 2.0)**2 + (delta_kappa / 2.0)**2
         - cos_phi + (kappa_bar / 2.0)**2)
    q = (kappa_bar / 4.0) * (delta_f * delta_kappa - 2.0 * sin_phi)

    roots = np.roots([1.0, 0.0, p, q]).astype(complex)
    roots = roots + (F_C - delta_f / 2.0)

    real_mask = np.abs(np.imag(roots)) < 1e-8
    real_roots = np.real(roots[real_mask]) if real_mask.any() else np.real(roots)
    real_roots.sort()

    if real_roots.size == 1:
        return np.repeat(real_roots, 3)
    if real_roots.size == 2:
        return np.array([real_roots[0], real_roots[0], real_roots[1]])
    return real_roots[:3]


def discriminant_grid(dk: np.ndarray, df: np.ndarray,
                      kappa_c: float, phi: float) -> np.ndarray:
    dk = np.asarray(dk, dtype=float)
    df = np.asarray(df, dtype=float)
    cp = float(np.cos(phi))
    sp = float(np.sin(phi))
    k = float(kappa_c)

    t1 = cp + (df**2)/4.0 - (dk**2)/4.0 - ((dk - k)**2)/4.0
    t2 = (dk - k)**2
    t3 = (4.0*sp - 2.0*df*dk)**2
    return 4.0*(t1**3) - (27.0*t2*t3)/64.0


def compute_theory_curves_dimless(
    kappa_c: float, dk_dimless_vals: np.ndarray,
) -> Dict[str, np.ndarray]:
    n = len(dk_dimless_vals)
    df_tilde = 1e-6

    nu_p = np.zeros(n)
    nu_m = np.zeros(n)
    for i, dk in enumerate(dk_dimless_vals):
        roots = _all_roots(df_tilde, dk, kappa_c, PHI)
        nu_m[i] = roots[0]
        nu_p[i] = roots[2]

    return {"nu_plus": nu_p, "nu_minus": nu_m}


def compute_peak_spread_dimless(
    kappa_c: float, dk_dimless_vals: np.ndarray,
    n_samples: int, additive_std: float,
    param_dk_std: float, param_df_std: float,
) -> Dict[str, np.ndarray]:
    rng = np.random.default_rng(42)
    n_dk = len(dk_dimless_vals)
    df_base = 1e-6

    nu_plus_mean = np.zeros(n_dk)
    nu_plus_std = np.zeros(n_dk)
    nu_minus_mean = np.zeros(n_dk)
    nu_minus_std = np.zeros(n_dk)

    for i, dkt in enumerate(dk_dimless_vals):
        sp, sm = [], []
        for _ in range(n_samples):
            dk_n = dkt + rng.normal(0, param_dk_std)
            df_n = df_base + rng.normal(0, param_df_std)
            roots = _all_roots(df_n, dk_n, kappa_c, PHI)
            sm.append(roots[0] + rng.normal(0, additive_std))
            sp.append(roots[2] + rng.normal(0, additive_std))
        nu_plus_mean[i] = np.mean(sp)
        nu_plus_std[i] = np.std(sp, ddof=1)
        nu_minus_mean[i] = np.mean(sm)
        nu_minus_std[i] = np.std(sm, ddof=1)

    return {
        "nu_plus_mean": nu_plus_mean, "nu_plus_std": nu_plus_std,
        "nu_minus_mean": nu_minus_mean, "nu_minus_std": nu_minus_std,
    }


# =============================================================================
# PANEL: DISC CONTOUR
# =============================================================================

def draw_disc_panel(
    ax: plt.Axes,
    kappa_c: float,
    show_legend: bool,
) -> None:
    dk_tpd = tpd_location_formula(kappa_c)

    tcfg = THEORY_CONFIG
    xlim = tcfg["xlim_dimless"]

    dk_dimless = np.linspace(
        dk_tpd + xlim[0],
        dk_tpd + xlim[1],
        DISC_GRID_N,
    )
    y_half = DISC_Y_HALFSPAN_DIMLESS
    df_dimless = np.linspace(-y_half, y_half, DISC_GRID_N)

    DK, DF = np.meshgrid(dk_dimless, df_dimless)
    disc = discriminant_grid(DK, DF, kappa_c, PHI)

    # Plot axes: x = Δ̃κ − Δ̃κ_TPD (dimensionless), y = Δ̃f (dimensionless)
    x_plot = dk_dimless - dk_tpd
    y_plot = df_dimless

    X_P, Y_P = np.meshgrid(x_plot, y_plot)

    # Disc = 0 contour
    ax.contour(X_P, Y_P, disc, levels=[0.0],
               colors=COLOR_DISC, linewidths=LW_DISC_CONTOUR, linestyles="--")

    # ±σ band
    sigma = SIGMA_DF_DIMLESS
    ax.fill_between(x_plot, -sigma, sigma,
                    color=COLOR_DISC_BAND, alpha=ALPHA_DISC_BAND, zorder=1)
    # ax.axhline(+sigma, color=COLOR_SIGMA_LINES, lw=LW_SIGMA_LINE,
    #            solid_capstyle="round")
    # ax.axhline(-sigma, color=COLOR_SIGMA_LINES, lw=LW_SIGMA_LINE,
    #            solid_capstyle="round")

    # TPD marker
    ax.scatter([0.0], [0.0], s=130, facecolors="none",
               edgecolors="red", linewidths=2.6, zorder=5)

    # Zero line
    ax.axhline(0.0, color="lightgray", lw=0.8, ls="--", zorder=0)

    ax.set_xlim(xlim)
    ax.set_ylim(-y_half, y_half)
    ax.tick_params(labelsize=FONTSIZE_DISC_TICK)
    ax.tick_params(axis="x", labelbottom=False)

    # Legend only on middle panel (κ̃_c = 2.0)
    if show_legend:
        handles = [
            Line2D([0], [0], marker="o", mfc="none", mec="red",
                   lw=0, ms=10, markeredgewidth=2.2, label="TPD"),
            Line2D([0, 1], [0, 1], color=COLOR_DISC, lw=LW_DISC_CONTOUR,
                   ls="--", label="Disc = 0"),
            # Line2D([0, 1], [0, 1], color=COLOR_SIGMA_LINES,
            #        lw=LW_SIGMA_LINE, ls="-",
            #        label=r"$\tilde{\Delta}_f = \pm10^{-3}$"),
            mpatches.Patch(fc=COLOR_DISC_BAND, alpha=ALPHA_DISC_BAND,
                           label=r"$\pm1\sigma(\tilde{\delta}(\tilde{\Delta}_f))$"),
        ]
        ax.legend(handles=handles, loc="upper right",
                  fontsize=FONTSIZE_DISC_LEGEND, framealpha=1.0,
                  edgecolor="lightgray", borderpad=0.2, labelspacing=0.1,
                  handlelength=1.8, handletextpad=0.5,
                  bbox_to_anchor=(1.015, 1.06))


# =============================================================================
# PANEL: THEORY — dimensionless axes
# =============================================================================

def draw_theory_panel(
    ax: plt.Axes,
    kappa_c: float,
    color: str,
    is_left: bool,
    show_legend: bool = False,
) -> None:
    tcfg = THEORY_CONFIG

    dk_tpd = tpd_location_formula(kappa_c)

    dk_range = tcfg["dk_range_from_tpd_dimless"]
    dk_dimless_abs = np.linspace(
        dk_tpd + dk_range[0],
        dk_tpd + dk_range[1],
        tcfg["n_dk_points"],
    )
    dk_rel = dk_dimless_abs - dk_tpd   # relative to TPD

    theory = compute_theory_curves_dimless(kappa_c, dk_dimless_abs)
    nu_plus = theory["nu_plus"]
    nu_minus = theory["nu_minus"]

    ax.plot(dk_rel, nu_plus, "-", color=color, lw=LW_THEORY,
            label=r"$\tilde\nu_\pm$")
    ax.plot(dk_rel, nu_minus, "-", color=color, lw=LW_THEORY,
            label="_nolegend_")

    if tcfg["enable_noise_spread"]:
        print(f"    Computing noise spread ({tcfg['n_noise_samples']} MC samples)...")
        spread = compute_peak_spread_dimless(
            kappa_c, dk_dimless_abs,
            tcfg["n_noise_samples"],
            tcfg["additive_noise_std"],
            tcfg["parametric_noise_dk_std"],
            tcfg["parametric_noise_df_std"],
        )

        ax.fill_between(dk_rel,
                        spread["nu_plus_mean"] - spread["nu_plus_std"],
                        spread["nu_plus_mean"] + spread["nu_plus_std"],
                        alpha=ALPHA_SPREAD, color=color,
                        label=r"$\pm 1\sigma(\tilde \nu_\pm)$")
        ax.fill_between(dk_rel,
                        spread["nu_minus_mean"] - spread["nu_minus_std"],
                        spread["nu_minus_mean"] + spread["nu_minus_std"],
                        alpha=ALPHA_SPREAD, color=color)

    ax.axhline(0, color="lightgray", lw=1, ls="--", zorder=0)
    ax.axvline(0, color=COLOR_TPD, lw=LW_TPD, ls="-", label="TPD")

    ax.set_xlabel(
        r"$\tilde{\Delta}_\kappa - \tilde{\Delta}_\kappa^{\mathrm{TPD}}$",
        fontsize=FONTSIZE_AXIS_LABEL)
    ax.tick_params(labelsize=FONTSIZE_TICK)

    if is_left:
        ax.set_ylabel(r"Frequency / $J$", fontsize=FONTSIZE_AXIS_LABEL)

    if tcfg["xlim_dimless"]:
        ax.set_xlim(tcfg["xlim_dimless"])
    if tcfg["ylim_dimless"]:
        ax.set_ylim(tcfg["ylim_dimless"])

    if show_legend:
        ax.legend(loc="lower right", fontsize=FONTSIZE_LEGEND,
                  framealpha=1.0, bbox_to_anchor=(1.019, -0.02))

    ax.text(0.97, 0.96,
            rf"$\tilde{{\kappa}}_c = {kappa_c}$",
            transform=ax.transAxes, fontsize=FONTSIZE_ANNOTATION,
            ha="right", va="top",
            bbox=dict(boxstyle="round", facecolor="white", alpha=1.0,
                      edgecolor="gray"))


# =============================================================================
# MAIN
# =============================================================================

def build_figure():
    plt.rcParams.update(plt.rcParamsDefault)
    plt.rcParams["figure.facecolor"] = "white"
    plt.rcParams["axes.facecolor"] = "white"

    print("=" * 60)
    print("BUILDING 6-PANEL FIGURE (2×3)")
    print("=" * 60)

    fig = plt.figure(figsize=FIGURE_SIZE)

    gs = GridSpec(
        2, 3,
        figure=fig,
        height_ratios=[HEIGHT_RATIO_DISC, HEIGHT_RATIO_THEORY],
        width_ratios=[1.0, 1.0, 1.0],
        left=0.08,
        right=0.99,
        bottom=0.165,
        top=0.945,
        wspace=0.03,
        hspace=0.06,
    )

    # Create axes with shared Y per row and shared X per column
    ax_disc = [None, None, None]
    ax_theory = [None, None, None]

    # Column 0 (left): masters
    ax_disc[0] = fig.add_subplot(gs[0, 0])
    ax_theory[0] = fig.add_subplot(gs[1, 0], sharex=ax_disc[0])

    # Column 1 (middle): share Y with col 0
    ax_disc[1] = fig.add_subplot(gs[0, 1], sharey=ax_disc[0])
    ax_theory[1] = fig.add_subplot(gs[1, 1], sharex=ax_disc[1], sharey=ax_theory[0])

    # Column 2 (right): share Y with col 0
    ax_disc[2] = fig.add_subplot(gs[0, 2], sharey=ax_disc[0])
    ax_theory[2] = fig.add_subplot(gs[1, 2], sharex=ax_disc[2], sharey=ax_theory[0])

    # Hide Y ticks on non-leftmost panels
    for col in [1, 2]:
        plt.setp(ax_disc[col].get_yticklabels(), visible=False)
        plt.setp(ax_theory[col].get_yticklabels(), visible=False)

    # Y label + ticks for disc (left only)
    ax_disc[0].set_ylabel(r"$\tilde{\Delta}_f$", fontsize=FONTSIZE_DISC_AXIS, labelpad=-14)
    ax_disc[0].set_yticks([-1e-3, 0, 1e-3])
    ax_disc[0].set_yticklabels([r"$-10^{-3}$", r"$0$", r"$10^{-3}$"])
    ax_disc[0].tick_params(axis='y', pad=3)

    # ---- Draw panels ----
    sub_labels = ["i", "ii", "iii"]

    for col, kc in enumerate(KAPPA_C_VALUES):
        color = COLUMN_COLORS[col]

        print(f"\n  Drawing column {col+1}: κ̃_c = {kc}...")

        # Legend only on left panel (a.i)
        show_legend = (col == 0)
        draw_disc_panel(ax_disc[col], kc, show_legend=show_legend)

        # Theory panel (row b) — legend only on left (b.i)
        is_left = (col == 0)
        draw_theory_panel(ax_theory[col], kc, color, is_left=is_left,
                          show_legend=is_left)

    # ---- Labels ----
    # (a) on left disc panel
    ax_disc[0].text(PANEL_LABEL_X, PANEL_LABEL_Y, PANEL_LABELS["disc_row"],
                    transform=ax_disc[0].transAxes,
                    fontsize=PANEL_LABEL_FONTSIZE,
                    fontweight=PANEL_LABEL_FONTWEIGHT,
                    ha="right", va="bottom")

    # (b) on left theory panel
    ax_theory[0].text(PANEL_LABEL_X, PANEL_LABEL_Y, PANEL_LABELS["theory_row"],
                      transform=ax_theory[0].transAxes,
                      fontsize=PANEL_LABEL_FONTSIZE,
                      fontweight=PANEL_LABEL_FONTWEIGHT,
                      ha="right", va="bottom")

    # Sub-labels i/ii/iii inside each panel
    for col in range(3):
        for ax in [ax_disc[col], ax_theory[col]]:
            ax.text(SUBLABEL_X, SUBLABEL_Y, sub_labels[col],
                    transform=ax.transAxes,
                    fontsize=SUBLABEL_FONTSIZE,
                    fontweight=SUBLABEL_FONTWEIGHT,
                    ha="left", va="top")

    # Save
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUTPUT_DIR / OUTPUT_FILENAME
    fig.savefig(out_path, dpi=DPI, facecolor="white")
    fig.savefig(OUTPUT_DIR / OUTPUT_FILENAME_SVG, facecolor="white")
    plt.close(fig)

    print(f"\nSaved figure to: {out_path}")
    print("=" * 60)
    return out_path


if __name__ == "__main__":
    build_figure()
