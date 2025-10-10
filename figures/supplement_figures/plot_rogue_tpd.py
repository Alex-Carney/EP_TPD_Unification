#!/usr/bin/env python3
"""
plot_rogue_tpd.py

Supplemental figure illustrating rogue TPDs. Left panel is the PF map for
phi = 0 and kappa_c_tilde = 2.5. Right panel sweeps Delta_f_tilde at fixed
Delta_kappa_tilde = 0 and overlays eigenvalue imaginaries and peak locations.
"""

from __future__ import annotations

import sys
from pathlib import Path
import importlib
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.ticker import FormatStrFormatter

# -----------------------------------------------------------------------------
# Repo imports
# -----------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from figures.large_figures.figure_2_theory import mesh_nd  # noqa: E402
from figures.large_figures.figure_2_theory.mesh_nd import get as get_mesh  # noqa: E402
from figures.large_figures.figure_2_theory import abs_contours  # noqa: E402
from figures.large_figures.figure_2_theory.tpd_locations_nd import (  # noqa: E402
    tpd_location,
    DegeneracyType,
)
fig2_settings = importlib.import_module("figures.large_figures.figure_2_theory.settings")  # noqa: E402
sys.modules.setdefault("settings", fig2_settings)
STYLE = fig2_settings.STYLE

from fitting.peak_fitting import peak_location, eigenvalues  # noqa: E402


# -----------------------------------------------------------------------------
# Global figure controls (edit these)
# -----------------------------------------------------------------------------
FIGSIZE       = (5*2.5, 5)   # overall figure size
WSPACE        = 0.5          # horizontal spacing between the two panels
HSPACE        = 0.20          # vertical spacing (unused for 1x2 but kept for symmetry)
LEFT          = 0.085          # figure margin left
RIGHT         = 0.99          # figure margin right
TOP           = 0.97          # figure margin top
BOTTOM        = 0.175          # figure margin bottom

FS_LABEL      = 26            # axis label font size
FS_TICKS      = 20            # tick label font size
FS_LEGEND     = 18            # legend font size
FS_CB_LABEL   = 26            # colorbar label font size
FS_CORNER     = 16            # corner tag font size
FS_PANEL_LAB  = 28           # (a), (b) font size
PANEL_LAB_WT  = "bold"
PANEL_LAB_X   = -0.12         # offset in axes fraction for panel label x
PANEL_LAB_Y   = 1.02          # offset in axes fraction for panel label y

LW_MAIN       = 2.0           # line width for main curves
LW_REF        = 1.9           # line width for reference lines

CB_GUTTER     = 0.01         # gap between left axes and colorbar axes (in figure coords)
CB_WIDTH      = 0.01         # colorbar width (in figure coords)

LEFT_XTICKS_N = 5             # number of x ticks on the left panel


# -----------------------------------------------------------------------------
# Physics/config
# -----------------------------------------------------------------------------
PHI = 0.0
KAPPA_TILDE_C = 2.5
DELTA_KAPPA_FIXED = 0.0
J_COUPLING = 1.0
F_C = 0.0

DELTA_F_RANGE = (-3.5, 3.5)
DELTA_F_SAMPLES = 1601
SPLIT_EPS = 1e-10


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------
def _break_at_transitions(y: np.ndarray, split: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """Insert NaNs when the peak splitting toggles between zero and non-zero."""
    y = y.copy()
    rising = np.where((split[1:] > eps) & (split[:-1] <= eps))[0] + 1
    falling = np.where((split[1:] <= eps) & (split[:-1] > eps))[0] + 1
    indices = np.unique(np.concatenate([rising, falling]))
    y[indices] = np.nan
    return y


def _compute_peak_data(delta_f_vals: np.ndarray) -> dict[str, np.ndarray]:
    """Return peak locations, eigenvalues, and splitting versus Delta_f_tilde."""
    nu_plus = np.full_like(delta_f_vals, np.nan, dtype=float)
    nu_minus = np.full_like(delta_f_vals, np.nan, dtype=float)
    eig_plus = np.full_like(delta_f_vals, np.nan, dtype=float)
    eig_minus = np.full_like(delta_f_vals, np.nan, dtype=float)

    for idx, df in enumerate(delta_f_vals):
        peaks = peak_location(J_COUPLING, F_C, KAPPA_TILDE_C, df, DELTA_KAPPA_FIXED, PHI)
        if len(peaks) == 2:
            lo, hi = sorted(float(val) for val in peaks)
        else:
            lo = hi = float(peaks[0])
        nu_minus[idx] = lo
        nu_plus[idx] = hi

        lam = eigenvalues(J_COUPLING, F_C, KAPPA_TILDE_C, df, DELTA_KAPPA_FIXED, PHI)
        eig_plus[idx] = float(np.imag(lam[1]))
        eig_minus[idx] = float(np.imag(lam[2]))

    splitting = nu_plus - nu_minus
    splitting[splitting < 0.0] = 0.0

    return {
        "delta_f": delta_f_vals,
        "nu_plus": _break_at_transitions(nu_plus, splitting, SPLIT_EPS),
        "nu_minus": _break_at_transitions(nu_minus, splitting, SPLIT_EPS),
        "eig_plus": eig_plus,
        "eig_minus": eig_minus,
    }


def _rogue_tpd_positions() -> list[float]:
    """Rogue TPD Delta_f_tilde locations for the chosen parameters."""
    degeneracies = tpd_location(PHI, KAPPA_TILDE_C)
    return [
        deg.Delta_tilde_f
        for deg in degeneracies
        if deg.degeneracy_type is DegeneracyType.ROGUE_TPD
           and np.isclose(deg.Delta_tilde_kappa, DELTA_KAPPA_FIXED)
    ]


def _add_panel_labels(fig: plt.Figure, axes, labels=("a", "b"), fs=FS_PANEL_LAB, weight=PANEL_LAB_WT):
    """Place bold panel labels just outside the top-left corner of each axes."""
    for ax, lab in zip(axes, labels):
        bbox = ax.get_position()
        fig.text(
            bbox.x0 - 0.05,
            bbox.y1 - 0.06,
            lab,
            fontsize=fs,
            fontweight=weight,
            ha="center",
            va="bottom",
        )


# -----------------------------------------------------------------------------
# Build
# -----------------------------------------------------------------------------
def build() -> Path:
    plt.rcParams.update(plt.rcParamsDefault)
    plt.rcParams["figure.facecolor"] = "white"
    plt.rcParams["axes.facecolor"] = "white"

    fig, (ax_left, ax_right) = plt.subplots(nrows=1, ncols=2, figsize=FIGSIZE, constrained_layout=False)

    # manage spacing and margins explicitly
    fig.subplots_adjust(left=LEFT, right=RIGHT, top=TOP, bottom=BOTTOM, wspace=WSPACE, hspace=HSPACE)

    # Left panel: PF map
    mesh = get_mesh(
        PHI,
        kappa_tilde_c=KAPPA_TILDE_C,
        delta_tilde_kappa_lim=(-4, 4),
        delta_tilde_f_lim=(-4, 4),
        N=1001,
    )
    mappable = abs_contours.plot(
        ax_left,
        mesh,
        PHI,
        figure_mode=False,
        legend=False,
        kappa_tilde_c=KAPPA_TILDE_C,
        return_mappable=True,
        include_p=False,
    )
    ax_left.set_xlabel(r"$\tilde{\Delta}_\kappa$", fontsize=FS_LABEL)
    ax_left.set_ylabel(r"$\tilde{\Delta}_f$", fontsize=FS_LABEL)
    ax_left.tick_params(labelsize=FS_TICKS)
    # Highlight the q = 0 trajectory we operate along (Δ̃_κ = 0).
    q_color = getattr(STYLE, "q_color", "magenta")
    q_linewidth = getattr(STYLE, "contour_lw", LW_MAIN * 1.4)
    ax_left.axvline(
        DELTA_KAPPA_FIXED,
        color=q_color,
        linewidth=q_linewidth,
        linestyle="-",
        zorder=4,  # above the filled contour, below markers
    )

    # force a sensible number of X ticks on the left plot and restore visible labels
    ax_left.xaxis.set_major_locator(mticker.MaxNLocator(LEFT_XTICKS_N))
    ax_left.xaxis.set_major_formatter(mticker.ScalarFormatter())
    ax_left.tick_params(axis="x", labelsize=FS_TICKS)

    # Right panel: eigenvalues and peaks vs Delta_f_tilde
    delta_f_vals = np.linspace(*DELTA_F_RANGE, DELTA_F_SAMPLES)
    curves = _compute_peak_data(delta_f_vals)

    ax_right.plot(
        curves["delta_f"],
        curves["eig_plus"],
        color="black",
        linestyle="--",
        linewidth=LW_MAIN,
        label=r"$\mathrm{Im}(\tilde{\lambda}_\pm)$",
    )
    ax_right.plot(
        curves["delta_f"],
        curves["eig_minus"],
        color="black",
        linestyle="--",
        linewidth=LW_MAIN
    )
    ax_right.plot(
        curves["delta_f"],
        curves["nu_plus"],
        color=q_color,
        linewidth=LW_MAIN * 1.25,
        linestyle="-",
        label=r"$\tilde{\nu}_\pm^{\text{Root}}$",
    )
    ax_right.plot(
        curves["delta_f"],
        curves["nu_minus"],
        color=q_color,
        linewidth=LW_MAIN * 1.25,
        linestyle="-",
        label="_nolegend_",
    )

    for tpd_f in _rogue_tpd_positions():
        ax_right.axvline(tpd_f, color="cyan", linewidth=LW_REF, linestyle="-", label="Rogue TPD")

    ax_right.set_xlabel(r"$\tilde{\Delta}_f$", fontsize=FS_LABEL)
    ax_right.set_ylabel("Frequency / J", fontsize=FS_LABEL)
    ax_right.tick_params(labelsize=FS_TICKS)
    ax_right.set_xlim(DELTA_F_RANGE)
    ax_right.set_ylim(-3.5, 3.5)
    # mesh_nd.corner_tag(ax_right, PHI, KAPPA_TILDE_C)

    # Legend (deduped)
    handles, labels = ax_right.get_legend_handles_labels()
    uniq_h, uniq_l = [], []
    seen = set()
    for h, l in zip(handles, labels):
        if l not in seen:
            uniq_h.append(h)
            uniq_l.append(l)
            seen.add(l)
    ax_right.legend(uniq_h, uniq_l, fontsize=FS_LEGEND, loc="lower left", framealpha=0.95, borderpad=0.2)

    # Colorbar in a fixed gutter to the right of the left axes
    left_pos = ax_left.get_position()
    cax = fig.add_axes([left_pos.x1 + CB_GUTTER, left_pos.y0, CB_WIDTH, left_pos.height])
    cb = fig.colorbar(mappable, cax=cax)
    cb.set_label("PF (Clipped)", fontsize=FS_CB_LABEL)
    cb.ax.tick_params(labelsize=FS_TICKS)
    vmin, vmax = mappable.get_clim()
    vmin = max(1.0, vmin)
    cb.set_ticks(np.linspace(vmin, vmax, 5))
    cb.ax.yaxis.set_major_formatter(FormatStrFormatter("%.1f"))

    # Panel labels
    _add_panel_labels(fig, (ax_left, ax_right), labels=("a", "b"))

    out_dir = Path(__file__).resolve().parents[1] / ".figures"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "SUPP_rogue_tpd.png"
    fig.savefig(out_path, dpi=400, facecolor="white")
    plt.close(fig)
    return out_path


if __name__ == "__main__":
    build()
