#!/usr/bin/env python3
"""
combined_ep_tpd_grid.py

Top row: Perfect EP (left) and Imperfect EP (right) showing |Im(lambda_+)| and |Im(lambda_-)| vs Delta_kappa.
Rows 2-4: TPD peak locations and splitting for Perfect, Imperfect, and Robust TPD scenarios.

Styling, fonts, and reference lines are consistent across the full 4x2 figure.
Row-specific x sweeps are used where needed. NaNs are inserted to break trajectories at teleport boundaries.
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import matplotlib.pyplot as plt

from fitting.transition_fitting import TPD_location
from fitting.peak_fitting import eigenvalues

# -----------------------------------------------------------------------------
# Shared configuration
# -----------------------------------------------------------------------------
J_COUPLING = 1.0
F_C = 0.0

# colors and fonts
LINECOLOR_REF = "c"           # vertical reference lines
LINECOLOR_REF_EP = "r"        # EP reference line
LINECOLOR_MAIN = "k"          # main curves
HILITE_COLOR = "forestgreen"  # highlight color used in TPD middle row

FS_TICKS = 20
FS_LABEL = 22
FS_SCENARIO = 18
FS_MIN_TEXT = 20
FS_LEGEND = 19

# -----------------------------------------------------------------------------
# EP row configuration (row 0)
# -----------------------------------------------------------------------------
PHI_EP = 0.0
DELTA_KAPPA_EP = np.linspace(-2.025, -1.975, 4001)
EP_POSITIONS = (-2.0 * J_COUPLING, 2.0 * J_COUPLING)  # draw left one only in-range

EP_SCENARIOS = (
    {"name": "Perfect EP",   "subtitle": r"$\tilde{\Delta}_f = 0$",       "delta_f": 0.0},
    {"name": "Imperfect EP", "subtitle": r"$\tilde{\Delta}_f = 10^{-3}$", "delta_f": 1.0e-3},
)

def eigenvalue_magnitudes(delta_f: float) -> tuple[np.ndarray, np.ndarray]:
    """Return |Im(lambda_+)| and |Im(lambda_-)| over the EP sweep."""
    plus_vals = np.empty_like(DELTA_KAPPA_EP)
    minus_vals = np.empty_like(DELTA_KAPPA_EP)
    for idx, dk in enumerate(DELTA_KAPPA_EP):
        lambdas = eigenvalues(
            J_COUPLING, F_C,
            kappa_c=0.0,
            delta_f=delta_f,
            delta_kappa=dk,
            phi=PHI_EP,
        )
        plus_vals[idx] = (np.imag(lambdas[1]))
        minus_vals[idx] = (np.imag(lambdas[2]))
    return plus_vals, minus_vals

# -----------------------------------------------------------------------------
# TPD rows configuration (rows 1-3)
# -----------------------------------------------------------------------------
DELTA_KAPPA_TOP = np.linspace(-0.85, -0.80, 10001)  # for Perfect and Imperfect TPD rows
DELTA_KAPPA_BOTTOM = np.linspace(-0.05, 0.05, 10001)  # for Robust TPD row

TPD_SCENARIOS = (
    {
        "name": "Perfect TPD",
        "description": r"$\tilde{\kappa}_c = 1.0$, $\tilde{\Delta}_f = 0$",
        "phi": 0.0,
        "kappa_c": 1.0,
        "delta_f": 0.0,
        "dk_sweep": DELTA_KAPPA_TOP,
    },
    {
        "name": "Imperfect TPD",
        "description": r"$\tilde{\kappa}_c = 1.0$, $\tilde{\Delta}_f = 10^{-3}$",
        "phi": 0.0,
        "kappa_c": 1.0,
        "delta_f": 1e-3,
        "dk_sweep": DELTA_KAPPA_TOP,
    },
    {
        "name": "Robust TPD",
        "description": r"$\tilde{\kappa}_c = 2.0$, $\tilde{\Delta}_f = 10^{-3}$",
        "phi": 0.0,
        "kappa_c": 2.0,
        "delta_f": 1e-3,
        "dk_sweep": DELTA_KAPPA_BOTTOM,
    },
)

def _break_at_transitions(y: np.ndarray, splitting: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """Insert NaNs where the system switches between degenerate and split roots."""
    y = y.copy()
    s = splitting
    up = np.where((s[1:] > eps) & (s[:-1] <= eps))[0] + 1
    down = np.where((s[1:] <= eps) & (s[:-1] > eps))[0] + 1
    idx = np.unique(np.concatenate([up, down]))
    y[idx] = np.nan
    return y

def _all_roots(delta_f: float, delta_kappa: float, kappa_c: float, phi: float) -> np.ndarray:
    """Return three real roots (peak, trough, peak). Repeat values in 1 or 2 root regions."""
    cos_phi = np.cos(phi)
    sin_phi = np.sin(phi)
    kappa_bar = kappa_c - delta_kappa

    p = (-(delta_f / 2) ** 2 + (delta_kappa / 2) ** 2 - cos_phi * J_COUPLING ** 2 + (kappa_bar / 2) ** 2)
    q = (kappa_bar / 4) * (delta_f * delta_kappa - 2 * J_COUPLING ** 2 * sin_phi)

    coeffs = [1.0, 0.0, p, q]
    roots = np.roots(coeffs).astype(complex)
    roots = roots + (F_C - delta_f / 2)

    real_mask = np.abs(np.imag(roots)) < 1e-8
    real_roots = np.real(roots[real_mask]) if real_mask.any() else np.real(roots)
    real_roots.sort()

    if real_roots.size == 1:
        return np.repeat(real_roots, 3)
    if real_roots.size == 2:
        return np.array([real_roots[0], real_roots[0], real_roots[1]])
    return real_roots[:3]

def simulate_tpd_row(phi: float, kappa_c: float, delta_f: float, dk_vals: np.ndarray) -> dict[str, np.ndarray]:
    """Compute all three roots and the splitting for a TPD sweep."""
    nu_plus = np.full_like(dk_vals, np.nan, dtype=float)
    nu_mid = np.full_like(dk_vals, np.nan, dtype=float)
    nu_minus = np.full_like(dk_vals, np.nan, dtype=float)

    for idx, delta_kappa in enumerate(dk_vals):
        roots = _all_roots(delta_f, delta_kappa, kappa_c, phi)
        nu_minus[idx], nu_mid[idx], nu_plus[idx] = roots

    splitting = nu_plus - nu_minus
    splitting[splitting < 0] = 0.0

    return {
        "delta_kappa": dk_vals,
        "nu_plus": _break_at_transitions(nu_plus, splitting),
        "nu_mid": _break_at_transitions(nu_mid, splitting),
        "nu_minus": _break_at_transitions(nu_minus, splitting),
        "splitting": _break_at_transitions(splitting, splitting),
    }

def tpd_location(phi: float, kappa_c: float) -> float:
    return float(TPD_location(phi, kappa_c, J_COUPLING))

def _format_common(ax):
    ax.axhline(0.0, color="lightgray", linewidth=1.0, linestyle="--", zorder=0)
    ax.tick_params(labelsize=FS_TICKS)

# -----------------------------------------------------------------------------
# Figure assembly
# -----------------------------------------------------------------------------
def build():
    plt.rcParams.update(plt.rcParamsDefault)
    plt.rcParams["figure.facecolor"] = "white"
    plt.rcParams["axes.facecolor"] = "white"

    # 4 rows x 2 cols: row 0 = EP, rows 1..3 = TPD
    fig = plt.figure(figsize=(14, 16))
    gs = fig.add_gridspec(
        nrows=4, ncols=2,
        hspace=0.2, wspace=0.25,   # spacing between rows and columns
        left=0.1, right=0.96,      # figure margins
        bottom=0.06, top=0.98
    )

    axes = np.empty((4, 2), dtype=object)

    # EP row (independent x scales)
    axes[0, 0] = fig.add_subplot(gs[0, 0])
    axes[0, 1] = fig.add_subplot(gs[0, 1])

    # TPD rows. For each column, share x within the three TPD rows only.
    for col in range(2):
        ax1 = fig.add_subplot(gs[1, col])               # first TPD row in that column
        ax2 = fig.add_subplot(gs[2, col], sharex=ax1)   # share x with ax1
        ax3 = fig.add_subplot(gs[3, col])               # bottom TPD row has different sweep, do not share
        axes[1, col] = ax1
        axes[2, col] = ax2
        axes[3, col] = ax3

    # ---------------- EP row ----------------
    for col, sc in enumerate(EP_SCENARIOS):
        ax = axes[0, col]
        _format_common(ax)
        plus_vals, minus_vals = eigenvalue_magnitudes(sc["delta_f"])
        ax.plot(DELTA_KAPPA_EP, plus_vals, color=LINECOLOR_MAIN, linewidth=4.0, label=r"$\operatorname{Im}(\lambda)$")
        ax.plot(DELTA_KAPPA_EP, minus_vals, color=LINECOLOR_MAIN, linewidth=4.0)
        # reference line at EP in-range
        ax.axvline(EP_POSITIONS[0], color=LINECOLOR_REF_EP, linewidth=2.0, label=r"$\tilde \Delta_\kappa^\text{EP}$")
        ax.set_xlim(DELTA_KAPPA_EP[0], DELTA_KAPPA_EP[-1])
        ax.set_ylabel(r"Frequency / J", fontsize=FS_LABEL)

        # scenario label inside bottom-left
        ax.text(0.02, 0.02, sc["name"] + "\n" + sc["subtitle"],
                transform=ax.transAxes, ha="left", va="bottom", fontsize=FS_SCENARIO)

        # legend only on left EP panel
        if col == 0:
            ax.legend(loc="upper left", fontsize=FS_LEGEND, frameon=False)

    # x label only on the bottom row of the whole figure
    # so no x labels on EP row

    # ---------------- TPD rows ----------------
    for row_idx, sc in enumerate(TPD_SCENARIOS, start=1):
        dk_vals = sc["dk_sweep"]
        curves = simulate_tpd_row(sc["phi"], sc["kappa_c"], sc["delta_f"], dk_vals)
        nu_plus = curves["nu_plus"]
        nu_mid = curves["nu_mid"]
        nu_minus = curves["nu_minus"]
        splitting = curves["splitting"]
        tpd_x = tpd_location(sc["phi"], sc["kappa_c"])

        # left column: peak locations (three roots)
        ax_loc = axes[row_idx, 0]
        _format_common(ax_loc)
        line_plus,  = ax_loc.plot(dk_vals, nu_plus,  color=LINECOLOR_MAIN, linewidth=4.0, label=r"$\tilde{\nu}_{+}^{\text{Root}}$")
        # need to put = in the subscript because of a formatting glitch on my computer
        line_minus, = ax_loc.plot(dk_vals, nu_minus, color=LINECOLOR_MAIN, linewidth=4.0, linestyle="--", label=r"$\tilde{\nu}_{=}^{\text{Root}}$")
        line_mid,   = ax_loc.plot(dk_vals, nu_mid,   color=LINECOLOR_MAIN, linewidth=4.0, linestyle=":",  label=r"$\tilde{\eta}^{\text{Root}}$")
        line_tpd = ax_loc.axvline(tpd_x, color=LINECOLOR_REF, linewidth=2.0, label=r"$\tilde \Delta_\kappa^\text{TPD}$")
        ax_loc.set_xlim(dk_vals.min(), dk_vals.max())
        ax_loc.set_ylabel("Frequency / J", fontsize=FS_LABEL)

        # scenario text bottom-left
        ax_loc.text(0.02, 0.02, sc["name"] + "\n" + sc["description"],
                    transform=ax_loc.transAxes, ha="left", va="bottom", fontsize=FS_SCENARIO)

        # right column: splitting
        ax_split = axes[row_idx, 1]
        _format_common(ax_split)
        ax_split.plot(dk_vals, splitting, color=LINECOLOR_MAIN, linewidth=4.0)
        ax_split.axvline(tpd_x, color=LINECOLOR_REF, linewidth=2.0)
        ax_split.set_xlim(dk_vals.min(), dk_vals.max())
        ax_split.set_ylabel(r"$\tilde{\Delta}_\nu$", fontsize=FS_LABEL)

        # legend only on the first TPD row
        if row_idx == 1:
            ax_loc.legend(loc="upper left", fontsize=FS_LEGEND, frameon=False, handles=[line_plus, line_minus, line_mid])
            ax_split.legend(loc="upper left", fontsize=FS_LEGEND, frameon=False, handles=[line_tpd])

        # highlight min splitting only on the middle TPD row (Imperfect TPD)
        if row_idx == 2:
            mask = np.isfinite(splitting) & (splitting > 0.0)
            if np.any(mask):
                idxs = np.flatnonzero(mask)
                i0 = idxs[np.nanargmin(splitting[mask])]
                dk0 = dk_vals[i0]
                split_min = float(splitting[i0])
                y_lo = float(nu_minus[i0])
                y_hi = float(nu_plus[i0])
                y_mid = 0.5 * (y_lo + y_hi)

                # vertical bracket on left plot
                ax_loc.vlines(dk0, y_lo, y_hi, color=HILITE_COLOR, linewidth=3.0)
                cap = 0.001 * dk_vals.ptp()
                ax_loc.hlines([y_lo, y_hi], dk0 - cap, dk0 + cap, color=HILITE_COLOR, linewidth=3.0)
                ax_loc.text(dk0 + 0.005 * dk_vals.ptp(), y_mid,
                            r"$\min(\tilde{\Delta}_\nu)$",
                            color='black', fontsize=FS_MIN_TEXT, va="center", ha="left")

                # horizontal dashed line on right plot
                ax_split.axhline(split_min, color=HILITE_COLOR, linestyle="--", linewidth=2.5,
                                 label=r"$\min(\tilde{\Delta}_\nu)$")
                ax_split.legend(loc="upper left", fontsize=FS_LEGEND, frameon=False)

    # x labels only on the very bottom row
    axes[3, 0].set_xlabel(r"$\tilde{\Delta}_\kappa$", fontsize=FS_LABEL)
    axes[3, 1].set_xlabel(r"$\tilde{\Delta}_\kappa$", fontsize=FS_LABEL)

    fig.tight_layout(w_pad=2.6, h_pad=2.2)

    out_dir = Path(__file__).resolve().parents[1] / ".figures"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "SUPP_hypersensitivity.png"
    fig.savefig(out_path, dpi=400, facecolor="white")
    plt.close(fig)

if __name__ == "__main__":
    build()
