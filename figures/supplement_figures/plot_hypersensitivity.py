#!/usr/bin/env python3
"""
combined_ep_tpd_grid.py

Rows 0-1: Perfect and imperfect EP. Column 0 plots the imaginary eigenvalue
locations, column 1 plots their splitting.
Rows 2-4: Perfect, imperfect, and robust TPD peak locations (column 0) and
splitting (column 1).

Styling, fonts, and reference lines are consistent across the full 5x2 figure.
Row-specific x sweeps are used where needed. NaNs are inserted to break
trajectories at teleport boundaries.
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import matplotlib.pyplot as plt
from typing import Optional, Tuple
from matplotlib.lines import Line2D

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
PANEL_LABEL_FS = 27
FS_SCENARIO = 20
FS_MIN_TEXT = 20
FS_LEGEND = 19

# Puiseux display configuration
PUISEUX_CONFIG = {
    "show_uncertainty": False,
    "show_rms": False,
    "text_offsets": {
        "default": (0.92, 0.5),
        "ep": (0.92, 0.5),
        "ted": (0.92, 0.4),
    },
}

# Puiseux expansion configuration: number of terms (>=1) to include per fit.
# Terms progress as (x - x0)^(1/2), (x - x0)^(1), (x - x0)^(3/2), ...
PUISSEUX_TERMS = 2

# -----------------------------------------------------------------------------
# EP row configuration (row 0)
# -----------------------------------------------------------------------------
PHI_EP = 0.0
DELTA_KAPPA_EP = np.linspace(-2.025, -1.975, 50001)
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
DELTA_KAPPA_TOP = np.linspace(-0.8, -0.70, 10001)  # for Perfect and Imperfect TPD rows
DELTA_KAPPA_BOTTOM = np.linspace(-0.05, 0.05, 10001)  # for Robust TPD row

DELTA_F_SWEEP = np.linspace(0.003, 0.003, 10001)

TPD_SCENARIOS = (
    {
        "name": "TPD",
        "description": r"$\tilde{\kappa}_c = 1.0$, $\tilde{\Delta}_f = 0$",
        "phi": 0.0,
        "kappa_c": 1.12,
        "delta_f": 0.0 * np.ones_like(DELTA_KAPPA_TOP),
        "dk_sweep": DELTA_KAPPA_TOP,
    },
    {
        "name": "TED",
        "description": r"$\tilde{\kappa}_c = 1.0$, $\tilde{\Delta}_f = 10^{-3}$",
        "phi": 0.0,
        "kappa_c": 1.12,
        "delta_f": DELTA_F_SWEEP,
        "dk_sweep": DELTA_KAPPA_TOP,
    },
    {
        "name": "Robust TPD",
        "description": r"$\tilde{\kappa}_c = 2.0$, $\tilde{\Delta}_f = 10^{-3}$",
        "phi": 0.0,
        "kappa_c": 2,
        "delta_f": 1e-3 * np.ones_like(DELTA_KAPPA_BOTTOM),
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

def simulate_tpd_row(phi: float, kappa_c: float, df_vals: np.ndarray, dk_vals: np.ndarray) -> dict[str, np.ndarray]:
    """Compute all three roots and the splitting for a TPD sweep."""
    nu_plus = np.full_like(dk_vals, np.nan, dtype=float)
    nu_mid = np.full_like(dk_vals, np.nan, dtype=float)
    nu_minus = np.full_like(dk_vals, np.nan, dtype=float)

    for idx, (delta_kappa, df_val) in enumerate(zip(dk_vals, df_vals)):
        roots = _all_roots(df_val, delta_kappa, kappa_c, phi)
        nu_minus[idx], nu_mid[idx], nu_plus[idx] = roots

    splitting = nu_plus - nu_minus
    splitting[splitting < 0] = 0.0

    slope_raw = np.gradient(splitting, dk_vals, edge_order=2)

    return {
        "delta_kappa": dk_vals,
        "nu_plus": _break_at_transitions(nu_plus, splitting),
        "nu_mid": _break_at_transitions(nu_mid, splitting),
        "nu_minus": _break_at_transitions(nu_minus, splitting),
        "splitting": _break_at_transitions(splitting, splitting),
        "splitting_raw": splitting,
        "slope": _break_at_transitions(slope_raw, splitting),
        "slope_raw": slope_raw,
    }

def tpd_location(phi: float, kappa_c: float) -> float:
    return float(TPD_location(phi, kappa_c, J_COUPLING))

def _format_common(ax):
    ax.axhline(0.0, color="lightgray", linewidth=1.0, linestyle="--", zorder=0)
    ax.tick_params(labelsize=FS_TICKS)

def _puiseux_powers(n_terms: int) -> np.ndarray:
    """Return an array of Puiseux powers starting at 1/2 with step 1/2."""
    n_terms = max(1, int(n_terms))
    return np.array([0.5 * (i + 1) for i in range(n_terms)], dtype=float)


def _puiseux_fit(
        ax: plt.Axes,
        x_vals: np.ndarray,
        y_vals: np.ndarray,
        *,
        base: Optional[float] = None,
        span: float = 0.05,
        color: str = "tab:orange",
        label: str = r"Fit",
        y_offset: float = 0.0,
        n_terms: int = 2,
        text_key: str = "default",
) -> Optional[dict]:
    """Fit a Puiseux series near the onset and overlay the fitted curve."""
    finite_mask = np.isfinite(x_vals) & np.isfinite(y_vals)
    if not finite_mask.any():
        return None
    x = x_vals[finite_mask]
    y = y_vals[finite_mask]

    positive_mask = y > 1e-6
    if not positive_mask.any():
        return None
    if base is None:
        base = float(x[positive_mask][0])

    current_span = span
    for _ in range(5):
        mask = (x >= base) & (x <= base + current_span)
        if mask.sum() >= 8:
            break
        current_span *= 1.5
    else:
        return None

    x_fit = x[mask]
    y_fit = y[mask]
    sqrt_term = np.sqrt(np.clip(x_fit - base, 0.0, None))
    if np.allclose(sqrt_term, 0.0):
        return None

    powers = _puiseux_powers(n_terms)
    text_offsets = PUISEUX_CONFIG["text_offsets"]
    text_offset = text_offsets.get(text_key, text_offsets.get("default", (0.95, 0.85)))
    # Build design matrix with columns (x - base)^{power}
    delta = np.clip(x_fit - base, 0.0, None)
    design_columns = []
    for p in powers:
        design_columns.append(np.power(delta, p))
    A = np.column_stack(design_columns)
    # Remove columns that are entirely zero (outside span)
    zero_cols = np.all(np.isclose(A, 0.0, atol=1e-12, rtol=0.0), axis=0)
    nonzero_cols = np.logical_not(zero_cols)
    if not np.any(nonzero_cols):
        return None
    A = A[:, nonzero_cols]
    effective_powers = powers[nonzero_cols]

    y_work = y_fit - y_offset
    try:
        coeffs, _, _, _ = np.linalg.lstsq(A, y_work, rcond=None)
    except np.linalg.LinAlgError:
        return None
    fit_vals = y_offset + A @ coeffs

    line_handle = ax.plot(
        x_fit,
        fit_vals,
        color=color,
        linewidth=3.0,
        linestyle="--",
        label=label,
    )[0]

    resid = y_fit - fit_vals
    dof = max(len(y_fit) - len(coeffs), 1)
    sigma_sq = float(np.dot(resid, resid) / dof)
    rms_resid = float(np.sqrt(max(sigma_sq, 0.0)))
    try:
        cov = sigma_sq * np.linalg.inv(A.T @ A)
    except np.linalg.LinAlgError:
        cov = np.full((len(coeffs), len(coeffs)), np.nan)

    show_unc = PUISEUX_CONFIG.get("show_uncertainty", True)
    show_rms = PUISEUX_CONFIG.get("show_rms", True)
    lines = []
    for idx, (power, coeff) in enumerate(zip(effective_powers, coeffs)):
        sigma_i = float(np.sqrt(max(cov[idx, idx], 0.0))) if np.isfinite(cov[idx, idx]) else float("nan")
        # Format exponent nicely: 1/2, 1, 3/2 ...
        if abs(power - int(power)) < 1e-9:
            exp_str = f"{int(power)}"
        else:
            numerator = int(round(power * 2))
            exp_str = rf"{numerator}/2"
        coeff_term = rf"$c_{{{exp_str}}} = {coeff:.1f}$"
        if show_unc and np.isfinite(sigma_i):
            coeff_term = rf"$c_{{{exp_str}}} = {coeff:.1f} \pm {sigma_i:.1f}$"
        lines.append(coeff_term)
    if show_rms:
        lines.append(rf"$\mathrm{{RMS}} = {rms_resid:.1f}$")
    txt = "\n".join(lines)

    ax.text(
        text_offset[0],
        text_offset[1],
        txt,
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=FS_SCENARIO,
    )
    return {
        "powers": effective_powers,
        "coeffs": coeffs,
        "cov": cov,
        "rms": rms_resid,
        "x0": base,
        "x_span": (x_fit[0], x_fit[-1]),
        "line_handle": line_handle,
    }

# -----------------------------------------------------------------------------
# Figure assembly
# -----------------------------------------------------------------------------
def build():
    plt.rcParams.update(plt.rcParamsDefault)
    plt.rcParams["figure.facecolor"] = "white"
    plt.rcParams["axes.facecolor"] = "white"

    # 5 rows x 2 cols: rows 0-1 = EP, rows 2-4 = TPD
    fig = plt.figure(figsize=(14, 18))
    gs = fig.add_gridspec(
        nrows=5, ncols=2,
        hspace=0.24, wspace=0.25,   # spacing between rows and columns
        left=0.1, right=0.96,      # figure margins
        bottom=0.06, top=0.98
    )

    axes = np.empty((5, 2), dtype=object)

    # EP rows (each row shares x across its two columns)
    for row in range(2):
        ax_loc = fig.add_subplot(gs[row, 0])
        ax_split = fig.add_subplot(gs[row, 1], sharex=ax_loc)
        axes[row, 0] = ax_loc
        axes[row, 1] = ax_split

    # TPD rows. Share x within the top two TPD rows when sweeps match.
    base_row2 = []
    for col in range(2):
        base = fig.add_subplot(gs[2, col])
        axes[2, col] = base
        base_row2.append(base)

    for col in range(2):
        axes[3, col] = fig.add_subplot(gs[3, col], sharex=base_row2[col])

    for col in range(2):
        base = fig.add_subplot(gs[4, col])
        axes[4, col] = base

# ---------------- EP rows ----------------
    for row_idx, sc in enumerate(EP_SCENARIOS):
        ax_loc = axes[row_idx, 0]
        ax_split = axes[row_idx, 1]
        _format_common(ax_loc)
        _format_common(ax_split)

        plus_vals, minus_vals = eigenvalue_magnitudes(sc["delta_f"])
        splitting = np.abs(plus_vals - minus_vals)

        ax_loc.plot(
            DELTA_KAPPA_EP,
            plus_vals,
            color=LINECOLOR_MAIN,
            linewidth=4.0,
            label=r"$\operatorname{Im}(\tilde{\lambda}_\pm)$",
        )
        ax_loc.plot(
            DELTA_KAPPA_EP,
            minus_vals,
            color=LINECOLOR_MAIN,
            linewidth=4.0,
        )

        ax_split.plot(
            DELTA_KAPPA_EP,
            splitting,
            color=LINECOLOR_MAIN,
            linewidth=4.0,
        )

        ep_label = r"$\tilde \Delta_\kappa^\text{EP}$" if row_idx == 0 else "_nolegend_"
        for ax in (ax_loc, ax_split):
            ax.axvline(
                EP_POSITIONS[0],
                color=LINECOLOR_REF_EP,
                linewidth=2.5,
                label=ep_label,
            )

        ax_loc.set_xlim(DELTA_KAPPA_EP[0], DELTA_KAPPA_EP[-1])
        ax_split.set_xlim(DELTA_KAPPA_EP[0], DELTA_KAPPA_EP[-1])

        ax_loc.set_ylabel(r"Frequency / J", fontsize=FS_LABEL)
        ax_split.set_ylabel(
            r"$\tilde{\Delta}_\lambda$",
            fontsize=FS_LABEL,
        )

        ax_loc.text(
            0.02,
            0.02,
            sc["name"] + "\n" + sc["subtitle"],
            transform=ax_loc.transAxes,
            ha="left",
            va="bottom",
            fontsize=FS_SCENARIO,
        )

        if row_idx == 0:
            ax_loc.legend(loc="upper left", fontsize=FS_LEGEND, frameon=False)

        if row_idx == 0:
            fit_ep = _puiseux_fit(
                ax_split,
                DELTA_KAPPA_EP,
                splitting,
                base=EP_POSITIONS[0],
                span=0.01,
                color="tab:orange",
                label=None,
                y_offset=0.0,
                n_terms=1,
                text_key="ep",
            )
            if fit_ep:
                ax_split.legend([fit_ep["line_handle"]], ["Fit"], fontsize=FS_LEGEND, frameon=False, loc="upper left")
        ax_split.set_xlim(DELTA_KAPPA_EP[0], DELTA_KAPPA_EP[-1])

    # x label only on the bottom row of the whole figure
    # so no x labels on EP row

    # ---------------- TPD rows ----------------
    for row_idx, sc in enumerate(TPD_SCENARIOS, start=2):
        dk_vals = sc["dk_sweep"]
        curves = simulate_tpd_row(sc["phi"], sc["kappa_c"], sc["delta_f"], dk_vals)
        nu_plus = curves["nu_plus"]
        nu_mid = curves["nu_mid"]
        nu_minus = curves["nu_minus"]
        splitting = curves["splitting"]
        splitting_raw = curves["splitting_raw"]
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
                    transform=ax_loc.transAxes, ha="left", va="bottom", fontsize=FS_SCENARIO,
                    color='black' if row_idx < 4 else 'crimson')

        # right column: splitting
        ax_split = axes[row_idx, 1]
        _format_common(ax_split)
        ax_split.plot(dk_vals, splitting, color=LINECOLOR_MAIN, linewidth=4.0)
        ax_split.axvline(tpd_x, color=LINECOLOR_REF, linewidth=2.0)
        ax_split.set_ylabel(r"$\tilde{\Delta}_\nu$", fontsize=FS_LABEL)

        fit_span = 0.02 if row_idx < 4 else 0.05
        mask_pos = np.isfinite(splitting_raw) & (splitting_raw > 1e-6)
        if np.any(mask_pos):
            first_idx = int(np.nonzero(mask_pos)[0][0])
            base_x = float(dk_vals[first_idx])
            base_y = float(splitting_raw[first_idx])
        else:
            base_x = float(tpd_x)
            base_y = 0.0
        if row_idx == 2:
            base_x = float(tpd_x)
        if row_idx != 3:
            base_y = 0.0
        text_key = "ted" if row_idx == 3 else "default"
        fit = _puiseux_fit(
            ax_split,
            dk_vals,
            splitting_raw,
            base=base_x,
            span=fit_span,
            color="tab:orange",
            label=None,
            y_offset=base_y,
            n_terms=PUISSEUX_TERMS if row_idx == 3 else 1,
            text_key=text_key,
        )
        ax_split.set_xlim(dk_vals.min(), dk_vals.max())

        # legend only on the first TPD row
        if row_idx == 2:
            ax_loc.legend(loc="upper left", fontsize=FS_LEGEND, frameon=False, handles=[line_plus, line_minus, line_mid],
                          bbox_to_anchor=(-0.02, 1.05))
            ax_split.legend(loc="upper left", fontsize=FS_LEGEND, frameon=False, handles=[line_tpd])

        # highlight min splitting only on the middle TPD row (Imperfect TPD)
        if row_idx == 3:
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
                cap = 0.001 * np.ptp(dk_vals)
                ax_loc.hlines([y_lo, y_hi], dk0 - cap, dk0 + cap, color=HILITE_COLOR, linewidth=3.0)
                ax_loc.text(dk0 + 0.005 * np.ptp(dk_vals), y_mid,
                            r"$\tilde{\Delta}_\nu^\text{TED}$",
                            color='black', fontsize=FS_MIN_TEXT, va="center", ha="left")

                # horizontal dashed line on right plot
                ax_split.axhline(split_min, color=HILITE_COLOR, linestyle="--", linewidth=2.5,
                                 label=r"$\tilde{\Delta}_\nu^\text{TED}$")
        if row_idx == 3:
            handles, labels = ax_split.get_legend_handles_labels()
            entries = []
            seen = set()
            for h, l in zip(handles, labels):
                if not l:
                    continue
                if l in seen:
                    continue
                seen.add(l)
                entries.append((h, l))
            if fit and np.any(np.isclose(fit["powers"], 1.0, atol=1e-6)):
                pass
                # lin_label = r"$c_{1}(\Delta\tilde\kappa - \tilde\Delta_\kappa^\mathrm{TED})$"
                # linear_handle = Line2D([0, 1], [0, 1], color="tab:orange", linestyle=":", linewidth=3.0)
                # entries.append((linear_handle, lin_label))
            if entries:
                handles_out, labels_out = zip(*entries)
                ax_split.legend(handles_out, labels_out, fontsize=FS_LEGEND, frameon=False, loc="upper left")


    # x labels only on the very bottom row
    axes[4, 0].set_xlabel(r"$\tilde{\Delta}_\kappa$", fontsize=FS_LABEL)
    axes[4, 1].set_xlabel(r"$\tilde{\Delta}_\kappa$", fontsize=FS_LABEL)
    fig.tight_layout(w_pad=2.6, h_pad=2.2)

    # Panel labels a–j (five rows × two columns)
    panel_labels = list("abcdefghij")
    idx = 0
    for row in range(5):
        for col in range(2):
            ax = axes[row, col]
            letter = panel_labels[idx]
            idx += 1
            bb = ax.get_position()
            fig.text(
                bb.x0 - 0.05,
                bb.y1 - 0.01,
                letter,
                fontsize=PANEL_LABEL_FS,
                fontweight="bold",
                ha="center",
                va="bottom",
            )

    out_dir = Path(__file__).resolve().parents[1] / ".figures"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "SUPP_hypersensitivity.png"
    fig.savefig(out_path, dpi=400, facecolor="white")
    plt.close(fig)

if __name__ == "__main__":
    build()
