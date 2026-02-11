#!/usr/bin/env python3
"""
figure_5_bottom_panels.py

Two horizontal panels:
  (c) kappa_tilde_c = 1.12
  (d) kappa_tilde_c = 2.0

Each panel:
  - main: three real roots vs delta_kappa_tilde
          cyan vertical line: TPD
          green vertical dashed line: TED (only on left panel)
  - inset top-left (i): local discriminant contour Disc = 0 in (delta_kappa_tilde, delta_f_tilde)
          red circle: TPD
          magenta horizontal line: sweep path at delta_f used for the roots sweep
  - inset top-right (ii): splitting vs delta_kappa_tilde
          cyan vertical line: TPD
          green horizontal dashed line: Delta_nu^TED
          Puiseux coefficients shown inside the inset (no fit curve plotted)

Requested formatting:
  - one-row legend at top
  - shared y-axis between the two main panels (no repeated y label/ticks on right)
  - inset tick labels removed (axis labels kept)
  - no orange fit curve
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional, Tuple, Dict, Any

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from fitting.transition_fitting import TPD_location


# =============================================================================
# ONE PLACE TO EDIT FORMATTING, LIMITS, POSITIONS
# =============================================================================

CONFIG: Dict[str, Any] = {
    "model": {
        "J": 1.0,
        "F_c": 0.0,
        "phi": 0.0,
    },

    "style": {
        "line_main_color": "k",
        "tpd_color": "c",
        "ted_color": "forestgreen",
        "disc_color": "c",
        "disc_ls": "--",
        "sweep_color": "magenta",
        "sweep_ls": "-",

        "lw_main": 4.0,
        "lw_ref": 2.0,
        "lw_inset": 2.5,
        "lw_inset_ref": 1.6,
        "lw_disc": 2.0,
        "lw_sweep": 2.0,

        "fs_ticks": 26,
        "fs_label": 28,
        "fs_panel_text": 24,
        "fs_inset_label": 18,
        "fs_inset_roman": 16,
        "fs_panel_label": 30,
        "fs_legend": 22,
        "fs_coeff": 14,

        # Label padding controls (smaller = closer)
        "ylabel_pad_main": -10,
        "xlabel_pad_main": -1,
        "ylabel_pad_inset": 1,
        "xlabel_pad_inset": 1,
    },

    # Global inset layout: single place to keep them synchronized
    # Rect is (x0, y0, w, h) in parent-axes fraction coordinates.
    "inset_layout": {
        "disc_rect": (0.06, 0.62, 0.34, 0.34),
        "split_rect": (0.60, 0.62, 0.34, 0.34),
    },

    "insets": {
        "roman_pos": (0.02, 0.92),
        "coeff_pos": (0.98, 0.98),  # inside splitting inset
        "coeff_ha": "right",
        "coeff_va": "top",
    },

    "figure": {
        "figsize": (18, 6),
        "subplots_adjust": {
            "left": 0.075,
            "right": 0.98,
            "bottom": 0.15,
            "top": 0.85,
            "wspace": 0.09,
        },

        # Panel label position relative to axes bbox (figure coords)
        # x = bb.x0 + dx, y = bb.y1 + dy
        "panel_label_offset": (-0.042, -0.025),
    },

    "legend": {
        "enabled": True,
        "loc": "upper center",
        "frameon": False,
        "bbox_to_anchor": (0.5, 1.02),
        "ncol": 9,  # keep one row
        "columnspacing": 1.15,
        "handletextpad": 0.5,
    },

    "puiseux": {
        "n_terms": 2,
        "span_frac_of_main_xrange": 0.02,  # one-sided fit to the right of TED
    },

    "disc_inset": {
        # If True, ensure the y-span includes the sweep path value (df_sweep)
        "auto_include_sweep_path": True,
        "sweep_pad_factor": 1.05,
    },

    "panels": [
        {
            "label": "(a)",
            "kappa_c": 1.12,
            "title": r"$\tilde{\kappa}_c = 1.12$",

            "dk_sweep": (-0.801, -0.701, 10001),
            "delta_f_for_roots": 1e-3,

            "main_xlim": (-0.751, -0.721),
            "main_ylim": (-0.10, 0.24),

            "disc_xlim_halfspan": 0.02,
            "disc_ylim_halfspan": 2e-3,
            "disc_grid_n": 1001,

            "split_xlim": (-0.76, -0.72),
            "split_ylim": (0.0, 0.50),
        },
        {
            "label": "(b)",
            "kappa_c": 2.0,
            "title": r"$\tilde{\kappa}_c = 2.0$",

            "dk_sweep": (-0.051, 0.051, 10001),
            "delta_f_for_roots": 1e-3,

            "main_xlim": (-0.011, 0.021),
            "main_ylim": (-0.10, 0.24),

            "disc_xlim_halfspan": 0.02,
            "disc_ylim_halfspan": 2e-3,
            "disc_grid_n": 241,

            "split_xlim": (-0.02, 0.02),
            "split_ylim": (0.0, 0.50),
        },
    ],

    "output": {
        "filename": "FIG_5_bottom_panels.png",
        "dpi": 400,
        "dir_relative_to_this_file": "../.figures",
    },
}


# =============================================================================
# Helpers: transitions, roots, discriminant, puiseux (coeffs only)
# =============================================================================

def _break_at_transitions(y: np.ndarray, splitting: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    y = y.copy()
    s = splitting
    up = np.where((s[1:] > eps) & (s[:-1] <= eps))[0] + 1
    down = np.where((s[1:] <= eps) & (s[:-1] > eps))[0] + 1
    idx = np.unique(np.concatenate([up, down]))
    y[idx] = np.nan
    return y


def _all_roots(delta_f: float, delta_kappa: float, kappa_c: float, phi: float) -> np.ndarray:
    J = float(CONFIG["model"]["J"])
    F_c = float(CONFIG["model"]["F_c"])

    cos_phi = float(np.cos(phi))
    sin_phi = float(np.sin(phi))
    kappa_bar = kappa_c - delta_kappa

    p = (-(delta_f / 2.0) ** 2 + (delta_kappa / 2.0) ** 2 - cos_phi * J ** 2 + (kappa_bar / 2.0) ** 2)
    q = (kappa_bar / 4.0) * (delta_f * delta_kappa - 2.0 * J ** 2 * sin_phi)

    coeffs = [1.0, 0.0, p, q]
    roots = np.roots(coeffs).astype(complex)
    roots = roots + (F_c - delta_f / 2.0)

    real_mask = np.abs(np.imag(roots)) < 1e-8
    real_roots = np.real(roots[real_mask]) if real_mask.any() else np.real(roots)
    real_roots.sort()

    if real_roots.size == 1:
        return np.repeat(real_roots, 3)
    if real_roots.size == 2:
        return np.array([real_roots[0], real_roots[0], real_roots[1]])
    return real_roots[:3]


def simulate_roots_and_splitting(phi: float, kappa_c: float, df_val: float, dk_vals: np.ndarray) -> Dict[str, np.ndarray]:
    nu_plus = np.full_like(dk_vals, np.nan, dtype=float)
    nu_mid = np.full_like(dk_vals, np.nan, dtype=float)
    nu_minus = np.full_like(dk_vals, np.nan, dtype=float)

    for idx, dk in enumerate(dk_vals):
        roots = _all_roots(df_val, float(dk), kappa_c, phi)
        nu_minus[idx], nu_mid[idx], nu_plus[idx] = roots

    splitting_raw = nu_plus - nu_minus
    splitting_raw[splitting_raw < 0.0] = 0.0
    splitting_plot = _break_at_transitions(splitting_raw, splitting_raw)

    return {
        "dk": dk_vals,
        "nu_plus": _break_at_transitions(nu_plus, splitting_raw),
        "nu_mid": _break_at_transitions(nu_mid, splitting_raw),
        "nu_minus": _break_at_transitions(nu_minus, splitting_raw),
        "splitting_raw": splitting_raw,
        "splitting_plot": splitting_plot,
    }


def tpd_x_location(phi: float, kappa_c: float) -> float:
    J = float(CONFIG["model"]["J"])
    return float(TPD_location(phi, kappa_c, J))


def discriminant_field_grid(dk_grid: np.ndarray, df_grid: np.ndarray, kappa_c: float, phi: float) -> np.ndarray:
    kappa_bar = kappa_c - dk_grid
    p = (-df_grid ** 2 / 4.0 + dk_grid ** 2 / 4.0 - np.cos(phi) + kappa_bar ** 2 / 4.0)
    q = -kappa_bar * (2.0 * np.sin(phi) - df_grid * dk_grid) / 4.0
    disc = -4.0 * (p ** 3) - 27.0 * (q ** 2)
    return disc.astype(float)


def _puiseux_powers(n_terms: int) -> np.ndarray:
    n_terms = max(1, int(n_terms))
    return np.array([0.5 * (i + 1) for i in range(n_terms)], dtype=float)


def puiseux_fit_coeffs(
    x_vals: np.ndarray,
    y_vals: np.ndarray,
    *,
    base: float,
    span: float,
    n_terms: int,
    y_offset: float,
) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    finite = np.isfinite(x_vals) & np.isfinite(y_vals)
    if not np.any(finite):
        return None

    x = x_vals[finite]
    y = y_vals[finite]

    mask = (x >= base) & (x <= base + span)
    if mask.sum() < 10:
        return None

    x_fit = x[mask]
    y_fit = y[mask]

    delta = np.clip(x_fit - base, 0.0, None)
    powers = _puiseux_powers(n_terms)
    cols = [np.power(delta, p) for p in powers]
    A = np.column_stack(cols)

    zero_cols = np.all(np.isclose(A, 0.0, atol=1e-12, rtol=0.0), axis=0)
    keep = np.logical_not(zero_cols)
    if not np.any(keep):
        return None

    A = A[:, keep]
    eff_powers = powers[keep]

    y_work = y_fit - y_offset
    try:
        coeffs, _, _, _ = np.linalg.lstsq(A, y_work, rcond=None)
    except np.linalg.LinAlgError:
        return None

    return eff_powers, coeffs


def compute_ted(dk: np.ndarray, splitting_raw: np.ndarray) -> Optional[Tuple[float, float]]:
    mask = np.isfinite(splitting_raw) & (splitting_raw > 0.0)
    if not np.any(mask):
        return None
    idxs = np.flatnonzero(mask)
    i0 = idxs[np.argmin(splitting_raw[mask])]
    dk_ted = float(dk[i0])
    split_min = float(splitting_raw[i0])
    return dk_ted, split_min


def format_coeff_text(powers: np.ndarray, coeffs: np.ndarray) -> str:
    powers_to_include = [0.5]
    lines = []
    for power, coeff in zip(powers, coeffs):
        if abs(power - int(power)) < 1e-9:
            exp_str = f"{int(power)}"
        else:
            numerator = int(round(power * 2))
            exp_str = f"{numerator}/2"
        if power in powers_to_include:
            lines.append(rf"$c_{{{exp_str}}} = {coeff:.2f}$")
    return "\n".join(lines)


# =============================================================================
# Insets
# =============================================================================

def add_discriminant_inset(
    ax_parent: plt.Axes,
    *,
    kappa_c: float,
    phi: float,
    tpd_dk: float,
    df_sweep: float,
    inset_rect: Tuple[float, float, float, float],
    dk_halfspan: float,
    df_halfspan: float,
    n_grid: int,
) -> plt.Axes:
    st = CONFIG["style"]
    ins_cfg = CONFIG["insets"]
    disc_cfg = CONFIG["disc_inset"]

    df_span = float(df_halfspan)
    if disc_cfg.get("auto_include_sweep_path", True):
        pad = float(disc_cfg.get("sweep_pad_factor", 1.05))
        df_span = max(df_span, abs(float(df_sweep)) * pad)

    axins = ax_parent.inset_axes(inset_rect)

    dk = np.linspace(tpd_dk - dk_halfspan, tpd_dk + dk_halfspan, int(n_grid))
    df = np.linspace(-df_span, df_span, int(n_grid))
    DK, DF = np.meshgrid(dk, df)

    disc = discriminant_field_grid(DK, DF, kappa_c=kappa_c, phi=phi)

    axins.contour(
        DK,
        DF,
        disc,
        levels=[0.0],
        colors=st["disc_color"],
        linestyles=st["disc_ls"],
        linewidths=st["lw_disc"],
    )

    axins.scatter(
        [tpd_dk],
        [0.0],
        s=80,
        facecolors="none",
        edgecolors="red",
        linewidths=2.0,
        zorder=5,
    )

    axins.axhline(
        float(df_sweep),
        color=st["sweep_color"],
        linestyle=st["sweep_ls"],
        linewidth=st["lw_sweep"],
        zorder=4,
    )

    axins.set_xlim(tpd_dk - dk_halfspan, tpd_dk + dk_halfspan)
    axins.set_ylim(-df_span, df_span)

    axins.tick_params(labelbottom=False, labelleft=False)
    axins.set_xlabel(r"$\tilde{\Delta}_\kappa$", fontsize=st["fs_inset_label"], labelpad=st["xlabel_pad_inset"])
    axins.set_ylabel(r"$\tilde{\Delta}_f$", fontsize=st["fs_inset_label"], labelpad=st["ylabel_pad_inset"])

    axins.text(
        ins_cfg["roman_pos"][0],
        ins_cfg["roman_pos"][1],
        "i",
        transform=axins.transAxes,
        fontsize=st["fs_inset_roman"],
        fontweight="bold",
        ha="left",
        va="top",
    )

    return axins


def add_splitting_inset(
    ax_parent: plt.Axes,
    *,
    dk: np.ndarray,
    splitting_plot: np.ndarray,
    tpd_dk: float,
    split_min: Optional[float],
    coeff_text: Optional[str],
    inset_rect: Tuple[float, float, float, float],
    xlim: Optional[Tuple[float, float]],
    ylim: Optional[Tuple[float, float]],
) -> plt.Axes:
    st = CONFIG["style"]
    ins_cfg = CONFIG["insets"]

    axins = ax_parent.inset_axes(inset_rect)

    axins.plot(dk, splitting_plot, color=st["line_main_color"], linewidth=st["lw_inset"])
    axins.axvline(tpd_dk, color=st["tpd_color"], linewidth=st["lw_inset_ref"])

    if split_min is not None:
        axins.axhline(split_min, color=st["ted_color"], linestyle="--", linewidth=1.8)

    if xlim is not None:
        axins.set_xlim(float(xlim[0]), float(xlim[1]))
    if ylim is not None:
        axins.set_ylim(float(ylim[0]), float(ylim[1]))

    axins.tick_params(labelbottom=False, labelleft=False)
    axins.set_ylabel(r"$\tilde{\Delta}_\nu$", fontsize=st["fs_inset_label"], labelpad=st["ylabel_pad_inset"])
    axins.set_xlabel(r"$\tilde{\Delta}_\kappa$", fontsize=st["fs_inset_label"], labelpad=st["xlabel_pad_inset"])

    axins.text(
        ins_cfg["roman_pos"][0],
        ins_cfg["roman_pos"][1],
        "ii",
        transform=axins.transAxes,
        fontsize=st["fs_inset_roman"],
        fontweight="bold",
        ha="left",
        va="top",
    )

    if coeff_text is not None:
        axins.text(
            ins_cfg["coeff_pos"][0],
            ins_cfg["coeff_pos"][1],
            coeff_text,
            transform=axins.transAxes,
            fontsize=st["fs_coeff"],
            ha=ins_cfg["coeff_ha"],
            va=ins_cfg["coeff_va"],
        )

    return axins


# =============================================================================
# Panel builder
# =============================================================================

def build_panel(ax: plt.Axes, panel_cfg: Dict[str, Any], *, is_left: bool) -> None:
    st = CONFIG["style"]
    phi = float(CONFIG["model"]["phi"])

    kappa_c = float(panel_cfg["kappa_c"])
    dk0, dk1, npts = panel_cfg["dk_sweep"]
    dk_vals = np.linspace(float(dk0), float(dk1), int(npts))
    df_sweep = float(panel_cfg["delta_f_for_roots"])

    curves = simulate_roots_and_splitting(phi, kappa_c, df_sweep, dk_vals)

    dk = curves["dk"]
    nu_plus = curves["nu_plus"]
    nu_mid = curves["nu_mid"]
    nu_minus = curves["nu_minus"]
    splitting_raw = curves["splitting_raw"]
    splitting_plot = curves["splitting_plot"]

    tpd_dk = tpd_x_location(phi, kappa_c)

    ted = compute_ted(dk, splitting_raw)
    dk_ted = None
    split_min = None
    if ted is not None:
        dk_ted, split_min = ted

    ax.axhline(0.0, color="lightgray", linewidth=1.0, linestyle="--", zorder=0)
    ax.tick_params(labelsize=st["fs_ticks"])

    ax.plot(dk, nu_plus, color=st["line_main_color"], linewidth=st["lw_main"], linestyle="-")
    ax.plot(dk, nu_minus, color=st["line_main_color"], linewidth=st["lw_main"], linestyle="--")
    ax.plot(dk, nu_mid, color=st["line_main_color"], linewidth=st["lw_main"], linestyle=":")

    ax.axvline(tpd_dk, color=st["tpd_color"], linewidth=st["lw_ref"])

    if is_left and (dk_ted is not None):
        ax.axvline(dk_ted, color=st["ted_color"], linestyle="--", linewidth=st["lw_ref"])

    if panel_cfg["main_xlim"] is not None:
        ax.set_xlim(float(panel_cfg["main_xlim"][0]), float(panel_cfg["main_xlim"][1]))
    if panel_cfg["main_ylim"] is not None:
        ax.set_ylim(float(panel_cfg["main_ylim"][0]), float(panel_cfg["main_ylim"][1]))

    ax.set_xlabel(r"$\tilde{\Delta}_\kappa$", fontsize=st["fs_label"], labelpad=st["xlabel_pad_main"])

    if is_left:
        ax.set_ylabel("Frequency / J", fontsize=st["fs_label"], labelpad=st["ylabel_pad_main"])
    else:
        ax.set_ylabel("")
        ax.tick_params(labelleft=False)

    ax.text(
        0.02,
        0.02,
        panel_cfg["title"],
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=st["fs_panel_text"],
    )

    coeff_text = None
    if (dk_ted is not None) and (split_min is not None):
        span = float(CONFIG["puiseux"]["span_frac_of_main_xrange"]) * float(dk.ptp())
        out = puiseux_fit_coeffs(
            dk,
            splitting_raw,
            base=float(dk_ted),
            span=float(span),
            n_terms=int(CONFIG["puiseux"]["n_terms"]),
            y_offset=float(split_min),
        )
        if out is not None:
            powers, coeffs = out
            coeff_text = format_coeff_text(powers, coeffs)

    disc_rect = tuple(CONFIG["inset_layout"]["disc_rect"])
    split_rect = tuple(CONFIG["inset_layout"]["split_rect"])

    _ = add_discriminant_inset(
        ax,
        kappa_c=kappa_c,
        phi=phi,
        tpd_dk=tpd_dk,
        df_sweep=df_sweep,
        inset_rect=disc_rect,
        dk_halfspan=float(panel_cfg["disc_xlim_halfspan"]),
        df_halfspan=float(panel_cfg["disc_ylim_halfspan"]),
        n_grid=int(panel_cfg["disc_grid_n"]),
    )

    _ = add_splitting_inset(
        ax,
        dk=dk,
        splitting_plot=splitting_plot,
        tpd_dk=tpd_dk,
        split_min=split_min,
        coeff_text=coeff_text,
        inset_rect=split_rect,
        xlim=tuple(panel_cfg["split_xlim"]) if panel_cfg["split_xlim"] is not None else None,
        ylim=tuple(panel_cfg["split_ylim"]) if panel_cfg["split_ylim"] is not None else None,
    )


# =============================================================================
# Figure assembly
# =============================================================================

def build() -> None:
    plt.rcParams.update(plt.rcParamsDefault)
    plt.rcParams["figure.facecolor"] = "white"
    plt.rcParams["axes.facecolor"] = "white"

    fig_cfg = CONFIG["figure"]
    st = CONFIG["style"]

    fig, axs = plt.subplots(1, 2, figsize=tuple(fig_cfg["figsize"]), sharey=True)
    fig.subplots_adjust(**fig_cfg["subplots_adjust"])

    for i, panel_cfg in enumerate(CONFIG["panels"]):
        build_panel(axs[i], panel_cfg, is_left=(i == 0))

    if CONFIG["legend"]["enabled"]:
        handles = [
            Line2D([0, 1], [0, 1], color=st["line_main_color"], lw=3.0, ls="-", label=r"$\tilde{\nu}_{+}^{\mathrm{Root}}$"),
            Line2D([0, 1], [0, 1], color=st["line_main_color"], lw=3.0, ls="--", label=r"$\tilde{\nu}_{-}^{\mathrm{Root}}$"),
            Line2D([0, 1], [0, 1], color=st["line_main_color"], lw=3.0, ls=":", label=r"$\tilde{\eta}^{\mathrm{Root}}$"),
            Line2D([0, 1], [0, 1], color=st["tpd_color"], lw=2.0, ls="-", label=r"$\tilde{\Delta}_\kappa^{\mathrm{TPD}}$"),
            Line2D([0, 1], [0, 1], color=st["ted_color"], lw=2.0, ls="--", label=r"$\tilde{\Delta}_\nu^{\mathrm{TED}}$"),
            Line2D([0, 1], [0, 1], color=st["disc_color"], lw=2.0, ls=st["disc_ls"], label=r"$\mathrm{Disc}=0$"),
            Line2D([0, 1], [0, 1], color=st["sweep_color"], lw=2.0, ls=st["sweep_ls"], label="Sweep Path Bound"),
            Line2D([0, 1], [0, 1], color="red", marker="o", mfc="none", mec="red", lw=0.0, label="TPD (inset)"),
        ]
        leg_cfg = CONFIG["legend"]
        fig.legend(
            handles=handles,
            loc=leg_cfg["loc"],
            ncol=int(leg_cfg["ncol"]),
            frameon=leg_cfg["frameon"],
            fontsize=st["fs_legend"],
            bbox_to_anchor=tuple(leg_cfg["bbox_to_anchor"]),
            columnspacing=float(leg_cfg["columnspacing"]),
            handletextpad=float(leg_cfg["handletextpad"]),
        )

    dx, dy = CONFIG["figure"]["panel_label_offset"]
    for i, panel_cfg in enumerate(CONFIG["panels"]):
        bb = axs[i].get_position()
        fig.text(
            bb.x0 + float(dx),
            bb.y1 + float(dy),
            str(panel_cfg["label"]),
            fontsize=st["fs_panel_label"],
            fontweight="bold",
            ha="left",
            va="bottom",
        )

    out_dir = (Path(__file__).resolve().parent / Path(CONFIG["output"]["dir_relative_to_this_file"])).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / str(CONFIG["output"]["filename"])
    fig.savefig(out_path, dpi=int(CONFIG["output"]["dpi"]), facecolor="white")
    plt.close(fig)


if __name__ == "__main__":
    build()
