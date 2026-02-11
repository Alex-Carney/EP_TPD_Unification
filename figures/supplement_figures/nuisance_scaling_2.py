#!/usr/bin/env python3
from __future__ import annotations

import math
from pathlib import Path
from typing import Dict

import matplotlib.pyplot as plt
import numpy as np


# -----------------------------------------------------------------------------
# User configuration (change everything here)
# -----------------------------------------------------------------------------
CFG: Dict[str, object] = {
    # Physical parameters
    "J_coupling": 1.0,
    "F_c": 0.0,
    "kappa_tilde_c": 1.0,
    "phis": [0.0, math.pi],

    # Which TPD branch to use when there are two symmetric solutions
    # phi = 0: delta_kappa_tpd = (kappa +/- sqrt(8 - kappa^2))/2
    "tpd_branch_phi0": "minus",  # "minus" or "plus"
    # phi = pi: delta_f_tpd = +/- sqrt(4 + kappa^2) at delta_kappa = 0
    "tpd_branch_phipi": "minus",  # "minus" or "plus"

    # Sampling
    "n_sense": 12001,
    "n_nuis": 8001,

    # Sweep half-spans around the TPD point for each row (phi case)
    # Row 0 (phi=0): sensing variable is delta_kappa around dk_tpd
    "phi0_sense_halfspan": 0.14,
    # Row 0 (phi=0): nuisance variable is delta_f around 0
    "phi0_nuis_halfspan": 0.02,

    # Row 1 (phi=pi): sensing variable is delta_f around df_tpd
    "phipi_sense_halfspan": 0.35,
    # Row 1 (phi=pi): nuisance variable is delta_kappa around 0
    "phipi_nuis_halfspan": 0.08,

    # Fit windows (positive x window used for sqrt, symmetric window used for cbrt)
    "sqrt_fit_span": 0.12,
    "cbrt_fit_span": 0.02,

    # Plot styling
    "figure_size": (13.5, 8.2),
    "dpi": 300,
    "line_w_data": 4.4,
    "line_w_fit": 3.8,
    "line_w_ref": 2.8,
    "grid_lw": 1.1,

    # Fonts
    "rc_fonts": {
        "font.family": "sans-serif",
        "font.size": 12,
        "axes.labelsize": 18,
        "xtick.labelsize": 14,
        "ytick.labelsize": 14,
        "legend.fontsize": 14,
    },

    # Panel labels outside axes
    "panel_labels": {
        "labels": [["(a)", "(b)"], ["(c)", "(d)"]],
        "fontsize": 26,
        "fontweight": "bold",
        # Offsets in figure coordinates relative to each axes bbox corner
        "dx": -0.045,
        "dy": 0.010,
    },

    # Textbox font
    "textbox_fs": 14,

    # Colors
    "color_data": "black",
    "color_fit": "tab:orange",
    "color_ref": "c",

    # Legends: per-axis placement (loc, bbox_to_anchor)
    # keys: (row_index, col_index)
    "legend_cfg": {
        (0, 0): {"loc": "lower right", "bbox_to_anchor": (1.0, 0.02)},
        (0, 1): {"loc": "lower right", "bbox_to_anchor": (1.0, 0.02)},
        (1, 0): {"loc": "lower right", "bbox_to_anchor": (1.0, 0.02)},
        (1, 1): {"loc": "lower right", "bbox_to_anchor": (1.0, 0.02)},
    },

    # Text box placement (separable by phase and by column)
    # Row 0 is phi=0, row 1 is phi=pi
    # Keys: phase -> col -> (x, y) in axes coordinates
    "text_cfg_phase": {
        "phi0": {
            0: (0.02, 0.98),
            1: (0.02, 0.98),
        },
        "phipi": {
            0: (0.02, 0.98),
            1: (0.02, 0.98),
        },
    },
    # Optional per-panel overrides (row, col) -> (x, y)
    "text_cfg_overrides": {
        # Example:
        # (0, 1): (0.02, 0.94),
    },

    # Output
    "out_name": "tpd_epsilon_vs_delta_grid_2x2.png",
    "out_dir": ".figures",
}


# -----------------------------------------------------------------------------
# Core model: roots from the same cubic you have been using
# -----------------------------------------------------------------------------
def all_roots(
    delta_f_tilde: float,
    delta_kappa_tilde: float,
    kappa_tilde_c: float,
    phi: float,
    *,
    j_coupling: float,
    f_c: float,
) -> np.ndarray:
    """
    Return three real roots (peak, trough, peak) using the depressed cubic.
    Repeat values in 1- or 2-root regions to match prior plotting convention.
    """
    cos_phi = math.cos(phi)
    sin_phi = math.sin(phi)
    kappa_bar = kappa_tilde_c - delta_kappa_tilde

    p = (
        - (delta_f_tilde / 2.0) ** 2
        + (delta_kappa_tilde / 2.0) ** 2
        - cos_phi * (j_coupling ** 2)
        + (kappa_bar / 2.0) ** 2
    )
    q = (kappa_bar / 4.0) * (delta_f_tilde * delta_kappa_tilde - 2.0 * (j_coupling ** 2) * sin_phi)

    coeffs = [1.0, 0.0, p, q]
    roots = np.roots(coeffs).astype(complex)
    roots = roots + (f_c - delta_f_tilde / 2.0)

    real_mask = np.abs(np.imag(roots)) < 1e-6
    real_roots = np.real(roots[real_mask]) if real_mask.any() else np.real(roots)
    real_roots.sort()

    if real_roots.size == 1:
        return np.repeat(real_roots, 3)
    if real_roots.size == 2:
        return np.array([real_roots[0], real_roots[0], real_roots[1]])
    return real_roots[:3]


# -----------------------------------------------------------------------------
# TPD locations (analytic, consistent with the cubic)
# -----------------------------------------------------------------------------
def tpd_location_phi0_delta_kappa(kappa_tilde_c: float, branch: str) -> float:
    s = math.sqrt(max(8.0 - kappa_tilde_c * kappa_tilde_c, 0.0))
    if branch == "plus":
        return 0.5 * (kappa_tilde_c + s)
    return 0.5 * (kappa_tilde_c - s)


def tpd_location_phipi_delta_f(kappa_tilde_c: float, branch: str) -> float:
    # At phi=pi, along delta_kappa=0, p=0 => delta_f^2 = 4 + kappa_c^2
    val = math.sqrt(4.0 + kappa_tilde_c * kappa_tilde_c)
    if branch == "plus":
        return val
    return -val


# -----------------------------------------------------------------------------
# Analytic coefficients (local Puiseux leading terms implied by the depressed cubic)
# -----------------------------------------------------------------------------
def analytic_a_sqrt_tpd(phi: float, kappa_tilde_c: float, *, branch_phi0: str, branch_phipi: str) -> float:
    if abs(phi) < 1e-12:
        s = math.sqrt(max(8.0 - kappa_tilde_c * kappa_tilde_c, 0.0))
        return math.sqrt(2.0 * s)
    df0 = abs(tpd_location_phipi_delta_f(kappa_tilde_c, branch_phipi))
    return math.sqrt(2.0 * df0)


def analytic_b_cbrt_tpd(phi: float, kappa_tilde_c: float, *, branch_phi0: str, branch_phipi: str) -> float:
    if abs(phi) < 1e-12:
        dk0 = tpd_location_phi0_delta_kappa(kappa_tilde_c, branch_phi0)
        kbar = kappa_tilde_c - dk0
        c_lin = (kbar * dk0) / 4.0
        return float(np.cbrt(-c_lin))
    df0 = tpd_location_phipi_delta_f(kappa_tilde_c, branch_phipi)
    c_lin = (kappa_tilde_c * df0) / 4.0
    return float(np.cbrt(-c_lin))


# -----------------------------------------------------------------------------
# Simple one-term fits (no intercept)
# -----------------------------------------------------------------------------
def fit_sqrt_coefficient(x: np.ndarray, y: np.ndarray) -> float:
    m = np.isfinite(x) & np.isfinite(y) & (x >= 0.0)
    x2 = x[m]
    y2 = y[m]
    sx = np.sqrt(np.clip(x2, 0.0, None))
    denom = float(np.dot(sx, sx))
    if denom <= 0.0:
        return float("nan")
    return float(np.dot(sx, y2) / denom)


def fit_cuberoot_coefficient(x: np.ndarray, y: np.ndarray) -> float:
    m = np.isfinite(x) & np.isfinite(y) & (np.abs(x) > 1e-14) & np.isfinite(y)
    x2 = x[m]
    y2 = y[m]
    cx = np.cbrt(x2)
    denom = float(np.dot(cx, cx))
    if denom <= 0.0:
        return float("nan")
    return float(np.dot(cx, y2) / denom)


def choose_split_direction(var0: float, eval_fn) -> int:
    eps = 1e-4
    s_plus = float(eval_fn(var0 + eps))
    s_minus = float(eval_fn(var0 - eps))
    if s_plus >= s_minus:
        return 1
    return -1


def phase_key_from_row(row: int) -> str:
    return "phi0" if row == 0 else "phipi"


def get_text_pos(row: int, col: int) -> tuple[float, float]:
    overrides = CFG.get("text_cfg_overrides", {})
    if isinstance(overrides, dict) and (row, col) in overrides:
        xy = overrides[(row, col)]
        return (float(xy[0]), float(xy[1]))
    phase_key = phase_key_from_row(row)
    phase_cfg = CFG["text_cfg_phase"][phase_key]  # type: ignore[index]
    xy = phase_cfg[col]
    return (float(xy[0]), float(xy[1]))


def phi_suffix(phi: float) -> str:
    if abs(phi) < 1e-12:
        return r"(\phi = 0)"
    return r"(\phi = \pi)"


# -----------------------------------------------------------------------------
# Figure builder
# -----------------------------------------------------------------------------
def build() -> Path:
    j_coupling = float(CFG["J_coupling"])
    f_c = float(CFG["F_c"])
    kappa_c = float(CFG["kappa_tilde_c"])
    phis = [float(p) for p in CFG["phis"]]

    branch_phi0 = str(CFG["tpd_branch_phi0"])
    branch_phipi = str(CFG["tpd_branch_phipi"])

    plt.rcParams.update(CFG["rc_fonts"])  # type: ignore[arg-type]

    fig, axes = plt.subplots(2, 2, figsize=CFG["figure_size"])  # type: ignore[arg-type]
    fig.patch.set_facecolor("white")

    # Precompute analytic coefficients for display
    a0_analytic = analytic_a_sqrt_tpd(0.0, kappa_c, branch_phi0=branch_phi0, branch_phipi=branch_phipi)
    b0_analytic = analytic_b_cbrt_tpd(0.0, kappa_c, branch_phi0=branch_phi0, branch_phipi=branch_phipi)
    api_analytic = analytic_a_sqrt_tpd(math.pi, kappa_c, branch_phi0=branch_phi0, branch_phipi=branch_phipi)
    bpi_analytic = analytic_b_cbrt_tpd(math.pi, kappa_c, branch_phi0=branch_phi0, branch_phipi=branch_phipi)

    # -------------------------------------------------------------------------
    # Row 0: phi = 0
    #   col 0: epsilon(delta_kappa) -> Peak Splitting sqrt
    #   col 1: tilde(delta)(delta_f) -> Peak location cube root
    # -------------------------------------------------------------------------
    phi0 = phis[0]
    dk_tpd_phi0 = tpd_location_phi0_delta_kappa(kappa_c, branch_phi0)

    # (0,0)
    ax = axes[0, 0]
    dk_half = float(CFG["phi0_sense_halfspan"])
    dk = np.linspace(dk_tpd_phi0 - dk_half, dk_tpd_phi0 + dk_half, int(CFG["n_sense"]))

    nu_minus = np.empty_like(dk)
    nu_plus = np.empty_like(dk)
    for i, dki in enumerate(dk):
        r = all_roots(0.0, float(dki), kappa_c, phi0, j_coupling=j_coupling, f_c=f_c)
        nu_minus[i] = r[0]
        nu_plus[i] = r[2]

    splitting = nu_plus - nu_minus
    splitting[splitting < 0.0] = 0.0

    def split_eval_phi0(dk_val: float) -> float:
        r = all_roots(0.0, float(dk_val), kappa_c, phi0, j_coupling=j_coupling, f_c=f_c)
        return max(float(r[2] - r[0]), 0.0)

    dir_phi0 = choose_split_direction(dk_tpd_phi0, split_eval_phi0)
    sqrt_span = float(CFG["sqrt_fit_span"])
    x_sense = dir_phi0 * (dk - dk_tpd_phi0)
    fit_mask = (x_sense >= 0.0) & (x_sense <= sqrt_span) & (splitting > 1e-10)
    a_fit_phi0 = fit_sqrt_coefficient(x_sense[fit_mask], splitting[fit_mask])

    dk_fit = dk_tpd_phi0 + dir_phi0 * np.linspace(0.0, sqrt_span, 500)
    split_fit = a_fit_phi0 * np.sqrt(np.clip(dir_phi0 * (dk_fit - dk_tpd_phi0), 0.0, None))

    ax.plot(dk, splitting, lw=CFG["line_w_data"], color=CFG["color_data"], label="Peak Splitting")  # type: ignore[arg-type]
    ax.plot(
        dk_fit,
        split_fit,
        lw=CFG["line_w_fit"],
        ls="--",
        color=CFG["color_fit"],
        label=r"$\tilde a_{\text{sqrt}}^{\mathrm{TPD}}\sqrt{\epsilon}$",
    )  # type: ignore[arg-type]
    ax.axvline(
        dk_tpd_phi0,
        lw=CFG["line_w_ref"],
        color=CFG["color_ref"],
        label=r"$\tilde{\Delta}_\kappa^{\mathrm{TPD}}$ " + phi_suffix(phi0),
    )  # type: ignore[arg-type]
    ax.axhline(0.0, lw=CFG["grid_lw"], color="lightgray", ls="--", zorder=0)  # type: ignore[arg-type]

    ax.set_xlabel(r"$\tilde{\Delta}_\kappa$")
    ax.set_ylabel(r"Peak Splitting / $J$")
    ax.set_xlim(float(dk[0]), float(dk[-1]))
    ax.set_ylim(-0.02, 1.12 * float(np.nanmax(splitting)))

    tx, ty = get_text_pos(0, 0)
    txt_lines = [
        r"$\phi=0,\ \tilde{\kappa}_c=1$",
        r"$\epsilon(\tilde{\Delta}_\kappa)$ (sensing)",
        r"$\tilde a_{\text{sqrt}}^{\mathrm{TPD}}$ fit $= $" + f"{a_fit_phi0:.4f}",
        r"$\tilde a_{\text{sqrt}}^{\mathrm{TPD}}$ analytic $= $" + f"{a0_analytic:.4f}",
    ]
    ax.text(
        tx,
        ty,
        "\n".join(txt_lines),
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=int(CFG["textbox_fs"]),
        bbox=dict(facecolor="white", edgecolor="none", pad=0.2),
        zorder=5,
    )

    leg_cfg = CFG["legend_cfg"][(0, 0)]  # type: ignore[index]
    ax.legend(
        loc=str(leg_cfg["loc"]),
        bbox_to_anchor=tuple(leg_cfg["bbox_to_anchor"]),
        frameon=True,
        framealpha=1.0,
        facecolor="white",
        edgecolor="none",
    )

    # (0,1)
    ax = axes[0, 1]
    df_half = float(CFG["phi0_nuis_halfspan"])
    df = np.linspace(-df_half, df_half, int(CFG["n_nuis"]))

    nu_mid = np.empty_like(df)
    for i, dfi in enumerate(df):
        r = all_roots(float(dfi), float(dk_tpd_phi0), kappa_c, phi0, j_coupling=j_coupling, f_c=f_c)
        nu_mid[i] = r[1]
    y0 = float(all_roots(0.0, float(dk_tpd_phi0), kappa_c, phi0, j_coupling=j_coupling, f_c=f_c)[1])

    cbrt_span = float(CFG["cbrt_fit_span"])
    fit_mask = (df >= -cbrt_span) & (df <= cbrt_span)
    b_fit_phi0 = fit_cuberoot_coefficient(df[fit_mask], nu_mid[fit_mask] - y0)

    df_fit = np.linspace(-cbrt_span, cbrt_span, 600)
    nu_fit = y0 + b_fit_phi0 * np.cbrt(df_fit)

    ax.plot(df, nu_mid, lw=CFG["line_w_data"], color=CFG["color_data"], label=r"$\tilde{\nu}$")  # type: ignore[arg-type]
    ax.plot(
        df_fit,
        nu_fit,
        lw=CFG["line_w_fit"],
        ls="--",
        color=CFG["color_fit"],
        label=r"$b_{\mathrm{cbrt}}^{\mathrm{TPD}}\sqrt[3]{\tilde{\delta}}$",
    )  # type: ignore[arg-type]
    ax.axvline(
        0.0,
        lw=CFG["line_w_ref"],
        color=CFG["color_ref"],
        label=r"$\tilde{\Delta}_f^{\mathrm{TPD}}$ " + phi_suffix(phi0),
    )  # type: ignore[arg-type]
    ax.axhline(0.0, lw=CFG["grid_lw"], color="lightgray", ls="--", zorder=0)  # type: ignore[arg-type]
    ax.axvline(0.0, lw=CFG["grid_lw"], color="lightgray", ls="--", zorder=0)  # type: ignore[arg-type]

    ax.set_xlabel(r"$\tilde{\Delta}_f$")
    ax.set_ylabel(r"Peak location / $J$")
    ax.set_xlim(float(df[0]), float(df[-1]))

    tx, ty = get_text_pos(0, 1)
    txt_lines = [
        r"$\phi=0,\ \tilde{\kappa}_c=1$",
        r"$\tilde{\delta}(\tilde{\Delta}_f)$ (nuisance)",
        r"$b_{\mathrm{cbrt}}^{\mathrm{TPD}}$ fit $= $" + f"{b_fit_phi0:.4f}",
        r"$b_{\mathrm{cbrt}}^{\mathrm{TPD}}$ analytic $= $" + f"{b0_analytic:.4f}",
    ]
    ax.text(
        tx,
        ty,
        "\n".join(txt_lines),
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=int(CFG["textbox_fs"]),
        bbox=dict(facecolor="white", edgecolor="none", pad=0.2),
        zorder=5,
    )

    leg_cfg = CFG["legend_cfg"][(0, 1)]  # type: ignore[index]
    ax.legend(
        loc=str(leg_cfg["loc"]),
        bbox_to_anchor=tuple(leg_cfg["bbox_to_anchor"]),
        frameon=True,
        framealpha=1.0,
        facecolor="white",
        edgecolor="none",
    )

    # -------------------------------------------------------------------------
    # Row 1: phi = pi
    #   col 0: epsilon(delta_f) -> Peak Splitting sqrt
    #   col 1: tilde(delta)(delta_kappa) -> Peak location cube root
    # -------------------------------------------------------------------------
    phipi = phis[1]
    df_tpd_phipi = tpd_location_phipi_delta_f(kappa_c, branch_phipi)

    # (1,0)
    ax = axes[1, 0]
    df_half = float(CFG["phipi_sense_halfspan"])
    df = np.linspace(df_tpd_phipi - df_half, df_tpd_phipi + df_half, int(CFG["n_sense"]))

    nu_minus = np.empty_like(df)
    nu_plus = np.empty_like(df)
    for i, dfi in enumerate(df):
        r = all_roots(float(dfi), 0.0, kappa_c, phipi, j_coupling=j_coupling, f_c=f_c)
        nu_minus[i] = r[0]
        nu_plus[i] = r[2]

    splitting = nu_plus - nu_minus
    splitting[splitting < 0.0] = 0.0

    def split_eval_phipi(df_val: float) -> float:
        r = all_roots(float(df_val), 0.0, kappa_c, phipi, j_coupling=j_coupling, f_c=f_c)
        return max(float(r[2] - r[0]), 0.0)

    dir_phipi = choose_split_direction(df_tpd_phipi, split_eval_phipi)
    sqrt_span = float(CFG["sqrt_fit_span"])
    x_sense = dir_phipi * (df - df_tpd_phipi)
    fit_mask = (x_sense >= 0.0) & (x_sense <= sqrt_span) & (splitting > 1e-10)
    a_fit_phipi = fit_sqrt_coefficient(x_sense[fit_mask], splitting[fit_mask])

    df_fit = df_tpd_phipi + dir_phipi * np.linspace(0.0, sqrt_span, 500)
    split_fit = a_fit_phipi * np.sqrt(np.clip(dir_phipi * (df_fit - df_tpd_phipi), 0.0, None))

    ax.plot(df, splitting, lw=CFG["line_w_data"], color=CFG["color_data"], label="Peak Splitting")  # type: ignore[arg-type]
    ax.plot(
        df_fit,
        split_fit,
        lw=CFG["line_w_fit"],
        ls="--",
        color=CFG["color_fit"],
        label=r"$\tilde a_{\text{sqrt}}^{\mathrm{TPD}}\sqrt{\epsilon}$",
    )  # type: ignore[arg-type]
    ax.axvline(
        df_tpd_phipi,
        lw=CFG["line_w_ref"],
        color=CFG["color_ref"],
        label=r"$\tilde{\Delta}_f^{\mathrm{TPD}}$ " + phi_suffix(phipi),
    )  # type: ignore[arg-type]
    ax.axhline(0.0, lw=CFG["grid_lw"], color="lightgray", ls="--", zorder=0)  # type: ignore[arg-type]

    ax.set_xlabel(r"$\tilde{\Delta}_f$")
    ax.set_ylabel(r"Peak Splitting / $J$")
    ax.set_xlim(float(df[0]), float(df[-1]))
    ax.set_ylim(-0.02, 1.12 * float(np.nanmax(splitting)))

    tx, ty = get_text_pos(1, 0)
    txt_lines = [
        r"$\phi=\pi,\ \tilde{\kappa}_c=1$",
        r"$\epsilon(\tilde{\Delta}_f)$ (sensing)",
        r"$\tilde a_{\text{sqrt}}^{\mathrm{TPD}}$ fit $= $" + f"{a_fit_phipi:.4f}",
        r"$\tilde a_{\text{sqrt}}^{\mathrm{TPD}}$ analytic $= $" + f"{api_analytic:.4f}",
    ]
    ax.text(
        tx,
        ty,
        "\n".join(txt_lines),
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=int(CFG["textbox_fs"]),
        bbox=dict(facecolor="white", edgecolor="none", pad=0.2),
        zorder=5,
    )

    leg_cfg = CFG["legend_cfg"][(1, 0)]  # type: ignore[index]
    ax.legend(
        loc=str(leg_cfg["loc"]),
        bbox_to_anchor=tuple(leg_cfg["bbox_to_anchor"]),
        frameon=True,
        framealpha=1.0,
        facecolor="white",
        edgecolor="none",
    )

    # (1,1)
    ax = axes[1, 1]
    dk_half = float(CFG["phipi_nuis_halfspan"])
    dk = np.linspace(-dk_half, dk_half, int(CFG["n_nuis"]))

    nu_mid = np.empty_like(dk)
    for i, dki in enumerate(dk):
        r = all_roots(float(df_tpd_phipi), float(dki), kappa_c, phipi, j_coupling=j_coupling, f_c=f_c)
        nu_mid[i] = r[1]
    y0 = float(all_roots(float(df_tpd_phipi), 0.0, kappa_c, phipi, j_coupling=j_coupling, f_c=f_c)[1])

    cbrt_span = float(CFG["cbrt_fit_span"])
    fit_mask = (dk >= -cbrt_span) & (dk <= cbrt_span)
    b_fit_phipi = fit_cuberoot_coefficient(dk[fit_mask], nu_mid[fit_mask] - y0)

    dk_fit = np.linspace(-cbrt_span, cbrt_span, 600)
    nu_fit = y0 + b_fit_phipi * np.cbrt(dk_fit)

    ax.plot(dk, nu_mid, lw=CFG["line_w_data"], color=CFG["color_data"], label=r"$\tilde{\nu}$")  # type: ignore[arg-type]
    ax.plot(
        dk_fit,
        nu_fit,
        lw=CFG["line_w_fit"],
        ls="--",
        color=CFG["color_fit"],
        label=r"$b_{\mathrm{cbrt}}^{\mathrm{TPD}}\sqrt[3]{\tilde{\delta}}$",
    )  # type: ignore[arg-type]
    ax.axvline(
        0.0,
        lw=CFG["line_w_ref"],
        color=CFG["color_ref"],
        label=r"$\tilde{\Delta}_\kappa^{\mathrm{TPD}}$ " + phi_suffix(phipi),
    )  # type: ignore[arg-type]
    ax.axhline(0.0, lw=CFG["grid_lw"], color="lightgray", ls="--", zorder=0)  # type: ignore[arg-type]
    ax.axvline(0.0, lw=CFG["grid_lw"], color="lightgray", ls="--", zorder=0)  # type: ignore[arg-type]

    ax.set_xlabel(r"$\tilde{\Delta}_\kappa$")
    ax.set_ylabel(r"Peak location / $J$")
    ax.set_xlim(float(dk[0]), float(dk[-1]))

    tx, ty = get_text_pos(1, 1)
    txt_lines = [
        r"$\phi=\pi,\ \tilde{\kappa}_c=1$",
        r"$\tilde{\delta}(\tilde{\Delta}_\kappa)$ (nuisance)",
        r"$b_{\mathrm{cbrt}}^{\mathrm{TPD}}$ fit $= $" + f"{b_fit_phipi:.4f}",
        r"$b_{\mathrm{cbrt}}^{\mathrm{TPD}}$ analytic $= $" + f"{bpi_analytic:.4f}",
    ]
    ax.text(
        tx,
        ty,
        "\n".join(txt_lines),
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=int(CFG["textbox_fs"]),
        bbox=dict(facecolor="white", edgecolor="none", pad=0.2),
        zorder=5,
    )

    leg_cfg = CFG["legend_cfg"][(1, 1)]  # type: ignore[index]
    ax.legend(
        loc=str(leg_cfg["loc"]),
        bbox_to_anchor=tuple(leg_cfg["bbox_to_anchor"]),
        frameon=True,
        framealpha=1.0,
        facecolor="white",
        edgecolor="none",
    )

    # Layout first, then draw panel labels outside each axes bbox
    fig.tight_layout()

    pl_cfg = CFG["panel_labels"]  # type: ignore[assignment]
    dx = float(pl_cfg["dx"])
    dy = float(pl_cfg["dy"])
    fs = int(pl_cfg["fontsize"])
    fw = str(pl_cfg["fontweight"])
    labels = pl_cfg["labels"]

    for r in range(2):
        for c in range(2):
            ax = axes[r, c]
            bb = ax.get_position()
            fig.text(
                bb.x0 + dx,
                bb.y1 + dy,
                labels[r][c],
                fontsize=fs,
                fontweight=fw,
                ha="left",
                va="bottom",
                color="black",
            )

    out_dir = Path(str(CFG["out_dir"]))
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / str(CFG["out_name"])
    fig.savefig(out_path, dpi=int(CFG["dpi"]), facecolor="white")
    plt.close(fig)
    return out_path


def main() -> None:
    out = build()
    print("saved:", out)


if __name__ == "__main__":
    main()
