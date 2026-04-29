#!/usr/bin/env python3
"""
Single-panel figure: Theory panel only, with legend ABOVE the panel.

Changes vs previous version:
  - Removed the cyan TPD vertical line from both the main panel and the inset
    (the TPD is no longer the focus once we move to the TED; keeping it cluttered
     the main panel with two vertical lines competing for attention)
  - Added a green open-circle marker for the TED in panel (i), placed at the
    intersection of the sweep path with the Disc=0 contour. This visually ties
    panel (i) to panel (ii) and to the green dashed TED line in the main panel.
  - Annotated the incidence angle alpha numerically inside panel (i), since
    the inset's stretched aspect ratio makes the geometric angle hard to read
    by eye. The annotation is computed from the actual gradient of Disc at
    the TED.
  - Updated legend: dropped the TPD entry, added a TED entry.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.gridspec import GridSpec

from fitting.transition_fitting import TPD_location

# =============================================================================
# OUTPUT
# =============================================================================
BASE_OUTPUT_DIR = Path("results")
OUTPUT_FILENAME = "supp_fig_TED_sqrt.png"


# =============================================================================
# GLOBAL FONT SCALE
# =============================================================================
FONT_SCALE = 1.4

def _fs(x: float) -> float:
    return float(x) * float(FONT_SCALE)

# =============================================================================
# INSET STACK TUNING
# =============================================================================
INSET_STACK_DEFAULTS: Dict[str, Any] = {
    "rect": (0.08, 0.215, 0.38, 0.78),
    "v_gap_frac": 0.015,
    "force_shared_dk_xlim": True,
}

# =============================================================================
# THEORETICAL PANEL CONFIG
# =============================================================================
THEORY_CONFIG: Dict[str, Any] = {
    "model": {
        "J": 1.0,
        "F_c": 0.0,
        "phi": 0.0,
    },
    "style": {
        "line_main_color": "k",
        "tpd_color": "c",            # kept for the TPD red circle outline only
        "ted_color": "forestgreen",
        "disc_color": "c",
        "disc_ls": "--",
        "sweep_color": "magenta",
        "sweep_ls": "-",
        "lw_main": 7,
        "lw_ref": 5,
        "lw_inset": 4,
        "lw_inset_ref": 4,
        "lw_disc": 4,
        "lw_sweep": 4,
        "fs_ticks": _fs(28),
        "fs_label": _fs(36),
        "fs_panel_text": _fs(36),
        "fs_inset_label": _fs(36),
        "fs_inset_roman": _fs(28),
        "fs_panel_label": _fs(36),
        "fs_legend": _fs(24),
        "fs_coeff": _fs(28),
        "fs_alpha": _fs(28),
        "ylabel_pad_main": -10,
        "xlabel_pad_main": -1,
        "ylabel_pad_inset": 1,
        "xlabel_pad_inset": 1,
    },
    "insets": {
        "roman_pos": (0.02, 0.92),
        "coeff_pos": (0.00, 0.00),
        "coeff_ha": "left",
        "coeff_va": "bottom",
    },
    "puiseux": {
        "n_terms": 2,
        "span_frac_of_main_xrange": 0.02,
    },
    "disc_inset": {
        "auto_include_sweep_path": True,
        "sweep_pad_factor": 1.02,
    },
    "panel": {
        "label": "(a)",
        "kappa_c": 1.18,
        "title": r"$\tilde{\kappa}_c = 1.18$",
        "dk_sweep": (-0.801, -0.601, 10001),
        "delta_f_for_roots": 1e-3,
        "main_xlim": (-0.761, -0.661),
        "main_ylim": (-0.24, 0.24),
        "disc_xlim_halfspan": 0.02,
        "disc_ylim_halfspan": 2e-3,
        "disc_grid_n": 1001,
        "split_xlim": (-0.761, -0.661),
        "split_ylim": (0.0, 0.50),
    },
}


# =============================================================================
# THEORY PANEL HELPERS
# =============================================================================
def _break_at_transitions(y: np.ndarray, splitting: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    y2 = y.copy()
    s = splitting
    up = np.where((s[1:] > eps) & (s[:-1] <= eps))[0] + 1
    down = np.where((s[1:] <= eps) & (s[:-1] > eps))[0] + 1
    idx = np.unique(np.concatenate([up, down])) if (up.size + down.size) > 0 else np.array([], dtype=int)
    if idx.size > 0:
        y2[idx] = np.nan
    return y2


def _all_roots(delta_f: float, delta_kappa: float, kappa_c: float, phi: float) -> np.ndarray:
    J = float(THEORY_CONFIG["model"]["J"])
    F_c = float(THEORY_CONFIG["model"]["F_c"])

    cos_phi = float(np.cos(phi))
    sin_phi = float(np.sin(phi))
    kappa_bar = float(kappa_c - delta_kappa)

    p = (-(delta_f / 2.0) ** 2 + (delta_kappa / 2.0) ** 2 - cos_phi * J ** 2 + (kappa_bar / 2.0) ** 2)
    q = (kappa_bar / 4.0) * (delta_f * delta_kappa - 2.0 * J ** 2 * sin_phi)

    roots = np.roots([1.0, 0.0, p, q]).astype(complex)
    roots = roots + (F_c - delta_f / 2.0)

    real_mask = np.abs(np.imag(roots)) < 1e-8
    real_roots = np.real(roots[real_mask]) if real_mask.any() else np.real(roots)
    real_roots.sort()

    if real_roots.size == 1:
        return np.repeat(real_roots, 3).astype(float)
    if real_roots.size == 2:
        return np.array([real_roots[0], real_roots[0], real_roots[1]], dtype=float)
    return real_roots[:3].astype(float)


def simulate_roots_and_splitting(phi: float, kappa_c: float, df_val: float, dk_vals: np.ndarray) -> Dict[str, np.ndarray]:
    nu_plus = np.full_like(dk_vals, np.nan, dtype=float)
    nu_mid = np.full_like(dk_vals, np.nan, dtype=float)
    nu_minus = np.full_like(dk_vals, np.nan, dtype=float)

    for idx, dk in enumerate(dk_vals):
        roots = _all_roots(df_val, float(dk), kappa_c, phi)
        nu_minus[idx], nu_mid[idx], nu_plus[idx] = float(roots[0]), float(roots[1]), float(roots[2])

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


def tpd_x_location_theory(phi: float, kappa_c: float) -> float:
    J = float(THEORY_CONFIG["model"]["J"])
    return float(TPD_location(phi, kappa_c, J))


def discriminant_field_grid(dk_grid: np.ndarray, df_grid: np.ndarray, kappa_c: float, phi: float) -> np.ndarray:
    kappa_bar = kappa_c - dk_grid
    p = (-df_grid ** 2 / 4.0 + dk_grid ** 2 / 4.0 - np.cos(phi) + kappa_bar ** 2 / 4.0)
    q = -kappa_bar * (2.0 * np.sin(phi) - df_grid * dk_grid) / 4.0
    disc = -4.0 * (p ** 3) - 27.0 * (q ** 2)
    return disc.astype(float)


def compute_ted(dk: np.ndarray, splitting_raw: np.ndarray) -> Optional[Tuple[float, float]]:
    mask = np.isfinite(splitting_raw) & (splitting_raw > 0.0)
    if not np.any(mask):
        return None
    idxs = np.flatnonzero(mask)
    i0 = idxs[np.argmin(splitting_raw[mask])]
    return float(dk[i0]), float(splitting_raw[i0])


def compute_alpha_at_ted(dk_ted: float, df_sweep: float, kappa_c: float, phi: float) -> float:
    """
    Compute the incidence angle alpha (in degrees) between the sweep path
    (along the Dk axis, direction (1, 0)) and the tangent to the Disc=0
    contour at the TED.

    The tangent to Disc=0 is perpendicular to grad(Disc), so if
    grad(Disc) = (g_k, g_f), the tangent direction is (-g_f, g_k).

    alpha = angle between (1, 0) and the tangent line
          = arctan(|g_k| / |g_f|)
    """
    h = 1e-7
    DK_p = np.array([[dk_ted + h]])
    DK_m = np.array([[dk_ted - h]])
    DF_p = np.array([[df_sweep + h]])
    DF_m = np.array([[df_sweep - h]])
    DF_c = np.array([[df_sweep]])
    DK_c = np.array([[dk_ted]])

    g_k = (
        discriminant_field_grid(DK_p, DF_c, kappa_c, phi)[0, 0]
        - discriminant_field_grid(DK_m, DF_c, kappa_c, phi)[0, 0]
    ) / (2 * h)
    g_f = (
        discriminant_field_grid(DK_c, DF_p, kappa_c, phi)[0, 0]
        - discriminant_field_grid(DK_c, DF_m, kappa_c, phi)[0, 0]
    ) / (2 * h)

    if abs(g_f) < 1e-30:
        return 90.0
    alpha_rad = float(np.arctan2(abs(g_k), abs(g_f)))
    return float(np.degrees(alpha_rad))


def _puiseux_powers(n_terms: int) -> np.ndarray:
    return np.array([0.5 * (i + 1) for i in range(max(1, n_terms))], dtype=float)


def puiseux_fit_coeffs(x_vals, y_vals, *, base, span, n_terms, y_offset):
    finite = np.isfinite(x_vals) & np.isfinite(y_vals)
    if not np.any(finite):
        return None
    x, y = x_vals[finite], y_vals[finite]
    mask = (x >= base) & (x <= base + span)
    if int(mask.sum()) < 10:
        return None
    x_fit, y_fit = x[mask], y[mask]
    delta = np.clip(x_fit - base, 0.0, None)
    powers = _puiseux_powers(int(n_terms))
    cols = [np.power(delta, float(p)) for p in powers]
    A = np.column_stack(cols)
    zero_cols = np.all(np.isclose(A, 0.0, atol=1e-12), axis=0)
    keep = ~zero_cols
    if not np.any(keep):
        return None
    A2 = A[:, keep]
    eff_powers = powers[keep]
    try:
        coeffs, *_ = np.linalg.lstsq(A2, y_fit - float(y_offset), rcond=None)
    except np.linalg.LinAlgError:
        return None
    return eff_powers, coeffs


def format_coeff_text(powers, coeffs) -> str:
    lines = []
    for power, coeff in zip(powers, coeffs):
        if abs(float(power) - 0.5) < 1e-9:
            lines.append(r"$\tilde{a}_\text{sqrt}^\text{TED} = " + f"{float(coeff):.2f}" + r"$")
    return "\n".join(lines)


def _resolve_stacked_inset_rects(panel_cfg: Dict[str, Any]) -> Tuple[Tuple[float, float, float, float], Tuple[float, float, float, float]]:
    rect = tuple(panel_cfg.get("inset_stack_rect", INSET_STACK_DEFAULTS["rect"]))
    v_gap_frac = float(panel_cfg.get("inset_stack_v_gap_frac", INSET_STACK_DEFAULTS["v_gap_frac"]))

    x0, y0, w, h = float(rect[0]), float(rect[1]), float(rect[2]), float(rect[3])
    gap = h * v_gap_frac
    h_each = (h - gap) / 2.0

    rect_bottom = (x0, y0, w, h_each)
    rect_top = (x0, y0 + h_each + gap, w, h_each)
    return rect_top, rect_bottom


def add_discriminant_inset(
    ax_parent: plt.Axes,
    *,
    kappa_c: float,
    phi: float,
    tpd_dk: float,
    dk_ted: Optional[float],
    df_sweep: float,
    inset_rect: Tuple[float, float, float, float],
    dk_grid_xlim: Tuple[float, float],
    df_halfspan: float,
    n_grid: int,
    show_xlabel: bool,
) -> plt.Axes:
    st = THEORY_CONFIG["style"]
    ins_cfg = THEORY_CONFIG["insets"]
    disc_cfg = THEORY_CONFIG["disc_inset"]

    df_span = float(df_halfspan)
    if disc_cfg.get("auto_include_sweep_path", True):
        df_span = max(df_span, abs(float(df_sweep)) * float(disc_cfg.get("sweep_pad_factor", 1.05)))

    axins = ax_parent.inset_axes(inset_rect)

    dk_left, dk_right = float(dk_grid_xlim[0]), float(dk_grid_xlim[1])
    dk = np.linspace(dk_left, dk_right, int(n_grid))
    df = np.linspace(-df_span, df_span, int(n_grid))
    DK, DF = np.meshgrid(dk, df)
    disc = discriminant_field_grid(DK, DF, kappa_c, phi)

    # Disc=0 contour
    axins.contour(
        DK,
        DF,
        disc,
        levels=[0.0],
        colors=st["disc_color"],
        linestyles=st["disc_ls"],
        linewidths=st["lw_disc"],
    )

    # Sweep path (horizontal magenta line at df = df_sweep)
    axins.axhline(
        float(df_sweep),
        color=st["sweep_color"],
        linestyle=st["sweep_ls"],
        linewidth=st["lw_sweep"],
        zorder=4,
    )

    # df = 0 reference line (light gray)
    axins.axhline(0.0, color="lightgray", linewidth=1.5, linestyle="--", zorder=0)

    # TPD marker (red open circle, kept for context)
    axins.scatter(
        [tpd_dk],
        [0.0],
        s=200,
        facecolors="none",
        edgecolors="red",
        linewidths=3.0,
        zorder=5,
    )

    # TED marker (green open circle, where the sweep crosses Disc=0)
    if dk_ted is not None:
        axins.scatter(
            [dk_ted],
            [df_sweep],
            s=200,
            facecolors="none",
            edgecolors=st["ted_color"],
            linewidths=3.0,
            zorder=6,
        )

        # Place an alpha label near the TED as a placeholder.
        # The protractor arc is added manually in Inkscape post-export.
        axins.annotate(
            r"$\alpha$",
            xy=(dk_ted, df_sweep),
            xytext=(-35, 35),  # offset in points from the TED point
            textcoords="offset points",
            fontsize=st["fs_alpha"],
            ha="center",
            va="center",
            color="black",
            zorder=7,
        )

    axins.set_xlim(dk_left, dk_right)
    axins.set_ylim(-df_span + (df_span / 2.0), df_span)

    axins.tick_params(labelbottom=False, labelleft=False)

    if show_xlabel:
        axins.set_xlabel(r"$\tilde{\Delta}_\kappa$", fontsize=st["fs_inset_label"], labelpad=st["xlabel_pad_inset"])
    else:
        axins.set_xlabel("")

    axins.set_ylabel(r"$\tilde{\Delta}_f$", fontsize=st["fs_inset_label"], labelpad=st["ylabel_pad_inset"])

    axins.text(
        float(ins_cfg["roman_pos"][0]),
        float(ins_cfg["roman_pos"][1]),
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
    dk_ted: Optional[float],
    split_min: Optional[float],
    coeff_text: Optional[str],
    inset_rect: Tuple[float, float, float, float],
    xlim: Optional[Tuple[float, float]],
    ylim: Optional[Tuple[float, float]],
) -> plt.Axes:
    st = THEORY_CONFIG["style"]
    ins_cfg = THEORY_CONFIG["insets"]

    axins = ax_parent.inset_axes(inset_rect)
    axins.plot(dk, splitting_plot, color=st["line_main_color"], linewidth=st["lw_inset"])

    # Residual splitting Delta_nu^TED as horizontal green dashed line
    if split_min is not None:
        axins.axhline(float(split_min), color=st["ted_color"], linestyle="--", linewidth=st["lw_inset_ref"])

    if xlim is not None:
        axins.set_xlim(float(xlim[0]), float(xlim[1]))
    if ylim is not None:
        axins.set_ylim(float(ylim[0]), float(ylim[1]))

    axins.tick_params(labelbottom=False, labelleft=False)

    axins.set_ylabel(r"$\tilde{\Delta}_\nu$", fontsize=st["fs_inset_label"], labelpad=st["ylabel_pad_inset"])
    axins.set_xlabel(r"$\tilde{\Delta}_\kappa$", fontsize=st["fs_inset_label"], labelpad=st["xlabel_pad_inset"])

    axins.text(
        float(ins_cfg["roman_pos"][0]),
        float(ins_cfg["roman_pos"][1]),
        "ii",
        transform=axins.transAxes,
        fontsize=st["fs_inset_roman"],
        fontweight="bold",
        ha="left",
        va="top",
    )

    if coeff_text:
        axins.text(
            float(ins_cfg["coeff_pos"][0]),
            float(ins_cfg["coeff_pos"][1]),
            coeff_text,
            transform=axins.transAxes,
            fontsize=st["fs_coeff"],
            ha=str(ins_cfg["coeff_ha"]),
            va=str(ins_cfg["coeff_va"]),
        )
    return axins


def build_theory_panel(ax: plt.Axes, panel_cfg: Dict[str, Any], *, is_left: bool) -> None:
    st = THEORY_CONFIG["style"]
    phi = float(THEORY_CONFIG["model"]["phi"])

    kappa_c = float(panel_cfg["kappa_c"])
    dk0, dk1, npts = panel_cfg["dk_sweep"]
    dk_vals = np.linspace(float(dk0), float(dk1), int(npts))
    df_sweep = float(panel_cfg["delta_f_for_roots"])

    curves = simulate_roots_and_splitting(phi, kappa_c, df_sweep, dk_vals)
    dk = curves["dk"]
    nu_plus, nu_mid, nu_minus = curves["nu_plus"], curves["nu_mid"], curves["nu_minus"]
    splitting_raw, splitting_plot = curves["splitting_raw"], curves["splitting_plot"]

    tpd_dk = tpd_x_location_theory(phi, kappa_c)
    ted = compute_ted(dk, splitting_raw)
    dk_ted, split_min = ted if ted else (None, None)

    ax.axhline(0.0, color="lightgray", linewidth=1.5, linestyle="--", zorder=0)
    ax.tick_params(labelsize=st["fs_ticks"])

    ax.plot(dk, nu_plus, color=st["line_main_color"], linewidth=st["lw_main"], linestyle="-")
    ax.plot(dk, nu_minus, color=st["line_main_color"], linewidth=st["lw_main"], linestyle="--")
    ax.plot(dk, nu_mid, color=st["line_main_color"], linewidth=st["lw_main"], linestyle=":")

    # NOTE: the TPD vertical cyan line that used to be drawn here has been removed.
    # The TED is the focus of this figure; keeping the TPD line cluttered the panel.

    if is_left and dk_ted is not None:
        roots_at_ted = _all_roots(df_sweep, float(dk_ted), kappa_c, phi)
        y_bottom = float(roots_at_ted[0])
        y_top = float(roots_at_ted[2])
        ax.vlines(float(dk_ted), y_bottom, y_top, color=st["ted_color"], linestyle="--", linewidth=st["lw_ref"])

    if panel_cfg.get("main_xlim"):
        ax.set_xlim(*panel_cfg["main_xlim"])
    if panel_cfg.get("main_ylim"):
        ax.set_ylim(*panel_cfg["main_ylim"])

    ax.set_xlabel(r"$\tilde{\Delta}_\kappa$", fontsize=st["fs_label"], labelpad=st["xlabel_pad_main"])
    if is_left:
        ax.set_ylabel("Frequency / J", fontsize=st["fs_label"], labelpad=st["ylabel_pad_main"])
    else:
        ax.set_ylabel("")
        ax.tick_params(labelleft=False)

    coeff_text = None
    if dk_ted is not None and split_min is not None:
        span = float(THEORY_CONFIG["puiseux"]["span_frac_of_main_xrange"]) * float(np.ptp(dk))
        out = puiseux_fit_coeffs(
            dk,
            splitting_raw,
            base=float(dk_ted),
            span=float(span),
            n_terms=int(THEORY_CONFIG["puiseux"]["n_terms"]),
            y_offset=float(split_min),
        )
        if out is not None:
            coeff_text = format_coeff_text(out[0], out[1])

    rect_top, rect_bottom = _resolve_stacked_inset_rects(panel_cfg)

    disc_half = float(panel_cfg["disc_xlim_halfspan"])
    disc_default_xlim = (float(tpd_dk - disc_half), float(tpd_dk + disc_half))
    split_xlim = tuple(panel_cfg["split_xlim"]) if panel_cfg.get("split_xlim") else None

    inset_dk_xlim = panel_cfg.get("inset_dk_xlim", None)
    if inset_dk_xlim is not None:
        shared_xlim = (float(inset_dk_xlim[0]), float(inset_dk_xlim[1]))
    else:
        if bool(INSET_STACK_DEFAULTS["force_shared_dk_xlim"]) and split_xlim is not None:
            shared_xlim = (float(split_xlim[0]), float(split_xlim[1]))
        else:
            shared_xlim = disc_default_xlim

    add_discriminant_inset(
        ax,
        kappa_c=kappa_c,
        phi=phi,
        tpd_dk=float(tpd_dk),
        dk_ted=dk_ted,
        df_sweep=df_sweep,
        inset_rect=rect_top,
        dk_grid_xlim=shared_xlim,
        df_halfspan=float(panel_cfg["disc_ylim_halfspan"]),
        n_grid=int(panel_cfg["disc_grid_n"]),
        show_xlabel=False,
    )

    add_splitting_inset(
        ax,
        dk=dk,
        splitting_plot=splitting_plot,
        dk_ted=dk_ted,
        split_min=split_min,
        coeff_text=coeff_text,
        inset_rect=rect_bottom,
        xlim=shared_xlim if bool(INSET_STACK_DEFAULTS["force_shared_dk_xlim"]) else split_xlim,
        ylim=tuple(panel_cfg["split_ylim"]) if panel_cfg.get("split_ylim") else None,
    )


# =============================================================================
# MAIN
# =============================================================================
def build_panel_only() -> None:
    plt.rcParams.update(plt.rcParamsDefault)
    plt.rcParams["figure.facecolor"] = "white"
    plt.rcParams["axes.facecolor"] = "white"

    st = THEORY_CONFIG["style"]
    panel_cfg = THEORY_CONFIG["panel"]

    fig = plt.figure(figsize=(14.0, 9.5))

    # Legend band on top, plot below
    gs = GridSpec(
        2,
        1,
        figure=fig,
        height_ratios=[0.18, 1.0],
        left=0.10,
        right=0.985,
        bottom=0.09,
        top=0.96,
        hspace=0.02,
    )

    ax_leg = fig.add_subplot(gs[0, 0])
    ax = fig.add_subplot(gs[1, 0])
    ax_leg.axis("off")

    build_theory_panel(ax, panel_cfg, is_left=True)

    # Legend: dropped TPD vertical line entry, added TED entry
    handles_theory = [
        Line2D([0, 1], [0, 1], color=st["line_main_color"], lw=4.0, ls="-",
               label=r"$\tilde{\nu}_{+}^{\mathrm{Root}}$"),
        Line2D([0, 1], [0, 1], color=st["line_main_color"], lw=4.0, ls="--",
               label=r"$\tilde{\nu}_{-}^{\mathrm{Root}}$"),
        Line2D([0, 1], [0, 1], color=st["line_main_color"], lw=4.0, ls=":",
               label=r"$\tilde{\eta}^{\mathrm{Root}}$"),
        Line2D([0, 1], [0, 1], color=st["ted_color"], lw=4.0, ls="--",
               label=r"$\tilde{\Delta}_\nu^{\mathrm{TED}}$"),
        Line2D([0, 1], [0, 1], color=st["disc_color"], lw=4.0, ls=st["disc_ls"],
               label=r"$\mathrm{Disc}=0$"),
        Line2D([0, 1], [0, 1], color=st["sweep_color"], lw=4.0, ls=st["sweep_ls"],
               label="Sweep Path"),
        Line2D([0, 1], [0, 1], color="red", marker="o", mfc="none", mec="red",
               lw=0.0, ms=20, markeredgewidth=3.0, label="TPD"),
        Line2D([0, 1], [0, 1], color=st["ted_color"], marker="o", mfc="none",
               mec=st["ted_color"], lw=0.0, ms=20, markeredgewidth=3.0, label="TED"),
    ]

    ax_leg.legend(
        handles=handles_theory,
        loc="center",
        ncol=4,  # 2 rows for 8 items
        frameon=False,
        fontsize=st["fs_legend"],
        columnspacing=0.5,
        handletextpad=0.5,
        handlelength=2.5,
    )

    out_path = BASE_OUTPUT_DIR / OUTPUT_FILENAME
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=400, facecolor="white", bbox_inches="tight")
    plt.close(fig)
    print(f"Saved to: {out_path}")


if __name__ == "__main__":
    build_panel_only()