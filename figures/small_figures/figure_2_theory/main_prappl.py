#!/usr/bin/env python3
from __future__ import annotations

import importlib
import sys
from dataclasses import replace
from pathlib import Path
from typing import Tuple

import matplotlib.gridspec as gridspec
import matplotlib.patheffects as patheffects
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.ticker import FormatStrFormatter

# ---------------------------------------------------------------------------
# Configuration block
# ---------------------------------------------------------------------------
CFG = {
    "figure_size": (9, 15),
    "grid": {
        "left": 0.09,
        "right": 0.865,
        "bottom": 0.06,
        "top": 0.95,
        "wspace": 0.09,
        "hspace_top": 0.10,      # spacing BETWEEN the 3 colorplot rows only
        "block_hspace": 0.15,    # spacing BETWEEN colorplots block and bottom cut only
    },
    "fonts": {
        "font.family": "sans-serif",
        "font.size": 12,
        "axes.labelsize": 12,
        "axes.titlesize": 9,
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
        "legend.fontsize": 8,
    },
    "axis_spacing": {
        "y_labelpad": -12.5,
        "x_labelpad": 5,
    },
    "style_scale": {
        "tick_font": 18,
        "label_font": 27,
        "legend_font": 12,
        "scatter_size": 250,
        "scatter_lw": 4,
        "scatter_lw_tpd": 4,
        "contour_lw": 4,
        "corner_tag_font": 18,
    },
    "panel_label": {
        "text": ["(a)", "(b)", "(c)", "(d)", "(e)", "(f)", "(g)", "(h)"],
        "fontsize": 25,
        "fontweight": "bold",
        "x": 0.04,
        "y": 0.93,
        "pathwidth": 1.0,
    },
    "panel_label_g": {
        "x": 0.021125,
        "y": 0.93,
    },
    "colorbar": {
        "x": 0.87,
        "y": 0.322,
        "width": 0.02,
        "height": 0.6275,
        "label": "Petermann Noise Factor",
    },
    "legend": {
        "fontsize": 18,
        "ncol": 4,
        "columnspacing": 0.8,
        "handletextpad": 0.4,
        "bbox_to_anchor": (0.48, 1.008),
    },
    # --- User Controls ---
    "y_limits": {
        "cut_panel": (-1.2, 1.2),
    },
    "top_x_limits": (-4.0, 4.0),
    "bottom_x_limits": (-5.0, 4.0),
    "sample_points": 2501,
    "bottom_legend": {
        "bbox_to_anchor": (-0.015, -0.02),
        "fontsize_scale": 1.5,  # bigger legend
    },
}

_THIS_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _THIS_DIR.parents[2]
_LARGE_FIG_DIR = _PROJECT_ROOT / "figures" / "large_figures" / "figure_2_theory"

for candidate in (_PROJECT_ROOT, _LARGE_FIG_DIR):
    candidate_str = str(candidate)
    if candidate_str not in sys.path:
        sys.path.insert(0, candidate_str)

# Import local aliases
abs_contours = importlib.import_module("abs_contours")
mesh_module = importlib.import_module("mesh_nd")
settings = importlib.import_module("figures.large_figures.figure_2_theory.settings")

# Also import the fully qualified modules to ensure we patch the specific instances
# that abs_contours might be using internally
try:
    import figures.large_figures.figure_2_theory.mesh_nd as pkg_mesh_nd
    import figures.large_figures.figure_2_theory.settings as pkg_settings
except ImportError:
    pkg_mesh_nd = mesh_module
    pkg_settings = settings

STYLE_SMALL = replace(
    settings.STYLE,
    tick_font=CFG["style_scale"]["tick_font"],
    label_font=CFG["style_scale"]["label_font"],
    legend_font=CFG["style_scale"]["legend_font"],
    corner_tag_font=CFG["style_scale"]["corner_tag_font"],
    scatter_size=CFG["style_scale"]["scatter_size"],
    scatter_lw=CFG["style_scale"]["scatter_lw"],
    scatter_lw_tpd=CFG["style_scale"]["scatter_lw_tpd"],
    contour_lw=CFG["style_scale"]["contour_lw"],
)

# --- PATCHING EVERYTHING ---
settings.STYLE = STYLE_SMALL
mesh_module.STYLE = STYLE_SMALL
abs_contours.STYLE = STYLE_SMALL
pkg_settings.STYLE = STYLE_SMALL
pkg_mesh_nd.STYLE = STYLE_SMALL

STYLE = STYLE_SMALL
get_mesh = mesh_module.get
PANE_LABELS = CFG["panel_label"]["text"]


# ---------------------------------------------------------------------------
# Helper Functions
# ---------------------------------------------------------------------------

def _break_at_transitions(y: np.ndarray, splitting: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """Insert NaNs where the system switches between degenerate and split roots."""
    y2 = y.copy()
    s = splitting
    up = np.where((s[1:] > eps) & (s[:-1] <= eps))[0] + 1
    down = np.where((s[1:] <= eps) & (s[:-1] > eps))[0] + 1
    idx = np.unique(np.concatenate([up, down])) if (up.size + down.size) > 0 else np.array([], dtype=int)
    if idx.size > 0:
        y2[idx] = np.nan
    return y2


def _all_roots(delta_f: float, delta_kappa: float, kappa_c: float, phi: float) -> np.ndarray:
    """
    Return three real roots (peak, trough, peak) for the TPD characteristic equation.
    """
    K_tilde = delta_kappa
    D_tilde = delta_f
    kappa_tilde_bar = kappa_c - K_tilde

    p = (-D_tilde ** 2 / 4.0
         + K_tilde ** 2 / 4.0
         - np.cos(phi)
         + kappa_tilde_bar ** 2 / 4.0)

    q = -kappa_tilde_bar * (2.0 * np.sin(phi) - D_tilde * K_tilde) / 4.0

    coeffs = [1.0, 0.0, p, q]
    roots = np.roots(coeffs).astype(complex)
    roots = roots - (delta_f / 2.0)

    real_mask = np.abs(np.imag(roots)) < 1e-8
    real_roots = np.real(roots[real_mask]) if real_mask.any() else np.real(roots)
    real_roots.sort()

    if real_roots.size == 1:
        return np.repeat(real_roots, 3)
    if real_roots.size == 2:
        return np.array([real_roots[0], real_roots[0], real_roots[1]])
    return real_roots[:3]


def calculate_1d_cut(phi: float, kappa_tilde_c: float, dk_array: np.ndarray) -> Tuple[np.ndarray, dict, np.ndarray]:
    """
    Calculates eigenvalue imaginary magnitude and peak locations along Delta_tilde_f = 0.
    """
    n = len(dk_array)

    ev_mag = np.zeros(n)
    unstable_mask = np.zeros(n, dtype=bool)

    nu_plus = np.full(n, np.nan)
    nu_mid = np.full(n, np.nan)
    nu_minus = np.full(n, np.nan)

    df = 0.0

    for i, dk in enumerate(dk_array):
        # Eigenvalues (EP world): |Im(lambda)|
        rad_val = dk * dk - 4.0
        if rad_val < 0.0:
            ev_mag[i] = np.sqrt(abs(rad_val)) / 2.0
        else:
            ev_mag[i] = 0.0

        # Instability
        K_tilde = dk
        D_tilde = df
        radicand_tilde = (-D_tilde ** 2 + 2j * D_tilde * K_tilde + K_tilde ** 2 - 4.0 * np.exp(1j * phi))
        Lambda_tilde = np.sqrt(radicand_tilde)

        thresh_real = -(K_tilde - kappa_tilde_c)
        if Lambda_tilde.real > thresh_real:
            unstable_mask[i] = True

        # Peaks (TPD world)
        roots = _all_roots(df, dk, kappa_tilde_c, phi)
        nu_minus[i], nu_mid[i], nu_plus[i] = roots

    splitting = nu_plus - nu_minus
    splitting[splitting < 0.0] = 0.0

    tpd_data = {
        "nu_plus": _break_at_transitions(nu_plus, splitting),
        "nu_mid": _break_at_transitions(nu_mid, splitting),
        "nu_minus": _break_at_transitions(nu_minus, splitting),
    }

    return ev_mag, tpd_data, unstable_mask


def _legend_handles() -> list[Line2D]:
    st = STYLE
    marker_size = 13.5
    edge_width = 3.5
    contour_legend_lw = 4
    return [
        Line2D([], [], color=st.primary_ep_color,
               marker=st.primary_ep_marker, linestyle="none",
               markersize=marker_size, markeredgewidth=edge_width,
               label=st.primary_ep_label),
        Line2D([], [], color=st.secondary_ep_color,
               marker=st.secondary_ep_marker, linestyle="none",
               markersize=marker_size, markeredgewidth=edge_width,
               label=st.secondary_ep_label),
        Line2D([], [], color=st.primary_tpd_color,
               marker=st.primary_tpd_marker, linestyle="none",
               markerfacecolor="none",
               markersize=marker_size, markeredgewidth=edge_width,
               label=st.primary_tpd_label),
        Line2D([], [], color=st.secondary_tpd_color,
               marker=st.secondary_tpd_marker, linestyle="none",
               markerfacecolor="none",
               markersize=marker_size, markeredgewidth=edge_width,
               label=st.secondary_tpd_label),
        Line2D([], [], color=st.rogue_tpd_color,
               marker=st.rogue_tpd_marker, linestyle="none",
               markerfacecolor="none",
               markersize=marker_size, markeredgewidth=edge_width,
               label=st.rogue_tpd_label),
        Line2D([], [], color=st.q_color, ls=st.q_ls, lw=contour_legend_lw, label=r"$\tilde q = 0$"),
        Line2D([], [], color=st.split_col, ls=st.split_ls, lw=contour_legend_lw, label="Disc = 0"),
        Line2D([], [], color=st.stability_col, ls=st.stability_ls, lw=contour_legend_lw, label="Instability"),
    ]


def _label_panel(ax: plt.Axes, idx: int, color: str = "white") -> None:
    cfg = CFG["panel_label"]
    x_location = cfg["x"] if PANE_LABELS[idx] != "(g)" else CFG["panel_label_g"]["x"]
    y_location = cfg["y"] if PANE_LABELS[idx] != "(g)" else CFG["panel_label_g"]["y"]
    ax.text(
        x_location, y_location, PANE_LABELS[idx],
        transform=ax.transAxes,
        fontsize=cfg["fontsize"],
        fontweight=cfg["fontweight"],
        color=color,
        path_effects=[patheffects.withStroke(linewidth=cfg["pathwidth"], foreground="black")] if color == "white" else [],
        ha="left", va="top",
    )


def _remove_legend_edge(leg: plt.Legend) -> None:
    if leg is None:
        return
    frame = leg.get_frame()
    frame.set_linewidth(0.0)
    frame.set_edgecolor("none")


def build_single_column_figure(
    filename: str = "../../.figures/FIG_2_theory_small.png",
    *,
    N: int = CFG["sample_points"],
) -> None:
    """Render the PRL single-column variant of the theory figure."""
    plt.rcParams.update(CFG["fonts"])

    kappa_tilde_vals = {
        0.0: (0.67, 1.96),
        np.pi: (0.83, 1.66),
        np.pi / 2.0: (1.30, 2.32),
    }
    phi_vals = (0.0, np.pi, np.pi / 2.0)

    fig = plt.figure(figsize=CFG["figure_size"])

    # -----------------------------------------------------------------------
    # NESTED GRIDSPEC:
    #   outer: [top colorplots block] + [bottom 1D cut block]
    #   top block: 3x2 with its own hspace/wspace
    # -----------------------------------------------------------------------
    outer = gridspec.GridSpec(
        2,
        1,
        figure=fig,
        left=CFG["grid"]["left"],
        right=CFG["grid"]["right"],
        bottom=CFG["grid"]["bottom"],
        top=CFG["grid"]["top"],
        hspace=CFG["grid"]["block_hspace"],
        height_ratios=[3.1, 1.0],
    )

    gs_top = gridspec.GridSpecFromSubplotSpec(
        3,
        2,
        subplot_spec=outer[0, 0],
        wspace=CFG["grid"]["wspace"],
        hspace=CFG["grid"]["hspace_top"],
    )

    gs_bot = gridspec.GridSpecFromSubplotSpec(
        1,
        1,
        subplot_spec=outer[1, 0],
    )

    column_mappables = [None, None]
    bounds_top = (float(CFG["top_x_limits"][0]), float(CFG["top_x_limits"][1]))

    # --- ROWS 0-2: CONTOUR PLOTS ---
    for row, phi in enumerate(phi_vals):
        meshes = [
            get_mesh(
                phi,
                kappa_tilde_c=kappa_tilde_vals[phi][0],
                N=N,
                delta_tilde_kappa_lim=bounds_top,
                delta_tilde_f_lim=bounds_top,
            ),
            get_mesh(
                phi,
                kappa_tilde_c=kappa_tilde_vals[phi][1],
                N=N,
                delta_tilde_kappa_lim=bounds_top,
                delta_tilde_f_lim=bounds_top,
            ),
        ]

        for col in range(2):
            ax = fig.add_subplot(gs_top[row, col])

            mappable = abs_contours.plot(
                ax,
                meshes[col],
                phi,
                figure_mode=False,
                legend=False,
                kappa_tilde_c=kappa_tilde_vals[phi][col],
                return_mappable=True,
                include_p=False,
            )
            if column_mappables[col] is None:
                column_mappables[col] = mappable

            _label_panel(ax, row * 2 + col)

            if col == 0:
                ax.set_ylabel(r"$\tilde \Delta_f$", fontsize=STYLE.label_font, labelpad=CFG["axis_spacing"]["y_labelpad"])
            else:
                ax.set_ylabel("")
                ax.set_yticklabels([])

            # Put x axis back on the bottom colorplot row only (row 2)
            if row == 2:
                ax.set_xlabel(
                    r"$\tilde \Delta_\kappa$",
                    fontsize=STYLE.label_font,
                    labelpad=CFG["axis_spacing"]["x_labelpad"],
                )
                ax.tick_params(axis="x", labelsize=STYLE.tick_font)
            else:
                ax.set_xlabel("")
                ax.set_xticklabels([])

            ax.tick_params(axis="both", labelsize=STYLE.tick_font)

    # --- BOTTOM: 1D CUT (SINGLE WIDE PANEL) ---
    phi_cut = 0.0
    kappa_c_cut = 0.67

    dk0_bot, dk1_bot = float(CFG["bottom_x_limits"][0]), float(CFG["bottom_x_limits"][1])
    dk_sweep_bot = np.linspace(dk0_bot, dk1_bot, int(CFG["sample_points"]))

    ev_mag, tpd_data, unstable_mask = calculate_1d_cut(phi_cut, kappa_c_cut, dk_sweep_bot)

    split_lw = 4
    instability_ls = "--"  # solid
    instability_col = STYLE.stability_col
    ep_col = "red"
    tpd_col = "cyan"

    idx_unstable = np.where(np.diff(unstable_mask.astype(int)) != 0)[0]

    ax_cut = fig.add_subplot(gs_bot[0, 0])

    # Instability shading
    ax_cut.fill_between(dk_sweep_bot, -10, 10, where=unstable_mask, color="gray", alpha=0.3, zorder=0)

    # Eigenvalues: dashed black
    ax_cut.plot(dk_sweep_bot, ev_mag, color="black", lw=split_lw, ls="--", zorder=2)
    ax_cut.plot(dk_sweep_bot, -ev_mag, color="black", lw=split_lw, ls="--", zorder=2)

    # Peaks: royalblue
    ax_cut.plot(dk_sweep_bot, tpd_data["nu_plus"], color="royalblue", lw=split_lw, ls="-", zorder=3)
    ax_cut.plot(dk_sweep_bot, tpd_data["nu_minus"], color="royalblue", lw=split_lw, ls="-", zorder=3)

    # EP marker
    ep_loc = -2.0
    ax_cut.axvline(ep_loc, color=ep_col, ls="-", lw=3, zorder=4)

    # TPD marker
    tpd_loc = (kappa_c_cut - np.sqrt(8.0 - kappa_c_cut * kappa_c_cut)) / 2.0
    ax_cut.axvline(tpd_loc, color=tpd_col, ls="-", lw=3, zorder=4)

    # Instability marker (first boundary)
    inst_loc = None
    if len(idx_unstable) > 0:
        inst_loc = float(dk_sweep_bot[idx_unstable[0]])
        ax_cut.axvline(inst_loc, color=instability_col, ls=instability_ls, lw=3, zorder=4)

    if np.any(unstable_mask):
        ax_cut.text(
            dk1_bot - 0.1,
            CFG["y_limits"]["cut_panel"][1] - 0.05,
            "Unstable",
            color="black",
            fontweight="bold",
            fontsize=STYLE.tick_font,
            ha="right",
            va="top",
            rotation=0,
        )

    ax_cut.set_ylim(CFG["y_limits"]["cut_panel"])
    ax_cut.set_xlim(CFG["bottom_x_limits"])

    ax_cut.set_xlabel(r"$\tilde \Delta_\kappa$", fontsize=STYLE.label_font, labelpad=CFG["axis_spacing"]["x_labelpad"])

    # y label on the right, ticks on both sides
    ax_cut.set_ylabel("Frequency / J", fontsize=STYLE.label_font, labelpad=CFG["axis_spacing"]["y_labelpad"] + 15)
    ax_cut.yaxis.set_label_position("right")
    ax_cut.tick_params(axis="both", labelsize=STYLE.tick_font)
    ax_cut.tick_params(axis="y", left=True, labelleft=True, right=True, labelright=True)

    _label_panel(ax_cut, 6, color="black")
    mesh_module.corner_tag(ax_cut, phi_cut, kappa_c_cut)

    # Bottom combined legend (opaque, no edge, nu_pm first)
    leg_handles = [
        Line2D([], [], color="royalblue", lw=split_lw, ls="-", label=r"$\tilde{\nu}_\pm$"),
        Line2D([], [], color="black", lw=split_lw, ls="--", label=r"$|\mathrm{Im}(\tilde{\lambda})|$"),
        Line2D([], [], color=ep_col, lw=3, ls="-", label=r"$\tilde{\Delta}_\kappa^\mathrm{EP}$"),
        Line2D([], [], color=tpd_col, lw=3, ls="-", label=r"$\tilde{\Delta}_\kappa^\mathrm{TPD}$"),
        Line2D([], [], color=instability_col, lw=3, ls=instability_ls, label=r"$\tilde{\Delta}_\kappa^\mathrm{NL}$"),
    ]
    leg = ax_cut.legend(
        handles=leg_handles,
        loc="lower left",
        bbox_to_anchor=CFG["bottom_legend"]["bbox_to_anchor"],
        frameon=True,
        framealpha=1.0,
        facecolor="white",
        fontsize=float(STYLE.legend_font) * float(CFG["bottom_legend"]["fontsize_scale"]),
        handlelength=2.2,
        handletextpad=0.6,
        borderpad=0.25,
        labelspacing=0.35,
    )
    _remove_legend_edge(leg)

    # --- SHARED COLORBAR ---
    for col, mappable in enumerate(column_mappables):
        if mappable is None:
            continue
        cax_cfg = CFG["colorbar"]
        cax = fig.add_axes([cax_cfg["x"], cax_cfg["y"], cax_cfg["width"], cax_cfg["height"]])
        cbar = fig.colorbar(mappable, cax=cax)
        cbar.set_label(cax_cfg["label"], fontsize=STYLE.label_font)
        cbar.ax.tick_params(labelsize=STYLE.tick_font)
        cbar.ax.yaxis.set_major_formatter(FormatStrFormatter("%.1f"))
        break

    # --- TOP LEGEND (CONTOUR SYMBOLS) ---
    legend = fig.legend(
        handles=_legend_handles(),
        loc="upper center",
        ncol=CFG["legend"]["ncol"],
        fontsize=CFG["legend"]["fontsize"],
        columnspacing=CFG["legend"]["columnspacing"],
        handletextpad=CFG["legend"]["handletextpad"],
        frameon=False,
        bbox_to_anchor=CFG["legend"]["bbox_to_anchor"],
    )
    if legend:
        for text in legend.get_texts():
            text.set_fontsize(CFG["legend"]["fontsize"])

    output_path = Path(filename)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    for suffix in [".png", ".svg"]:
        out = output_path.with_suffix(suffix)
        print(f"Saving figure to: {out.as_posix()}")
        fig.savefig(out, dpi=1000 if suffix == ".png" else None, facecolor="white")

    plt.close(fig)


if __name__ == "__main__":
    default_path = Path("../../.figures/FIG_2_theory_small_prappl.png")
    default_path.parent.mkdir(parents=True, exist_ok=True)
    build_single_column_figure(filename=str(default_path), N=max(2501, STYLE.GRID_SIZE // 7))
