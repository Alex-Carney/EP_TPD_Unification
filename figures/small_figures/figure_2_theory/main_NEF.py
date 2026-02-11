# main_NEF.py
#
# Copy of the existing PF main, but with the color field replaced by a
# noise enhancement factor (NEF) computed on the same (Delta_kappa/J, Delta_f/J) mesh.
# Everything else (contours, markers, legend, layout) is kept the same.
#
# Stable region colormap: jet
# Unstable region colormap: gray

from __future__ import annotations

import importlib
import sys
from dataclasses import replace
from pathlib import Path
from matplotlib.colors import LogNorm


import matplotlib.gridspec as gridspec
import matplotlib.patheffects as patheffects
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.ticker import FormatStrFormatter

# ---------------------------------------------------------------------------
# Configuration block - keep identical to PF main except colormap + colorbar label.
# ---------------------------------------------------------------------------
CFG = {
    "figure_size": (9, 12),
    "grid": {
        "left": 0.09,
        "right": 0.84,
        "bottom": 0.075,
        "top": 0.93,
        "wspace": 0.09,
        "hspace": 0.08,
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
        "text": ["(a)", "(b)", "(c)", "(d)", "(e)", "(f)"],
        "fontsize": 27,
        "x": 0.04,
        "y": 0.93,
        "fontweight": "bold",
        "pathwidth": 1.5,
    },
    "colorbar": {
        "x": 0.845,
        "y": 0.074,
        "width": 0.035,
        "height": 0.856,
        "label": "Thermal Noise Efficiency (TNE)",
    },
    "legend": {
        "fontsize": 18,
        "ncol": 4,
        "columnspacing": 0.8,
        "handletextpad": 0.4,
        "bbox_to_anchor": (0.48, 1.008),
    },
    "sample_points": 101,
    # NEF parameters (kept simple and constant for now)
    "n_th": 1.0,
    "f_tilde_c": 5.0,  # must match mesh_nd.get default if not overridden
}

_THIS_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _THIS_DIR.parents[2]
_LARGE_FIG_DIR = _PROJECT_ROOT / "figures" / "large_figures" / "figure_2_theory"

for candidate in (_PROJECT_ROOT, _LARGE_FIG_DIR):
    candidate_str = str(candidate)
    if candidate_str not in sys.path:
        sys.path.insert(0, candidate_str)

abs_contours = importlib.import_module("abs_contours")
mesh_module = importlib.import_module("mesh_nd")
settings = importlib.import_module("figures.large_figures.figure_2_theory.settings")

# Scale style for the small-figure layout, but switch stable colormap to jet.
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
    cmap="inferno",
)

# Push the scaled style into shared modules so helper functions use it.
settings.STYLE = STYLE_SMALL
mesh_module.STYLE = STYLE_SMALL
abs_contours.STYLE = STYLE_SMALL

STYLE = STYLE_SMALL
get_mesh = mesh_module.get
MeshND = mesh_module.MeshND

PANE_LABELS = CFG["panel_label"]["text"]


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


def _label_panel(ax: plt.Axes, idx: int) -> None:
    cfg = CFG["panel_label"]
    ax.text(
        cfg["x"], cfg["y"], PANE_LABELS[idx],
        transform=ax.transAxes,
        fontsize=cfg["fontsize"],
        fontweight=cfg["fontweight"],
        color="white",
        ha="left", va="top",
    )

def _nef_field_for_mesh(mesh: MeshND, *, phi: float, kappa_tilde_c: float, f_tilde_c: float, n_th: float) -> np.ndarray:
    """
    Compute NEF on the (Delta_kappa/J, Delta_f/J) mesh using the SAME definition as the 1D scripts:
      - peak frequencies come from peaks.peak_location(...)
      - NEF is evaluated at those peak frequencies via the noise expression
    """
    import fitting.peak_fitting as peaks  # local import so main_NEF stays self-contained

    K = mesh.G  # Delta_kappa/J
    D = mesh.D  # Delta_f/J

    # Coupled constraints on the mesh
    kappa_y = -2.0 * K + kappa_tilde_c
    f_y = f_tilde_c - D

    i = 1.0j

    def nef_at_fd(fd: float, fy_val: float, ky_val: float) -> float:
        # This is your MATLAB-derived expression, in tilde units (J = 1).
        den = (
            4.0 * f_tilde_c * fd * i
            - 4.0 * f_tilde_c * fy_val * i
            + 4.0 * fd * fy_val * i
            - 2.0 * f_tilde_c * ky_val
            + 2.0 * fd * kappa_tilde_c
            + 2.0 * fd * ky_val
            - 2.0 * fy_val * kappa_tilde_c
            + kappa_tilde_c * ky_val * i
            + 4.0 * np.exp(phi * i) * i
            - 4.0 * (fd ** 2) * i
        )

        den_abs_sq = (np.abs(den) ** 2)

        term1 = (4.0 * ky_val * ((2.0 * f_tilde_c - 2.0 * fd) ** 2 + (kappa_tilde_c ** 2))) / den_abs_sq
        term2 = (16.0 * kappa_tilde_c) / den_abs_sq

        out = ky_val * n_th * (term1 + term2)

        if not np.isfinite(out):
            return float("nan")
        return float(np.abs(out))

    nef = np.empty_like(K, dtype=float)

    nrows, ncols = K.shape
    for r in range(nrows):
        for c in range(ncols):
            dk_val = float(K[r, c])
            df_val = float(D[r, c])

            fy_val = float(f_y[r, c])
            ky_val = float(kappa_y[r, c])

            # Peak locations from your peak finder (this is the key point)
            J_val = 1
            peak_list = peaks.peak_location(J_val, f_tilde_c, kappa_tilde_c, df_val, dk_val, float(phi))

            if len(peak_list) == 2:
                nu_plus = float(max(peak_list))
                nu_minus = float(min(peak_list))

                a = nef_at_fd(nu_plus, fy_val, ky_val)
                b = nef_at_fd(nu_minus, fy_val, ky_val)

                # Single scalar for colormap -- take the max of the two peaks, even if they differ
                nef[r, c] = np.nanmax(np.array([a, b], dtype=float))
            elif len(peak_list) == 1:
                # If your peak finder returns one peak, use that peak only
                nu = float(peak_list[0])
                nef[r, c] = nef_at_fd(nu, fy_val, ky_val)
            else:
                nef[r, c] = float("nan")

    return nef



def _mesh_with_nef(mesh: MeshND, *, phi: float, kappa_tilde_c: float) -> MeshND:
    nef = _nef_field_for_mesh(
        mesh,
        phi=phi,
        kappa_tilde_c=kappa_tilde_c,
        f_tilde_c=float(CFG["f_tilde_c"]),
        n_th=float(CFG["n_th"]),
    )
    # Replace diff_abs (used by abs_contours.plot as the color field) with NEF.
    return MeshND(
        mesh.G, mesh.D,
        mesh.diff_imag,
        mesh.diff_real,
        nef,
        mesh.max_real,
        mesh.thresh_imag,
        mesh.thresh_real,
        mesh.disc_field,
        mesh.q_field,
        mesh.p_field,
        mesh.unstable,
    )


def build_single_column_figure(
    filename: str = "../../.figures/FIG_2_theory_small_NEF.png",
    *,
    N: int = CFG["sample_points"],
) -> None:
    plt.rcParams.update(CFG["fonts"])

    kappa_tilde_vals = {
        0.0: (0.67, 1.96),
        np.pi: (0.83, 1.66),
        np.pi / 2: (1.30, 2.32),
    }
    phi_vals = (0.0, np.pi, np.pi / 2)

    fig = plt.figure(figsize=CFG["figure_size"])
    gs = gridspec.GridSpec(
        3,
        2,
        figure=fig,
        left=CFG["grid"]["left"],
        right=CFG["grid"]["right"],
        bottom=CFG["grid"]["bottom"],
        top=CFG["grid"]["top"],
        wspace=CFG["grid"]["wspace"],
        hspace=CFG["grid"]["hspace"],
    )

    axes: list[list[plt.Axes]] = [[None, None] for _ in phi_vals]  # type: ignore
    column_mappables = [None, None]
    bounds = (-4, 4)

    for row, phi in enumerate(phi_vals):
        meshes_raw = [
            get_mesh(
                phi,
                kappa_tilde_c=kappa_tilde_vals[phi][0],
                N=N,
                delta_tilde_kappa_lim=bounds,
                delta_tilde_f_lim=bounds,
                f_tilde_c=float(CFG["f_tilde_c"]),
            ),
            get_mesh(
                phi,
                kappa_tilde_c=kappa_tilde_vals[phi][1],
                N=N,
                delta_tilde_kappa_lim=bounds,
                delta_tilde_f_lim=bounds,
                f_tilde_c=float(CFG["f_tilde_c"]),
            ),
        ]

        # Replace PF field with NEF field (only change to the plotted data).
        meshes = [
            _mesh_with_nef(meshes_raw[0], phi=phi, kappa_tilde_c=kappa_tilde_vals[phi][0]),
            _mesh_with_nef(meshes_raw[1], phi=phi, kappa_tilde_c=kappa_tilde_vals[phi][1]),
        ]

        for col in range(2):
            ax = fig.add_subplot(gs[row, col])
            axes[row][col] = ax

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
                ax.set_ylabel(r"$\tilde \Delta_f$", fontsize=STYLE.label_font,
                              labelpad=CFG["axis_spacing"]["y_labelpad"])
            else:
                ax.set_ylabel("")
                ax.set_yticklabels([])

            if row < len(phi_vals) - 1:
                ax.set_xlabel("")
                ax.set_xticklabels([])
            else:
                ax.set_xlabel(r"$\tilde \Delta_\kappa$", fontsize=STYLE.label_font,
                              labelpad=CFG["axis_spacing"]["x_labelpad"])

            ax.tick_params(axis="both", labelsize=STYLE.tick_font)

    # Single shared colorbar (same geometry)
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
    plt.tight_layout()
    fig.savefig(output_path, dpi=STYLE.save_dpi, facecolor="white")
    plt.close(fig)


if __name__ == "__main__":
    default_path = Path("../../.figures/FIG_2_theory_small_NEF.png")
    default_path.parent.mkdir(parents=True, exist_ok=True)
    build_single_column_figure(filename=str(default_path), N=2501)
