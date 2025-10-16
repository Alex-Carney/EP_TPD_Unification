from __future__ import annotations

import importlib
import sys
from dataclasses import replace
from pathlib import Path
from typing import Iterable

import matplotlib.gridspec as gridspec
import matplotlib.patheffects as patheffects
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.ticker import FormatStrFormatter

# ---------------------------------------------------------------------------
# Configuration block – tweak these values to adjust the PRL layout quickly.
# ---------------------------------------------------------------------------
CFG = {
    "figure_size": (9, 12),        # Overall figure size in inches
    "grid": {                          # GridSpec margins and spacing
        "left": 0.09,
        "right": 0.85,
        "bottom": 0.075,
        "top": 0.93,
        "wspace": 0.09,
        "hspace": 0.08,
    },
    "fonts": {                         # Matplotlib rcParams overrides
        "font.family": "sans-serif",
        "font.size": 12,
        "axes.labelsize": 12,
        "axes.titlesize": 9,
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
        "legend.fontsize": 8,
    },
    "axis_spacing": {                  # Spacing between labels and other elements
        "y_labelpad": -12.5,               # Space between y-axis label and tick labels (can be negative)
        "x_labelpad": 5,               # Space between x-axis label and tick labels
    },
    "style_scale": {                   # Down-scaling of large-figure STYLE
        "tick_font": 18,
        "label_font": 27,
        "legend_font": 12,
        "scatter_size": 250,
        "scatter_lw": 4,
        "scatter_lw_tpd": 4,
        "contour_lw": 4,
        "corner_tag_font": 18,         # Adjusted to smaller font size for small figures
    },
    "panel_label": {                   # (a)–(f) annotations
        "text": ["(a)", "(b)", "(c)", "(d)", "(e)", "(f)"],
        "fontsize": 27,
        "x": 0.04,
        "y": 0.93,
        "pathwidth": 1.5,
    },
    "colorbar": {                      # Shared colour bar geometry
        "x": 0.86,
        "y": 0.074,
        "width": 0.035,
        "height": 0.856,
        "label": "Petermann Noise Factor",
    },
    "legend": {                        # Global legend layout (top of page)
        "fontsize": 18,
        "ncol": 4,
        "columnspacing": 0.8,
        "handletextpad": 0.4,
        "bbox_to_anchor": (0.48, 1.008),
    },
    "sample_points": 601,              # Mesh density when re-rendering
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

STYLE_SMALL = replace(
    settings.STYLE,
    tick_font=CFG["style_scale"]["tick_font"],
    label_font=CFG["style_scale"]["label_font"],
    legend_font=CFG["style_scale"]["legend_font"],
    corner_tag_font=CFG["style_scale"]["corner_tag_font"],  # Added corner tag font
    scatter_size=CFG["style_scale"]["scatter_size"],
    scatter_lw=CFG["style_scale"]["scatter_lw"],
    scatter_lw_tpd=CFG["style_scale"]["scatter_lw_tpd"],
    contour_lw=CFG["style_scale"]["contour_lw"],
)

# Push the scaled style into the shared modules so that helper functions use it.
settings.STYLE = STYLE_SMALL
mesh_module.STYLE = STYLE_SMALL
abs_contours.STYLE = STYLE_SMALL

STYLE = STYLE_SMALL
get_mesh = mesh_module.get

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
        Line2D([], [], color=st.stability_col, ls=st.stability_ls, lw=contour_legend_lw,
               label="Instability"),
    ]


def _label_panel(ax: plt.Axes, idx: int) -> None:
    cfg = CFG["panel_label"]
    ax.text(
        cfg["x"], cfg["y"], PANE_LABELS[idx],
        transform=ax.transAxes,
        fontsize=cfg["fontsize"],
        color="white",
        path_effects=[patheffects.withStroke(linewidth=cfg["pathwidth"],
                                             foreground="black")],
        ha="left", va="top",
    )


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
        meshes = [
            get_mesh(
                phi,
                kappa_tilde_c=kappa_tilde_vals[phi][0],
                N=N,
                delta_tilde_kappa_lim=bounds,
                delta_tilde_f_lim=bounds,
            ),
            get_mesh(
                phi,
                kappa_tilde_c=kappa_tilde_vals[phi][1],
                N=N,
                delta_tilde_kappa_lim=bounds,
                delta_tilde_f_lim=bounds,
            ),
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
                # Keep the label empty for second column
                ax.set_ylabel("")
                # Remove y-ticks for second column
                ax.set_yticklabels([])

            if row < len(phi_vals) - 1:
                ax.set_xlabel("")
                ax.set_xticklabels([])
            else:
                ax.set_xlabel(r"$\tilde \Delta_\kappa$", fontsize=STYLE.label_font,
                             labelpad=CFG["axis_spacing"]["x_labelpad"])

            ax.tick_params(axis="both", labelsize=STYLE.tick_font)

    for col, mappable in enumerate(column_mappables):
        if mappable is None:
            continue
        cax_cfg = CFG["colorbar"]
        cax = fig.add_axes([cax_cfg["x"], cax_cfg["y"], cax_cfg["width"], cax_cfg["height"]])
        cbar = fig.colorbar(mappable, cax=cax)
        cbar.set_label(cax_cfg["label"], fontsize=STYLE.label_font)
        cbar.ax.tick_params(labelsize=STYLE.tick_font)
        cbar.ax.yaxis.set_major_formatter(FormatStrFormatter("%.1f"))
        break  # single shared colour bar

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
    default_path = Path("../../.figures/FIG_2_theory_small.png")
    default_path.parent.mkdir(parents=True, exist_ok=True)
    build_single_column_figure(filename=str(default_path), N=max(2501, STYLE.GRID_SIZE // 7))