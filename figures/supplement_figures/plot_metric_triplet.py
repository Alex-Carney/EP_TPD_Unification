#!/usr/bin/env python3
"""
plot_metric_triplet.py

Supplemental figure rendering the three theory-only metrics from Fig. 4:
TPD strength, Petermann factor at the TPD, and the two-norm distance between
the EP and the TPD.  Curves reuse the existing metric utilities and styling,
including reference markers (star/triangle and Ref. 19 star).
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import importlib

# -----------------------------------------------------------------------------
# Import project modules (ensure repo root is on sys.path first)
# -----------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from figures.large_figures.figure_3_experiment.experiment_tpds import (  # noqa: E402
    standard_tpd_locations,
)
_metrics_settings = importlib.import_module("figures.large_figures.figure_4_metrics.settings")  # noqa: E402
sys.modules.setdefault("settings", _metrics_settings)
STYLE = _metrics_settings.STYLE
from figures.large_figures.figure_4_metrics.metric_markers import (  # noqa: E402
    scatter_metric_markers,
)
from figures.large_figures.figure_4_metrics.metric_calculations import (  # noqa: E402
    petermann_factor_at_tpd,
    splitting_strength_at_tpd_arc_2,
    distance_between_ep_and_tpd,
)

# -----------------------------------------------------------------------------
# Parameter grids
# -----------------------------------------------------------------------------
PHI_SET = (0.0, np.pi, np.pi / 2.0)
KAPPA_TILDE_C = np.linspace(0.0, 2.8, 1_000)
LEFT_TPD_ONLY = True

def build() -> Path:
    """Generate the three-metric supplemental plot and return the output path."""
    strength_curves = np.empty((len(PHI_SET), KAPPA_TILDE_C.size))
    petermann_curves = np.empty_like(strength_curves)
    distance_curves = np.empty_like(strength_curves)

    for j, phi in enumerate(PHI_SET):
        for i, kappa_c in enumerate(KAPPA_TILDE_C):
            tpd_loc = standard_tpd_locations(phi, kappa_c, left_tpd=LEFT_TPD_ONLY)
            distance_curves[j, i] = distance_between_ep_and_tpd(phi, kappa_c, left_tpd=LEFT_TPD_ONLY)
            petermann_curves[j, i] = petermann_factor_at_tpd(tpd_loc, phi)
            strength_curves[j, i] = splitting_strength_at_tpd_arc_2(phi, kappa_c, left_tpd=LEFT_TPD_ONLY)

    plt.rcParams.update(plt.rcParamsDefault)
    plt.rcParams["figure.facecolor"] = "white"
    plt.rcParams["axes.facecolor"] = "white"

    fig, axes = plt.subplots(
        nrows=1,
        ncols=3,
        figsize=(15, 5),
        sharex=False,
    )

    metric_data = (
        (strength_curves, r"TPD Strength / J", None),
        (petermann_curves, r"PF(TPD)", (0.75, 5.0)),
        (distance_curves, r"$\mathrm{dist}_2(\mathrm{EP\!, \!TPD})$", None),
    )

    phi_labels = {
        0.0: r"$\phi = 0$",
        np.pi: r"$\phi = \pi$",
        np.pi / 2.0: r"$\phi = \pi/2$",
    }

    legend_handles: list[plt.Line2D] = []
    legend_labels: list[str] = []

    for ax, (metric, ylabel, ylim) in zip(axes, metric_data):
        for j, phi in enumerate(PHI_SET):
            color = STYLE.curve_color_map(phi)
            line, = ax.plot(
                KAPPA_TILDE_C,
                metric[j],
                color=color,
                lw=STYLE.curve_lw,
                label=phi_labels[phi],
                zorder=20,
            )
            scatter_metric_markers(
                ax,
                KAPPA_TILDE_C,
                metric[j],
                phi,
                color,
            )

            if ax is axes[0]:
                legend_handles.append(line)
                legend_labels.append(phi_labels[phi])

        ax.set_xlabel(r"$\tilde{\kappa}_c$", fontsize=STYLE.label_font)
        ax.set_ylabel(ylabel, fontsize=STYLE.label_font)
        ax.tick_params(labelsize=STYLE.tick_font)
        if ylim is not None:
            ax.set_ylim(*ylim)

    # Horizontal PF = 1 reference line on the middle (Petermann) panel
    pf_ax = axes[1]
    pf_line = pf_ax.axhline(
        1.0,
        color="gray",
        linestyle="--",
        linewidth=2.0,
        label=r"$\mathrm{PF}=1$",
    )
    pf_ax.legend(
        [pf_line],
        [r"$\mathrm{PF}=1$"],
        fontsize=STYLE.theory_legend_font,
        frameon=False,
        loc="upper right",
    )

    if legend_handles:
        star_proxy = plt.Line2D(
            [0],
            [0],
            marker="o",
            color="none",
            markerfacecolor="white",
            markeredgecolor="gray",
            markersize=STYLE.star_ms**0.5,
            markeredgewidth=2.0,
            linestyle="None",
            label=r"$\star$ Min splitting",
        )
        tri_proxy = plt.Line2D(
            [0],
            [0],
            marker="^",
            color="none",
            markerfacecolor="white",
            markeredgecolor="gray",
            markersize=STYLE.tri_ms**0.5,
            markeredgewidth=2.0,
            linestyle="None",
            label=r"$\triangle$ Min splitting",
        )
        ref_proxy = plt.Line2D(
            [0],
            [0],
            marker="*",
            color="none",
            markerfacecolor="gold",
            markeredgecolor="black",
            markersize=STYLE.ref_ms**0.5,
            markeredgewidth=2.0,
            linestyle="None",
            label=r"$\star$ Ref. 19",
        )
        legend_handles.extend([star_proxy, tri_proxy, ref_proxy])
        legend_labels.extend([
            r"Small $\tilde{\kappa}_c$",
            r"Large $\tilde{\kappa}_c$",
            r"Ref. 19 $\tilde{\kappa}_c$",
        ])
        axes[0].legend(
            legend_handles,
            legend_labels,
            fontsize=STYLE.theory_legend_font,
            loc="lower left",
            framealpha=1.0,
            borderpad=0.2,
            bbox_to_anchor=(-0.025, -0.025)
        )

    fig.tight_layout()

    # ── Add bold panel letters a, b, c ─────────────────────────────
    for ax, letter in zip(axes, "abc"):
        pos = ax.get_position()
        fig.text(pos.x0 - 0.025, pos.y1 - 0.05,
                 letter, fontsize=30, fontweight="bold",
                 ha="right", va="bottom")

    output_dir = Path(__file__).resolve().parents[1] / ".figures"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "SUPP_metric_triplet.png"
    fig.savefig(output_path, dpi=STYLE.save_dpi, facecolor="white")
    plt.close(fig)
    return output_path



if __name__ == "__main__":
    build()
