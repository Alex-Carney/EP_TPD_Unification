from __future__ import annotations

from pathlib import Path
import sys
from typing import Iterable, Sequence

import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.lines import Line2D
import numpy as np
from sqlalchemy import create_engine, select
from sqlalchemy.orm import selectinload, sessionmaker

# ---------------------------------------------------------------------------
# Formatting controls - tweak these to adjust the PRL small-multipanel layout.
# ---------------------------------------------------------------------------
CFG = {
    "figure_size": (9, 6),
    "grid": {
        "left": 0.08,
        "right": 0.98,
        "bottom": 0.125,
        "top": 0.98,
        "hspace": 0.32,
        "wspace": 0.27,
    },
    "fonts": {
        "font.family": "sans-serif",
        "font.size": 9,
        "axes.labelsize": 10,
        "axes.titlesize": 10,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "legend.fontsize": 1,
        "axes.labelpad": 0.25,  # distance from axis label to tick labels - smaller values bring labels closer
    },
    "panel_labels": {
        "labels": ["(a)", "(b)", "(c)", "(d)"],
        "fontsize": 20,
        "x_offset_left": 0.075,
        "x_offset_right": 0.102,
        "y_offset": 0.015,
        "color": "black",
    },
    "axis_spacing": {
        "top_subplot_x_labelpad": -7.0,  # Special padding just for the top subplot's x-axis label
    },
    "legend_kappas": {
        0.0: (0.67, 1.96),
        np.pi / 2.0: (1.30, 2.32),
        np.pi: (0.83, 1.66),
    },
}

_THIS_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _THIS_DIR.parents[2]

for candidate in (_PROJECT_ROOT, _THIS_DIR):
    cand_str = str(candidate)
    if cand_str not in sys.path:
        sys.path.insert(0, cand_str)

from figures.small_figures.figure_4_metrics.settings import STYLE
import metric_markers
import metric_calculations
import splitting_data_pane
import experiment_tpds
from models.analysis import AnalyzedExperiment

DATA_LOCATION = _PROJECT_ROOT / "data" / "ep_tpd_transformed_data.db"

ROW_EXPS: tuple[tuple[str, str], ...] = (
    ("cab5729f-0491-41f9-9fc8-acf0096754ab",
     "62156c79-a18b-4c96-b4b5-2a43519d1900"),
    ("02f5e503-3e4e-4651-9ef8-d90104b3721f",
     "47809d5e-9041-456b-80d8-becf48bf1cfd"),
    ("25e2d559-5960-48c9-8cd2-9441cc067b61",
     "3ef835ef-e2be-4eb3-929e-3a5149a00e87"),
)
PHI_SET = (0.0, np.pi / 2.0, np.pi)

KAPPA_TILDE_C = np.linspace(0.0, 2.8, 1_000)
LEFT_TPD_ONLY = True
DELTA_F_UNC = 1.0e-5 / 1.0e-3
DELTA_K_UNC = 1.0e-5 / 1.0e-3
THEORY_ZORDER = 20

engine = create_engine(f"sqlite:///{DATA_LOCATION.as_posix()}")
Session = sessionmaker(bind=engine, future=True)


def _fetch_experiments(ids: Iterable[str]) -> dict[str, AnalyzedExperiment]:
    id_list = list(ids)
    with Session() as session:
        stmt = (
            select(AnalyzedExperiment)
            .options(
                selectinload(AnalyzedExperiment.analyzed_aggregate_traces),
                selectinload(AnalyzedExperiment.theory_data_points),
            )
            .where(AnalyzedExperiment.analyzed_experiment_id.in_(id_list))
        )
        rows = session.scalars(stmt).all()
    found = {row.analyzed_experiment_id for row in rows}
    missing = sorted(set(id_list) - found)
    if missing:
        raise ValueError(f"Missing experiments: {missing}")
    return {row.analyzed_experiment_id: row for row in rows}


def _compute_theory_curves(phi_values: Sequence[float], kappa_values: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    petermann = np.empty((len(phi_values), kappa_values.size))
    strength = np.empty_like(petermann)
    min_split = np.empty_like(petermann)
    max_resp = np.empty_like(petermann)
    ep_dist = np.empty_like(petermann)

    for j, phi in enumerate(phi_values):
        for i, kappa_c in enumerate(kappa_values):
            tpd = experiment_tpds.standard_tpd_locations(phi, kappa_c, left_tpd=LEFT_TPD_ONLY)
            ep_dist[j, i] = metric_calculations.distance_between_ep_and_tpd(phi, kappa_c, left_tpd=LEFT_TPD_ONLY)
            petermann[j, i] = metric_calculations.petermann_factor_at_tpd(tpd, phi)
            strength[j, i] = metric_calculations.splitting_strength_at_tpd_arc_2(phi, kappa_c, left_tpd=LEFT_TPD_ONLY)
            min_split[j, i] = metric_calculations.min_split_over_J(
                phi, kappa_c, DELTA_F_UNC, DELTA_K_UNC, left_tpd=LEFT_TPD_ONLY
            )
            max_resp[j, i] = metric_calculations.calculate_max_response_tilde(strength[j, i], min_split[j, i])

    return petermann, strength, min_split, max_resp, ep_dist


def _figure_text_label(fig: plt.Figure, ax: plt.Axes, text: str, *, column_index: int | None = None) -> None:
    cfg = CFG["panel_labels"]
    bb = ax.get_position()
    x_offset = cfg["x_offset_left"]
    if column_index is not None and column_index == 1:
        x_offset = cfg["x_offset_right"]
    fig.text(
        bb.x0 - x_offset,
        bb.y1 + cfg["y_offset"],
        text,
        fontsize=cfg["fontsize"],
        color=cfg["color"],
        ha="left",
        va="top",
    )


def _apply_panel_labels(fig: plt.Figure, axes: Sequence[plt.Axes]) -> None:
    labels = CFG["panel_labels"]["labels"]
    if len(labels) < len(axes):
        raise ValueError("Not enough labels configured for the displayed panels")
    for idx, (text, ax) in enumerate(zip(labels, axes)):
        column_index = idx % 2
        _figure_text_label(fig, ax, text, column_index=column_index)


def build_single_column_figure(filename: str = "../../.figures/FIG_4_metrics_small.png") -> None:
    plt.rcParams.update(CFG["fonts"])

    all_ids = [exp_id for pair in ROW_EXPS for exp_id in pair]
    experiments = _fetch_experiments(all_ids)
    _, _, min_split, _, _ = _compute_theory_curves(PHI_SET, KAPPA_TILDE_C)

    fig = plt.figure(figsize=CFG["figure_size"])
    grid = GridSpec(
        2,
        2,
        figure=fig,
        left=CFG["grid"]["left"],
        right=CFG["grid"]["right"],
        bottom=CFG["grid"]["bottom"],
        top=CFG["grid"]["top"],
        hspace=CFG["grid"]["hspace"],
        wspace=CFG["grid"]["wspace"],
    )

    # Top row: (a) phi = 0 data, (b) phi = pi data
    def _plot_splitting(ax: plt.Axes, exp_ids: Sequence[str], phi: float) -> None:
        target_kappas = CFG["legend_kappas"].get(phi)
        for exp_id in exp_ids:
            splitting_data_pane.plot_splitting_pane(
                ax,
                analyzed_experiment=experiments[exp_id],
                phi=phi,
                legend_kappas=target_kappas,
            )

    def _apply_custom_x_labelpad(ax: plt.Axes, is_top_row: bool = False):
        # Re-apply the existing x-label with custom padding
        if is_top_row:
            ax.set_xlabel(ax.get_xlabel(), labelpad=CFG["axis_spacing"]["top_subplot_x_labelpad"])

    ax_phi0 = fig.add_subplot(grid[0, 0])
    _plot_splitting(ax_phi0, ROW_EXPS[0], 0.0)
    _apply_custom_x_labelpad(ax_phi0, is_top_row=True)
    ax_phi0.set_xlim([-2.3, 0.8])

    ax_phi_pi = fig.add_subplot(grid[0, 1])
    _plot_splitting(ax_phi_pi, ROW_EXPS[1], np.pi)
    _apply_custom_x_labelpad(ax_phi_pi, is_top_row=True)
    ax_phi_pi.set_xlim([1.65, 3.15])

    # Bottom-left: (c) phi = pi/2 data
    ax_phi_half = fig.add_subplot(grid[1, 0])
    _plot_splitting(ax_phi_half, ROW_EXPS[2], np.pi / 2.0)

    # Bottom-right: (d) theory min splitting
    ax_max = fig.add_subplot(grid[1, 1])
    for j, phi in enumerate(PHI_SET):
        if np.isclose(phi, 0.0):
            label = r"$\phi=0$"
        elif np.isclose(phi, np.pi / 2.0):
            label = r"$\phi=\pi/2$"
        else:
            label = r"$\phi=\pi$"
        col = STYLE.curve_color_map(phi)
        ax_max.plot(KAPPA_TILDE_C, min_split[j], color=col, lw=STYLE.curve_lw, label=label, zorder=THEORY_ZORDER)
        metric_markers.scatter_metric_markers(ax_max, KAPPA_TILDE_C, min_split[j], phi, col)

    star_proxy = Line2D(
        [0],
        [0],
        marker="o",
        color="none",
        markerfacecolor="white",
        markeredgecolor="gray",
        markersize=STYLE.star_ms ** 0.5,
        markeredgewidth=2,
        linewidth=1.4,
        linestyle="None",
        label=r"$\star$ Min splitting",
    )
    tri_proxy = Line2D(
        [0],
        [0],
        marker="^",
        color="none",
        markerfacecolor="white",
        markeredgecolor="gray",
        markersize=STYLE.tri_ms ** 0.5,
        markeredgewidth=2,
        linewidth=1.4,
        linestyle="None",
        label=r"$\triangle$ Min splitting",
    )
    filled_star_proxy = Line2D(
        [0],
        [0],
        marker="*",
        color="none",
        markerfacecolor="gold",
        markeredgecolor="black",
        markersize=STYLE.star_ms ** 0.5,
        markeredgewidth=2,
        linestyle="None",
        label=r"$\star$ Min splitting",
    )

    handles, labels = ax_max.get_legend_handles_labels()
    handles.extend([star_proxy, tri_proxy, filled_star_proxy])
    labels.extend([r"Small $\tilde \kappa_c$", r"Large $\tilde \kappa_c$"])

    ax_max.set_xlabel(r"$\tilde{\kappa}_c$", fontsize=STYLE.label_font)
    ax_max.set_ylabel(r"$\min(\tilde\Delta_\nu)$", fontsize=STYLE.label_font)
    # ax_max.legend(
    #     handles=handles,
    #     labels=labels,
    #     fontsize=STYLE.theory_legend_font,
    #     loc="upper right",
    #     framealpha=1,
    #     borderpad=0.1,
    #     bbox_to_anchor=(1.0, 1.02),
    # )
    ax_max.tick_params(labelsize=STYLE.tick_font)

    for axis in (ax_phi0, ax_phi_half, ax_phi_pi):
        axis.tick_params(labelsize=STYLE.tick_font)

    # Panel labels in requested order
    axes_in_order = [ax_phi0, ax_phi_pi, ax_phi_half, ax_max]
    _apply_panel_labels(fig, axes_in_order)

    output_path = Path(filename)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=STYLE.save_dpi, facecolor="white")
    plt.close(fig)


if __name__ == "__main__":
    default_path = Path("../../.figures/FIG_4_metrics_small.png")
    default_path.parent.mkdir(parents=True, exist_ok=True)
    build_single_column_figure(filename=str(default_path))
