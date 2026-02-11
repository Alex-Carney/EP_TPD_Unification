from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.lines import Line2D

from sqlalchemy import create_engine, select
from sqlalchemy.orm import sessionmaker, selectinload

from figures.large_figures.figure_3_experiment.experiment_tpds import standard_tpd_locations
from figures.large_figures.figure_4_metrics.metric_calculations import nuisance_scaling_coefficient_at_tpd
from settings import STYLE
from metric_markers import scatter_metric_markers
from metric_calculations import (
    petermann_factor_at_tpd,
    splitting_strength_at_tpd_arc_2,
    min_split_over_J,
    thermal_noise_efficiency_at_tpd,
    distance_between_ep_and_tpd,
    distance_between_instability_and_tpd,
)
from splitting_data_pane import plot_splitting_pane
from models.analysis import AnalyzedExperiment


# --------------- database path and experiment IDs ----------------------
DATA_LOCATION = Path("../../../data/ep_tpd_transformed_data.db")

ROW_EXPS: tuple[tuple[str, str], ...] = (
    (
        "cab5729f-0491-41f9-9fc8-acf0096754ab",
        "62156c79-a18b-4c96-b4b5-2a43519d1900",
    ),  # phi = 0
    (
        "02f5e503-3e4e-4651-9ef8-d90104b3721f",
        "47809d5e-9041-456b-80d8-becf48bf1cfd",
    ),  # phi = pi
    (
        "25e2d559-5960-48c9-8cd2-9441cc067b61",
        "3ef835ef-e2be-4eb3-929e-3a5149a00e87",
    ),  # phi = pi/2
)

PHI_SET = (0.0, np.pi, np.pi / 2.0)  # same order as ROW_EXPS

# --------------- theory-grid parameters --------------------------------
KAPPA_TILDE_C = np.linspace(0.0, 2.8, 5_000)
LEFT_TPD_ONLY = True
DELTA_F_UNC = 1.0e-5 / 1.0e-3
DELTA_K_UNC = 1.0e-5 / 1.0e-3
THEORY_ZORDER = 20

F_TILDE_C_FOR_NE = 10.0


def phi_color(phi: float) -> str:
    if np.isclose(phi, 0.0):
        return "royalblue"
    if np.isclose(phi, np.pi):
        return "purple"
    if np.isclose(phi, np.pi / 2.0):
        return "darkgreen"
    return "black"


# Optional: patch STYLE color maps so splitting panes match the legend colors
# (plot_splitting_pane uses STYLE.theory_color_map(phi))
try:
    if hasattr(STYLE, "curve_color_map"):
        STYLE.curve_color_map = phi_color  # type: ignore[assignment]
    if hasattr(STYLE, "theory_color_map"):
        STYLE.theory_color_map = phi_color  # type: ignore[assignment]
except Exception:
    pass


# --------------- compute theory curves ---------------------------------
n_phi, n_kappa = len(PHI_SET), KAPPA_TILDE_C.size
petermann = np.empty((n_phi, n_kappa))
strength = np.empty_like(petermann)
nuisance_scaling = np.empty_like(petermann)
ne = np.empty_like(petermann)
ep_dist = np.empty_like(petermann)
inst_dist = np.empty_like(petermann)

for j, phi in enumerate(PHI_SET):
    for i, kappa_c in enumerate(KAPPA_TILDE_C):
        tpd = standard_tpd_locations(phi, kappa_c, left_tpd=LEFT_TPD_ONLY)

        petermann[j, i] = petermann_factor_at_tpd(tpd, phi)
        strength[j, i] = splitting_strength_at_tpd_arc_2(phi, kappa_c, left_tpd=LEFT_TPD_ONLY)
        nuisance_scaling[j, i] = nuisance_scaling_coefficient_at_tpd(phi, kappa_c, DELTA_F_UNC, DELTA_K_UNC, left_tpd=LEFT_TPD_ONLY)

        ne[j, i] = thermal_noise_efficiency_at_tpd(
            tpd,
            phi,
            kappa_c,
            f_tilde_c=F_TILDE_C_FOR_NE,
        )

        ep_dist[j, i] = distance_between_ep_and_tpd(phi, kappa_c, left_tpd=LEFT_TPD_ONLY)
        inst_dist[j, i] = distance_between_instability_and_tpd(phi, kappa_c, left_tpd=LEFT_TPD_ONLY)


# --------------- fetch experiments once --------------------------------
engine = create_engine(f"sqlite:///{DATA_LOCATION}")
Session = sessionmaker(bind=engine, future=True)


def _fetch_experiments_by_id(ids: list[str]) -> dict[str, AnalyzedExperiment]:
    with Session() as session:
        stmt = (
            select(AnalyzedExperiment)
            .options(
                selectinload(AnalyzedExperiment.analyzed_aggregate_traces),
                selectinload(AnalyzedExperiment.theory_data_points),
            )
            .where(AnalyzedExperiment.analyzed_experiment_id.in_(ids))
        )
        rows = session.scalars(stmt).all()

    found = {e.analyzed_experiment_id for e in rows}
    missing = set(ids) - found
    if missing:
        raise ValueError(f"Experiment IDs not found: {sorted(missing)}")
    return {e.analyzed_experiment_id: e for e in rows}


ALL_IDS = [x for pair in ROW_EXPS for x in pair]
EXP_MAP = _fetch_experiments_by_id(ALL_IDS)

# --------------- figure layout -----------------------------------------
fig = plt.figure(figsize=(18, 12))

# Leave space for the top legend
outer = gridspec.GridSpec(
    3,
    2,
    left=0.07,
    right=0.99,
    bottom=0.08,
    top=0.92,
    hspace=0.35,
    wspace=0.25,
    width_ratios=[1, 2],
)


def two_side(parent):
    inner = gridspec.GridSpecFromSubplotSpec(1, 2, subplot_spec=parent, wspace=0.45)
    return fig.add_subplot(inner[0, 0]), fig.add_subplot(inner[0, 1])


# --------------- Row-0 --------------------------------------------------
ax_split_0 = fig.add_subplot(outer[0, 0])
for exp_id in ROW_EXPS[0]:
    plot_splitting_pane(
        ax_split_0,
        analyzed_experiment=EXP_MAP[exp_id],
        phi=PHI_SET[0],
        include_legend=False,
        include_data_label=False,
    )

ax_str, ax_min = two_side(outer[0, 1])

# --------------- Row-1 --------------------------------------------------
ax_split_pi = fig.add_subplot(outer[1, 0])
for exp_id in ROW_EXPS[1]:
    plot_splitting_pane(
        ax_split_pi,
        analyzed_experiment=EXP_MAP[exp_id],
        phi=PHI_SET[1],
        include_legend=False,
        include_data_label=False,
    )

ax_ne, ax_pet = two_side(outer[1, 1])

# --------------- Row-2 --------------------------------------------------
ax_split_pi2 = fig.add_subplot(outer[2, 0])
for exp_id in ROW_EXPS[2]:
    plot_splitting_pane(
        ax_split_pi2,
        analyzed_experiment=EXP_MAP[exp_id],
        phi=PHI_SET[2],
        include_legend=False,
        include_data_label=False,
    )

ax_inst, ax_dist = two_side(outer[2, 1])

# --------------- plot theory curves ------------------------------------
for j, phi in enumerate(PHI_SET):
    col = phi_color(phi)

    ax_str.plot(KAPPA_TILDE_C, strength[j], color=col, lw=STYLE.curve_lw, zorder=THEORY_ZORDER)
    ax_min.plot(KAPPA_TILDE_C, nuisance_scaling[j], color=col, lw=STYLE.curve_lw, zorder=THEORY_ZORDER)

    scatter_metric_markers(ax_str, KAPPA_TILDE_C, strength[j], phi, col)
    scatter_metric_markers(ax_min, KAPPA_TILDE_C, nuisance_scaling[j], phi, col)

    if np.isclose(phi, np.pi):
        mask = KAPPA_TILDE_C > 0.0
        ax_ne.plot(KAPPA_TILDE_C[mask], ne[j, mask], color=col, lw=STYLE.curve_lw, zorder=THEORY_ZORDER)
        scatter_metric_markers(ax_ne, KAPPA_TILDE_C[mask], ne[j, mask], phi, col)
    else:
        ax_ne.plot(KAPPA_TILDE_C, ne[j], color=col, lw=STYLE.curve_lw, zorder=THEORY_ZORDER)
        scatter_metric_markers(ax_ne, KAPPA_TILDE_C, ne[j], phi, col)

    ax_pet.plot(KAPPA_TILDE_C, petermann[j], color=col, lw=STYLE.curve_lw, zorder=THEORY_ZORDER)
    scatter_metric_markers(ax_pet, KAPPA_TILDE_C, petermann[j], phi, col)

    ax_inst.plot(KAPPA_TILDE_C, inst_dist[j], color=col, lw=STYLE.curve_lw, zorder=THEORY_ZORDER)
    scatter_metric_markers(ax_inst, KAPPA_TILDE_C, inst_dist[j], phi, col)

    if np.isclose(phi, np.pi):
        mask_ep = np.isfinite(ep_dist[j]) & (KAPPA_TILDE_C > 0.0)
        ax_dist.plot(KAPPA_TILDE_C[mask_ep], ep_dist[j, mask_ep], color=col, lw=STYLE.curve_lw, zorder=THEORY_ZORDER)
        scatter_metric_markers(ax_dist, KAPPA_TILDE_C[mask_ep], ep_dist[j, mask_ep], phi, col)
    else:
        ax_dist.plot(KAPPA_TILDE_C, ep_dist[j], color=col, lw=STYLE.curve_lw, zorder=THEORY_ZORDER)
        scatter_metric_markers(ax_dist, KAPPA_TILDE_C, ep_dist[j], phi, col)

# --------------- axis labels/limits ------------------------------------
for a in (ax_str, ax_min, ax_ne, ax_pet, ax_inst, ax_dist):
    a.set_xlabel(r"$\tilde{\kappa}_c$", fontsize=STYLE.label_font)
    a.tick_params(labelsize=STYLE.tick_font)

ax_str.set_ylabel("Signal Str. \n" + r"$\tilde {a}_{\text{sqrt}}^\text{TPD}$", fontsize=STYLE.label_font)
ax_min.set_ylabel("Nuisance Str. \n" + r"$\tilde {b}_{\text{cbrt}}^\text{TPD}$", fontsize=STYLE.label_font)

ax_ne.set_ylabel("Thermal Noise \n Eff. at TPD", fontsize=STYLE.label_font)
ax_ne.set_ylim(0.0, 10.0)

ax_pet.set_ylabel("Petermann Factor \n at TPD", fontsize=STYLE.label_font)
ax_pet.set_ylim(0.75, 5.0)

ax_inst.set_ylabel("TPD Dist. \n to Instability", fontsize=STYLE.label_font)
ax_dist.set_ylabel("TPD Dist. \n to EP", fontsize=STYLE.label_font)

# --------------- unified top legend ------------------------------------
legend_font = getattr(STYLE, "theory_legend_font", STYLE.legend_font)

phi_line_0 = Line2D([0], [0], color="royalblue", lw=STYLE.curve_lw, linestyle="-", label=r"$\phi=0$")
phi_line_pi = Line2D([0], [0], color="purple", lw=STYLE.curve_lw, linestyle="-", label=r"$\phi=\pi$")
phi_line_pi2 = Line2D([0], [0], color="green", lw=STYLE.curve_lw, linestyle="-", label=r"$\phi=\pi/2$")

small_kappa = Line2D(
    [0],
    [0],
    marker="o",
    linestyle="None",
    color="none",
    markerfacecolor="none",
    markeredgecolor="black",
    markeredgewidth=2.0,
    markersize=20,
    label=r"Small $\tilde \kappa_c$",
)
large_kappa = Line2D(
    [0],
    [0],
    marker="^",
    linestyle="None",
    color="none",
    markerfacecolor="none",
    markeredgecolor="black",
    markeredgewidth=2.0,
    markersize=20,
    label=r"Large $\tilde \kappa_c$",
)

fig.legend(
    handles=[phi_line_0, phi_line_pi, phi_line_pi2, small_kappa, large_kappa],
    loc="upper center",
    ncol=5,
    frameon=False,
    fontsize=legend_font,
    columnspacing=1.45,
    handletextpad=0.6,
    handlelength=2.4,
    bbox_to_anchor=(0.52, 1.01),
)

# --------------- panel letters -----------------------------------------
def _panel_label(ax: plt.Axes, letter: str) -> None:
    bb = ax.get_position()
    fig.text(bb.x0 - 0.065, bb.y1 + 0.0125, letter, fontsize=29, fontweight="bold", va="top", ha="left")


_left_axes = (ax_split_0, ax_split_pi, ax_split_pi2)
for lbl, ax in zip(("(a)", "(b)", "(c)"), _left_axes):
    _panel_label(ax, lbl)

_panel_label(ax_str, "(d)")

# --------------- roman numerals (right-hand column) ---------------------
right_axes = (ax_str, ax_min, ax_ne, ax_pet, ax_inst, ax_dist)
right_tags = ("i", "ii", "iii", "iv", "v", "vi")
for tag, ax in zip(right_tags, right_axes):
    ax.text(0.015, 0.98, tag, transform=ax.transAxes, ha="left", va="top", fontsize=26, fontweight="bold")

# --------------- save ---------------------------------------------------
out_dir = Path("../../.figures")
out_dir.mkdir(parents=True, exist_ok=True)

out_path = out_dir / "FIG_4_metrics.png"
fig.savefig(out_path, dpi=STYLE.save_dpi)
print(f"figure saved to {out_path}")
plt.close(fig)
