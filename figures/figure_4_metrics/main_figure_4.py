"""
metrics_grid_figure.py
Full 3 x 2 "metrics grid" figure

LEFT column  -> experimental splitting data panes
RIGHT column -> theory/model curves (with star/triangle markers)
"""

from __future__ import annotations

import os
from pathlib import Path
from matplotlib.lines import Line2D
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from sqlalchemy import create_engine, select
from sqlalchemy.orm import sessionmaker, selectinload

from figures.figure_3_experiment.experiment_tpds import standard_tpd_locations
from settings import STYLE
from metric_markers import scatter_metric_markers
from metric_calculations import (
    petermann_factor_at_tpd,
    splitting_strength_at_tpd_arc_2,
    min_split_over_J,
    calculate_max_response_tilde,
    distance_between_ep_and_tpd,
)
from splitting_data_pane import plot_splitting_pane
from models.analysis import AnalyzedExperiment

# --------------- database path and experiment IDs ----------------------
DATA_LOCATION = Path("../../data/ep_tpd_transformed_data.db")

ROW_EXPS: tuple[tuple[str, str], ...] = (
    ("cab5729f-0491-41f9-9fc8-acf0096754ab",
     "62156c79-a18b-4c96-b4b5-2a43519d1900"),   # phi = 0
    ("02f5e503-3e4e-4651-9ef8-d90104b3721f",
     "47809d5e-9041-456b-80d8-becf48bf1cfd"),   # phi = pi
    ("25e2d559-5960-48c9-8cd2-9441cc067b61",
     "3ef835ef-e2be-4eb3-929e-3a5149a00e87"),   # phi = pi/2
)
PHI_SET = (0.0, np.pi, np.pi / 2.0)             # same order as ROW_EXPS

# --------------- theory-grid parameters --------------------------------
KAPPA_TILDE_C = np.linspace(0.0, 2.8, 1_000)
LEFT_TPD_ONLY = True
DELTA_F_UNC   = 1.0e-5 / 1.0e-3
DELTA_K_UNC   = 1.0e-5 / 1.0e-3
THEORY_ZORDER = 20

n_phi, n_kappa = len(PHI_SET), KAPPA_TILDE_C.size
petermann = np.empty((n_phi, n_kappa))
strength  = np.empty_like(petermann)
min_split = np.empty_like(petermann)
max_resp  = np.empty_like(petermann)
ep_dist   = np.empty_like(petermann)

# --------------- compute theory curves ---------------------------------
for j, phi in enumerate(PHI_SET):
    for i, kappa_c in enumerate(KAPPA_TILDE_C):
        tpd = standard_tpd_locations(phi, kappa_c, left_tpd=LEFT_TPD_ONLY)
        ep_dist  [j, i] = distance_between_ep_and_tpd(phi, kappa_c, left_tpd=LEFT_TPD_ONLY)
        petermann[j, i] = petermann_factor_at_tpd(tpd, phi)
        strength [j, i] = splitting_strength_at_tpd_arc_2(phi, kappa_c, left_tpd=LEFT_TPD_ONLY)
        min_split[j, i] = min_split_over_J(phi, kappa_c, DELTA_F_UNC, DELTA_K_UNC, left_tpd=LEFT_TPD_ONLY)
        max_resp [j, i] = calculate_max_response_tilde(strength[j, i], min_split[j, i])

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
outer = gridspec.GridSpec(
    3, 2,
    left=0.055, right=0.97, bottom=0.068, top=0.975,
    hspace=0.265, wspace=0.15, width_ratios=[1, 1]
)

def two_side(parent):
    inner = gridspec.GridSpecFromSubplotSpec(1, 2, subplot_spec=parent, wspace=0.3)
    return fig.add_subplot(inner[0, 0]), fig.add_subplot(inner[0, 1])

# --------------- Row-0 --------------------------------------------------
# left  = experimental splitting (phi = 0)
ax_split_0 = fig.add_subplot(outer[0, 0])
for exp_id in ROW_EXPS[0]:
    plot_splitting_pane(ax_split_0, analyzed_experiment=EXP_MAP[exp_id], phi=PHI_SET[0])

# right = theory (max-response)
ax_max = fig.add_subplot(outer[0, 1])
for j, phi in enumerate(PHI_SET):
    if np.isclose(phi, np.pi):
        label = r"$\phi=\pi$"
    elif np.isclose(phi, np.pi/2.0):
        label = r"$\phi=\pi/2$"
    elif np.isclose(phi, 0.0):
        label = r"$\phi=0$"
    else:
        label = r"$\phi$"
    col = STYLE.curve_color_map(phi)
    ax_max.plot(KAPPA_TILDE_C, max_resp[j], color=col, lw=STYLE.curve_lw, label=label, zorder=THEORY_ZORDER)
    scatter_metric_markers(ax_max, KAPPA_TILDE_C, max_resp[j], phi, col)

star_proxy = Line2D([0], [0], marker="o", color="none", markerfacecolor="white",
                    markeredgecolor="gray", markersize=STYLE.star_ms**0.5, markeredgewidth=2,
                    linewidth=1.4, linestyle="None", label=r"$\star$ Min splitting")

tri_proxy = Line2D([0], [0], marker="^", color="none", markerfacecolor="white",
                   markeredgecolor="gray", markersize=STYLE.tri_ms**0.5, markeredgewidth=2,
                   linewidth=1.4, linestyle="None", label=r"$\triangle$ Min splitting")

filled_star_proxy = Line2D([0], [0], marker="*", color="none",
                           markerfacecolor="gold", markeredgecolor="black",
                           markersize=STYLE.star_ms**0.5, markeredgewidth=2,
                           linestyle="None", label=r"$\star$ Min splitting")

handles, labels = ax_max.get_legend_handles_labels()
handles.extend([star_proxy, tri_proxy, filled_star_proxy])
labels.extend([r"Small $\tilde \kappa_c$", r"Large $\tilde \kappa_c$", r"Ref. 19 $\tilde \kappa_c$"])

ax_max.set_xlabel(r"$\tilde{\kappa}_c$", fontsize=STYLE.label_font)
ax_max.set_ylabel(r"$\max(\tilde\chi)$", fontsize=STYLE.label_font)
ax_max.legend(handles=handles, labels=labels, fontsize=STYLE.theory_legend_font,
              loc="upper right", framealpha=1, borderpad=.1, bbox_to_anchor=(1.01, 1.02))
ax_max.tick_params(labelsize=STYLE.tick_font)

# --------------- Row-1 --------------------------------------------------
# left  = experimental splitting (phi = pi/2)
ax_split_pi2 = fig.add_subplot(outer[2, 0])
for exp_id in ROW_EXPS[2]:
    plot_splitting_pane(ax_split_pi2, analyzed_experiment=EXP_MAP[exp_id], phi=PHI_SET[2])

# right = theory (strength & min-split)
ax_str, ax_min = two_side(outer[1, 1])
for j, phi in enumerate(PHI_SET):
    col = STYLE.curve_color_map(phi)
    ax_str.plot(KAPPA_TILDE_C, strength [j], color=col, lw=STYLE.curve_lw, zorder=THEORY_ZORDER)
    ax_min.plot(KAPPA_TILDE_C, min_split[j], color=col, lw=STYLE.curve_lw, zorder=THEORY_ZORDER)
    scatter_metric_markers(ax_str, KAPPA_TILDE_C, strength [j], phi, col)
    scatter_metric_markers(ax_min, KAPPA_TILDE_C, min_split [j], phi, col)

ax_str.set_xlabel(r"$\tilde{\kappa}_c$", fontsize=STYLE.label_font)
ax_min.set_xlabel(r"$\tilde{\kappa}_c$", fontsize=STYLE.label_font)
ax_str.set_ylabel(r"TPD Strength / J", fontsize=STYLE.label_font)
ax_min.set_ylabel(r"$\min(\tilde \Delta_\nu)$", fontsize=STYLE.label_font)
plt.setp(ax_str.get_xticklabels(), visible=True)
plt.setp(ax_min.get_xticklabels(), visible=True)
for a in (ax_str, ax_min):
    a.tick_params(labelsize=STYLE.tick_font)

# --------------- Row-2 --------------------------------------------------
# left  = experimental splitting (phi = pi)
ax_split_pi = fig.add_subplot(outer[1, 0])
for exp_id in ROW_EXPS[1]:
    plot_splitting_pane(ax_split_pi, analyzed_experiment=EXP_MAP[exp_id], phi=PHI_SET[1])

# right = theory (EP-distance & Petermann)
ax_dist, ax_pet = two_side(outer[2, 1])
for j, phi in enumerate(PHI_SET):
    col = STYLE.curve_color_map(phi)
    ax_dist.plot(KAPPA_TILDE_C, ep_dist  [j], color=col, lw=STYLE.curve_lw, zorder=THEORY_ZORDER)
    ax_pet .plot(KAPPA_TILDE_C, petermann[j], color=col, lw=STYLE.curve_lw, zorder=THEORY_ZORDER)
    scatter_metric_markers(ax_dist, KAPPA_TILDE_C, ep_dist  [j], phi, col)
    scatter_metric_markers(ax_pet , KAPPA_TILDE_C, petermann[j], phi, col)
ax_dist.set_ylabel(r"$\text{dist}_2(\mathrm{EP\!-\!TPD})$", fontsize=STYLE.label_font)
ax_dist.set_xlabel(r"$\tilde{\kappa}_c$", fontsize=STYLE.label_font)
ax_pet.set_xlabel(r"$\tilde{\kappa}_c$", fontsize=STYLE.label_font)
ax_pet .set_ylabel(r"PF(TPD)", fontsize=STYLE.label_font)
ax_pet.set_ylim(0.75, 5.0)
for a in (ax_dist, ax_pet):
    a.tick_params(labelsize=STYLE.tick_font)

# --------------- panel letters -----------------------------------------
def _panel_label(ax: plt.Axes, letter: str) -> None:
    bb = ax.get_position()
    fig.text(bb.x0 - 0.045, bb.y1 + .0125, letter, fontsize=28, fontweight="bold", va="top", ha="left")

_left_axes = (ax_split_0, ax_split_pi2, ax_split_pi)
for lbl, ax in zip(("A", "C", "B"), _left_axes):
    _panel_label(ax, lbl)
_panel_label(ax_max, "D")

# --------------- roman numerals (right-hand column) ---------------------
right_axes = (ax_max, ax_str, ax_min, ax_dist, ax_pet)
right_tags = ("i", "ii", "iii", "iv", "v")
for tag, ax in zip(right_tags, right_axes):
    ax.text(0.015, 0.98, tag, transform=ax.transAxes, ha="left", va="top", fontsize=22, fontweight="bold")

# --------------- save ---------------------------------------------------

if not os.path.exists("../../.figures"):
        os.makedirs("../../.figures")

fig.savefig("../../.figures/FIG_4_metrics.png", dpi=STYLE.save_dpi)
