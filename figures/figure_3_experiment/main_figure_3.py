# MAKE_FIG_EPS_updated.py
"""
Composite figure generator – 18-pane layout (three stacked 6-pane clusters)

• X values shown as X/J
• Y values shown as (value − f_c)/J
• Each axis is completely independent (NO sharex)
• Only the very bottom row shows x-tick labels **and** the x-axis label
"""
import os
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import matplotlib
import numpy as np
import pandas as pd
import matplotlib.patheffects as pe
from sqlalchemy import create_engine, select
from sqlalchemy.orm import sessionmaker, selectinload
from big_pane   import plot_big_pane
import small_panes
from models.analysis import AnalyzedExperiment


def _format_val_unc(val: float, unc: float) -> str:
    """
    Return LaTeX-ready string  ``x.xx(n)``, rounding *val* and *unc*
    so that **exactly one significant digit** of the uncertainty is kept.

    Examples
    --------
    >>> _format_val_unc(1.957, 0.043)   ->  '1.96(4)'
    >>> _format_val_unc(0.0314, 0.0021) ->  '0.031(2)'
    """
    if unc <= 0 or np.isnan(unc):
        return f"{val:.3g}"          # fall-back: just a value

    # order-of-magnitude of the 1-digit uncertainty
    exp  = int(np.floor(np.log10(unc)))
    step = 10**exp                   # e.g. 0.01
    unc_1dig = round(unc / step) * step     # e.g. → 0.04

    # make sure we still have 1 significant digit after rounding
    if unc_1dig >= 10 * step:
        unc_1dig /= 10
        step     /= 10
        exp      -= 1

    # round the value to the same decimal place
    val_rounded = round(val / step) * step

    # integer shown inside the parentheses
    paren = int(round(unc_1dig / step))

    # number of decimal places to print
    dec = max(-exp, 0)
    fmt = f"{{:.{dec}f}}"

    return f"{fmt.format(val_rounded)}({paren})"

# ──────────────────────────────────────────────────────────────────────
matplotlib.rcParams.update({
    "font.family"      : "sans-serif",
    "font.size"        : 20,
    "axes.labelsize"   : 18,
    "legend.fontsize"  : 22,
    "xtick.labelsize"  : 14,
    "ytick.labelsize"  : 14,
    "lines.linewidth"  : 1.3,
    "lines.markersize" : 4,
    "grid.color"       : "#999999",
    "grid.linestyle"   : ":",
    "grid.alpha"       : 0.4,
})

# ──────────────────────────────────────────────────────────────────────
DATA_LOCATION = Path("../../data/ep_tpd_transformed_data.db")

# ╭──────────────────────────────────────────────────────── experiment IDs
#   Row 1 (φ = 0)
EXP1_L, EXP1_R = (
    "cab5729f-0491-41f9-9fc8-acf0096754ab",
    "62156c79-a18b-4c96-b4b5-2a43519d1900",
)
#   Row 2 (φ = π)
EXP2_L, EXP2_R = (
    "02f5e503-3e4e-4651-9ef8-d90104b3721f",
    "47809d5e-9041-456b-80d8-becf48bf1cfd",
)
#   Row 3 (φ = π/2)
EXP3_L, EXP3_R = (
    "25e2d559-5960-48c9-8cd2-9441cc067b61",
    "3ef835ef-e2be-4eb3-929e-3a5149a00e87",
)
# ╰──────────────────────────────────────────────────────────────────────

DRAW_INSET       = (False, False, False)
DRAW_UNSTAB      = (True,  False, False)
INCLUDE_LEGEND   = (True,  False, False)
INCLUDE_LEGEND_BIG = (True, True, True)

# axis limits -----------------------------------------------------------
LEFT_X_LIMS        = (-2.25, 1)
LEFT_Y_FREQ_LIM    = (-1.05, 1.05)
LEFT_Y_RE_EIG_LIM  = (-2.1, 0.25)

RIGHT_X_LIMS       = (0.95, 3)
RIGHT_Y_FREQ_LIM   = (-1.5, 1)
RIGHT_Y_RE_EIG_LIM = (-1, 0.25)

BOTTOM_X_LIMS = (-1.5, -0.3)
BOTTOM_Y_FREQ_LIM = (-1, 3)
BOTTOM_Y_RE_EIG_LIM = (-2.1, 0.25)

OUTFIG  = Path("FIG_EPS/eps.png")
FIGSIZE = (13, 15)

# ──────────────────────────────────────────────────────────────────────
fig = plt.figure(figsize=FIGSIZE)
gs  = GridSpec(
    8, 4, figure=fig,
    width_ratios =[1.0, 1.7, 1.0, 1.7],
    height_ratios=[1, 1, 0.17, 1, 1, 0.17, 1, 1],
    left=0.08, right=.97, bottom=0.0475, top=0.965,
    wspace=0.55, hspace=0.3,
)

# ──────────────────────────────────────────────────────────────────────
def _add_cluster(row0: int,
                 phi_val: float,
                 exp_L: str, exp_R: str,
                 phi_lbl_left: str, phi_lbl_right: str,
                 *, draw_inset: bool, draw_unstable: bool,
                 include_legend: bool,
                 include_legend_big: bool,
                 is_bottom_cluster: bool):
    """Build one 6-pane cluster (rows row0 & row0+1)."""

    # Check to ensure that the transformed database has been created
    if not os.path.exists(DATA_LOCATION):
        raise FileNotFoundError(f"Database file not found at {DATA_LOCATION}. Please ensure the database has been created.")

    # Load data from database using sqlalchemy
    DataSession = sessionmaker(bind=create_engine(f"sqlite:///{DATA_LOCATION}"))
    with DataSession() as session:
        stmt = (
            select(AnalyzedExperiment)
            .options(
                selectinload(AnalyzedExperiment.analyzed_aggregate_traces),
                selectinload(AnalyzedExperiment.theory_data_points)
            )
            .where(AnalyzedExperiment.analyzed_experiment_id.in_([exp_L, exp_R]))
        )
        experiments = session.scalars(stmt).all()

    if len(experiments) != 2:
        found_ids = {e.analyzed_experiment_id for e in experiments}
        missing = {exp_L, exp_R} - found_ids
        raise ValueError(f"Experiment IDs not found: {sorted(missing)}")

    exp_map = {e.analyzed_experiment_id: e for e in experiments}
    expL = exp_map[exp_L]
    expR = exp_map[exp_R]
    J_L, fc_L, kc_L = expL.J_avg, expL.f_c_avg, expL.kappa_c_avg
    kappa_c_std_L = expL.kappa_c_std
    J_R, fc_R, kc_R = expR.J_avg, expR.f_c_avg, expR.kappa_c_avg
    kappa_c_std_R = expR.kappa_c_std

    # inside _add_cluster: select which small-panel function to use
    if np.isclose(phi_val, 0):
        small_func = small_panes.left_small_panels
    elif np.isclose(phi_val % (2*np.pi), np.pi):
        small_func = small_panes.right_small_panels
    else:
        small_func = small_panes.hybrid_small_panels       # <── NEW


    # ── decide x-axis for THIS ROW  ───────────────────────────────────
    if np.isclose(phi_val, np.pi):          # φ = π  ⇒  Δf/J
        x_key, x_lab   = "Delta_f",     r"$\tilde \Delta_f$"
        small_func     = small_panes.right_small_panels
        xlims          = RIGHT_X_LIMS
        ylims_freq     = RIGHT_Y_FREQ_LIM
        ylims_re_eig   = RIGHT_Y_RE_EIG_LIM
    elif np.isclose(phi_val, 0):
        x_key, x_lab   = "Delta_kappa", r"$\tilde \Delta_\kappa$"
        small_func     = small_panes.left_small_panels
        xlims          = LEFT_X_LIMS
        ylims_freq     = LEFT_Y_FREQ_LIM
        ylims_re_eig   = LEFT_Y_RE_EIG_LIM
    else:                                   # φ = 0  or  π/2  ⇒  Δκ/J
        x_key, x_lab   = "Delta_kappa", r"$\tilde \Delta_\kappa$ (Hyperbolic)"
        small_func     = small_panes.hybrid_small_panels
        xlims          = BOTTOM_X_LIMS
        ylims_freq     = BOTTOM_Y_FREQ_LIM
        ylims_re_eig   = BOTTOM_Y_RE_EIG_LIM

    # Helper to call the correct small-panel routine with matching kwargs
    def _draw_small(ax_top, ax_bot, *, analyzed_experiment, J_scale, f_c, phi):
        kw = dict(data_dir=DATA_LOCATION, analyzed_experiment=analyzed_experiment,
                  J_scale=J_scale, f_c=f_c,
                  xlims=xlims,
                  ylims_freq=ylims_freq,
                  ylims_re_eig=ylims_re_eig,
                  include_legend=include_legend)
        # right_small_panels needs these extras
        if small_func is small_panes.right_small_panels:
            kw.update(draw_inset=draw_inset, draw_unstable=draw_unstable)
        if small_func is small_panes.hybrid_small_panels:
            kw.update(phi=phi)
        small_func(ax_top, ax_bot, **kw)

    # -------------------------------------------------- left column
    ax_L_top = fig.add_subplot(gs[row0, 0])
    ax_L_bot = fig.add_subplot(gs[row0 + 1, 0])
    ax_L_big = fig.add_subplot(gs[row0:row0 + 2, 1])

    _draw_small(ax_L_top, ax_L_bot, analyzed_experiment=expL, J_scale=J_L, f_c=fc_L, phi=phi_val)
    plot_big_pane(
        ax_L_big, DATA_LOCATION, analyzed_experiment=expL,
        x_key=x_key, xlab=x_lab,
        J_scale=J_L, f_c=fc_L, kappa_c=kc_L,
        draw_unstable=draw_unstable,
        xlims=xlims, ylims_freq=ylims_freq,
        include_legend=include_legend_big,
        phi_val=phi_val,
    )

    # -------------------------------------------------- right column
    ax_R_top = fig.add_subplot(gs[row0, 2])
    ax_R_bot = fig.add_subplot(gs[row0 + 1, 2])
    ax_R_big = fig.add_subplot(gs[row0:row0 + 2, 3])

    _draw_small(ax_R_top, ax_R_bot, analyzed_experiment=expR, J_scale=J_R, f_c=fc_R, phi=phi_val)
    plot_big_pane(
        ax_R_big, DATA_LOCATION, analyzed_experiment=expR,
        x_key=x_key, xlab=x_lab,
        J_scale=J_R, f_c=fc_R, kappa_c=kc_R,
        draw_unstable=draw_unstable,
        xlims=xlims, ylims_freq=ylims_freq,
        include_legend=include_legend,
        phi_val=phi_val,
    )

    # -------------------------------------------------- tick / label housekeeping
    ax_L_top.tick_params(labelbottom=False)
    ax_R_top.tick_params(labelbottom=False)
    # if not is_bottom_cluster:
    #     for ax in (ax_L_bot, ax_R_bot, ax_L_big, ax_R_big):
    #         ax.tick_params(labelbottom=False)
    #         ax.set_xlabel("")

    # -------------------------------------------------- panel letters
    # big_map = {ax_L_top: phi_lbl_left, ax_R_top: phi_lbl_right}
    # for ax, lbl in big_map.items():
    #     pos = ax.get_position()
    #     fig.text(pos.x0 - 0.020, pos.y1 + .007,
    #              lbl, fontsize=27, fontweight="bold",
    #              fontfamily="sans-serif", ha="right", va="bottom")
    roman_axes = [ax_L_top, ax_L_bot, ax_L_big,
                  ax_R_top, ax_R_bot, ax_R_big]
    roman_lbls = ["i", "ii", "iii", "i", "ii", "iii"]
    VERT = -0.02
    offsets = {
        ax_L_top: (0.012, VERT), ax_R_top: (0.012, VERT),
        ax_L_bot: (0.017, VERT), ax_R_bot: (0.017, VERT),
        ax_L_big: (0.025, VERT), ax_R_big: (0.025, VERT),
    }
    for ax, lbl in zip(roman_axes, roman_lbls):
        dx, dy = offsets[ax]
        pos = ax.get_position()
        fig.text(pos.x0 + dx, pos.y1 + dy, lbl,
                 fontsize=18, fontweight="bold", fontfamily="sans-serif",
                 ha="right", va="bottom", zorder=20,
                 bbox=dict(facecolor="white", edgecolor="none", pad=2))

    # -------------------------------------------------- headers (φ + κ_c/J)
    kc_over_J_L  = kc_L / J_L
    kc_over_J_R  = kc_R / J_R
    dktc_L       = kappa_c_std_L / J_L          # uncertainty in κ̃c (left)
    dktc_R       = kappa_c_std_R / J_R          # uncertainty in κ̃c (right)
    kc_L_str = _format_val_unc(kc_over_J_L, dktc_L)
    kc_R_str = _format_val_unc(kc_over_J_R, dktc_R)

    # Format φ as LaTeX string if it's close to π or π/2
    if np.isclose(phi_val % (2*np.pi), np.pi):
        phi_str = r"\pi"
    elif np.isclose(phi_val % (2*np.pi), np.pi / 2):
        phi_str = r"\pi/2"
    else:
        phi_str = f"{phi_val:.0f}"

    small = r"\mathrm{ (Small)}"
    large = r"\mathrm{ (Large)}"

    hdr_L = rf"$\phi = {phi_str},\;\tilde \kappa_c = {kc_L_str}\,{small}$"
    hdr_R = rf"$\phi = {phi_str},\;\tilde \kappa_c = {kc_R_str}\,{large}$"

    mid_L = 0.5 * (ax_L_top.get_position().x0 + ax_L_big.get_position().x1)
    mid_R = 0.5 * (ax_R_top.get_position().x0 + ax_R_big.get_position().x1)
    y_hdr = ax_L_top.get_position().y1 + .019

    subpanel_letter_LR = {
        0: ("a", "b"),
        3: ("c", "d"),
        6: ("e", "f"),
    }
    lbl_L, lbl_R = subpanel_letter_LR[row0]

    LEFT_OFFSET = .22
    LETTER_FS = 24
    fig.text(mid_L - LEFT_OFFSET, y_hdr, lbl_L,
             fontsize=LETTER_FS, fontweight="bold", fontfamily="sans-serif",
             ha="right", va="center")

    fig.text(mid_R - LEFT_OFFSET, y_hdr, lbl_R,
             fontsize=LETTER_FS, fontweight="bold", fontfamily="sans-serif",
             ha="right", va="center")

    with matplotlib.rc_context({
        "text.usetex": True,
        "font.family": "serif",
        "font.serif": ["Computer Modern Roman"],
        "text.latex.preamble": r"\usepackage{amsmath}",
    }):
        for mid, hdr, lr in [(mid_L, hdr_L, 'l'), (mid_R, hdr_R, 'r')]:
            txt = fig.text(mid - 0.01, y_hdr, f"{hdr}",
                           ha="center", va="center",
                           fontsize=20, fontweight="bold",
                           fontfamily="sans-serif",
                           bbox=dict(boxstyle="round,pad=0.25",
                                     facecolor="white", edgecolor="none",
                                     alpha=0.9))
            txt.set_path_effects([pe.withStroke(linewidth=0.75,
                                                foreground="black")])

# ──────────────────────────────────────────────────────────────────────
# Build the three clusters (no sharex arguments)
_add_cluster(
    row0=0,  phi_val=0,
    exp_L=EXP1_L, exp_R=EXP1_R,
    phi_lbl_left=r"\phi = 0", phi_lbl_right=r"\phi = 0",
    draw_inset=DRAW_INSET[0], draw_unstable=DRAW_UNSTAB[0],
    include_legend=INCLUDE_LEGEND[0],
    include_legend_big=INCLUDE_LEGEND_BIG[0],
    is_bottom_cluster=False,
)
_add_cluster(
    row0=3,  phi_val=np.pi,
    exp_L=EXP2_L, exp_R=EXP2_R,
    phi_lbl_left=r"\phi = \pi", phi_lbl_right=r"\phi = \pi",
    draw_inset=DRAW_INSET[1], draw_unstable=DRAW_UNSTAB[1],
    include_legend=INCLUDE_LEGEND[1],
    include_legend_big=INCLUDE_LEGEND_BIG[1],
    is_bottom_cluster=False,
)
_add_cluster(
    row0=6,  phi_val=np.pi/2,
    exp_L=EXP3_L, exp_R=EXP3_R,
    phi_lbl_left=r"\phi = \pi/2", phi_lbl_right=r"\phi = \pi/2",
    draw_inset=DRAW_INSET[2], draw_unstable=DRAW_UNSTAB[2],
    include_legend=INCLUDE_LEGEND[2],
    include_legend_big=INCLUDE_LEGEND_BIG[2],
    is_bottom_cluster=True,
)

# ──────────────────────────────────────────────────────────────────────
OUTFIG.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(OUTFIG, dpi=400)
print("saved figure to", OUTFIG)
