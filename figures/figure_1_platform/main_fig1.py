# plot_from_single_db_grid.py
"""
3x2 NR heat-map grid with one shared vertical color-bar
and bold sans-serif panel letters (ASCII-only labels).

Reads from a single merged SQLite DB.
"""

import os
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from sqlalchemy import create_engine, text

# ───────────────────────────────────────── CONSTANTS ──────────────────────────
DATA_LOCATION = Path("../../data/fig1_raw_data.db")
TABLE_NAME     = "expr"

FIGSIZE   = (15, 20)   # inches
LABEL_FONT = 42
TICK_FONT  = 38
TAG_FONT   = LABEL_FONT + 27
SAVE_DPI   = 400

CMAP      = "inferno"
VMIN, VMAX = -25, 12   # color limits

# data window (match your legacy styling; tweak if needed)
FREQ_MIN, FREQ_MAX       = 6.001e9, 6.009e9
CURRENT_MIN, CURRENT_MAX = -99.9, 99.9

# Panel tags (legacy had b..g because (a) was elsewhere)
PANEL_TAGS = "bcdefg"

# ───────────────────────────────────────── HELPERS ────────────────────────────
def get_engine(db_path: str):
    return create_engine(f"sqlite:///{db_path}")

def get_nr_data(engine, experiment_id: str,
                fmin=FREQ_MIN, fmax=FREQ_MAX,
                cmin=CURRENT_MIN, cmax=CURRENT_MAX):
    """
    Return (power_grid, currents, freqs) for one experiment.
    If no data, return all None.
    """
    q = text(f"""
        SELECT frequency_hz, set_amperage, power_dBm
        FROM {TABLE_NAME}
        WHERE experiment_id = :exp_id
          AND readout_type = 'nr'
          AND set_amperage BETWEEN :cmin AND :cmax
          AND frequency_hz  BETWEEN :fmin AND :fmax
        ORDER BY set_amperage, frequency_hz
    """)
    df = pd.read_sql_query(q, engine, params=dict(
        exp_id=experiment_id, cmin=cmin, cmax=cmax, fmin=fmin, fmax=fmax
    ))
    if df.empty:
        return None, None, None
    piv = df.pivot_table(index="set_amperage",
                         columns="frequency_hz",
                         values="power_dBm",
                         aggfunc="first")
    return piv.values, piv.index.values, piv.columns.values

def find_experiment(engine, settings: dict):
    """
    Return first experiment_id matching *settings* among NR rows.
    """
    base = f"SELECT DISTINCT experiment_id FROM {TABLE_NAME} WHERE readout_type = 'nr'"
    conds, params = [], {}
    for k, v in settings.items():
        conds.append(f"{k} = :{k}")
        params[k] = v
    q = text(base + (" AND " + " AND ".join(conds) if conds else ""))
    df = pd.read_sql_query(q, engine, params=params)
    return None if df.empty else df.iloc[0, 0]

def plot_panel(ax, power, currents, freqs, show_xlabel, show_ylabel):
    """
    Draw one pcolormesh panel. Return QuadMesh handle for the shared color-bar.
    """
    pc = ax.pcolormesh(currents,
                       freqs / 1e9,
                       power.T,
                       cmap=CMAP,
                       shading="auto",
                       vmin=VMIN,
                       vmax=VMAX)
    if show_xlabel:
        # ASCII-only label (no unicode delta)
        ax.set_xlabel(r"$\Delta$ Current [mA]", fontsize=LABEL_FONT, labelpad=18)
    else:
        ax.set_xticklabels([])
    if show_ylabel:
        ax.set_ylabel("Frequency [GHz]", fontsize=LABEL_FONT)
    else:
        ax.set_yticklabels([])
    ax.tick_params(axis="both", labelsize=TICK_FONT)
    return pc

# ────────────────────────────────────────── MAIN ──────────────────────────────
def main():
    # All panels now read from the single merged DB
    engine = get_engine(DATA_LOCATION)

    # (row, col, explicit_id, settings_dict, tag)
    panels = [
        (0, 0, None, dict(set_loop_att=25, set_loop_phase_deg=320), "b"),
        (0, 1, None, dict(set_loop_att=20, set_loop_phase_deg=332), "c"),
        (1, 0, None, dict(set_loop_att=30, set_loop_phase_deg=140), "d"),
        (1, 1, None, dict(set_loop_att=25, set_loop_phase_deg=152), "e"),
        (2, 0, "82eadd98-9c98-40e7-9c0e-a57e2ef7b669", None, "f"),
        (2, 1, "5fa268ba-7872-4bfc-8bfe-53da27ce36aa", None, "g"),
    ]

    # ─── Figure layout (3 columns; last is colorbar gutter) ──────────────────
    fig = plt.figure(figsize=FIGSIZE)
    gs  = GridSpec(
        3, 3,
        width_ratios=[1, 1, 0.05],   # last col is color-bar gutter
        wspace=0.125,                # horizontal spacing
        hspace=0.15,                 # vertical spacing
        left=0.06, right=0.92, top=0.96, bottom=0.07
    )

    axes = np.empty((3, 2), dtype=object)
    pc_for_colorbar = None

    for row, col, exp_id, settings, tag in panels:
        ax = fig.add_subplot(gs[row, col])
        axes[row, col] = ax

        # Resolve experiment
        if exp_id is None and settings is not None:
            exp_id = find_experiment(engine, settings)

        # Data fetch + plot
        if exp_id is None:
            ax.text(0.5, 0.5, "No matching experiment",
                    ha="center", va="center", fontsize=16)
        else:
            power, cur, frq = get_nr_data(engine, exp_id)
            if power is None:
                ax.text(0.5, 0.5, "No data found",
                        ha="center", va="center", fontsize=16)
            else:
                pc = plot_panel(ax,
                                power,
                                cur,
                                frq,
                                show_xlabel=(row == 2),
                                show_ylabel=(col == 0))
                if pc_for_colorbar is None:
                    pc_for_colorbar = pc

        # Outside panel tag (bold sans-serif)
        pos = ax.get_position()
        x_offset = 0.03 if col == 0 else 0.0
        fig.text(pos.x0 - x_offset, pos.y1 - 0.005, tag,
                 fontsize=TAG_FONT, fontweight="bold",
                 fontfamily="sans-serif", ha="right", va="bottom")

    # ─── single full-height color-bar aligned to right column ────────────────
    top_right    = axes[0, 1].get_position()
    bottom_right = axes[2, 1].get_position()

    pad   = 0.0125
    width = 0.0175
    x0    = top_right.x1 + pad
    y0    = bottom_right.y0
    height = top_right.y1 - bottom_right.y0

    cax = fig.add_axes([x0, y0, width, height])
    cbar = fig.colorbar(pc_for_colorbar, cax=cax)
    cbar.set_label("Power [dBm]", fontsize=LABEL_FONT)
    cbar.ax.tick_params(labelsize=TICK_FONT)

    # ─── save ────────────────────────────────────────────────────────────────
    if not os.path.exists("../../.figures"):
        os.makedirs("../../.figures")

    out_path = os.path.abspath("../../.figures/FIG_1_platform.png")
    fig.savefig(out_path, dpi=SAVE_DPI, facecolor="white", bbox_inches="tight")
    plt.close(fig)
    print(f"Figure saved -> {out_path}")

if __name__ == "__main__":
    main()
