from __future__ import annotations

import importlib
from pathlib import Path
import sys
from typing import Iterable

import matplotlib.container as mcontainer
from matplotlib.ticker import FixedLocator, FormatStrFormatter
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.gridspec import GridSpec
from matplotlib.lines import Line2D
from sqlalchemy import create_engine, select
from sqlalchemy.orm import selectinload, sessionmaker

# ---------------------------------------------------------------------------
# Formatting control block – edit here to tune the PRL single-column styling.
# ---------------------------------------------------------------------------
DISPLAY_TITLE = False
CFG = {
    "figure_size": (4, 7),
    "grid": {
        "left": 0.175,
        "right": 0.96,
        "bottom": 0.07,
        "top": 0.98,
        "wspace": 0.185,
        "hspace": 0.275,
    },
    "fonts": {
        "font.family": "sans-serif",
        "font.size": 9,
        "axes.labelsize": 13.5,
        "axes.titlesize": 11,
        "xtick.labelsize": 10.5,
        "ytick.labelsize": 10.5,
        "legend.fontsize": 8,

        "xtick.major.pad": 1.0,
        "ytick.major.pad": 1.0,
        "xtick.minor.pad": 1.0,
        "ytick.minor.pad": 1.0,
        "axes.labelpad": 1.0,  # distance from axis label to tick labels
    },
    "panel_label": {
        "fontsize": 13.5,
        "x": 0.1,
        "y": 0.96,
        "pathwidth": 1.2,
        "color": "black",
    },
    "data_markers": {
        "marker": "o",
        "size": 3.0,
        "facecolor": "black",
        "edgecolor": "black",
        "errorbar_lw": 0.6,
    },
    "imag_linewidth": 1.5,
    "peak_linewidth": 1.5,
    "sample_points": 500,
}

_THIS_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _THIS_DIR.parents[2]
_LARGE_FIG_DIR = _PROJECT_ROOT / "figures" / "large_figures" / "figure_3_experiment"

for candidate in (_PROJECT_ROOT, _LARGE_FIG_DIR):
    cand_str = str(candidate)
    if cand_str not in sys.path:
        sys.path.insert(0, cand_str)

big_pane = importlib.import_module("big_pane")
phase_style = importlib.import_module("style_maps")
peaks = importlib.import_module("fitting.peak_fitting")
from models.analysis import AnalyzedExperiment

big_pane.VERT_W = 1.3
big_pane.BASE_LINEWIDTH = 1.3
big_pane.LEGEND_FONT_SIZE = 9

COL_EIG = "black"
LS_EIG = "--"

DATA_LOCATION = _PROJECT_ROOT / "data" / "ep_tpd_transformed_data.db"

EXP1_L = "cab5729f-0491-41f9-9fc8-acf0096754ab"
EXP1_R = "62156c79-a18b-4c96-b4b5-2a43519d1900"
EXP2_L = "02f5e503-3e4e-4651-9ef8-d90104b3721f"
EXP2_R = "47809d5e-9041-456b-80d8-becf48bf1cfd"
EXP3_L = "25e2d559-5960-48c9-8cd2-9441cc067b61"
EXP3_R = "3ef835ef-e2be-4eb3-929e-3a5149a00e87"

LEFT_X_LIMS = (-2.25, 1.0)
LEFT_Y_FREQ_LIM = (-1.05, 1.05)
RIGHT_X_LIMS = (0.95, 3.0)
RIGHT_Y_FREQ_LIM = (-1.5, 1.0)
BOTTOM_X_LIMS = (-1.5, -0.3)
BOTTOM_Y_FREQ_LIM = (-1.0, 3.0)

PANEL_SPECS = [
    {"label": "(a)", "exp_id": EXP1_L, "phi": 0.0,       "phi_text": "0",      "size_text": "Small",
     "x_key": "Delta_kappa", "x_label": r"$\tilde \Delta_\kappa$",
     "xlims": LEFT_X_LIMS, "ylims": LEFT_Y_FREQ_LIM, "draw_unstable": True, "include_legend": False,

     "legend": {
         "entries": [
             {"key": "tpd_line", "label": r"$\tilde \Delta_\kappa^\mathrm{TPD}$"},
             {"key": "ep_line", "label": r"$\tilde \Delta_\kappa^\mathrm{EP}$"},
             {"key": "instability_line", "label": r"$\tilde \Delta_\kappa^\mathrm{NL}$"},
         ],
         "loc": "lower left",
     }


     },
    {"label": "(b)", "exp_id": EXP1_R, "phi": 0.0,       "phi_text": "0",      "size_text": "Large",
     "x_key": "Delta_kappa", "x_label": r"$\tilde \Delta_\kappa$",
     "xlims": LEFT_X_LIMS, "ylims": LEFT_Y_FREQ_LIM, "draw_unstable": True, "include_legend": False,
     "legend": {
         "entries": [
             {"key": "nu_data", "label": r"$\tilde \nu_\pm$ Data"},
             {"key": "nu_theory", "label": r"$\tilde \nu_\pm$ Theory"},
             {"key": "imag_eigs", "label": r"$|\mathrm{Im}( \tilde \lambda_\pm)|$"},
         ],
         "loc": "lower left",
     }
     },
    {"label": "(c)", "exp_id": EXP2_L, "phi": np.pi,     "phi_text": "\pi",   "size_text": "Small",
     "x_key": "Delta_f",     "x_label": r"$\tilde \Delta_f$",
     "xlims": RIGHT_X_LIMS, "ylims": RIGHT_Y_FREQ_LIM, "draw_unstable": False, "include_legend": False,

     "legend": {
         "entries": [
             {"key": "tpd_line", "label": r"$\tilde \Delta_f^\mathrm{TPD}$"},
             {"key": "ep_line", "label": r"$\tilde \Delta_f^\mathrm{EP}$"},
             {"key": "instability_line", "label": r"$\tilde \Delta_f^\mathrm{NL}$"},
         ],
         "loc": "lower left",
     }


     },
    {"label": "(d)", "exp_id": EXP2_R, "phi": np.pi,     "phi_text": "\pi",   "size_text": "Large",
     "x_key": "Delta_f",     "x_label": r"$\tilde \Delta_f$",
     "xlims": RIGHT_X_LIMS, "ylims": RIGHT_Y_FREQ_LIM, "draw_unstable": False, "include_legend": False,

     "legend": {
         "entries": [
             {"key": "nu_theory", "label": r"$\tilde \nu_\pm$ Theory"},
         ],
         "loc": "lower left",
     }

     },
    {"label": "(e)", "exp_id": EXP3_L, "phi": np.pi/2,  "phi_text": "\pi/2", "size_text": "Small",
     "x_key": "Delta_kappa", "x_label": r"Hyperbolic $\tilde \Delta_\kappa$",
     "xlims": BOTTOM_X_LIMS, "ylims": BOTTOM_Y_FREQ_LIM, "draw_unstable": False, "include_legend": False,

    "legend": {
         "entries": [
             {"key": "tpd_line", "label": r"$\tilde \Delta_\kappa^\mathrm{TPD}$"},
             {"key": "ep_line", "label": r"$\tilde \Delta_\kappa^\mathrm{EP}$"},
         ],
         "loc": "lower left",
     }

     },
    {"label": "(f)", "exp_id": EXP3_R, "phi": np.pi/2,  "phi_text": "\pi/2", "size_text": "Large",
     "x_key": "Delta_kappa", "x_label": r"Hyperbolic $\tilde \Delta_\kappa$",
     "xlims": BOTTOM_X_LIMS, "ylims": BOTTOM_Y_FREQ_LIM, "draw_unstable": False, "include_legend": False,

     "legend": {
         "entries": [
             {"key": "nu_theory", "label": r"$\tilde \nu_\pm$ Theory"},
         ],
         "loc": "lower left",
     }

     },
]


def _format_val_unc(val: float, unc: float) -> str:
    if unc <= 0 or np.isnan(unc):
        return f"{val:.3g}"
    exp = int(np.floor(np.log10(unc)))
    step = 10 ** exp
    unc_rounded = round(unc / step) * step
    if unc_rounded >= 10 * step:
        unc_rounded /= 10
        step /= 10
        exp -= 1
    val_rounded = round(val / step) * step
    paren = int(round(unc_rounded / step))
    dec = max(-exp, 0)
    fmt = f"{{:.{dec}f}}"
    return f"{fmt.format(val_rounded)}({paren})"


def _load_experiments(exp_ids: Iterable[str]) -> dict[str, AnalyzedExperiment]:
    engine = create_engine(f"sqlite:///{DATA_LOCATION.as_posix()}")
    Session = sessionmaker(bind=engine)
    ids = list(exp_ids)
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
    if len(rows) != len(ids):
        found = {row.analyzed_experiment_id for row in rows}
        missing = sorted(set(ids) - found)
        raise ValueError(f"Missing experiments: {missing}")
    return {row.analyzed_experiment_id: row for row in rows}


def _frequency_offset(J: float, f_c: float, phi: float) -> float:
    phi_mod = phi % (2 * np.pi)
    if np.isclose(phi_mod, np.pi):
        return (f_c - (2 * J) / 2) / J
    if np.isclose(phi_mod, np.pi / 2):
        return (f_c - (-np.sqrt(2) * J) / 2) / J
    return f_c / J


def _sample_scan(exp: AnalyzedExperiment, spec: dict, *, num: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    x_min, x_max = spec["xlims"]
    x_grid = np.linspace(x_min, x_max, num)
    phi = spec["phi"]
    J = exp.J_avg

    if spec["x_key"] == "Delta_kappa":
        delta_kappa = x_grid * J
        if np.isclose(phi % (2 * np.pi), np.pi / 2):
            safe = np.abs(x_grid) > 1e-6
            x_grid = x_grid[safe]
            delta_kappa = delta_kappa[safe]
            delta_f = (2 * np.sin(phi) / x_grid) * J
        else:
            delta_f = np.zeros_like(delta_kappa)
    else:
        delta_f = x_grid * J
        delta_kappa = np.zeros_like(delta_f)

    return x_grid, delta_f, delta_kappa


def _imaginary_eigen_curves(exp: AnalyzedExperiment, spec: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    x_grid, delta_f, delta_kappa = _sample_scan(exp, spec, num=CFG["sample_points"])
    J = exp.J_avg
    f_c = exp.f_c_avg
    kappa_c = exp.kappa_c_avg
    offset = _frequency_offset(J, f_c, spec["phi"])

    im_hi, im_lo = [], []
    for df, dk in zip(delta_f, delta_kappa):
        _, lam1, lam2 = peaks.eigenvalues(J, f_c, kappa_c, df, dk, spec["phi"])
        eigs = sorted((lam1, lam2), key=lambda val: val.imag, reverse=True)
        im_hi.append(abs(eigs[0].imag) / J - offset)
        im_lo.append(abs(eigs[1].imag) / J - offset)

    return x_grid, np.asarray(im_hi), np.asarray(im_lo)


def _theory_peaks(exp: AnalyzedExperiment, spec: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    x_grid, delta_f, delta_kappa = _sample_scan(exp, spec, num=CFG["sample_points"])
    J = exp.J_avg
    f_c = exp.f_c_avg
    kappa_c = exp.kappa_c_avg
    offset = _frequency_offset(J, f_c, spec["phi"])

    hi, lo = [], []
    for df, dk in zip(delta_f, delta_kappa):
        vals = peaks.peak_location(J, f_c, kappa_c, df, dk, spec["phi"])
        vals = sorted(vals) if len(vals) == 2 else (vals[0], vals[0])
        lo.append(vals[0] / J - offset)
        hi.append(vals[1] / J - offset)

    return x_grid, np.asarray(lo), np.asarray(hi)


def _label_panel(ax: plt.Axes, text: str) -> None:
    cfg = CFG["panel_label"]
    ax.text(
        cfg["x"], cfg["y"], text,
        transform=ax.transAxes,
        fontsize=cfg["fontsize"],
        color=cfg["color"],
        ha='left', va='top',
    )


def _panel_title(exp: AnalyzedExperiment, phi_text: str, size_text: str) -> str:
    ratio = exp.kappa_c_avg / exp.J_avg
    sigma = exp.kappa_c_std / exp.J_avg if exp.kappa_c_std is not None else 0.0
    kappa_txt = _format_val_unc(ratio, sigma)
    return rf"$\phi = {phi_text},\;\tilde \kappa_c = {kappa_txt}\,\mathrm{{ ({size_text})}}$"


def _restyle_data(ax: plt.Axes) -> None:
    cfg = CFG["data_markers"]
    for container in ax.containers:
        if isinstance(container, mcontainer.ErrorbarContainer):
            line = container.lines[0]
            line.set_marker(cfg["marker"])
            line.set_markersize(cfg["size"])
            line.set_markerfacecolor(cfg["facecolor"])
            line.set_markeredgecolor(cfg["edgecolor"])
            line.set_linestyle("None")
            for err in container.lines[1:]:
                if hasattr(err, "set_linewidth"):
                    err.set_linewidth(cfg["errorbar_lw"])
                elif isinstance(err, (list, tuple)):
                    for sub_err in err:
                        if hasattr(sub_err, "set_linewidth"):
                            sub_err.set_linewidth(cfg["errorbar_lw"])


def _legend_proxy(key: str, spec: dict, *, peak_color: str) -> Line2D | None:
    data_cfg = CFG["data_markers"]
    if key == "nu_data":
        return Line2D(
            [], [],
            marker=data_cfg["marker"],
            markersize=data_cfg["size"],
            markerfacecolor=data_cfg["facecolor"],
            markeredgecolor=data_cfg["edgecolor"],
            linestyle="None",
            color=data_cfg["edgecolor"],
        )
    if key == "nu_theory":
        return Line2D([], [], color=peak_color, linewidth=CFG["peak_linewidth"])
    if key == "imag_eigs":
        return Line2D([], [], color=COL_EIG, linestyle=LS_EIG, linewidth=CFG["imag_linewidth"])
    if key == "tpd_line":
        return Line2D([], [], color="cyan", linestyle="-", linewidth=big_pane.VERT_W)
    if key == "ep_line":
        return Line2D([], [], color="red", linestyle="-", linewidth=big_pane.VERT_W)
    if key == "instability_line":
        return Line2D([], [], color="lime", linestyle="-", linewidth=big_pane.VERT_W)
    return None


def _apply_panel_legend(ax: plt.Axes, spec: dict, *, peak_color: str) -> None:
    legend_cfg = spec.get("legend")
    if not legend_cfg:
        return

    existing = ax.get_legend()
    if existing is not None:
        existing.remove()

    handles: list[Line2D] = []
    labels: list[str] = []
    for entry in legend_cfg.get("entries", []):
        key = entry.get("key")
        if not key:
            continue
        handle = _legend_proxy(key, spec, peak_color=peak_color)
        if handle is None:
            continue
        handles.append(handle)
        labels.append(entry.get("label", ""))

    if not handles:
        return

    legend_kwargs = {
        "loc": legend_cfg.get("loc", "lower left"),
        "fontsize": legend_cfg.get("fontsize", CFG["fonts"]["legend.fontsize"]),
        "framealpha": legend_cfg.get("framealpha", 1.0),
        "borderpad": legend_cfg.get("borderpad", 0.15),
        "handlelength": legend_cfg.get("handlelength", 1.5),
        "handletextpad": legend_cfg.get("handletextpad", 0.5),
    }
    ax.legend(handles, labels, **legend_kwargs)


def build_single_column_figure(filename: str = "../../.figures/FIG_3_experiment_small.png") -> None:
    plt.rcParams.update(CFG["fonts"])

    experiments = _load_experiments(spec["exp_id"] for spec in PANEL_SPECS)

    fig = plt.figure(figsize=CFG["figure_size"])
    gs = GridSpec(
        3,
        2,
        figure=fig,
        left=CFG["grid"]["left"],
        right=CFG["grid"]["right"],
        bottom=CFG["grid"]["bottom"],
        top=CFG["grid"]["top"],
        hspace=CFG["grid"]["hspace"],
        wspace=CFG["grid"]["wspace"],
    )

    for idx, spec in enumerate(PANEL_SPECS):
        row, col = divmod(idx, 2)
        ax = fig.add_subplot(gs[row, col])
        exp = experiments[spec["exp_id"]]

        big_pane.plot_big_pane(
            ax,
            DATA_LOCATION,
            analyzed_experiment=exp,
            x_key=spec["x_key"],
            xlab=spec["x_label"],
            J_scale=exp.J_avg,
            f_c=exp.f_c_avg,
            kappa_c=exp.kappa_c_avg,
            phi_val=spec["phi"],
            draw_unstable=spec["draw_unstable"],
            xlims=spec["xlims"],
            ylims_freq=spec["ylims"],
            include_legend=spec["include_legend"],
        )

        _restyle_data(ax)

        # Reinstate axis labels and ticks for every panel; PRL layout needs each row annotated.
        ax.set_xlabel(spec["x_label"])
        ax.tick_params(axis="x", which="both", bottom=True, labelbottom=True)

        x_vals, im_hi, im_lo = _imaginary_eigen_curves(exp, spec)
        ax.plot(x_vals, im_hi, color=COL_EIG, linestyle=LS_EIG,
                linewidth=CFG["imag_linewidth"], label=r"$|\mathrm{Im}(\tilde \lambda_\pm)|$")
        ax.plot(x_vals, im_lo, color=COL_EIG, linestyle=LS_EIG,
                linewidth=CFG["imag_linewidth"])

        peak_color = phase_style.phase_peak_theory_color_map(spec["phi"])
        x_vals_peak, peak_lo, peak_hi = _theory_peaks(exp, spec)
        ax.plot(x_vals_peak, peak_hi, color=peak_color,
                linewidth=CFG["peak_linewidth"], label=r"$\tilde \nu_\pm$ Theory")
        ax.plot(x_vals_peak, peak_lo, color=peak_color,
                linewidth=CFG["peak_linewidth"])

        if col == 1:
            ax.set_ylabel("")
            ax.tick_params(axis="y", labelleft=False)

        if row == 0:
            ticks = [-2.0, -0.75, -0.75 + 1.25]
            ax.xaxis.set_major_locator(FixedLocator(ticks))
            ax.xaxis.set_major_formatter(FormatStrFormatter("%.2f"))

        if row == 1:  # middle row only
            ticks = [1.25, 2.0, 2.75]  # or [1.25, 2.0, 2.75]
            ax.xaxis.set_major_locator(FixedLocator(ticks))
            ax.xaxis.set_major_formatter(FormatStrFormatter("%.2f"))

        if DISPLAY_TITLE:
            ax.set_title(_panel_title(exp, spec["phi_text"], spec["size_text"]), fontsize=9)
        _label_panel(ax, spec["label"])

        _apply_panel_legend(ax, spec, peak_color=peak_color)

    output_path = Path(filename)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=1000, facecolor="white")
    plt.close(fig)


if __name__ == "__main__":
    default_path = Path("../../.figures/FIG_3_experiment_small.png")
    default_path.parent.mkdir(parents=True, exist_ok=True)
    build_single_column_figure(filename=str(default_path))
