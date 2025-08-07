# small_panes.py  ────────────────────────────────────────────────────────────
"""Unified helper for drawing *small* dimension‑less panels.

The original implementation kept two nearly identical functions
(`left_small_panels` and `right_small_panels`) that specialised in
*either* a Δκ scan at φ = 0 *or* a Δf scan at φ = π.  That split became
awkward once we wanted additional phase values (π/2) and arbitrary scan
axes.

This rewrite introduces **`create_small_panel`** – a single, flexible
routine that accepts *parallel* arrays of ``Delta_f`` and ``Delta_kappa``
values plus the phase ``phi``.  If you want to scan only one axis, pass a
vector for the independent axis and a *same‑length* zero‑vector for the
other.  (That is exactly what the old split implementation did under the
hood.)

Legacy wrappers ``left_small_panels`` / ``right_small_panels`` are kept
so that existing scripts continue to run; internally they delegate to
``create_small_panel`` with the appropriate arguments.

Usage example ──────────────────────────────────────────────────────────
# >>> # scan Δκ at φ = 0
# >>> dk = np.linspace(-2.2e6, 0.1e6, 1_000)
# >>> df = np.zeros_like(dk)
# >>> create_small_panel(ax_top, ax_bot,
# ...     data_dir=DATA_DIR, exp_id=EXP_ID,
# ...     J_scale=J, f_c=f_c_HZ, phi=0.0,
# ...     delta_f=df, delta_kappa=dk,
# ...     x_label=r"$\Delta_\kappa/J$", ...)
# """
from __future__ import annotations

from pathlib import Path
from typing import Sequence, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import matplotlib.patches as mpatches
import matplotlib.transforms as mtrans
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import fitting.peak_fitting as peaks
from figures.figure_3_experiment.experiment_tpds import standard_tpd_locations

from figures.figure_3_experiment.style_maps import phase_peak_theory_color_map
from models.analysis import AnalyzedExperiment

# ────────────────────────────────────── plotting constants
# COL_PEAK             = "purple"
COL_EIG              = "black"
LS_EIG               = "--"

COL_TPD, LS_TPD      = "cyan", "-"
COL_EP,  LS_EP       = "red",  "-"
COL_INS, LS_INS      = "lime", "-"


LEGEND_FONT_SIZE     = 12
INSET_LABEL_SIZE     = 8

XRADIUS              = 0.05
YR_IM_RAD            = 0.50
YR_RE_RAD            = 0.50

VERT_W                           = 2.0
BASE_LINEWIDTH          = 2.5

# ────────────────────────────────────── helpers


def _fmt_axis(ax: plt.Axes, *, hide_x: bool = False):
    fmt = mtick.FuncFormatter(lambda v, _p: f"{v:.2f}")
    ax.yaxis.set_major_formatter(fmt)
    if hide_x:
        ax.tick_params(labelbottom=False)
    else:
        ax.xaxis.set_major_formatter(fmt)


# ────────────────────────────────────── core drawing routine

def create_small_panel(
        ax_top: plt.Axes,
        ax_bot: plt.Axes,
        *,
        data_dir: Path,
        analyzed_experiment: AnalyzedExperiment,
        J_scale: float,
        f_c: float,
        phi: float,
        delta_f: Sequence[float],
        delta_kappa: Sequence[float],
        x_label: str,
        draw_inset: bool = False,
        draw_unstable: bool = False,
        include_legend: bool = True,
        xlims: Tuple[float, float] | None = None,
        ylims_freq: Tuple[float, float] | None = None,
        ylims_re_eig: Tuple[float, float] | None = None,
):
    """Draw the pair of *small* panels (|Im λ| + ν, Re λ) into *ax_top* / *ax_bot*.

    Parameters
    ----------
    ax_top / ax_bot
        Matplotlib axes that will receive the plots.
    data_dir, exp_id
        Look‑up directory and *experiment‑ID* (file stem of csv files).
    J_scale, f_c
        Scaling constants so that quantities are rendered in *dimension‑less*
        form (value / J_scale − f_c/J_scale).
    phi
        Phase value in *radians*.
    delta_f, delta_kappa
        Equal‑length arrays defining the scan.  If you want a Δκ scan
        (traditional left column) pass ``delta_kappa`` as the vector and
        ``delta_f = np.zeros_like(delta_kappa)``.  Vice‑versa for a Δf
        scan.
    x_label
        LaTeX string for the x‑axis label (e.g. ``r"$\Delta_f/J$"``).
    draw_inset
        If *True* draw a zoom‑in rectangle and inset.  Uses hard‑coded
        radii constants above (identical to the original behaviour).
    draw_unstable
        Whether to annotate the *unstable* region (only makes sense for
        φ ≈ π scans).
    include_legend
        Add legends to both panels.
    xlims, ylims_freq, ylims_re_eig
        Axis limits (dimension‑less).  Pass *None* to auto‑scale.
    """

    COL_PEAK = phase_peak_theory_color_map(phi)

    # sanity check
    delta_f = np.asarray(delta_f, dtype=float)
    delta_kappa = np.asarray(delta_kappa, dtype=float)
    if delta_f.shape != delta_kappa.shape:
        raise ValueError("delta_f and delta_kappa must have the same shape")

    # ── load constants ───────────────────────────────────────────────
    J, f_c_HZ, k_c = analyzed_experiment.J_avg, analyzed_experiment.f_c_avg, analyzed_experiment.kappa_c_avg
    k_c_std = analyzed_experiment.kappa_c_std

    scale = J_scale

    # offset: shift frequencies by −f_c/J (φ ≠ π) or specialised shift
    # matching the old right‑panel behaviour when φ = π.
    if np.isclose(phi, np.pi):
        # TPD_DELTA_F = np.sqrt(4 * J**2 + k_c**2)
        # tpd_frequency = (f_c_HZ - TPD_DELTA_F/2) / scale
        ep_delta_f = 2 * J
        tpd_frequency = (f_c_HZ - ep_delta_f/2) / scale
    elif np.isclose(phi, np.pi/2):
        # tpd_location = standard_tpd_locations(phi, k_c/J)
        # tpd_frequency = (f_c_HZ - (tpd_location.Delta_tilde_f * J)/2) / scale
        ep_delta_f = -np.sqrt(2) * J
        tpd_frequency = (f_c_HZ - ep_delta_f/2) / scale
    else:
        tpd_frequency = f_c_HZ / scale

    # containers ------------------------------------------------------
    im_hi, im_lo, re_hi, re_lo = [], [], [], []
    pk_hi, pk_lo               = [], []

    # iterate over scan points ---------------------------------------
    for df_i, dk_i in zip(delta_f, delta_kappa):
        # physical (Hz) values already
        lam1, lam2 = peaks.eigenvalues(J, f_c_HZ, k_c, df_i, dk_i, phi)
        lp, lm = sorted([lam1, lam2], key=lambda z: z.imag, reverse=True)

        # if np.isclose(phi, np.pi/2):
        #     print(f"df: {df_i} dk: {dk_i} lp: {lp} lm: {lm}")
        #     print(abs(lp.imag) / scale)

        im_hi.append(abs(lp.imag) / scale - tpd_frequency)
        im_lo.append(abs(lm.imag) / scale - tpd_frequency)

        re_hi.append(lp.real / scale)
        re_lo.append(lm.real / scale)

        nu = peaks.peak_location(J, f_c_HZ, k_c, df_i, dk_i, phi)
        lo_val, hi_val = (sorted(nu) if len(nu) == 2 else (nu[0], nu[0]))
        pk_hi.append(hi_val / scale - tpd_frequency)
        pk_lo.append(lo_val / scale - tpd_frequency)

    # convert containers to numpy arrays for faster slicing / insets
    im_hi = np.asarray(im_hi)
    im_lo = np.asarray(im_lo)
    re_hi = np.asarray(re_hi)
    re_lo = np.asarray(re_lo)
    pk_hi = np.asarray(pk_hi)
    pk_lo = np.asarray(pk_lo)

    # x‑axis (dimension‑less)
    if np.allclose(delta_kappa, 0):
        x_vals = delta_f / scale
        scan_type = "df"
    else:
        x_vals = delta_kappa / scale
        scan_type = "dk"

    # ── TOP PLOT  |Im λ| + ν peaks -----------------------------------
    ax_top.plot(x_vals, im_hi, color=COL_EIG, linestyle=LS_EIG,
                label=r"$|\mathrm{Im}(\tilde \lambda_\pm)|$", lw=BASE_LINEWIDTH)
    ax_top.plot(x_vals, im_lo, color=COL_EIG, linestyle=LS_EIG, lw=BASE_LINEWIDTH)
    ax_top.plot(x_vals, pk_hi, color=COL_PEAK, lw=BASE_LINEWIDTH)
    ax_top.plot(x_vals, pk_lo, color=COL_PEAK, lw=BASE_LINEWIDTH)
    ax_top.set_ylabel(r"$(\mathrm{Frequency}-f_{EP})/J$")

    # ── BOTTOM PLOT  Re λ -------------------------------------------
    # This is to fix teh two dashed lines plotting on top of each other and becoming a solid line
    eps = 1e-6  # adjust as needed
    split_mask = np.abs(re_hi - re_lo) > eps
    re_lo_masked = np.where(split_mask, re_lo, np.nan)

    ax_bot.plot(x_vals, re_hi, color=COL_EIG, linestyle=LS_EIG, lw=BASE_LINEWIDTH,
                label=r"$\mathrm{Re}(\tilde \lambda_\pm)$")
    ax_bot.plot(x_vals, re_lo_masked, color=COL_EIG, linestyle=LS_EIG, lw=BASE_LINEWIDTH)
    ax_bot.axhline(0.0, color="k", ls=":", lw=BASE_LINEWIDTH, label=r"$\mathrm{Re}(\tilde \lambda)=0$")
    ax_bot.set_xlabel(x_label)
    ax_bot.set_ylabel(r"$\mathrm{Frequency}/J$")

    # ── vertical reference lines ------------------------------------
    x_v_ep = x_v_tpd = x_v_inst = None
    if scan_type == "dk":
        if np.isclose(phi, 0):
            # EP and TPD along Δκ
            x_v_ep  = (-2 * J) / scale
            # x_v_tpd = ((k_c - np.sqrt(max(0.0, 8*J**2 - k_c**2))) / 2) / scale
            tpd_location = standard_tpd_locations(phi, k_c/J, sigma_kappa_tilde_c=k_c_std/J)
            print(f"for phi=0, dk: {k_c/J}, tpd: {tpd_location}")
            x_v_tpd = tpd_location.Delta_tilde_kappa
            # Inst
            ktc = k_c/scale
            x_v_inst = ktc if ktc ** 2 - 4 < 0 else (ktc**2 + 4)/(2 * ktc)
        else:
            # Get the TPD location from the other script
            ep_location = standard_ep_locations(phi, k_c/J)
            tpd_location = standard_tpd_locations(phi, k_c/J, sigma_kappa_tilde_c=k_c_std/J)
            print(f"for phi={phi}, dk: {k_c/J}, ep: {ep_location.Delta_tilde_kappa}, tpd: {tpd_location}")
            x_v_ep = ep_location.Delta_tilde_kappa
            x_v_tpd = tpd_location.Delta_tilde_kappa
    else:  # Δf scan at phi = pi
        x_v_ep  = ( 2 * J) / scale
        x_v_tpd =  np.sqrt(4 * J**2 + k_c**2) / scale
        x_v_inst = np.sqrt(max(0.0, 4 * J**2 - k_c**2)) / scale
        tpd_location = standard_tpd_locations(phi, k_c/J, sigma_kappa_tilde_c=k_c_std/J)
        print(f"for phi={phi}, df: {k_c/J}, ep: {x_v_ep}, tpd: {tpd_location}, inst: {x_v_inst}")

    for a in (ax_top, ax_bot):
        a.axvline(x_v_tpd, color=COL_TPD, ls=LS_TPD, lw=VERT_W)
        a.axvline(x_v_ep,  color=COL_EP,  ls=LS_EP,  lw=VERT_W)
        if x_v_inst is not None:
            a.axvline(x_v_inst, color=COL_INS, ls=LS_INS, lw=VERT_W)

    if include_legend and k_c/J < 1:
        ax_top.legend(fontsize=LEGEND_FONT_SIZE, loc="lower right",
                      framealpha=1, borderpad=0.2, bbox_to_anchor=(1.04, -0.03))
        ax_bot.legend(fontsize=LEGEND_FONT_SIZE, loc="lower right",
                      framealpha=1, borderpad=0.2, bbox_to_anchor=(1.04, -0.03))

    # ── unstable region annotation (φ ≈ π & Δf scan) ----------------
    if draw_unstable and scan_type == "df" and np.isclose((phi % (2*np.pi)), np.pi):
        x_span = x_vals.max() - x_vals.min()
        off    = 0.18 * x_span
        transform = lambda ax: mtrans.blended_transform_factory(ax.transData, ax.transAxes)
        for a, yfrac in ((ax_top, 0.95), (ax_bot, 0.90)):
            a.text(x_v_inst - off, yfrac, "unstable", rotation=90,
                   transform=transform(a), ha="right", va="top",
                   fontsize=LEGEND_FONT_SIZE - 2, fontweight="bold")

    # ── optional insets --------------------------------------------
    if draw_inset:
        vlines = [(x_v_tpd,  COL_TPD, LS_TPD),
                  (x_v_ep,   COL_EP,  LS_EP)]
        if x_v_inst is not None:
            vlines.append((x_v_inst, COL_INS, LS_INS))
        _draw_inset(
            ax_top, x=x_vals, c1=im_hi, c2=im_lo,
            peaks_hi=pk_hi, peaks_lo=pk_lo,
            x0=x_v_ep, y_rad=YR_IM_RAD, x_rad=XRADIUS,
            vlines=vlines, draw_zero=False,
        )
        _draw_inset(
            ax_bot, x=x_vals, c1=re_hi, c2=re_lo,
            peaks_hi=None, peaks_lo=None,
            x0=x_v_ep, y_rad=YR_RE_RAD, x_rad=XRADIUS,
            vlines=vlines, draw_zero=True,
        )

    # ── cosmetics ---------------------------------------------------
    _fmt_axis(ax_top, hide_x=True)
    _fmt_axis(ax_bot)

    if xlims is not None:
        ax_top.set_xlim(*xlims)
        ax_bot.set_xlim(*xlims)
    if ylims_freq is not None:
        ax_top.set_ylim(*ylims_freq)
    if ylims_re_eig is not None:
        ax_bot.set_ylim(*ylims_re_eig)


# ────────────────────────────────────── legacy convenience wrappers

def left_small_panels(*args, **kwargs):
    """Shim that reproduces the old Δκ‑scan @ φ = 0 behaviour."""
    # extract axes from args[0:2]
    ax_top, ax_bot = args[:2]
    # remaining kwargs forwarded; we inject phi and scan arrays.
    dk = np.linspace(-2.2e6, 1e6, 1_000)
    df = np.zeros_like(dk)
    create_small_panel(
        ax_top, ax_bot,
        **{k: v for k, v in kwargs.items() if k not in ("xlims", "ylims_freq", "ylims_re_eig", "include_legend")},
        phi=0.0,
        delta_f=df,
        delta_kappa=dk,
        x_label=r"$\tilde \Delta_\kappa$",
        draw_inset=False,
        draw_unstable=False,
        include_legend=kwargs.get("include_legend", True),
        xlims=kwargs.get("xlims"),
        ylims_freq=kwargs.get("ylims_freq"),
        ylims_re_eig=kwargs.get("ylims_re_eig"),
    )


def right_small_panels(*args, **kwargs):
    """Shim reproducing the old Δf‑scan @ φ = π behaviour."""
    ax_top, ax_bot = args[:2]
    expr: AnalyzedExperiment = kwargs["analyzed_experiment"]
    J = expr.J_avg
    df = np.linspace(-2.2*J, 5*J, 1_000)  # crude but matches old range
    dk = np.zeros_like(df)
    create_small_panel(
        ax_top, ax_bot,
        **{k: v for k, v in kwargs.items() if k not in ("xlims", "ylims_freq", "ylims_re_eig", "include_legend", "draw_inset", "draw_unstable")},
        phi=np.pi,
        delta_f=df,
        delta_kappa=dk,
        x_label=r"$\tilde \Delta_f$",
        draw_inset=kwargs.get("draw_inset", True),
        draw_unstable=kwargs.get("draw_unstable", True),
        include_legend=kwargs.get("include_legend", True),
        xlims=kwargs.get("xlims"),
        ylims_freq=kwargs.get("ylims_freq"),
        ylims_re_eig=kwargs.get("ylims_re_eig"),
    )

# ----------------------------------------------------------------------
#  HYBRID phase (φ ≠ 0, π)  –  hyperbolic path through (Δκ, Δf)
# ----------------------------------------------------------------------
def hybrid_small_panels(ax_top: plt.Axes,
                        ax_bot: plt.Axes,
                        *, data_dir: Path, exp_id: str,
                        J_scale: float, f_c: float,
                        phi: float,
                        xlims: tuple[float, float] = None,
                        ylims_freq: tuple[float, float] = None,
                        ylims_re_eig: tuple[float, float] = None,
                        include_legend: bool = True):
    """
    Draw the small-panel pair for ‘intermediate’ phases (e.g. φ = π/2).

    A Δκ grid is generated, then the corresponding Δf is
        Δf = 2 sin(φ) / Δκ            (same units as J, so Hz here)
    """
    # 1) choose Δκ sweep (Hz).  Avoid 0 to prevent division-by-zero.
    dk_hz = np.linspace(-2.2, 0, 500, endpoint=False)   # left branch

    # 2) compute companion Δf  (Hz)
    df_hz = (2 * np.sin(phi)) / dk_hz

    # 3) call the generic helper
    create_small_panel(
        ax_top, ax_bot,
        data_dir=data_dir, exp_id=exp_id,
        J_scale=J_scale, f_c=f_c,
        phi=phi,
        delta_f=df_hz * J_scale,
        delta_kappa=dk_hz * J_scale,
        x_label=r"$\tilde \Delta_\kappa$ (Hyperbolic)",       # x-axis is Δκ
        draw_inset=False,                   # change if you want insets
        draw_unstable=False,
        include_legend=include_legend,
        xlims=xlims,
        ylims_freq=ylims_freq,
        ylims_re_eig=ylims_re_eig,
    )


# keep _draw_inset private below the wrappers -------------------------

def _draw_inset(ax_par: plt.Axes,
                *, x: np.ndarray,
                c1: np.ndarray, c2: np.ndarray,
                peaks_hi: np.ndarray | None,
                peaks_lo: np.ndarray | None,
                x0: float, y_rad: float, x_rad: float,
                vlines: Tuple[Tuple[float, str, str], ...],
                draw_zero: bool):
    """Draw zoom‑in rectangle + inset (copied verbatim from old code)."""
    y_mid = 0.5 * (np.interp(x0, x, c1) + np.interp(x0, x, c2))

    rect = mpatches.Rectangle((x0 - x_rad, y_mid - y_rad),
                              2 * x_rad, 2 * y_rad,
                              fc="none", ec="black", lw=1.2, zorder=3)
    ax_par.add_patch(rect)

    ins = inset_axes(ax_par, width=0.93, height=0.9,
                     bbox_to_anchor=(0.22, 0.01, 1.0, 1.0),
                     bbox_transform=ax_par.transAxes,
                     loc="lower left")

    ins.plot(x, c1, color=COL_EIG, linestyle=LS_EIG)
    ins.plot(x, c2, color=COL_EIG, linestyle=LS_EIG)
    if peaks_hi is not None:
        ins.plot(x, peaks_hi, color=COL_PEAK)
        ins.plot(x, peaks_lo, color=COL_PEAK)

    for xv, col, ls in vlines:
        ins.axvline(xv, color=col, ls=ls, lw=BASE_LINEWIDTH)
    if draw_zero:
        ins.axhline(0.0, color="k", ls="--", lw=BASE_LINEWIDTH)

    ins.set_xlim(x0 - x_rad, x0 + x_rad)
    ins.set_ylim(y_mid - y_rad, y_mid + y_rad)

    ins.xaxis.set_major_locator(mtick.MaxNLocator(nbins=3))
    ins.yaxis.set_major_locator(mtick.MaxNLocator(nbins=3))
    _fmt_axis(ins)
    ins.tick_params(axis="both", labelsize=INSET_LABEL_SIZE, pad=0)
