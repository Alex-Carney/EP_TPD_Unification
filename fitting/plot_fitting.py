"""pt_plot.py
Utility for visualising raw traces and fitted single-resonance models
produced by mpfit.FitOutcome.
"""

from __future__ import annotations
import os
from typing import Sequence, Optional, Tuple
import numpy as np
import matplotlib.pyplot as plt

from fitting.model_fitting import FitOutcome, CoupledFitOutcome

plt.rcParams["font.family"] = "sans-serif"


DB_EPS = 1e-18  # avoid log of zero


def _linear_to_db(x: np.ndarray) -> np.ndarray:
    return 10 * np.log10(np.clip(x, DB_EPS, None))


def plot_trace_with_model(
        freqs: Sequence[float],
        data_trace_dbm: Sequence[float],
        fit_outcome: FitOutcome,
        *,
        title: Optional[str] = None,
        voltage: Optional[float] = None,
        trace_label: str = "data",
        model_label: str = "fit",
        save_path: Optional[str | os.PathLike[str]] = None,
        ax: Optional[plt.Axes] = None,
) -> Tuple[plt.Figure, plt.Axes]:
    """Overlay measured trace (dB) with fitted model curve.

    Parameters
    ----------
    freqs : array-like
        Frequency axis (same units as in FitOutcome).
    data_trace_dbm : array-like
        Measured power in dB.
    fit_outcome : FitOutcome
        Result returned by mpfit.fit_cavity_trace / fit_yig_trace.
    title : str, optional
        Plot title.
    voltage : float, optional
        Voltage bias associated with the trace (included in annotation).
    trace_label, model_label : str, optional
        Legend labels.
    save_path : str or Path, optional
        If given, figure is saved there (directories auto-created).
    ax : matplotlib.axes.Axes, optional
        Existing axis to draw on; if *None*, a new figure is created.

    Returns
    -------
    fig, ax : matplotlib figure and axes.
    """
    freqs = np.asarray(freqs, dtype=float)
    data_trace_dbm = np.asarray(data_trace_dbm, dtype=float)

    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 4))
    else:
        fig = ax.figure

    # Plot raw data
    ax.plot(freqs, data_trace_dbm, label=trace_label, lw=1)

    # Convert model (linear) → dB and plot
    model_db = _linear_to_db(fit_outcome.model_curve)
    ax.plot(freqs, model_db, label=model_label, lw=2, ls="--")

    # Annotations ------------------------------------------------------
    txt_lines = [
        rf"$f_0 = {fit_outcome.f0:.6g} \pm {fit_outcome.f0_err:.2g}$",
        rf"$\kappa = {fit_outcome.kappa:.6g} \pm {fit_outcome.kappa_err:.2g}$",
    ]
    if voltage is not None:
        txt_lines.insert(0, rf"$V = {voltage:.4g}$")
    ax.text(0.02, 0.95, "\n".join(txt_lines), transform=ax.transAxes,
            va="top", ha="left", fontsize=10, bbox=dict(boxstyle="round", alpha=0.1))

    # Axis labels & legend
    ax.set_xlabel("Frequency (same units as input)")
    ax.set_ylabel("Power (dB)")
    if title is not None:
        ax.set_title(title)
    ax.legend()
    ax.grid(True, ls=":", alpha=0.4)

    # Save file if requested
    if save_path is not None:
        save_path = os.fspath(save_path)
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, dpi=400, bbox_inches="tight")

    plt.close(fig)
    return fig, ax


def plot_coupled_trace_with_model(
        freqs: Sequence[float],
        data_trace_dbm: Sequence[float],
        fit_outcome: CoupledFitOutcome,
        *,
        title: Optional[str] = None,
        voltage: Optional[float] = None,
        trace_label: str = "data",
        model_label: str = "coupled-fit",
        save_path: Optional[str | os.PathLike[str]] = None,
        ax: Optional[plt.Axes] = None,
        found_peaks: Optional[Sequence[float]] = None,
        found_peaks_maxima: Optional[Sequence[float]] = None,
        found_peaks_minima: Optional[Sequence[float]] = None,
        extra_info  = None,
        min_peak_height=10,
        plot_in_linear=False
) -> Tuple[plt.Figure, plt.Axes]:
    """Overlay raw trace, fitted model, stars for peak means,
    and vertical lines for [min,max] of each peak."""
    freqs = np.asarray(freqs, dtype=float)
    data_trace_dbm = np.asarray(data_trace_dbm, dtype=float)

    if plot_in_linear:
        print('in here should be linear')
        data_trace_dbm = 10**(data_trace_dbm/10)

    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 4))
    else:
        fig = ax.figure

    # raw data
    ax.plot(freqs, data_trace_dbm, label=trace_label, lw=1)

    # model (linear → dB)
    model_db = _linear_to_db(fit_outcome.model_curve) if not plot_in_linear else fit_outcome.model_curve
    ax.plot(freqs, model_db, label=model_label, lw=2, ls="--")

    # ------------------------------------------------------------------
    # Peaks: mean star + min/max vertical bands
    # ------------------------------------------------------------------
    if found_peaks is not None:
        n_peaks = len(found_peaks)
        for i, peak in enumerate(found_peaks):
            # star at the model curve
            ax.scatter(
                peak,
                model_db[np.abs(freqs - peak).argmin()],
                marker="*",
                color="red" if n_peaks == 2 else "blue",
                zorder=5,
            )

            # draw min / max bars if provided
            if found_peaks_minima and i < len(found_peaks_minima):
                ax.axvline(found_peaks_minima[i],
                           color="gray", ls="--", lw=1, alpha=0.5)
            if found_peaks_maxima and i < len(found_peaks_maxima):
                ax.axvline(found_peaks_maxima[i],
                           color="gray", ls="--", lw=1, alpha=0.5)

    # annotation panel
    txt = [
        rf"$J = {fit_outcome.J:.6g} \pm {fit_outcome.J_err:.2g}$",
        rf"$\chi_r^2 = {fit_outcome.redchi:.2f}$",
        rf"$\chi_r^2$ Norm = {fit_outcome.redchi_normalized:.2f}",
        rf"$\phi = {fit_outcome.phi:.6g} \pm {fit_outcome.phi_err:.2g}$",
    ]
    # Only add extra_info lines if extra_info is not None
    if extra_info is not None:
        txt.extend([
            rf"$f_0 = {extra_info['f0_c']:.6g} \pm {extra_info['f0_err_c']:.2g}$",
            rf"$\kappa = {extra_info['kappa_c']:.6g} \pm {extra_info['kappa_err_c']:.2g}$",
            rf"$\Delta_f = {extra_info['Delta_f']:.6g} \pm {extra_info['Delta_f_err']:.2g}$",
            rf"$\Delta_kappa = {extra_info['Delta_kappa']:.6g} \pm {extra_info['Delta_kappa_err']:.2g}$",
        ])
    if voltage is not None:
        txt.insert(0, rf"$V = {voltage:.4g}$")
    ax.text(
        0.02, 0.95, "\n".join(txt), transform=ax.transAxes,
        va="top", ha="left", fontsize=10,
        bbox=dict(boxstyle="round", alpha=0.1)
    )

    ax.set_xlabel("Frequency")
    ax.set_ylabel("Power (dB)")
    if title:
        ax.set_title(title)
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, -0.3))
    ax.grid(True, ls=":", alpha=0.4)

    # save fig
    if save_path is not None:
        save_path = os.fspath(save_path)
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, dpi=400, bbox_inches="tight")

    # Close fig
    plt.close(fig)

    return fig, ax



__all__ = ["plot_trace_with_model"]
