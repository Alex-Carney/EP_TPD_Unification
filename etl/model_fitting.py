"""
mpfit.py — v0.6  (baseline-free, optional phi fit)

Changes from v0.5
-----------------
* No baseline terms anywhere (already true since v0.5).
* `fit_coupled_trace` now has keyword `fit_phi` (default False).
    • If `fit_phi=False`  →  phi is fixed to the value you pass.
    • If `fit_phi=True`   →  phi becomes a third fitted parameter.
* `CoupledFitOutcome` now always carries `phi` and `phi_err`
  (phi_err = 0.0 when phi is fixed).

Public helpers
--------------
    fit_cavity_trace
    fit_yig_trace
    fit_coupled_trace
"""

from __future__ import annotations
import math
import numpy as np
from dataclasses import dataclass
from typing import Sequence, Tuple, Callable, Optional

import pandas as pd
from matplotlib import pyplot as plt
from scipy.optimize import curve_fit, OptimizeWarning
import warnings

__all__ = [
    "cavity_response",
    "yig_response",
    "fit_cavity_trace",
    "fit_yig_trace",
    "fit_coupled_trace",
    "FitOutcome",
    "CoupledFitOutcome",
]

_RNG = np.random.default_rng()


# ---------------------------------------------------------------------
# single-resonance core ------------------------------------------------
def _lorentzian(fd: np.ndarray | float, f0: float, kappa: float) -> np.ndarray:
    x = np.asarray(fd, dtype=float)
    return 4.0 / (kappa ** 2 + 4.0 * (f0 - x) ** 2)


cavity_response = _lorentzian


def yig_response(fd: np.ndarray | float, f_y: float, kappa_y: float) -> np.ndarray:
    return _lorentzian(fd, f_y, kappa_y)


# ---------------------------------------------------------------------
# dataclasses ----------------------------------------------------------
@dataclass
class FitOutcome:
    a: float
    a_err: float
    f0: float
    f0_err: float
    kappa: float
    kappa_err: float
    chi2: float
    redchi: float
    pcov: np.ndarray
    model_curve: np.ndarray
    fit_ok: bool


@dataclass
class CoupledFitOutcome:
    a: float
    a_err: float
    J: float
    J_err: float
    phi: float
    phi_err: float
    chi2: float
    redchi: float
    pcov: np.ndarray
    model_curve: np.ndarray
    fit_ok: bool
    peak_locations: Optional[np.ndarray] = None
    b: Optional[float] = None
    b_err: Optional[float] = None
    redchi_normalized: Optional[float] = None
    inflated_pcov: Optional[np.ndarray] = None


# ---------------------------------------------------------------------
# helpers --------------------------------------------------------------
def _initial_guess(fd: np.ndarray, y_lin: np.ndarray) -> Tuple[float, float, float]:
    idx = int(np.nanargmax(y_lin))
    f0 = float(fd[idx])
    peak = float(y_lin[idx])
    half = peak / 2.0
    left = fd[:idx][y_lin[:idx] <= half]
    right = fd[idx:][y_lin[idx:] <= half]
    if left.size and right.size:
        fwhm = float(right[0] - left[-1])
    else:
        fwhm = 0.01 * float(fd.ptp())
    kappa = max(fwhm, 1e-6)
    # ONLY DO F0 * A FOR WHEN WE HAVE A/X INSTEAD
    a = peak * (kappa ** 2) / 4.0
    return a, f0, kappa


def _loss(y_obs: np.ndarray, y_mod: np.ndarray) -> float:
    return float(np.sum((y_obs - y_mod) ** 2))


def _to_db(x: np.ndarray, eps: float = 1e-18) -> np.ndarray:
    # return 10.0 * np.log10(np.clip(x, eps, None))
    return 10.0 * np.log10(x)


# ---------------------------------------------------------------------
# single-resonance fit -------------------------------------------------
def _fit_single_resonance(
        fd: Sequence[float],
        power_dbm: Sequence[float],
        core: Callable[[np.ndarray, float, float], np.ndarray],
        *,
        fit_in_db: bool = False,
        n_starts: int = 8,
) -> FitOutcome:
    fd = np.asarray(fd, dtype=float)
    dbm = np.asarray(power_dbm, dtype=float)
    y_lin = 10 ** (dbm / 10.0)

    a0, f00, kappa0 = _initial_guess(fd, y_lin)
    guesses = [(a0, f00, kappa0)]
    for _ in range(n_starts - 1):
        guesses.append((
            a0 * _RNG.uniform(0.5, 1.5),
            f00 + _RNG.normal(scale=0.2 * fd.ptp()),
            kappa0 * _RNG.uniform(0.3, 3.0),
        ))

    def model(x, a, f0, kappa):
        return a * core(x, f0, kappa)

    best, best_loss = None, np.inf
    for p0 in guesses:
        try:
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=OptimizeWarning)
                popt, pcov = curve_fit(
                    model,
                    fd,
                    dbm if fit_in_db else y_lin,
                    p0=p0,
                    bounds=((0.0, fd.min(), 0.0), (np.inf, fd.max(), np.inf)),
                    maxfev=20000,
                )
        except Exception:
            continue
        pred = model(fd, *popt)
        loss = _loss(dbm if fit_in_db else y_lin, pred)
        if loss < best_loss and np.all(np.isfinite(np.diag(pcov))):
            best_loss, best = loss, (popt, pcov)

    if best is None:
        raise RuntimeError("single-resonance fit failed")

    popt, pcov = best
    perr = np.sqrt(np.diag(pcov))
    a_fit, f0_fit, kappa_fit = popt
    a_err, f0_err, kappa_err = perr

    model_lin = model(fd, *popt)
    resid = (dbm if fit_in_db else y_lin) - model_lin
    dof = max(1, len(fd) - 3)
    chi2 = float(np.sum(resid ** 2))
    redchi = float(chi2 / dof)

    return FitOutcome(
        a=a_fit,
        a_err=a_err,
        f0=f0_fit,
        f0_err=f0_err,
        kappa=kappa_fit,
        kappa_err=kappa_err,
        chi2=chi2,
        redchi=redchi,
        pcov=pcov,
        model_curve=model_lin if not fit_in_db else _to_db(model_lin),
        fit_ok=np.all(perr > 0) and np.all(np.isfinite(perr)),
    )


def fit_cavity_trace(
        fd: Sequence[float],
        power_dbm: Sequence[float],
        *,
        fit_in_db: bool = True,
        n_starts: int = 8,
) -> FitOutcome:
    return _fit_single_resonance(fd, power_dbm, _lorentzian, fit_in_db=fit_in_db, n_starts=n_starts)


def fit_yig_trace(
        fd: Sequence[float],
        power_dbm: Sequence[float],
        *,
        fit_in_db: bool = True,
        n_starts: int = 8,
) -> FitOutcome:
    return _fit_single_resonance(fd, power_dbm, _lorentzian, fit_in_db=fit_in_db, n_starts=n_starts)


# ---------------------------------------------------------------------
# coupled response -----------------------------------------------------
def _coupled_core(
        fd: np.ndarray | float,
        J: float,
        f_c: float,
        kappa_c: float,
        delta_f: float,
        delta_kappa: float,
        phi: float,
) -> np.ndarray:
    x = np.asarray(fd, dtype=float)
    cos_p = np.cos(phi)
    sin_p = np.sin(phi)
    A = (
            4.0 * cos_p * J ** 2
            - 4.0 * f_c ** 2
            + 8.0 * f_c * x
            + 4.0 * delta_f * f_c
            - 4.0 * x ** 2
            - 4.0 * delta_f * x
            + kappa_c ** 2
            - 2.0 * delta_kappa * kappa_c
    )
    B = (
            -4.0 * sin_p * J ** 2
            + 4.0 * delta_kappa * f_c
            - 4.0 * delta_kappa * x
            + 2.0 * delta_f * kappa_c
            - 4.0 * f_c * kappa_c
            + 4.0 * x * kappa_c
    )

    # This is what NUM is supposed to be, for full transmission
    NUM = 16 * J ** 2

    return NUM / (A ** 2 + B ** 2)


def fit_coupled_trace_fixed_J(
        fd: Sequence[float],
        power_dbm: Sequence[float],
        *,
        J: float,
        f_c: float,
        kappa_c: float,
        delta_f: float,
        delta_kappa: float,
        phi: float = 0.0,
        fit_phi: bool = False,
        vertical_offset: bool = False,
        fit_in_db: bool = False,
        n_starts: int = 8,
        phi_bounds: Tuple[float, float] = (0.0, 2 * np.pi),
        maxfev: int = 30_000,
        a0_guess: Optional[float] = None,
        current: Optional[float] = None,
) -> CoupledFitOutcome:
    fd = np.asarray(fd, dtype=float)
    dbm = np.asarray(power_dbm, dtype=float)
    y_lin = 10.0 ** (dbm / 10.0)

    a0 = float(np.nanmax(y_lin)) if a0_guess is None else a0_guess
    b0 = float(np.percentile(y_lin, 5))
    n_par = 1 + int(fit_phi) + int(vertical_offset)

    guesses = []
    for _ in range(n_starts):
        a_guess = a0 * _RNG.uniform(0.5, 1.5)
        phi_guess = phi + _RNG.normal(scale=0.5)
        b_guess = b0 * _RNG.uniform(0.5, 1.5)
        if fit_phi and vertical_offset:
            guesses.append((a_guess, phi_guess, b_guess))
        elif fit_phi:
            guesses.append((a_guess, phi_guess))
        elif vertical_offset:
            guesses.append((a_guess, b_guess))
        else:
            guesses.append((a_guess,))

    core = _coupled_core
    if fit_phi and vertical_offset:
        def model(x, a, phi_var, b):
            return a * core(x, J, f_c, kappa_c, delta_f, delta_kappa, phi_var) + b

        lower = (0.0, phi_bounds[0], 0.0)
        upper = (np.inf, phi_bounds[1], np.inf)
    elif fit_phi:
        def model(x, a, phi_var):
            return a * core(x, J, f_c, kappa_c, delta_f, delta_kappa, phi_var)

        lower = (0.0, phi_bounds[0])
        upper = (np.inf, phi_bounds[1])
    elif vertical_offset:
        def model(x, a, b):
            return a * core(x, J, f_c, kappa_c, delta_f, delta_kappa, phi) + b

        lower = (0.0, 0.0)
        upper = (np.inf, np.inf)
    else:
        def model(x, a):
            return a * core(x, J, f_c, kappa_c, delta_f, delta_kappa, phi)

        lower = (0.0,)
        upper = (np.inf,)

    best, best_loss = None, np.inf
    for p0 in guesses:
        try:
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=OptimizeWarning)
                popt, pcov = curve_fit(
                    model, fd,
                    dbm if fit_in_db else y_lin,
                    p0=p0, bounds=(lower, upper),
                    maxfev=maxfev,
                )
        except Exception as e:
            print(f"[WARN] Fit failed with exception: {e}")
            continue

        if not np.all(np.isfinite(np.diag(pcov))):
            print("[WARN] Covariance matrix is not finite, skipping")
            continue

        pred = model(fd, *popt)
        loss = _loss(dbm if fit_in_db else y_lin, pred)

        if loss < best_loss:
            best_loss, best = loss, (popt, pcov)

    if best is None:
        raise RuntimeError(f"coupled fit failed (no solution) at sweep value: {current}")

    popt, pcov = best
    perr = np.sqrt(np.diag(pcov))

    idx = 0
    a_fit, a_err = popt[idx], perr[idx];
    idx += 1

    if fit_phi:
        phi_fit, phi_err = popt[idx], perr[idx];
        idx += 1
    else:
        phi_fit, phi_err = phi, 0.0

    if vertical_offset:
        b_fit, b_err = popt[idx], perr[idx]
    else:
        b_fit, b_err = 0.0, 0.0

    dof = max(1, len(fd) - n_par)
    model_lin = model(fd, *popt)

    if not np.all(np.isfinite(model_lin)):
        print("[ERROR] model_lin contains NaNs or infs")
    if not np.all(np.isfinite(y_lin)):
        print("[ERROR] y_lin contains NaNs or infs")

    resid = y_lin - model_lin
    chi2 = float(np.sum(resid ** 2))
    redchi = float(chi2 / dof)

    resid_normalized = resid / y_lin
    chi2_normalized = float(np.sum(resid_normalized ** 2))
    redchi_normalized = float(chi2_normalized / dof)

    return CoupledFitOutcome(
        a=a_fit, a_err=a_err,
        J=J, J_err=0.0,
        phi=phi_fit, phi_err=phi_err,
        b=b_fit, b_err=b_err,
        chi2=chi2, redchi=redchi, pcov=pcov * redchi,
        model_curve=model_lin if not fit_in_db else _to_db(model_lin),
        fit_ok=np.all(perr > 0) and np.all(np.isfinite(perr)),
        redchi_normalized=redchi_normalized,
        inflated_pcov=redchi * pcov
    )


def fit_coupled_trace(
        fd: Sequence[float],
        power_dbm: Sequence[float],
        *,
        f_c: float,
        kappa_c: float,
        delta_f: float,
        delta_kappa: float,
        phi: float = 0.0,
        fit_phi: bool = False,
        vertical_offset: bool = False,
        initial_J: Optional[float] = None,
        fit_in_db: bool = False,
        n_starts: int = 8,
        # ------ hard bounds passed to the optimiser -----------------
        J_bounds: Tuple[float, float] = (0.0, np.inf),
        phi_bounds: Tuple[float, float] = (0.0, 2 * np.pi),
        # ------ *post-selection* window you want to keep ------------
        reported_J_bounds: Tuple[float, float] | None = None,
        # ------------------------------------------------------------
        maxfev: int = 30_000,
        a0_guess: Optional[float] = None,
        current: Optional[float] = None,  # only used in error message
) -> CoupledFitOutcome:
    """
    If *reported_J_bounds* is given, any trial whose best-fit Ĵ lies outside
    that interval is discarded even when it satisfies the wider optimiser
    bounds.  This lets you:

        1.  keep the optimiser box wide (better convergence);
        2.  still reject solutions you consider unphysical.
    """
    # -----------------------------------------------------------------
    fd = np.asarray(fd, dtype=float)
    dbm = np.asarray(power_dbm, dtype=float)
    y_lin = 10.0 ** (dbm / 10.0)

    # ---------- default reported window = optimiser box --------------
    if reported_J_bounds is None:
        reported_J_bounds = J_bounds
    keep_lo, keep_hi = reported_J_bounds

    # ---------- heuristics & seeds -----------------------------------
    if initial_J is None:
        initial_J = max(abs(delta_f) / 2.0, kappa_c / 4.0)

    a0 = float(np.nanmax(y_lin)) if a0_guess is None else a0_guess
    b0 = float(np.percentile(y_lin, 5))
    n_par = 2 + int(fit_phi) + int(vertical_offset)

    guesses = []
    for _ in range(n_starts):
        a_guess = a0 * _RNG.uniform(0.5, 1.5)
        J_guess = initial_J * _RNG.uniform(0.3, 3.0)
        phi_guess = phi + _RNG.normal(scale=0.5)
        b_guess = b0 * _RNG.uniform(0.5, 1.5)
        if fit_phi and vertical_offset:
            guesses.append((a_guess, J_guess, phi_guess, b_guess))
        elif fit_phi:
            guesses.append((a_guess, J_guess, phi_guess))
        elif vertical_offset:
            guesses.append((a_guess, J_guess, b_guess))
        else:
            guesses.append((a_guess, J_guess))

    # ---------- choose model & bounds --------------------------------
    core = _coupled_core
    if fit_phi and vertical_offset:
        def model(x, a, J, phi_var, b):
            return a * core(x, J, f_c, kappa_c,
                            delta_f, delta_kappa, phi_var) + b

        lower = (0.0, J_bounds[0], phi_bounds[0], 0.0)
        upper = (np.inf, J_bounds[1], phi_bounds[1], np.inf)
    elif fit_phi:
        def model(x, a, J, phi_var):
            return a * core(x, J, f_c, kappa_c,
                            delta_f, delta_kappa, phi_var)

        lower = (0.0, J_bounds[0], phi_bounds[0])
        upper = (np.inf, J_bounds[1], phi_bounds[1])
    elif vertical_offset:
        def model(x, a, J, b):
            return a * core(x, J, f_c, kappa_c,
                            delta_f, delta_kappa, phi) + b

        lower = (0.0, J_bounds[0], 0.0)
        upper = (np.inf, J_bounds[1], np.inf)
    else:
        def model(x, a, J):
            return a * core(x, J, f_c, kappa_c,
                            delta_f, delta_kappa, phi)

        lower = (0.0, J_bounds[0])
        upper = (np.inf, J_bounds[1])

    # ---------- multi-start search -----------------------------------
    best, best_loss = None, np.inf
    for p0 in guesses:
        try:
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=OptimizeWarning)
                popt, pcov = curve_fit(
                    model, fd,
                    dbm if fit_in_db else y_lin,
                    p0=p0, bounds=(lower, upper),
                    maxfev=maxfev,
                )
        except Exception:
            continue

        J_trial = popt[1]  # Ĵ is always second parameter
        if not (keep_lo <= J_trial <= keep_hi):
            continue  # throw away – outside reported window

        if not np.all(np.isfinite(np.diag(pcov))):
            continue  # singular covariance

        pred = model(fd, *popt)
        loss = _loss(dbm if fit_in_db else y_lin, pred)
        if loss < best_loss:
            best_loss, best = loss, (popt, pcov)

    if best is None:
        raise RuntimeError(
            f"coupled fit failed (Ĵ outside reported bounds) at sweep value: {current}"
        )

    popt, pcov = best
    perr = np.sqrt(np.diag(pcov))

    # ---------- unpack fit -------------------------------------------
    idx = 0
    a_fit, a_err = popt[idx], perr[idx];
    idx += 1
    J_fit, J_err = popt[idx], perr[idx];
    idx += 1

    if fit_phi:
        phi_fit, phi_err = popt[idx], perr[idx];
        idx += 1
    else:
        phi_fit, phi_err = phi, 0.0

    if vertical_offset:
        b_fit, b_err = popt[idx], perr[idx]
    else:
        b_fit, b_err = 0.0, 0.0

    dof = max(1, len(fd) - n_par)
    model_lin = model(fd, *popt)
    resid = (dbm if fit_in_db else y_lin) - model_lin
    chi2 = float(np.sum(resid ** 2))
    redchi = float(chi2 / dof)

    return CoupledFitOutcome(
        a=a_fit, a_err=a_err,
        J=J_fit, J_err=J_err,
        phi=phi_fit, phi_err=phi_err,
        b=b_fit, b_err=b_err,
        chi2=chi2, redchi=redchi, pcov=pcov,
        model_curve=model_lin if not fit_in_db else _to_db(model_lin),
        fit_ok=np.all(perr > 0) and np.all(np.isfinite(perr)),
    )


def merge_cavity_yig(
        cavity_df: pd.DataFrame,
        yig_df: pd.DataFrame,
        *,
        key: str = "voltage"  # column used to align the rows
) -> pd.DataFrame:
    """
    Merge cavity & YIG fit DataFrames and add Δf, Δκ (+1σ errors).

    Returns
    -------
    pd.DataFrame
        One row per *key* with every original column plus
        Delta_f, Delta_f_err, Delta_kappa, Delta_kappa_err.
    """
    # ------------------------------------------------------------------
    # 1.  Prefix columns so we keep everything from both fits
    # ------------------------------------------------------------------
    cav_pref = {c: f"{c}_c" for c in cavity_df.columns if c != key}
    yig_pref = {c: f"{c}_y" for c in yig_df.columns if c != key}

    cav = cavity_df.rename(columns=cav_pref)
    yig = yig_df.rename(columns=yig_pref)

    # ------------------------------------------------------------------
    # 2.  Inner-join on the sweep variable (voltage, current, …)
    # ------------------------------------------------------------------
    merged = cav.merge(yig, on=key, how="inner", suffixes=("_c", "_y"))

    # ------------------------------------------------------------------
    # 3.  Derived quantities + Gaussian error propagation
    # ------------------------------------------------------------------
    merged["Delta_f"] = merged["f0_c"] - merged["f0_y"]
    merged["Delta_f_err"] = np.sqrt(
        merged["f0_err_c"] ** 2 + merged["f0_err_y"] ** 2
    )

    merged["Delta_kappa"] = merged["kappa_c"] / 2 - merged["kappa_y"] / 2
    merged["Delta_kappa_err"] = 0.5 * np.sqrt(
        merged["kappa_err_c"] ** 2 + merged["kappa_err_y"] ** 2
    )

    return merged


def plot_coupled_response(
        f_c: float,
        kappa_c: float,
        delta_f: float,
        delta_kappa: float,
        J: float,
        phi: float = 0.0,
        a: float = 1.0,
        b: float = 0.0,
        *,
        f_span: float = 5.0,
        n_pts: int = 2001,
):
    """
    Draw |S|^2 vs drive frequency using the same formula as fit_coupled_trace.

    All frequency-like quantities must share units (GHz, Hz, whatever).
    """
    # frequency axis centered on the cavity bare resonance
    fd = np.linspace(f_c - f_span, f_c + f_span, n_pts)

    core = _coupled_core(f_c, kappa_c, delta_f, delta_kappa, phi)
    s2_lin = a * core(fd, J) + b
    s2_db = _to_db(s2_lin)

    plt.figure(figsize=(6, 4))
    plt.plot(fd, s2_db, label="coupled model")
    plt.xlabel("Drive frequency (same units as f_c)")
    plt.ylabel("Power (dB)")
    plt.title("Coupled response preview")
    plt.grid(ls=":", alpha=0.5)

    # annotate the parameters for easy eyeballing
    txt = (
            rf"$f_c={f_c}$, $\kappa_c={kappa_c}$" + "\n"
                                                    rf"$\Delta_f={delta_f}$, $\Delta_\kappa={delta_kappa}$" + "\n"
                                                                                                              rf"$J={J}$, $\phi={phi}$"
    )
    plt.text(0.02, 0.95, txt, transform=plt.gca().transAxes,
             va="top", ha="left", fontsize=9,
             bbox=dict(boxstyle="round", alpha=0.1))
    plt.tight_layout()
    plt.show()


# ----------------------------------------------------------------------
if __name__ == "__main__":
    # --- choose some numbers to convince yourself the equation is right ---
    plot_coupled_response(
        f_c=6006232776.660189,  # GHz
        kappa_c=692341.4724742259,  # GHz
        delta_f=-8698.764387130737,  # GHz  (YIG detuned below)
        delta_kappa=18478.67660910223,  # GHz
        J=1e6,  # GHz  (coupling strength)
        phi=np.deg2rad(10),  # 0 or pi in your experiments
        a=2745535792643.5615,
        b=2.0049192130989785e-11,
        f_span=3e6  # GHz half-width shown
    )
