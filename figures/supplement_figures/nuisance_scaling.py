#!/usr/bin/env python3
"""
verify_nuisance_scaling.py

Verification of TPD Scaling Laws:
1. Sensing Direction -> Square Root Splitting (Crossing the Cusp)
2. Nuisance Direction -> Cube Root Peak Shift (Tangent to the Cusp)
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple

# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------
J_COUPLING = 1.0
F_C = 0.0

SCENARIOS = [
    (0.0, 1.0),    # Scenario 1: phi=0
    (np.pi, 1.66),  # Scenario 2: phi=pi
]

SWEEP_POINTS = 5001
SWEEP_WIDTH = 0.2  # Wide enough to see the curvature

# -----------------------------------------------------------------------------
# Physics Engine
# -----------------------------------------------------------------------------

def solve_roots_sorted(df: float, dk: float, kc: float, phi: float) -> np.ndarray:
    """Returns real parts of the roots, sorted."""
    # Coefficients for characteristic eq: x^3 + p*x + q = 0
    # Derived from det(H - wI) = 0
    # Note: These p/q forms assume the specific 2x2 Hamiltonian structure
    cos_phi = np.cos(phi)
    sin_phi = np.sin(phi)
    k_bar = kc - dk

    p = - (df/2)**2 + (dk/2)**2 - cos_phi + (k_bar/2)**2
    q = (k_bar/4) * (df*dk - 2*sin_phi)

    # Solve depressed cubic
    roots = np.roots([1.0, 0.0, p, q])

    # In the rotating frame, these are detunings from center. 
    # Shift back if F_C is non-zero, though strictly we look at diffs.
    # We only care about Real parts (transmission peaks)
    # Sort them to track branches
    return np.sort(roots.real)

def get_tpd_center(phi: float, kc: float) -> Tuple[float, float, float]:
    """Finds the (dk, df) coordinates and frequency of the TPD."""
    if np.isclose(phi, 0.0) or np.isclose(phi, 2*np.pi):
        # TPD on df=0 axis
        dk_tpd = (kc - np.sqrt(8 - kc**2)) / 2.0
        df_tpd = 0.0
    elif np.isclose(phi, np.pi) or np.isclose(phi, -np.pi):
        # TPD on dk=0 axis
        dk_tpd = 0.0
        df_tpd = -np.sqrt(4 + kc**2)
    else:
        return 0,0,0

    # Frequency at TPD
    r = solve_roots_sorted(df_tpd, dk_tpd, kc, phi)
    nu_tpd = np.mean(r) # Degenerate
    return dk_tpd, df_tpd, nu_tpd

# -----------------------------------------------------------------------------
# Data Generation
# -----------------------------------------------------------------------------

def get_sweep_data(phi, kc, axis, width=0.1):
    dk0, df0, nu0 = get_tpd_center(phi, kc)

    if axis == 'sens':
        # Crossing the cusp (Splitting)
        if np.isclose(phi, 0):
            # phi=0: Sensing is Kappa
            x = np.linspace(dk0 - width/2, dk0 + width/2, SWEEP_POINTS)
            y_splitting = []
            for val in x:
                r = solve_roots_sorted(df0, val, kc, phi)
                y_splitting.append(r[-1] - r[0])
            return x, np.array(y_splitting), dk0, r"$\tilde{\Delta}_\kappa$"
        else:
            # phi=pi: Sensing is Freq
            x = np.linspace(df0 - width/2, df0 + width/2, SWEEP_POINTS)
            y_splitting = []
            for val in x:
                r = solve_roots_sorted(val, dk0, kc, phi)
                y_splitting.append(r[-1] - r[0])
            return x, np.array(y_splitting), df0, r"$\tilde{\Delta}_f$"

    elif axis == 'nuis':
        # Tangent to cusp (Shift)
        if np.isclose(phi, 0):
            # phi=0: Nuisance is Freq
            x = np.linspace(df0 - width/2, df0 + width/2, SWEEP_POINTS)
            y_shift = []
            for val in x:
                r = solve_roots_sorted(val, dk0, kc, phi)
                # Track middle root shift
                y_shift.append(r[1] - nu0)
            return x, np.array(y_shift), df0, r"$\tilde{\Delta}_f$"
        else:
            # phi=pi: Nuisance is Kappa
            x = np.linspace(dk0 - width/2, dk0 + width/2, SWEEP_POINTS)
            y_shift = []
            for val in x:
                r = solve_roots_sorted(df0, val, kc, phi)
                y_shift.append(r[1] - nu0)
            return x, np.array(y_shift), dk0, r"$\tilde{\Delta}_\kappa$"

# -----------------------------------------------------------------------------
# Fitting
# -----------------------------------------------------------------------------

def fit_sqrt(x, y, x0):
    """Fit y = A * sqrt(|x-x0|) on the splitting side."""
    # Only fit where splitting is substantial
    mask = y > 1e-4
    if np.sum(mask) < 10: return x, np.zeros_like(x), 0.0

    x_fit = x[mask]
    y_fit = y[mask]

    # Model: y = A * sqrt(|x-x0|)
    X_feat = np.sqrt(np.abs(x_fit - x0))
    A = np.linalg.lstsq(X_feat[:,None], y_fit, rcond=None)[0][0]

    y_pred = A * np.sqrt(np.abs(x - x0))
    y_pred[~mask] = 0 # Zero out non-splitting side
    return x, y_pred, A

def fit_cbrt(x, y, x0):
    """Fit y = A * sgn(x-x0) * |x-x0|^(1/3)."""
    # Model: y = A * cbrt(x-x0)
    X_feat = np.cbrt(x - x0)
    A = np.linalg.lstsq(X_feat[:,None], y, rcond=None)[0][0]

    y_pred = A * np.cbrt(x - x0)
    return x, y_pred, A

# -----------------------------------------------------------------------------
# Main Plotting
# -----------------------------------------------------------------------------

def run():
    fig, axes = plt.subplots(2, 2, figsize=(12, 10), constrained_layout=True)

    for i, (phi, kc) in enumerate(SCENARIOS):
        # 1. Sensing Direction (Splitting)
        xs, ys, x0s, lbl_s = get_sweep_data(phi, kc, 'sens', SWEEP_WIDTH)
        _, ys_fit, As = fit_sqrt(xs, ys, x0s)

        ax = axes[i, 0]
        ax.plot(xs, ys, 'k-', lw=3, alpha=0.6, label='Simulation')
        ax.plot(xs, ys_fit, 'r--', lw=2, label=rf'Fit $\propto \delta^{{1/2}}$ ($c={As:.2f}$)')

        ax.set_title(rf"SENSING (Splitting): $\phi={phi:.2g}, \tilde\kappa_c={kc}$", fontsize=14)
        ax.set_xlabel(f"Parameter {lbl_s}", fontsize=12)
        ax.set_ylabel(r"Splitting $\tilde{\Delta}_\nu$", fontsize=12)
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.axvline(x0s, color='k', ls=':', alpha=0.5)

        # 2. Nuisance Direction (Shift)
        xn, yn, x0n, lbl_n = get_sweep_data(phi, kc, 'nuis', SWEEP_WIDTH)
        _, yn_fit, An = fit_cbrt(xn, yn, x0n)

        ax = axes[i, 1]
        ax.plot(xn, yn, 'k-', lw=3, alpha=0.6, label='Simulation')
        ax.plot(xn, yn_fit, 'r--', lw=2, label=rf'Fit $\propto \delta^{{1/3}}$ ($c={An:.2f}$)')

        ax.set_title(rf"NUISANCE (Shift): $\phi={phi:.2g}, \tilde\kappa_c={kc}$", fontsize=14)
        ax.set_xlabel(f"Parameter {lbl_n}", fontsize=12)
        ax.set_ylabel(r"Peak Shift $\tilde{\nu} - \tilde{\nu}_{TPD}$", fontsize=12)
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.axvline(x0n, color='k', ls=':', alpha=0.5)

    plt.show()

if __name__ == "__main__":
    run()