#!/usr/bin/env python3
"""
WARNING - THIS PLOT TAKES A LONG TIME TO GENERATE!

supp_transmission_grid_fmt.py
Six-panel (3 × 2) figure comparing three transmission responses
with SciPy peak detection.

Key cosmetics (already in your version):
  • Magenta EP–TPD lines unchanged.
  • Stand-alone TPD lines in columns 2–3 are dashed-cyan (“False TPD”).
  • Main legend (no False TPD) at upper-left of first panel.
  • Small “False TPD” legend at upper-left of second panel.
  • Tick labels = 20 pt.
  • Panel letters a b c (top) and d e f (bottom) left-to-right.
"""
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.lines import Line2D
from scipy.signal import find_peaks

# ────────── grid resolution ──────────
X_RES = 3501
Y_RES = 3501
# ────────── physical constants ───────
J               = 1.0
kappa_c_over_J  = 2.0
kappa_c         = kappa_c_over_J * J
PROMINENCE      = 1e-10
OMEGA_RP  = np.linspace(-1.2,  1.2, X_RES)
OMEGA_ARP = np.linspace(-4.8,  4.8, X_RES)
TICK_SIZE       = 18
# ─────────────────────────────────────

plt.rcParams.update(plt.rcParamsDefault)
plt.rcParams["figure.facecolor"] = "white"
plt.rcParams["axes.facecolor"]   = "white"

# ────────── helpers ──────────
def A_matrix(df, dk, phi):
    return np.array(
        [[-kappa_c/2,            -1j*J],
         [-1j*J*np.exp(1j*phi),  dk - kappa_c/2 + 1j*df]],
        dtype=complex)

def transfer_mag_sq(omega, df, dk, phi, B, C):
    out = np.empty_like(omega, dtype=float)
    A   = A_matrix(df, dk, phi)
    for i, w in enumerate(omega):
        H = C @ np.linalg.inv(1j*w*np.eye(2) - A) @ B
        out[i] = np.abs(H[0, 0])**2
    return out

def eig_im(df, dk, phi):
    inside = (-df**2 + 2j*df*dk + dk**2 - 4*np.exp(1j*phi)*J**2)
    root   = np.sqrt(inside)
    part   = 0.5*np.imag(root)
    return df/2 + part, df/2 - part

def tpd_pos(phi, Jv=1.0, kc=1.0):
    if abs(phi) < 1e-12:
        v = np.sqrt(8*Jv**2 - kc**2)
        return [((kc+v)/2, 0.0), ((kc-v)/2, 0.0)]
    if abs(phi - np.pi) < 1e-12:
        v = np.sqrt(4*Jv**2 + kc**2)
        return [(0.0,  v), (0.0, -v)]
    return []

def strongest_two(omega, mag):
    idx, _ = find_peaks(mag, prominence=PROMINENCE)
    if idx.size == 0:
        return np.nan, np.nan
    idx = idx[np.argsort(mag[idx])[::-1]]
    vals = omega[idx[:2]]
    if vals.size == 1:
        return np.nan, vals[0]
    return np.sort(vals)

# ───── drive/read configurations ─────
COLS = (
    dict(lbl="Drive Cavity / Read YIG",
         B=np.array([[1],[0]], dtype=complex),
         C=np.array([[0,1]], dtype=complex)),
    dict(lbl="Drive Cavity / Read Cavity",
         B=np.array([[1],[0]], dtype=complex),
         C=np.array([[1,0]], dtype=complex)),
    dict(lbl="Drive YIG / Read YIG",
         B=np.array([[0],[1]], dtype=complex),
         C=np.array([[0,1]], dtype=complex)),
)

# ────────── main builder ──────────
def build():
    dk_vec = np.linspace(-2.1, 2.1, Y_RES)
    df_vec = np.linspace(-5.0, 5.0, Y_RES)
    ep_x   = (-2*J, 2*J)

    fig = plt.figure(figsize=(15, 9))
    gs  = GridSpec(2, 3, wspace=.20, hspace=.35,
                   left=.06, right=.98, top=.92, bottom=.10)

    axes_top    = []
    axes_bottom = []

    for col, cfg in enumerate(COLS):
        B, C = cfg["B"], cfg["C"]

        # ── φ = 0 row ─────────────────────────────────────────
        pk_lo, pk_hi = [], []
        for dk in dk_vec:
            mag = transfer_mag_sq(OMEGA_RP, 0.0, dk, 0.0, B, C)
            p1, p2 = strongest_two(OMEGA_RP, mag)
            pk_lo.append(p1); pk_hi.append(p2)
        pk_lo, pk_hi = map(np.array, (pk_lo, pk_hi))

        axT = fig.add_subplot(gs[0, col]); axes_top.append(axT)

        eig_p, eig_m = eig_im(0.0, dk_vec, 0.0)
        axT.plot(dk_vec, np.real(eig_p), 'k--', lw=2.5)
        axT.plot(dk_vec, np.real(eig_m), 'k--', lw=2.5)
        axT.plot(dk_vec, pk_lo, 'k-', lw=2.5)
        axT.plot(dk_vec, pk_hi, 'k-', lw=2.5)

        for x in ep_x:
            axT.axvline(x, color='r', lw=2)
        for x_tpd, _ in tpd_pos(0.0, J, kappa_c):
            is_ep = x_tpd in ep_x
            color = 'm' if is_ep else 'c'
            style = '--' if (not is_ep and col) else '-'
            axT.axvline(x_tpd, color=color, lw=2, linestyle=style)

        axT.set_title(rf"$\phi=0$, {cfg['lbl']}", fontsize=18)
        axT.set_xlabel(r"$\tilde \Delta_{\kappa}$", fontsize=22)
        if col == 0:
            axT.set_ylabel("Frequency / J", fontsize=20)
        axT.set_ylim(-1.2, 1.2)
        axT.tick_params(labelsize=TICK_SIZE)

        # ── φ = π row ─────────────────────────────────────────
        pk_lo, pk_hi = [], []
        for df in df_vec:
            mag = transfer_mag_sq(OMEGA_ARP, df, 0.0, np.pi, B, C)
            p1, p2 = strongest_two(OMEGA_ARP, mag)
            pk_lo.append(p1); pk_hi.append(p2)
        pk_lo, pk_hi = map(np.array, (pk_lo, pk_hi))

        axB = fig.add_subplot(gs[1, col]); axes_bottom.append(axB)

        eig_p, eig_m = eig_im(df_vec, 0.0, np.pi)
        axB.plot(df_vec, np.real(eig_p), 'k--', lw=2.5)
        axB.plot(df_vec, np.real(eig_m), 'k--', lw=2.5)
        axB.plot(df_vec, pk_lo, 'k-', lw=2.5)
        axB.plot(df_vec, pk_hi, 'k-', lw=2.5)

        for x in ep_x:
            axB.axvline(x, color='r', lw=2)
        for _, y_tpd in tpd_pos(np.pi, J, kappa_c):
            is_ep = y_tpd in ep_x
            color = 'm' if is_ep else 'c'
            style = '--' if (not is_ep and col) else '-'
            axB.axvline(y_tpd, color=color, lw=2, linestyle=style)

        axB.set_title(rf"$\phi=\pi$, {cfg['lbl']}", fontsize=18)
        axB.set_xlabel(r"$\tilde \Delta_{f}$", fontsize=22)
        if col == 0:
            axB.set_ylabel("Frequency / J", fontsize=20)
        axB.set_ylim(-4.8, 4.8)
        axB.tick_params(labelsize=TICK_SIZE)

        # ── legends ───────────────────────────────────
        if col == 0:
            main_handles = [
                Line2D([0],[0], color='r', lw=2),
                Line2D([0],[0], color='c', lw=2),
                Line2D([0],[0], color='m', lw=2),
                Line2D([0],[0], color='k', lw=2, linestyle='--'),
                Line2D([0],[0], color='k', lw=2, linestyle='-'),
            ]
            main_labels  = ['EP', 'TPD', 'EP-TPD',
                            r'$\lambda_\pm$', r'$\nu_\pm$']
            axT.legend(main_handles, main_labels,
                       fontsize=18, framealpha=1.0,
                       loc='upper left', bbox_to_anchor=(-0.02, 1.02),
                       borderpad=.15)
        elif col == 1:   # False-TPD key only in second column
            false_handle = Line2D([0],[0], color='c', lw=2, linestyle='--')
            axT.legend([false_handle], ['False TPD'],
                       fontsize=18, framealpha=1.0,
                       loc='upper left', bbox_to_anchor=(-0.02, 1.02),
                       borderpad=.15)

    # ── panel letters a-f (left→right) ───────────────────────
    for idx, ax in enumerate(axes_top + axes_bottom):  # row-major order
        letter = chr(ord('a') + idx)
        pos = ax.get_position()
        fig.text(pos.x0 + 0.01, pos.y1 + 0.045,
                 letter, size=26, weight='bold')

    filename = "../.figures/SUPP_transmission_grid_fmt.png"
    output_path = Path(filename)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=400, facecolor="white")
    plt.close(fig)

    fig.savefig(filename,
                dpi=400, bbox_inches="tight")
    plt.close(fig)

# ─────────────────────────────────────────
if __name__ == "__main__":
    build()
