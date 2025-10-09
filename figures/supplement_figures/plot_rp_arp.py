#!/usr/bin/env python3
"""
plot_rp_arp.py
Eight-panel plot (4 columns × 2 rows) of eigenvalues (dashed) and peak locations (solid)
for RP (φ = 0) and ARP (φ = π).  EP (red), TPD (cyan), and EP–TPD pair (magenta)
vertical lines are shown.  A legend is placed on the top-left panel, and bold
panel letters “a … h” are drawn directly on the canvas (titles now contain no
letters).  Complex values are allowed; we plot the real part.
ASCII-only code, no non-ASCII characters.
"""
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.lines import Line2D
import matplotlib as mpl

# ────────────────────────────────────────────────────────────────────────────
#  Physics constants
# ────────────────────────────────────────────────────────────────────────────
J = 1.0                               # coupling, nondimensionalised to unity
KAPPA_C_LIST = (0.0, 1.0, 2.0, 2.8)   # four κ̃c columns

# ────────────────────────────────────────────────────────────────────────────
#  Helper functions
# ────────────────────────────────────────────────────────────────────────────
def rp_peaks(delta_kappa, kappa_c):
    rad  = 4.0 * J**2 - delta_kappa**2 - (delta_kappa - kappa_c)**2
    root = 0.5 * np.sqrt(rad.astype(np.complex128))
    return -root, root


def arp_peaks(delta_f, kappa_c):
    rad  = delta_f**2 - 4.0 * J**2 - kappa_c**2
    root = 0.5 * np.sqrt(rad.astype(np.complex128))
    return -root + delta_f / 2.0, root + delta_f / 2.0


def eigen_imag_parts(delta_f, delta_kappa, phi):
    inside = (-delta_f**2
              + 2j * delta_f * delta_kappa
              + delta_kappa**2
              - 4.0 * np.exp(1j * phi) * J**2)
    root = np.sqrt(inside)
    imag_term = 0.5 * np.imag(root)
    return delta_f / 2.0 + imag_term, delta_f / 2.0 - imag_term


def tpd_positions(phi, J=1.0, kappa_c=1.0):
    if phi == 0.0:
        val = np.sqrt(8 * J**2 - kappa_c**2)
        return [((kappa_c + val) / 2.0, 0.0),
                ((kappa_c - val) / 2.0, 0.0)]
    if phi == np.pi:
        val = np.sqrt(4 * J**2 + kappa_c**2)
        return [(0.0,  val), (0.0, -val)]
    return []

# ────────────────────────────────────────────────────────────────────────────
#  Main plotting routine
# ────────────────────────────────────────────────────────────────────────────
def build_figure():
    dk_vec = np.linspace(-2.1, 2.1, 2001)    # Δκ̃ sweep (RP)
    df_vec = np.linspace(-5.0, 5.0, 2001)    # Δf̃ sweep (ARP)

    fig = plt.figure(figsize=(20, 9))
    gs  = GridSpec(2, 4, wspace=0.20, hspace=0.375,
                   left=0.06, right=0.98, top=0.92, bottom=0.10)

    ep_pos = (-2.0 * J, 2.0 * J)             # EP x-positions
    panel_letters = iter("abcdefgh")          # for bold labels later
    axes_for_labels = []                      # keep refs for placing letters

    for col, kappa_c in enumerate(KAPPA_C_LIST):
        # ── RP row (φ = 0) ────────────────────────────────────────────────
        ax_rp = fig.add_subplot(gs[0, col])
        axes_for_labels.append(ax_rp)

        nu_p, nu_m        = rp_peaks(dk_vec, kappa_c)
        eig_p, eig_m      = eigen_imag_parts(0.0, dk_vec, phi=0.0)

        ax_rp.plot(dk_vec, np.real(eig_p), 'k--', lw=3.0)
        ax_rp.plot(dk_vec, np.real(eig_m), 'k--', lw=3.0)
        ax_rp.plot(dk_vec, np.real(nu_p),  'k-',  lw=3.0)
        ax_rp.plot(dk_vec, np.real(nu_m),  'k-',  lw=3.0)

        for x_ep in ep_pos:
            if not (kappa_c == 2.0 and x_ep == 2.0):   # skip  overlap case
                ax_rp.axvline(x_ep, color='r', lw=2.5)

        for dk_tpd, _ in tpd_positions(0.0, J, kappa_c):
            col_code = 'm' if dk_tpd in ep_pos else 'c'
            ax_rp.axvline(dk_tpd, color=col_code, lw=2.5)

        ax_rp.set_title(rf"$\phi = 0,\tilde \kappa_c = {kappa_c:.1f}$",
                        fontsize=20)
        ax_rp.set_xlabel(r"$\tilde \Delta_{\kappa}$", fontsize=23)
        if col == 0:
            ax_rp.set_ylabel("Frequency / J", fontsize=20)
        ax_rp.set_ylim(-1, 1)
        ax_rp.tick_params(labelsize=18)

        # Legend only on first panel
        if col == 0:
            proxies = [
                Line2D([0], [0], color='r', lw=2.5),
                Line2D([0], [0], color='c', lw=2.5),
                Line2D([0], [0], color='m', lw=2.5),
                Line2D([0], [0], color='k', linestyle='--' ,lw=2.5),
                Line2D([0], [0], color='k', linestyle='-', lw=2.5)
            ]
            ax_rp.legend(proxies, ['EP', 'TPD', 'EP-TPD', r'|Im($\tilde \lambda_\pm$)|', r'$\tilde \nu_\pm$'],
                         loc='center', frameon=False, fontsize=20)

        # ── ARP row (φ = π) ───────────────────────────────────────────────
        ax_arp = fig.add_subplot(gs[1, col])
        axes_for_labels.append(ax_arp)

        nu_p, nu_m        = arp_peaks(df_vec, kappa_c)
        eig_p, eig_m      = eigen_imag_parts(df_vec, 0.0, phi=np.pi)

        ax_arp.plot(df_vec, np.real(eig_p), 'k--', lw=3.0)
        ax_arp.plot(df_vec, np.real(eig_m), 'k--', lw=3.0)
        ax_arp.plot(df_vec, np.real(nu_p),  'k-',  lw=3.0)
        ax_arp.plot(df_vec, np.real(nu_m),  'k-',  lw=3.0)

        if kappa_c != 0.0:
            for x_ep in ep_pos:
                ax_arp.axvline(x_ep, color='r', lw=2.5)

        for _, df_tpd in tpd_positions(np.pi, J, kappa_c):
            col_code = 'm' if df_tpd in ep_pos else 'c'
            ax_arp.axvline(df_tpd, color=col_code, lw=2.5)



        ax_arp.set_title(rf"$\phi = \pi, \tilde \kappa_c= {kappa_c:.1f}$",
                         fontsize=20)
        ax_arp.set_xlabel(r"$\tilde \Delta_f$", fontsize=23)
        if col == 0:
            ax_arp.set_ylabel("Frequency / J", fontsize=20)
        ax_arp.set_ylim(-4, 4)
        ax_arp.tick_params(labelsize=18)

    axes_for_labels2 = [
        fig.axes[0], fig.axes[2], fig.axes[4], fig.axes[6],  # top row (a b c d)
        fig.axes[1], fig.axes[3], fig.axes[5], fig.axes[7],  # bottom row (e f g h)
    ]

    # ── Place bold panel letters a … h inside each axis ─────────────────────
    for ax, letter in zip(axes_for_labels2, "abcdefgh"):
        pos = ax.get_position()
        fig.text(pos.x0 + 0.01, pos.y1 + 0.045,
                 letter, fontsize=26, fontweight="bold",
                 ha="left", va="top")

    filename = "../.figures/SUPP_rp_arp_figure_with_lines.png"
    output_path = Path(filename)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=400, facecolor="white")
    plt.close(fig)

    fig.savefig(output_path, dpi=400)
    plt.close(fig)

# ────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    build_figure()
