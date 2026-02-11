#!/usr/bin/env python3
"""
supp_combined_rp_arp_and_transmission.py

14-panel combined figure with per-row shared y-axis:
  Row 0: analytic RP  (4 cols)
  Row 1: analytic ARP (4 cols)
  Row 2: transmission RP  (3 cols)
  Row 3: transmission ARP (3 cols)

Row labels (a)-(d) outside, roman numerals inside each axis.
Legend figure-level above plots.

ASCII-only code.
"""

from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from scipy.signal import find_peaks


CFG = {
    "output_path": "../.figures/SUPP_combined_rp_arp_transmission.png",
    "save_dpi": 400,

    "figure": {
        "figsize": (14, 12),
        "left": 0.08,
        "right": 0.985,
        "bottom": 0.06,
        "top": 0.88,
        "wspace": 0.25,
        "hspace": 0.6,
    },

    "style": {
        "tick_size": 16,
        "label_size": 20,
        "title_size": 18,
        "roman_size": 20,
        "row_label_size": 24,
        "curve_lw": 2.75,
        "vline_lw": 2.75,

        "legend_size": 23,
        "legend_ncol": 6,
        "legend_columnspacing": 1,
        "legend_handletextpad": 0.8,
        "legend_handlelength": 2.6,
        "legend_bbox_to_anchor": (0.5, 0.985),
    },

    "J": 1.0,

    "top": {
        "kappa_c_list": (0.0, 1.0, 2.0, 2.8),
        "dk_lim": (-2.4, 2.4),
        "df_lim": (-5.4, 5.4),
        "dk_points": 2001 * 5,
        "df_points": 2001 * 5,
        "ylim_rp": (-1.2, 1.2),
        "ylim_arp": (-4.8, 4.8),
    },

    "bottom": {
        "kappa_c_over_J": 2.0,
        "prominence": 1e-10,
        "omega_res": 1201 * 5,
        "sweep_res": 1201 * 5,

        "omega_rp_lim": (-1.2, 1.2),
        "omega_arp_lim": (-4.8, 4.8),
        "dk_lim": (-2.4, 2.4),
        "df_lim": (-5.0, 5.0),

        "ylim_rp": (-1.2, 1.2),
        "ylim_arp": (-4.8, 4.8),
    },

    "fast_preview": False,
    "fast_preview_overrides": {
        "top.dk_points": 801,
        "top.df_points": 801,
        "bottom.omega_res": 601,
        "bottom.sweep_res": 401,
    },
}


def _apply_fast_preview_overrides(cfg: dict) -> None:
    if not cfg.get("fast_preview", False):
        return
    ov = cfg.get("fast_preview_overrides", {})
    for k, v in ov.items():
        parts = k.split(".")
        if len(parts) != 2:
            continue
        block, key = parts
        if block in cfg and isinstance(cfg[block], dict):
            cfg[block][key] = v


def _apply_rcparams(cfg: dict) -> None:
    st = cfg["style"]
    plt.rcParams.update(plt.rcParamsDefault)
    plt.rcParams["figure.facecolor"] = "white"
    plt.rcParams["axes.facecolor"] = "white"
    plt.rcParams["axes.titlesize"] = st["title_size"]
    plt.rcParams["axes.labelsize"] = st["label_size"]
    plt.rcParams["xtick.labelsize"] = st["tick_size"]
    plt.rcParams["ytick.labelsize"] = st["tick_size"]
    plt.rcParams["legend.fontsize"] = st["legend_size"]


def rp_peaks(delta_kappa: np.ndarray, kappa_c: float, J: float) -> tuple[np.ndarray, np.ndarray]:
    rad = 4.0 * J**2 - delta_kappa**2 - (delta_kappa - kappa_c)**2
    root = 0.5 * np.sqrt(rad.astype(np.complex128))
    return -root, root


def arp_peaks(delta_f: np.ndarray, kappa_c: float, J: float) -> tuple[np.ndarray, np.ndarray]:
    rad = delta_f**2 - 4.0 * J**2 - kappa_c**2
    root = 0.5 * np.sqrt(rad.astype(np.complex128))
    return -root + delta_f / 2.0, root + delta_f / 2.0


def eig_im_parts(delta_f: np.ndarray | float, delta_kappa: np.ndarray | float, phi: float, J: float) -> tuple[np.ndarray, np.ndarray]:
    inside = (
            -np.asarray(delta_f)**2
            + 2j * np.asarray(delta_f) * np.asarray(delta_kappa)
            + np.asarray(delta_kappa)**2
            - 4.0 * np.exp(1j * phi) * J**2
    )
    root = np.sqrt(inside)
    part = 0.5 * np.imag(root)
    df = np.asarray(delta_f)
    return df / 2.0 + part, df / 2.0 - part


def tpd_positions(phi: float, J: float, kappa_c: float) -> list[tuple[float, float]]:
    if abs(phi) < 1e-12:
        val = np.sqrt(8.0 * J**2 - kappa_c**2)
        return [((kappa_c + val) / 2.0, 0.0),
                ((kappa_c - val) / 2.0, 0.0)]
    if abs(phi - np.pi) < 1e-12:
        val = np.sqrt(4.0 * J**2 + kappa_c**2)
        return [(0.0,  val), (0.0, -val)]
    return []


def transfer_mag_sq_vec(
        omega: np.ndarray,
        df: float,
        dk: float,
        phi: float,
        B: np.ndarray,
        C: np.ndarray,
        J: float,
        kappa_c: float,
) -> np.ndarray:
    w = omega.astype(np.complex128)

    a = 1j * w + (kappa_c / 2.0)
    b = 1j * J
    c = 1j * J * np.exp(1j * phi)
    d = 1j * w - dk + (kappa_c / 2.0) - 1j * df

    det = a * d - b * c

    inv11 = d / det
    inv12 = (-b) / det
    inv21 = (-c) / det
    inv22 = a / det

    vb0 = inv11 * B[0, 0] + inv12 * B[1, 0]
    vb1 = inv21 * B[0, 0] + inv22 * B[1, 0]

    H = C[0, 0] * vb0 + C[0, 1] * vb1
    return (np.abs(H)**2).astype(float)


def strongest_two(omega: np.ndarray, mag: np.ndarray, prominence: float) -> tuple[float, float]:
    idx, _ = find_peaks(mag, prominence=prominence)
    if idx.size == 0:
        return np.nan, np.nan
    idx = idx[np.argsort(mag[idx])[::-1]]
    vals = omega[idx[:2]]
    if vals.size == 1:
        return np.nan, float(vals[0])
    vals = np.sort(vals)
    return float(vals[0]), float(vals[1])


def _legend_handles() -> tuple[list[Line2D], list[str]]:
    handles = [
        Line2D([0], [0], color="r", lw=2.25),
        Line2D([0], [0], color="c", lw=2.25),
        Line2D([0], [0], color="m", lw=2.25),
        Line2D([0], [0], color="c", lw=2.25, linestyle="--"),
        Line2D([0], [0], color="k", lw=2.5, linestyle="--"),
        Line2D([0], [0], color="k", lw=2.5, linestyle="-"),
    ]
    labels = ["EP", "TPD", "EP-TPD", "False TPD", r"$|\text{Im}(\lambda_\pm)|$", r"$\nu_\pm$"]
    return handles, labels


def _add_roman(ax: plt.Axes, roman: str, fontsize: int) -> None:
    ax.text(
        0.02, 0.95, roman,
        transform=ax.transAxes,
        ha="left", va="top",
        fontsize=fontsize,
        fontweight="bold",
    )


def _add_row_label(fig: plt.Figure, ax_ref: plt.Axes, label: str, fontsize: int, dx: float = 0.060, dy: float = 0.040) -> None:
    pos = ax_ref.get_position()
    x = pos.x0 - dx
    y = pos.y1 + dy
    fig.text(x, y, label, fontsize=fontsize, fontweight="bold", ha="left", va="top")


def _hide_repeated_y(ax: plt.Axes) -> None:
    ax.set_ylabel("")
    ax.tick_params(axis="y", which="both", left=False, labelleft=False)


def build_combined() -> None:
    _apply_fast_preview_overrides(CFG)
    _apply_rcparams(CFG)

    J = float(CFG["J"])
    st = CFG["style"]
    top_cfg = CFG["top"]
    bot_cfg = CFG["bottom"]
    fig_cfg = CFG["figure"]

    fig = plt.figure(figsize=fig_cfg["figsize"])
    gs = fig.add_gridspec(
        nrows=4,
        ncols=12,
        left=fig_cfg["left"],
        right=fig_cfg["right"],
        bottom=fig_cfg["bottom"],
        top=fig_cfg["top"],
        wspace=fig_cfg["wspace"],
        hspace=fig_cfg["hspace"],
    )

    # Figure-level legend
    handles, labels = _legend_handles()
    fig.legend(
        handles=handles,
        labels=labels,
        loc="upper center",
        ncol=int(st["legend_ncol"]),
        frameon=False,
        bbox_to_anchor=st["legend_bbox_to_anchor"],
        columnspacing=float(st["legend_columnspacing"]),
        handletextpad=float(st["legend_handletextpad"]),
        handlelength=float(st["legend_handlelength"]),
        fontsize=float(st["legend_size"]),
    )

    romans_top = ["i", "ii", "iii", "iv"]
    romans_bot = ["i", "ii", "iii"]
    row_labels = ["(a)", "(b)", "(c)", "(d)"]

    ep_pos = (-2.0 * J, 2.0 * J)

    # Store leftmost axis per row for row label placement
    leftmost_axes = [None, None, None, None]

    # =========================
    # Rows 0-1: Top block 2x4
    # =========================
    dk_vec_top = np.linspace(top_cfg["dk_lim"][0], top_cfg["dk_lim"][1], int(top_cfg["dk_points"]))
    df_vec_top = np.linspace(top_cfg["df_lim"][0], top_cfg["df_lim"][1], int(top_cfg["df_points"]))

    ax_row0 = [None] * 4
    ax_row1 = [None] * 4

    for col, kappa_c in enumerate(top_cfg["kappa_c_list"]):
        col_start = col * 3
        col_stop = (col + 1) * 3

        # Row 0, shared y
        if col == 0:
            ax0 = fig.add_subplot(gs[0, col_start:col_stop])
            leftmost_axes[0] = ax0
        else:
            ax0 = fig.add_subplot(gs[0, col_start:col_stop], sharey=ax_row0[0])
        ax_row0[col] = ax0

        nu_p, nu_m = rp_peaks(dk_vec_top, float(kappa_c), J)
        eig_p, eig_m = eig_im_parts(0.0, dk_vec_top, phi=0.0, J=J)

        ax0.plot(dk_vec_top, np.real(eig_p), "k--", lw=st["curve_lw"])
        ax0.plot(dk_vec_top, np.real(eig_m), "k--", lw=st["curve_lw"])
        ax0.plot(dk_vec_top, np.real(nu_p), "k-", lw=st["curve_lw"])
        ax0.plot(dk_vec_top, np.real(nu_m), "k-", lw=st["curve_lw"])

        for x_ep in ep_pos:
            if not (float(kappa_c) == 2.0 and x_ep == 2.0):
                ax0.axvline(x_ep, color="r", lw=st["vline_lw"])

        for dk_tpd, _ in tpd_positions(0.0, J, float(kappa_c)):
            col_code = "m" if dk_tpd in ep_pos else "c"
            ax0.axvline(dk_tpd, color=col_code, lw=st["vline_lw"])

        ax0.set_title(rf"$\phi = 0,\ \tilde \kappa_c = {float(kappa_c):.1f}$", fontsize=st["title_size"])
        ax0.set_xlabel(r"$\tilde \Delta_{\kappa}$", fontsize=st["label_size"])
        ax0.set_ylim(top_cfg["ylim_rp"])
        ax0.tick_params(labelsize=st["tick_size"])
        _add_roman(ax0, romans_top[col], st["roman_size"])
        if col == 0:
            ax0.set_ylabel("Frequency / J", fontsize=st["label_size"])
        else:
            _hide_repeated_y(ax0)

        # Row 1, shared y
        if col == 0:
            ax1 = fig.add_subplot(gs[1, col_start:col_stop])
            leftmost_axes[1] = ax1
        else:
            ax1 = fig.add_subplot(gs[1, col_start:col_stop], sharey=ax_row1[0])
        ax_row1[col] = ax1

        nu_p, nu_m = arp_peaks(df_vec_top, float(kappa_c), J)
        eig_p, eig_m = eig_im_parts(df_vec_top, 0.0, phi=np.pi, J=J)

        ax1.plot(df_vec_top, np.real(eig_p), "k--", lw=st["curve_lw"])
        ax1.plot(df_vec_top, np.real(eig_m), "k--", lw=st["curve_lw"])
        ax1.plot(df_vec_top, np.real(nu_p), "k-", lw=st["curve_lw"])
        ax1.plot(df_vec_top, np.real(nu_m), "k-", lw=st["curve_lw"])

        if float(kappa_c) != 0.0:
            for x_ep in ep_pos:
                ax1.axvline(x_ep, color="r", lw=st["vline_lw"])

        for _, df_tpd in tpd_positions(np.pi, J, float(kappa_c)):
            col_code = "m" if df_tpd in ep_pos else "c"
            ax1.axvline(df_tpd, color=col_code, lw=st["vline_lw"])

        ax1.set_title(rf"$\phi = \pi,\ \tilde \kappa_c = {float(kappa_c):.1f}$", fontsize=st["title_size"])
        ax1.set_xlabel(r"$\tilde \Delta_f$", fontsize=st["label_size"])
        ax1.set_ylim(top_cfg["ylim_arp"])
        ax1.tick_params(labelsize=st["tick_size"])
        _add_roman(ax1, romans_top[col], st["roman_size"])
        if col == 0:
            ax1.set_ylabel("Frequency / J", fontsize=st["label_size"])
        else:
            _hide_repeated_y(ax1)

    # =========================
    # Rows 2-3: Bottom block 2x3
    # =========================
    kappa_c = float(bot_cfg["kappa_c_over_J"]) * J
    prominence = float(bot_cfg["prominence"])

    omega_rp = np.linspace(bot_cfg["omega_rp_lim"][0], bot_cfg["omega_rp_lim"][1], int(bot_cfg["omega_res"]))
    omega_arp = np.linspace(bot_cfg["omega_arp_lim"][0], bot_cfg["omega_arp_lim"][1], int(bot_cfg["omega_res"]))

    dk_vec = np.linspace(bot_cfg["dk_lim"][0], bot_cfg["dk_lim"][1], int(bot_cfg["sweep_res"]))
    df_vec = np.linspace(bot_cfg["df_lim"][0], bot_cfg["df_lim"][1], int(bot_cfg["sweep_res"]))

    cols = (
        dict(
            lbl="Drive Cavity / Read YIG",
            B=np.array([[1], [0]], dtype=np.complex128),
            C=np.array([[0, 1]], dtype=np.complex128),
        ),
        dict(
            lbl="Drive Cavity / Read Cavity",
            B=np.array([[1], [0]], dtype=np.complex128),
            C=np.array([[1, 0]], dtype=np.complex128),
        ),
        dict(
            lbl="Drive YIG / Read YIG",
            B=np.array([[0], [1]], dtype=np.complex128),
            C=np.array([[0, 1]], dtype=np.complex128),
        ),
    )

    ax_row2 = [None] * 3
    ax_row3 = [None] * 3

    for col, cfg in enumerate(cols):
        B, C = cfg["B"], cfg["C"]
        col_start = col * 4
        col_stop = (col + 1) * 4

        # Row 2, shared y
        if col == 0:
            ax2 = fig.add_subplot(gs[2, col_start:col_stop])
            leftmost_axes[2] = ax2
        else:
            ax2 = fig.add_subplot(gs[2, col_start:col_stop], sharey=ax_row2[0])
        ax_row2[col] = ax2

        pk_lo = np.empty_like(dk_vec, dtype=float)
        pk_hi = np.empty_like(dk_vec, dtype=float)
        for i, dk in enumerate(dk_vec):
            mag = transfer_mag_sq_vec(omega_rp, 0.0, float(dk), 0.0, B, C, J, kappa_c)
            p1, p2 = strongest_two(omega_rp, mag, prominence)
            pk_lo[i] = p1
            pk_hi[i] = p2

        eig_p, eig_m = eig_im_parts(0.0, dk_vec, 0.0, J)
        ax2.plot(dk_vec, np.real(eig_p), "k--", lw=st["curve_lw"])
        ax2.plot(dk_vec, np.real(eig_m), "k--", lw=st["curve_lw"])
        ax2.plot(dk_vec, pk_lo, "k-", lw=st["curve_lw"])
        ax2.plot(dk_vec, pk_hi, "k-", lw=st["curve_lw"])

        for x in ep_pos:
            ax2.axvline(x, color="r", lw=st["vline_lw"])

        for x_tpd, _ in tpd_positions(0.0, J, kappa_c):
            is_ep = x_tpd in ep_pos
            color = "m" if is_ep else "c"
            style = "--" if ((not is_ep) and (col > 0)) else "-"
            ax2.axvline(x_tpd, color=color, lw=st["vline_lw"], linestyle=style)

        ax2.set_title(rf"$\phi = 0$, {cfg['lbl']}", fontsize=st["title_size"])
        ax2.set_xlabel(r"$\tilde \Delta_{\kappa}$", fontsize=st["label_size"])
        ax2.set_ylim(bot_cfg["ylim_rp"])
        ax2.tick_params(labelsize=st["tick_size"])
        _add_roman(ax2, romans_bot[col], st["roman_size"])
        if col == 0:
            ax2.set_ylabel("Frequency / J", fontsize=st["label_size"])
        else:
            _hide_repeated_y(ax2)

        # Row 3, shared y
        if col == 0:
            ax3 = fig.add_subplot(gs[3, col_start:col_stop])
            leftmost_axes[3] = ax3
        else:
            ax3 = fig.add_subplot(gs[3, col_start:col_stop], sharey=ax_row3[0])
        ax_row3[col] = ax3

        pk_lo = np.empty_like(df_vec, dtype=float)
        pk_hi = np.empty_like(df_vec, dtype=float)
        for i, df in enumerate(df_vec):
            mag = transfer_mag_sq_vec(omega_arp, float(df), 0.0, np.pi, B, C, J, kappa_c)
            p1, p2 = strongest_two(omega_arp, mag, prominence)
            pk_lo[i] = p1
            pk_hi[i] = p2

        eig_p, eig_m = eig_im_parts(df_vec, 0.0, np.pi, J)
        ax3.plot(df_vec, np.real(eig_p), "k--", lw=st["curve_lw"])
        ax3.plot(df_vec, np.real(eig_m), "k--", lw=st["curve_lw"])
        ax3.plot(df_vec, pk_lo, "k-", lw=st["curve_lw"])
        ax3.plot(df_vec, pk_hi, "k-", lw=st["curve_lw"])

        for x in ep_pos:
            ax3.axvline(x, color="r", lw=st["vline_lw"])

        for _, y_tpd in tpd_positions(np.pi, J, kappa_c):
            is_ep = y_tpd in ep_pos
            color = "m" if is_ep else "c"
            style = "--" if ((not is_ep) and (col > 0)) else "-"
            ax3.axvline(y_tpd, color=color, lw=st["vline_lw"], linestyle=style)

        ax3.set_title(rf"$\phi = \pi$, {cfg['lbl']}", fontsize=st["title_size"])
        ax3.set_xlabel(r"$\tilde \Delta_f$", fontsize=st["label_size"])
        ax3.set_ylim(bot_cfg["ylim_arp"])
        ax3.tick_params(labelsize=st["tick_size"])
        _add_roman(ax3, romans_bot[col], st["roman_size"])
        if col == 0:
            ax3.set_ylabel("Frequency / J", fontsize=st["label_size"])
        else:
            _hide_repeated_y(ax3)

    # Row labels outside each row
    for r in range(4):
        ax_ref = leftmost_axes[r]
        if ax_ref is None:
            continue
        _add_row_label(fig, ax_ref, row_labels[r], st["row_label_size"], dx=0.060, dy=0.040)

    output_path = Path(CFG["output_path"])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=int(CFG["save_dpi"]), facecolor="white")
    plt.close(fig)
    print("saved figure to", output_path)


if __name__ == "__main__":
    build_combined()
