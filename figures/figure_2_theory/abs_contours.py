"""
Absolute splitting with contours panel.

Colour-map  : clipped |Lambda| (petermann).
Contours    : instability (green), discriminant (cyan), optimal path (magenta).
"""

from __future__ import annotations
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from tpd_locations_nd import ep_location, DegeneracyType, tpd_location
from mesh_nd import MeshND, corner_tag
from settings import STYLE


def plot(ax: plt.Axes,
         mesh: MeshND,
         phi: float,
         *,
         figure_mode: bool,
         legend: bool,
         kappa_tilde_c       : float,
         return_mappable: bool = False,
         include_p: bool = False, ):

    # --- colour map (clip 90-th percentile) ----------------------------
    pclip = np.nanpercentile(mesh.diff_abs, 90.0)
    data  = np.clip(mesh.diff_abs, 0.0, pclip)

    stable   = np.ma.masked_where(mesh.unstable,  data)
    unstable = np.ma.masked_where(~mesh.unstable, data)

    im = ax.contourf(mesh.G, mesh.D, stable, 50,
                     cmap=STYLE.cmap,
                     vmin=1, vmax=pclip)
    ax.contourf(mesh.G, mesh.D, unstable, 50,
                cmap=STYLE.gray_cmap,
                vmin=1, vmax=pclip)

    # --- contour lines -------------------------------------------------
    ax.contour(mesh.G, mesh.D, mesh.diff_real - mesh.thresh_real,
               levels=[0], colors=STYLE.stability_col,
               linewidths=STYLE.contour_lw, linestyles=STYLE.stability_ls)

    ax.contour(mesh.G, mesh.D, mesh.disc_field,
               levels=[0], colors=STYLE.split_col,
               linewidths=STYLE.contour_lw, linestyles=STYLE.stability_ls)

    ax.contour(mesh.G, mesh.D, mesh.q_field,
               levels=[0], colors=STYLE.q_color,
               linewidths=STYLE.contour_lw, linestyles=STYLE.stability_ls)

    if include_p:
        ax.contour(mesh.G, mesh.D, mesh.p_field,
                   levels=[0], colors=STYLE.p_color,
                   linewidths=STYLE.contour_lw, linestyles=STYLE.stability_ls)

    eps = ep_location(phi)
    for idx, degeneracy in enumerate(eps):
        marker_type = STYLE.primary_ep_marker if degeneracy.degeneracy_type == DegeneracyType.PRIMARY_EP else STYLE.secondary_ep_marker
        marker_color = STYLE.primary_ep_color if degeneracy.degeneracy_type == DegeneracyType.PRIMARY_EP else STYLE.secondary_ep_color
        label = STYLE.primary_ep_label if degeneracy.degeneracy_type == DegeneracyType.PRIMARY_EP else STYLE.secondary_ep_label
        ep_x = degeneracy.Delta_tilde_kappa
        ep_y = degeneracy.Delta_tilde_f
        ax.scatter(
            ep_x, ep_y,
            color=marker_color, marker=marker_type,
            s=STYLE.scatter_size, zorder=5, label=label,
            linewidths=STYLE.scatter_lw
        )

    tpd_marker_map = {
        DegeneracyType.PRIMARY_TPD:    STYLE.primary_tpd_marker,
        DegeneracyType.SECONDARY_TPD:  STYLE.secondary_tpd_marker,
        DegeneracyType.ROGUE_TPD:      STYLE.rogue_tpd_marker,
    }
    tpd_color_map = {
        DegeneracyType.PRIMARY_TPD:    STYLE.primary_tpd_color,
        DegeneracyType.SECONDARY_TPD:  STYLE.secondary_tpd_color,
        DegeneracyType.ROGUE_TPD:      STYLE.rogue_tpd_color,
    }
    tpd_label_map = {
        DegeneracyType.PRIMARY_TPD:    STYLE.primary_tpd_label,
        DegeneracyType.SECONDARY_TPD:  STYLE.secondary_tpd_label,
        DegeneracyType.ROGUE_TPD:      STYLE.rogue_tpd_label,
    }
    tpds = tpd_location(phi, kappa_tilde_c)
    for idx, degeneracy in enumerate(tpds):
        marker_type = tpd_marker_map[degeneracy.degeneracy_type]
        marker_color = tpd_color_map[degeneracy.degeneracy_type]
        label = tpd_label_map[degeneracy.degeneracy_type]
        tpd_x = degeneracy.Delta_tilde_kappa
        tpd_y = degeneracy.Delta_tilde_f
        ax.scatter(
            tpd_x, tpd_y,
            color=marker_color, marker=marker_type,
            s=STYLE.scatter_size, zorder=5, label=label,
            linewidths=STYLE.scatter_lw,
            facecolors='none'
        )

    # --- cosmetics -----------------------------------------------------
    if phi == 0:
        ax.set_xticklabels([])
        ax.set_yticklabels([])
    if phi == np.pi or phi == np.pi/2:
        ax.set_xlabel(r"$\Delta_\kappa/J$", fontsize=STYLE.label_font)
        ax.set_yticklabels([])
        ax.tick_params(axis="x", labelsize=STYLE.tick_font)



    if legend:
        handles = [
            Line2D([], [], color=STYLE.stability_col,
                   lw=STYLE.contour_lw, ls=STYLE.stability_ls,
                   label="Instability"),
            Line2D([], [], color=STYLE.split_col,
                   lw=STYLE.contour_lw, ls=STYLE.stability_ls,
                   label="Splitting"),
            Line2D([], [], color="magenta",
                   lw=STYLE.contour_lw, ls=STYLE.stability_ls,
                   label="Optimal Path"),
            Line2D([], [], marker=STYLE.primary_,
                   color='none', markerfacecolor='none',
                   markeredgecolor='red',
                   markersize=STYLE.scatter_size**0.4,
                   markeredgewidth=STYLE.scatter_lw - 3,
                   lw=0,
                   label="TPD"),
            Line2D([], [], marker=STYLE.primary_ep_marker,
                   color='red',
                   markerfacecolor='red',
                   markeredgecolor='red',
                   markersize=STYLE.scatter_size**0.4,
                   markeredgewidth=STYLE.scatter_lw - 3,
                   lw=0,
                   label="EP")
        ]

        ax.legend(handles=handles,
                  fontsize=STYLE.legend_font - 6,
                  loc="upper left",
                  bbox_to_anchor=(-0.0175, 1.0175),  # tweak these values
                  framealpha=1.0,
                  borderpad=0.18,)

    corner_tag(ax, phi, kappa_tilde_c)
    if return_mappable:
        return im
