import matplotlib.pyplot as plt
import numpy as np
import matplotlib.gridspec as gridspec
from matplotlib.lines import Line2D
from figures.figure_2_theory.tpd_locations_nd import ep_location, DegeneracyType
from settings import STYLE
from matplotlib.ticker import FormatStrFormatter
from mesh_nd import get as get_mesh, corner_tag
import abs_contours

def _cbar_label(func):
    if func is _imag_only:
        return r'|Im$(\tilde \Delta_\lambda)$|'
    if func is _real_only:
        return r'Re$(\tilde \Delta_\lambda)$'
    if func is abs_contours.plot:
        return r'$\bar{K}_2$'
    return


def _legend_handles():
    st = STYLE
    return [
        Line2D([], [], color=st.primary_ep_color,
               marker=st.primary_ep_marker, linestyle='none',
               markersize=15, markeredgewidth=5,
               label=st.primary_ep_label),
        Line2D([], [], color=st.secondary_ep_color,
               marker=st.secondary_ep_marker, linestyle='none',
               markersize=15, markeredgewidth=5,
               label=st.secondary_ep_label),
        Line2D([], [], color=st.primary_tpd_color,
               marker=st.primary_tpd_marker, linestyle='none',
               markerfacecolor='none',
               markersize=15, markeredgewidth=5,
               label=st.primary_tpd_label),
        Line2D([], [], color=st.secondary_tpd_color,
               marker=st.secondary_tpd_marker, linestyle='none',
               markerfacecolor='none',
               markersize=15, markeredgewidth=5,
               label=st.secondary_tpd_label),
        Line2D([], [], color=st.rogue_tpd_color,
               marker=st.rogue_tpd_marker, linestyle='none',
               markerfacecolor='none',
               markersize=15, markeredgewidth=5,
               label=st.rogue_tpd_label),
        Line2D([], [], color=st.q_color, ls=st.q_ls, lw=4, label='q = 0'),
        Line2D([], [], color=st.split_col, ls=st.split_ls, lw=4, label='Disc = 0'),
        Line2D([], [], color=st.stability_col, ls=st.stability_ls, lw=4,
               label='Instability'),
    ]

def _imag_only(ax, mesh, phi, *, figure_mode=False, legend=False,
               kappa_tilde_c=None, return_mappable=False, include_p=False):
    im = ax.contourf(mesh.G, mesh.D, mesh.diff_imag, 50, cmap=STYLE.cmap)
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
    corner_tag(ax, phi, kappa_c=kappa_tilde_c, phi_only=True)
    return im if return_mappable else None

def _real_only(ax, mesh, phi, *, figure_mode=False, legend=False,
               kappa_tilde_c=None, return_mappable=False, include_p=False):
    im = ax.contourf(mesh.G, mesh.D, mesh.diff_real, 50, cmap=STYLE.cmap)

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
    corner_tag(ax, phi, kappa_c=kappa_tilde_c, phi_only=True)
    return im if return_mappable else None


def build_figure(filename="theory_fixed.png", *, N=101):
    kappa_tilde_vals = {
        0.0      : (0.67,    1.96),
        np.pi : (0.83,  1.66),
        np.pi/2: (1.30,   2.32),
    }

    phi_vals  = (0.0, np.pi, np.pi/2)
    col_funcs = (_imag_only, _real_only, abs_contours.plot, abs_contours.plot)

    roman     = ['i','ii','iii','iv']
    row_let   = ['a','b','c']

    fig = plt.figure(figsize=(22,15))
    gs  = gridspec.GridSpec(3, 4,
                            left=0.06, right=0.975,
                            bottom=0.075, top=0.94,
                            wspace=0.38, hspace=0.1
                            )

    pad, bar_w = 0.0025, 0.007

    for r, phi in enumerate(phi_vals):
        # build the three meshes
        B  = (-4,4)
        meshes = [
            get_mesh(phi, kappa_tilde_c=kappa_tilde_vals[phi][0],    N=N, delta_tilde_kappa_lim=B, delta_tilde_f_lim=B),
            get_mesh(phi, kappa_tilde_c=kappa_tilde_vals[phi][1],    N=N, delta_tilde_kappa_lim=B, delta_tilde_f_lim=B),
        ]
        axes_row = []
        caxes_row = []

        for c in range(4):
            # for c = 0 to 3, use the first mesh. THen for c = 4, use the second
            this_mesh = meshes[0] if c < 3 else meshes[1]
            this_tilde_kappa_c = kappa_tilde_vals[phi][0] if c < 3 else kappa_tilde_vals[phi][1]
            ax = fig.add_subplot(gs[r,c])
            axes_row.append(ax)

            mappable = col_funcs[c](
                ax, this_mesh, phi,
                figure_mode=False,
                legend=False,
                kappa_tilde_c=this_tilde_kappa_c,
                return_mappable=True,
                include_p=False
            )

            # roman numeral inside
            roman_color = 'black' if c < 2 else 'white'
            ax.text(0.02, 0.895, roman[c],
                    transform=ax.transAxes,
                    fontsize=STYLE.label_font-3,
                    fontweight='bold',
                    color=roman_color)

            # axes labels
            if c>0:
                ax.set_ylabel(''); ax.set_yticklabels([])
            else:
                ax.set_ylabel(r"$\tilde \Delta_f$", fontsize=STYLE.label_font)
            if r<2:
                ax.set_xlabel(''); ax.set_xticklabels([])
            else:
                ax.set_xlabel(r"$\tilde \Delta_\kappa$", fontsize=STYLE.label_font)
            ax.tick_params(axis='both', labelsize=STYLE.tick_font)

            # colorbar for cols 0,1,3
            if c!=2:
                pos = ax.get_position()
                cax = fig.add_axes([pos.x1+pad, pos.y0, bar_w, pos.height])
                cb  = fig.colorbar(mappable, cax=cax)
                cb.set_label(_cbar_label(col_funcs[c]),
                             fontsize=STYLE.label_font, labelpad=3)
                cb.ax.tick_params(labelsize=STYLE.tick_font)
                # cb.ax.tick_params(labelsize=STYLE.tick_font)
                cb.ax.yaxis.set_major_formatter(FormatStrFormatter("%.1f"))  # ← rounds to 2 decimals
                caxes_row.append(cax)

        # shift col 3 (index 3) leftwards to tighten spacing
        pos2 = axes_row[2].get_position()
        pos3 = axes_row[3].get_position()
        new_left = pos2.x1 + 0.02
        axes_row[3].set_position([new_left, pos3.y0, pos3.width, pos3.height])
        # also shift its colorbar
        cax3 = caxes_row[2]  # the 3rd colorbar we created corresponds to col3
        cax3.set_position([new_left+pos3.width+pad, pos3.y0, bar_w, pos3.height])

        # row‐letter 'a','b','c'
        p0 = axes_row[0].get_position()
        y_mid = p0.y0 + 0.55*p0.height
        fig.text(p0.x0-0.03, y_mid + .12, row_let[r],
                 fontsize=STYLE.label_font+16,
                 fontweight='bold',
                 ha='center', va='center',
                 transform=fig.transFigure)

    # global legend
    fig.legend(handles=_legend_handles(),
               loc="upper center", ncol=8, frameon=False,
               fontsize=STYLE.legend_font,
               bbox_to_anchor=(0.5,1.00),
               columnspacing=1, handletextpad=0.4)

    fig.savefig(filename, dpi=STYLE.save_dpi)
    plt.close(fig)

if __name__=="__main__":
    # make the figures directory if it doesn't exist
    import os
    if not os.path.exists("../../.figures"):
        os.makedirs("../../.figures")

    build_figure(N=STYLE.GRID_SIZE, filename="../../.figures/theory.png")
