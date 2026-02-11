from dataclasses import dataclass
import numpy as np

@dataclass(frozen=True)
class FigMetricStyle:
    tick_font     : int   = 19
    label_font    : int   = 31
    legend_font   : int   = 25
    theory_legend_font: int = 30
    save_dpi      : int   = 400

    # colors / line-widths ------------------------------------------------
    curve_lw      : float = 5
    theory_lw     : float = 5
    min_split_lw  : float = 5

    data_marker   : str   = "o"
    data_color    : str   = "black"
    error_color   : str   = "black"
    data_ms       : int   = 16
    data_lw       : float = 3

    min_split_color : str = "black"   # dashed h-line

    # per-phase colours (match left column curves)
    # USE THE SAME COLORS AS FIGURE 3!!!!
    def curve_color_map(self, phi: float) -> str:
        if np.isclose(phi, 0.0):
            return "royalblue"
        if np.isclose(phi, np.pi/2):
            return "darkgreen"
        return "purple"

    def theory_color_map(self, phi: float) -> str:
        return self.curve_color_map(phi)   # reuse


    # two “chosen” κ̃ per φ
    star_kappa = {
        0.0      : 0.67,
        np.pi/2  : 1.30,
        np.pi    : 0.83,
    }
    tri_kappa  = {
        0.0      : 1.96,
        np.pi/2  : 2.32,
        np.pi    : 1.66,
    }
    reference_paper_kappa = {
        0.0: 0.137
    }

    star_ms   : int   = 400
    tri_ms    : int   = 400
    ref_ms: int = 0


STYLE = FigMetricStyle()