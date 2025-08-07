from dataclasses import dataclass
import numpy as np

@dataclass(frozen=True)
class FigMetricStyle:
    tick_font     : int   = 18
    label_font    : int   = 22
    legend_font   : int   = 15
    theory_legend_font: int = 17
    save_dpi      : int   = 400

    # colors / line-widths ------------------------------------------------
    curve_lw      : float = 3.0
    theory_lw     : float = 3.0
    min_split_lw  : float = 3.0

    data_marker   : str   = "o"
    data_color    : str   = "black"
    error_color   : str   = "black"
    data_ms       : int   = 10

    min_split_color : str = "black"   # dashed h-line

    # per-phase colours (match left column curves)
    def curve_color_map(self, phi: float) -> str:
        if np.isclose(phi, 0.0):
            return "royalblue"
        if np.isclose(phi, np.pi/2):
            return "darkorange"
        return "forestgreen"

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

    star_ms   : int   = 220
    tri_ms    : int   = 220


STYLE = FigMetricStyle()
