"""Central styling constants used by all panels."""
from pathlib import Path
from dataclasses import dataclass

@dataclass(frozen=True)
class FigTheoryStyle:

    # Default: 500, Paper: 2501 (takes a while)
    GRID_SIZE = 500

    tick_font     : int = 24
    label_font    : int = 34
    legend_font   : int = 24
    cmap          : str = "inferno"
    gray_cmap     : str = "gray"
    scatter_size  : int = 350

    primary_ep_color      : str = "red"
    primary_tpd_color : str = "red"
    secondary_ep_color: str = "gray"
    secondary_tpd_color: str = "gray"
    rogue_tpd_color: str = "gray"

    primary_ep_marker : str = "x"
    primary_tpd_marker     : str = "o"
    secondary_ep_marker: str = "x"
    secondary_tpd_marker: str = "o"
    rogue_tpd_marker : str = "D"

    primary_ep_label  : str = "Primary EP"
    primary_tpd_label      : str = "Primary TPD"
    secondary_ep_label: str = "Other EP"
    secondary_tpd_label: str = "Other TPD"
    rogue_tpd_label : str = "Rogue TPD"

    scatter_lw    : int = 5
    scatter_lw_tpd: int = 4
    contour_lw    : int = 5
    save_dpi      : int = 400
    stability_col : str = "lime"
    stability_ls  : str = "--"
    split_col     : str = "cyan"
    split_ls      : str = "--"
    fake_split_col: str = "royalblue"
    fake_split_ls : str = "--"
    q_color       : str = "magenta"
    q_ls          : str = "--"
    p_color       : str = "gold"
    p_ls          : str = "--"

STYLE = FigTheoryStyle()
ROOT   = Path(__file__).resolve().parent
