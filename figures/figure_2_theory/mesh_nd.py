# mesh.py  – numerical grid + pre-computed fields
# ASCII-only, no external dependencies beyond NumPy / dataclasses.

from dataclasses import dataclass
import numpy as np
from matplotlib import pyplot as plt

from settings import STYLE
import matplotlib.patheffects as pe

# put this once, right after your regular imports
import matplotlib as mpl


@dataclass
class MeshND:
    G: np.ndarray  # Δκ grid
    D: np.ndarray  # Δf grid
    diff_imag: np.ndarray  # |Im(Λ)|        (Λ from analytic √R)
    diff_real: np.ndarray  # Re(Λ)           "
    diff_abs: np.ndarray  # clipped |Λ|
    max_real: np.ndarray  # max Re(eigvals) of full matrix
    thresh_imag: np.ndarray  # |κ̄|
    thresh_real: np.ndarray  # −(Δκ−κ_c)
    disc_field: np.ndarray  # cubic discriminant
    q_field: np.ndarray  # q parameter (for optimal path)
    p_field: np.ndarray  # p parameter (for alternative path)
    unstable: np.ndarray  # boolean mask, max_real > 0


# ------------------------------------------------------------------
def _eigvals(delta_tilde_kappa: float,
             delta_tilde_f: float,
             kappa_tilde_c: float,
             phi: float,
             f_tilde_c: float) -> np.ndarray:
    """Eigen-values of the 2×2 non-Hermitian matrix."""
    kappa_tilde_y = -2.0 * delta_tilde_kappa + kappa_tilde_c
    f_tilde_y = f_tilde_c - delta_tilde_f
    mat = np.array([[-kappa_tilde_c / 2.0 - 1j * f_tilde_c, -1j],
                    [-1j * np.exp(1j * phi), -kappa_tilde_y / 2.0 - 1j * f_tilde_y]])
    return np.linalg.eigvals(mat)


# ------------------------------------------------------------------
def get(phi: float,
        *,
        kappa_tilde_c: float = 1.0,
        f_tilde_c: float = 5.0,
        delta_tilde_kappa_lim: tuple[float, float] = (-5, 5),
        delta_tilde_f_lim: tuple[float, float] = (-5, 5),
        N: int = 101) -> MeshND:
    """
    Build the mesh and all derived fields.

    Parameters
    ----------
    phi  : 0.0 or np.pi         phase
    J    : coupling constant
    kappa_tilde_c, f_tilde_c : cavity C parameters
    N    : points per axis      (501 → 250k points; adjust as needed)
    """
    # 1. grids ---------------------------------------------------------
    dk = np.linspace(*delta_tilde_kappa_lim, N)
    df = np.linspace(*delta_tilde_f_lim, N)
    K_tilde, D_tilde = np.meshgrid(dk, df)

    # 2. analytic Λ (square-root branch) -------------------------------
    radicand_tilde = (-D_tilde ** 2
         + 2j * D_tilde * K_tilde
         + K_tilde ** 2
         - 4.0 * np.exp(1j * phi))
    Lambda_tilde = np.sqrt(radicand_tilde)
    Lam_mag2 = np.abs(Lambda_tilde) ** 2 + 1e-12  # regularised |Λ|²

    diff_imag = np.abs(Lambda_tilde.imag)
    diff_real = Lambda_tilde.real
    petermann_factor_val = (D_tilde ** 2 + K_tilde ** 2 + 4.0 + Lam_mag2) / (2.0 * Lam_mag2)

    # 3. cubic discriminant & q ---------------------------------------
    kappa_tilde_bar = kappa_tilde_c - K_tilde
    p_field = (-D_tilde ** 2 / 4.0
               + K_tilde ** 2 / 4.0
               - np.cos(phi)
               + kappa_tilde_bar ** 2 / 4.0)
    q_field = -kappa_tilde_bar * (2 * np.sin(phi) - D_tilde * K_tilde) / 4.0
    discriminant_field = -4.0 * p_field ** 3 - 27.0 * q_field ** 2

    # 4. max Re(eigenvalues) – loop (vectorised loop would eat RAM) ---
    max_real = np.empty_like(K_tilde)
    # for i in range(N):
    #     for j in range(N):
    #         max_real[i, j] = _eigvals(K_tilde[i, j], D_tilde[i, j], kappa_tilde_c, phi, f_tilde_c).real.max()

    # 5. thresholds / masks -------------------------------------------
    thresh_imag = np.abs(kappa_tilde_c - K_tilde)
    thresh_real = -(K_tilde - kappa_tilde_c)
    unstable = diff_real > thresh_real

    return MeshND(K_tilde, D_tilde,
                  diff_imag.astype(float),
                  diff_real.astype(float),
                  petermann_factor_val.astype(float),
                  max_real.astype(float),
                  thresh_imag.astype(float),
                  thresh_real.astype(float),
                  discriminant_field.astype(float),
                  q_field.astype(float),
                  p_field.astype(float),
                  unstable)

def corner_tag(ax, phi: float, kappa_c: float, phi_only=False) -> None:
    """Draw the 'φ = …, κ̃_c/J ≈ …' label in the lower-right corner of *ax*."""
    if np.isclose(phi, 0.0):
        phi_txt = "0"
    elif np.isclose(phi, np.pi):
        phi_txt = r"\pi"
    elif np.isclose(phi, np.pi / 2):
        phi_txt = r"\pi/2"
    else:
        phi_txt = rf"{phi:.2g}"
    kappa_txt = rf"{kappa_c:.3g}"
    if phi_only:
        tag_string = (
            rf"$\phi = {phi_txt}$"
        )
    else:
        tag_string = (
                rf"$\phi = {phi_txt}$" + "\n" +
                rf"$\tilde \kappa_c = {kappa_txt}$"
        )

    # --- *only* inside this `with` block do we activate LaTeX rendering ----
    with mpl.rc_context({
        "text.usetex": True,
        "font.family": "serif",
        "font.serif": ["Computer Modern Roman"],
        "text.latex.preamble": r"\usepackage{amsmath}"
    }):
        text = ax.text(
            0.97, 0.03, tag_string,
            transform=ax.transAxes,
            ha="right", va="bottom",
            fontsize=STYLE.tick_font + 5,
            bbox=dict(boxstyle="round,pad=0.25",
                      facecolor="white", edgecolor="none", alpha=0.9),
            zorder=99
        )

        text.set_path_effects([
            pe.withStroke(linewidth=.75, foreground='black')
        ])
