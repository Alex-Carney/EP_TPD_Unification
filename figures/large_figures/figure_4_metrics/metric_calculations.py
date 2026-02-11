import numpy as np

import matplotlib.pyplot as plt

from figures.small_figures.figure_4_metrics.experiment_tpds import standard_ep_locations, standard_tpd_locations, \
    TPDLocation
from fitting.peak_fitting import peak_location

from scipy.optimize import curve_fit

def _instability_field_max_real_eig(
    dk_grid: np.ndarray,
    df_grid: np.ndarray,
    phi: float,
    kappa_tilde_c: float,
) -> np.ndarray:
    """
    Vectorized max(real(eigs)) on a dk/df grid, using the same analytic eigenvalue expression
    as fitting.peak_fitting._eigenvalues_core, but avoiding per-point calls.

    Assumes tilde units with J = 1 and f_tilde_c = 0. Real parts are independent of f_tilde_c anyway.
    """
    # delta_lambda = sqrt(-df^2 + 2i df dk + dk^2 - 4 exp(i phi))
    rad = -df_grid**2 + 2j * df_grid * dk_grid + dk_grid**2 - 4.0 * np.exp(1j * phi)
    delta_lambda = np.sqrt(rad)

    # lambda_0 = (dk - kappa_c)/2 + i*(df/2 - f_c)
    # real(lambda_0) = (dk - kappa_c)/2
    lambda0_real = 0.5 * (dk_grid - kappa_tilde_c)

    # eigenvalues are lambda0 +/- delta_lambda/2, so real parts are:
    lam_plus_real = lambda0_real + 0.5 * np.real(delta_lambda)
    lam_minus_real = lambda0_real - 0.5 * np.real(delta_lambda)

    return np.maximum(lam_plus_real, lam_minus_real)


def _zero_crossing_points_xy(
    dk_grid: np.ndarray,
    df_grid: np.ndarray,
    field: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Extract approximate (dk, df) points where field crosses 0 using sign changes
    on grid edges plus linear interpolation.

    Returns (dk_pts, df_pts). Can be empty.
    """
    dk_pts = []
    df_pts = []

    # Horizontal edges: between [:, j] and [:, j+1]
    f0 = field[:, :-1]
    f1 = field[:, 1:]
    cross = (f0 == 0.0) | (f1 == 0.0) | (f0 * f1 < 0.0)
    if np.any(cross):
        denom = (f0 - f1)
        denom_safe = np.where(denom == 0.0, np.nan, denom)
        t = f0 / denom_safe
        t = np.where(np.isfinite(t), t, 0.5)

        dk0 = dk_grid[:, :-1]
        dk1 = dk_grid[:, 1:]
        df0 = df_grid[:, :-1]

        dk_cross = dk0 + t * (dk1 - dk0)
        df_cross = df0

        dk_pts.append(dk_cross[cross])
        df_pts.append(df_cross[cross])

    # Vertical edges: between [i, :] and [i+1, :]
    f0 = field[:-1, :]
    f1 = field[1:, :]
    cross = (f0 == 0.0) | (f1 == 0.0) | (f0 * f1 < 0.0)
    if np.any(cross):
        denom = (f0 - f1)
        denom_safe = np.where(denom == 0.0, np.nan, denom)
        t = f0 / denom_safe
        t = np.where(np.isfinite(t), t, 0.5)

        df0 = df_grid[:-1, :]
        df1 = df_grid[1:, :]
        dk0 = dk_grid[:-1, :]

        df_cross = df0 + t * (df1 - df0)
        dk_cross = dk0

        dk_pts.append(dk_cross[cross])
        df_pts.append(df_cross[cross])

    if not dk_pts:
        return np.array([], dtype=float), np.array([], dtype=float)

    return np.concatenate(dk_pts).astype(float), np.concatenate(df_pts).astype(float)

def eigenvalue_splitting_eqn(dk_vals, df_vals, phi):
    """Evaluate imag(sqrt(...)) on a 1D grid of (dk, df) points."""
    # inside = -df^2 + 2i * df * dk + dk^2 - 4 exp(i phi)
    inside = -df_vals**2 + 2j * df_vals * dk_vals + dk_vals**2 - 4 * np.exp(1j * phi)
    return np.imag(np.sqrt(inside))

def sqrt_model(dk, a, dk0):
    out = np.zeros_like(dk)
    valid = dk >= dk0
    out[valid] = a * np.sqrt(dk[valid] - dk0)
    return out

def sqrt_model_right(dk, a, dk0):
    out = np.zeros_like(dk)
    valid = dk <= dk0
    out[valid] = a * np.sqrt(-dk0 - dk[valid])
    return out

def arc_derivative_unit_vector(phi, dk_val):
    dx = 1
    dy = -(2 * np.sin(phi)) / dk_val**2
    norm = np.sqrt(dx**2 + dy**2)
    return dx / norm, dy / norm

def arc_unit_vector(phi, dk):
    """Unit tangent to the TPD-crossing curve at (dk, df)."""
    dx = 1.0
    dy = -(2.0 * np.sin(phi)) / dk**2           # derivative of   f = 2 sin φ / k
    norm = np.hypot(dx, dy)                     # = √(dx² + dy²)
    return dx / norm, dy / norm                 # components of a unit vector

def step_along_curve(phi, dk0, eps):
    """
    Return (dk, df) that is a Euclidean distance `eps`
    away from the TPD point (dk0, df0) *along the curve*.
    The step is accurate to O(eps^2).
    """
    dx_hat, dy_hat = arc_unit_vector(phi, dk0)
    dk  = dk0 + eps * dx_hat                    # x–increment
    df  = (2.0 * np.sin(phi)) / dk              # puts the point back ON the curve
    return dk, df

def distance_between_ep_and_tpd(phi, kappa_tilde_c, left_tpd=True):
    left_ep_loc = standard_ep_locations(phi, left_ep=True)
    right_ep_loc = standard_ep_locations(phi, left_ep=False)
    tpd_loc = standard_tpd_locations(phi, kappa_tilde_c, left_tpd=left_tpd)

    dist_left = np.hypot(
        tpd_loc.Delta_tilde_f     - left_ep_loc.Delta_tilde_f,
        tpd_loc.Delta_tilde_kappa - left_ep_loc.Delta_tilde_kappa,
        )
    dist_right = np.hypot(
        tpd_loc.Delta_tilde_f     - right_ep_loc.Delta_tilde_f,
        tpd_loc.Delta_tilde_kappa - right_ep_loc.Delta_tilde_kappa,
        )

    return min(dist_left, dist_right)

def distance_between_instability_and_tpd(
    phi: float,
    kappa_tilde_c: float,
    *,
    left_tpd: bool = True,
    lim: float = 5.0,
    N: int = 241,
) -> float:
    """
    dist_2 in (Delta_tilde_kappa, Delta_tilde_f) space between the TPD and the nearest
    instability boundary point, where instability boundary is max(real(eigs)) = 0.

    The grid is centered around the TPD location with at least [-lim, lim] coverage.
    """
    tpd = standard_tpd_locations(phi, kappa_tilde_c, left_tpd=left_tpd)
    dk_tpd = float(tpd.Delta_tilde_kappa)
    df_tpd = float(tpd.Delta_tilde_f)

    if not (np.isfinite(dk_tpd) and np.isfinite(df_tpd)):
        return float("nan")

    # Ensure the grid covers the TPD even if it drifts outside [-lim, lim]
    dk_min = min(-lim, dk_tpd - 1.0)
    dk_max = max( lim, dk_tpd + 1.0)
    df_min = min(-lim, df_tpd - 1.0)
    df_max = max( lim, df_tpd + 1.0)

    dk = np.linspace(dk_min, dk_max, int(N))
    df = np.linspace(df_min, df_max, int(N))
    dk_grid, df_grid = np.meshgrid(dk, df, indexing="xy")

    field = _instability_field_max_real_eig(dk_grid, df_grid, phi, kappa_tilde_c)

    dk_pts, df_pts = _zero_crossing_points_xy(dk_grid, df_grid, field)
    if dk_pts.size == 0:
        return float("nan")

    dist = np.sqrt((dk_pts - dk_tpd) ** 2 + (df_pts - df_tpd) ** 2)
    return float(np.min(dist))


def thermal_noise_efficiency_at_tpd(
    tpd_location: TPDLocation,
    phi: float,
    kappa_tilde_c: float,
    *,
    f_tilde_c: float = 0.0,
):
    """
    Thermal Noise Efficiency (NE) evaluated at the TPD.

    Numerical substitutions only:
      - Delta_tilde_f, Delta_tilde_kappa come from tpd_location
      - kappa_tilde_c passed explicitly
      - f_tilde_d is taken from peak_location(...)
      - f_tilde_c is an external parameter (default 10.0 for now)

    Returns a single scalar for plotting. If peak_location returns two peaks,
    we evaluate NE at both and return max(abs(NE)).
    """
    df = float(tpd_location.Delta_tilde_f)
    dk = float(tpd_location.Delta_tilde_kappa)

    # Use the same peak locator already used throughout this file.
    # This gives the peak frequency locations f_tilde_d to substitute into NE.
    fds = peak_location(1, f_tilde_c, kappa_tilde_c, df, dk, phi)

    if fds is None or len(fds) == 0:
        return np.nan

    def _ne_single_fd(f_tilde_d: float) -> float:
        i = 1j

        # Numerator
        a = (2.0 * dk - kappa_tilde_c)
        num = (
            4.0 * a * (
                4.0 * (f_tilde_c ** 2) * a
                - 4.0 * kappa_tilde_c
                + 4.0 * (f_tilde_d ** 2) * a
                + (kappa_tilde_c ** 2) * a
                - 8.0 * f_tilde_c * f_tilde_d * a
            )
        )

        # Denominator factors (kept as written, purely numerical)
        den1 = (
            4.0 * dk * f_tilde_c
            - 4.0 * df * f_tilde_c * i
            + 4.0 * df * f_tilde_d * i
            - 4.0 * np.exp((-phi * i)) * i
            - 4.0 * dk * f_tilde_d
            + 2.0 * df * kappa_tilde_c
            + 2.0 * dk * kappa_tilde_c * i
            - 8.0 * f_tilde_c * f_tilde_d * i
            - 4.0 * f_tilde_c * kappa_tilde_c
            + 4.0 * f_tilde_d * kappa_tilde_c
            + 4.0 * (f_tilde_c ** 2) * i
            + 4.0 * (f_tilde_d ** 2) * i
            - (kappa_tilde_c ** 2) * i
        )

        den2 = (
            4.0 * np.exp((phi * i)) * i
            + 4.0 * df * f_tilde_c * i
            - 4.0 * df * f_tilde_d * i
            + 4.0 * dk * f_tilde_c
            - 4.0 * dk * f_tilde_d
            + 2.0 * df * kappa_tilde_c
            - 2.0 * dk * kappa_tilde_c * i
            + 8.0 * f_tilde_c * f_tilde_d * i
            - 4.0 * f_tilde_c * kappa_tilde_c
            + 4.0 * f_tilde_d * kappa_tilde_c
            - 4.0 * (f_tilde_c ** 2) * i
            - 4.0 * (f_tilde_d ** 2) * i
            + (kappa_tilde_c ** 2) * i
        )

        den = den1 * den2

        if den == 0:
            return np.nan

        val = num / den

        # NE should plot as a positive scalar. Use magnitude to be safe.
        out = float(np.abs(val))
        if not np.isfinite(out):
            return np.nan
        return out

    if len(fds) == 1:
        return _ne_single_fd(float(fds[0]))

    # At a true TPD the peaks coincide, but numerically you might get two values.
    vals = np.array([_ne_single_fd(float(fd)) for fd in fds], dtype=float)
    return float(np.nanmax(vals))

def petermann_factor_at_tpd(tpd_location: TPDLocation, phi):
    df = tpd_location.Delta_tilde_f
    dk = tpd_location.Delta_tilde_kappa
    ss = np.sqrt( -df **2 + 2 * df * dk * 1j + dk ** 2 - 4 * np.exp(1j * phi) )
    v = np.abs(ss)
    k2tilde = ( df **2 + dk ** 2 + v**2 + 4 ) / ( 2 * v**2 )
    return k2tilde

def splitting_strength_at_tpd_arc(phi, kappa_tilde_c):
    if phi == 0:
        return np.sqrt(2) * (8 - kappa_tilde_c**2)**0.25
    if phi == np.pi:
        return np.sqrt(2) * (kappa_tilde_c**2 + 4)**0.25

    # Get the TPD location
    tpd_loc = standard_tpd_locations(phi, kappa_tilde_c, left_tpd=True)
    dk0 = tpd_loc.Delta_tilde_kappa
    df0 = (2 * np.sin(phi)) / dk0

    # Arc direction vector (unit vector)
    dx_hat, dy_hat = arc_derivative_unit_vector(phi, dk0)

    # Step forward along arc
    arc_steps = np.linspace(0, 0.001, 2500)  # arc length distance from TPD
    dks_sim = dk0 + arc_steps * dx_hat
    dfs_sim = df0 + arc_steps * dy_hat

    # Compute splitting at each step
    splittings = np.zeros_like(dks_sim)
    for i in range(len(dks_sim)):
        fpts = peak_location(1, 0, kappa_tilde_c, dfs_sim[i], dks_sim[i], phi)
        splittings[i] = np.abs(fpts[1] - fpts[0]) if len(fpts) > 1 else 0

    # Plot for debug
    # plt.plot(arc_steps, splittings)
    # plt.xlabel("Arc Distance from TPD")
    # plt.ylabel("Splitting")
    # plt.title("Splitting vs Arc Distance")
    # plt.show()

    # Fit to sqrt model
    valid = splittings > 0
    xdata = arc_steps[valid]
    ydata = splittings[valid]

    try:
        popt, pcov = curve_fit(sqrt_model, xdata, ydata, p0=[3.0, 0.0], maxfev=10000)
    except RuntimeError as e:
        print('Fitting failed for phi =', phi, 'kappa_tilde_c =', kappa_tilde_c)
        print(str(e))
        return np.nan

    a_fit, s0_fit = popt

    # Optional plot of fit
    # fit_vals = sqrt_model(xdata, *popt)
    # plt.plot(xdata, ydata, 'o', label='data')
    # plt.plot(xdata, fit_vals, '-', label='fit')
    # plt.xlabel("Arc Distance from TPD")
    # plt.ylabel("Splitting")
    # plt.legend()
    # plt.title("Fit to Square Root Model")
    # plt.show()

    return a_fit

def eigenvalue_splitting_strength(phi, eps_min=1e-7, eps_max=1e-3, n_pts=60):
    # 1) Get the EP coordinates
    ep = standard_ep_locations(phi)
    dk0 = ep.Delta_tilde_kappa
    df0 = (2 * np.sin(phi)) / dk0

    # 2) Generate ε-values logarithmically spaced
    eps_samples = np.geomspace(eps_min, eps_max, n_pts)

    # 3) Get arc-following (dk, df) steps for each ε
    ks, fs = step_along_curve(phi, dk0, eps_samples)  # returns arrays of same length as eps_samples

    # 4) Compute the eigenvalue splitting using the analytic expression
    splittings = eigenvalue_splitting_eqn(ks, fs, phi)

    # 5) Filter out bad points and fit the sqrt law
    valid = splittings > 0.0
    if np.count_nonzero(valid) < 8:
        return np.nan

    popt, _ = curve_fit(sqrt_model,
                        eps_samples[valid], splittings[valid],
                        p0=(3.0, 0.0), maxfev=10000)

    a_fit, _ = popt
    return a_fit


def splitting_strength_at_tpd_arc_2(phi, kappa_tilde_c,
                                  eps_min=1e-7, eps_max=1e-3, n_pts=60, left_tpd=True):
    """
    Extract the prefactor  a  in  Δλ ≈ a √ε  at a TPD for arbitrary φ.
    Closed-form answers for φ = 0, π are kept for speed/clarity.
    """
    # closed-form cases ------------------------------------------------------
    if phi == 0.0:
        return np.sqrt(2.0) * (8.0 - kappa_tilde_c**2)**0.25
    if phi == np.pi:
        return np.sqrt(2.0) * (kappa_tilde_c**2 + 4.0)**0.25

    # hyperbolic case --------------------------------------------------------
    # 1) locate the TPD
    tpd = standard_tpd_locations(phi, kappa_tilde_c, left_tpd=left_tpd)
    dk0 = tpd.Delta_tilde_kappa
    df0 = (2.0 * np.sin(phi)) / dk0            # guaranteed to sit on the curve

    # 2) generate ε-values logarithmically spaced over [eps_min, eps_max]
    factor = 1 if left_tpd else -1
    eps_min = factor * eps_min
    eps_max = factor * eps_max

    eps_samples = np.geomspace(eps_min, eps_max, n_pts)

    # 3) fetch (dk, df) along the curve for each ε, then compute splitting
    ks, fs = step_along_curve(phi, dk0, eps_samples)   # vectorised helper

    # `peak_location` is probably the only slow call – vectorise it if possible
    splittings = np.array([
        (lambda pts: np.abs(pts[1] - pts[0]) if len(pts) > 1 else 0.0)(
            peak_location(1, 0, kappa_tilde_c, f, k, phi)
        )
        for k, f in zip(ks, fs)
    ])

    # 4) fit the square-root law
    valid = splittings > 0.0
    if np.count_nonzero(valid) < 8:            # not enough data for a robust fit
        return np.nan

    popt, _ = curve_fit(sqrt_model if left_tpd else sqrt_model_right,
                         eps_samples[valid], splittings[valid],
                        p0=(3.0, 0.0), maxfev=20000)

    # Debug, make a plot of eveyrhting
    # plt.plot(eps_samples, splittings)
    # plt.plot(eps_samples, sqrt_model(eps_samples, *popt) if left_tpd else sqrt_model_right(eps_samples, *popt))
    # plt.show()

    a_fit, _ = popt
    return a_fit

def q_tilde(delta_f, delta_kappa, kappa_tilde_c, phi):
    return (
            0.5 * delta_kappa * np.sin(phi)
            - 0.25 * delta_f * delta_kappa**2
            - 0.5 * kappa_tilde_c * np.sin(phi)
            + 0.25 * delta_f * delta_kappa * kappa_tilde_c
    )


def nuisance_unit_vector(phi, dk):
    """
    Unit normal to the q=0 sensing path at point (dk, df).

    The q=0 curve is df = 2*sin(φ)/dk, with tangent (1, -2*sin(φ)/dk²).
    The normal is perpendicular: rotate 90° CCW.
    """
    tx = 1.0
    ty = -(2.0 * np.sin(phi)) / dk ** 2
    # Rotate 90° CCW: (tx, ty) → (-ty, tx)
    nx = -ty  # = 2*sin(φ)/dk²
    ny = tx  # = 1
    norm = np.hypot(nx, ny)
    return nx / norm, ny / norm


def cbrt_model(eps, b, eps0=0):
    """Cube-root model: y = b * |eps - eps0|^(1/3)"""
    return b * np.abs(eps - eps0) ** (1 / 3)


def nuisance_scaling_coefficient_at_tpd(phi, kappa_tilde_c,
                                        eps_min=1e-7, eps_max=1e-3, n_pts=60,
                                        left_tpd=True):
    """
    Extract the prefactor b in ν̃₀^Root ≈ b * |δ|^(1/3) at a TPD.

    For φ = 0: nuisance is Δf direction, closed form
    For φ = π: nuisance is Δκ direction, closed form
    For other φ: nuisance is orthogonal to q=0 hyperbola, numerical fit
    """
    # Closed-form cases
    if np.isclose(phi, 0.0):
        return np.abs(kappa_tilde_c ** 2 - 4) ** (1 / 3) / 2.0

    if np.isclose(phi, np.pi):
        return (2 ** (1 / 3) * kappa_tilde_c ** (1 / 3) * (kappa_tilde_c ** 2 + 4) ** (1 / 6)) / 2.0

    # General case: numerical approach
    # 1) Get TPD location
    tpd = standard_tpd_locations(phi, kappa_tilde_c, left_tpd=left_tpd)
    dk0 = float(tpd.Delta_tilde_kappa)
    df0 = float(tpd.Delta_tilde_f)

    if not (np.isfinite(dk0) and np.isfinite(df0)):
        return np.nan

    # 2) Get nuisance direction (normal to sensing path)
    nx, ny = nuisance_unit_vector(phi, dk0)

    # 3) Get reference peak at TPD
    peaks_tpd = peak_location(1, 0, kappa_tilde_c, df0, dk0, phi)
    if peaks_tpd is None or len(peaks_tpd) == 0:
        return np.nan
    nu_tpd = float(np.mean(peaks_tpd))

    # 4) Step along nuisance direction and track peak shift
    eps_samples = np.geomspace(eps_min, eps_max, n_pts)
    shifts = np.zeros(n_pts)

    for i, eps in enumerate(eps_samples):
        dk = dk0 + eps * nx
        df = df0 + eps * ny

        peaks = peak_location(1, 0, kappa_tilde_c, df, dk, phi)

        if peaks is None or len(peaks) == 0:
            shifts[i] = np.nan
        else:
            # Should be single peak (or near-degenerate) in nuisance direction
            nu = float(peaks[0]) if len(peaks) == 1 else float(np.mean(peaks))
            shifts[i] = np.abs(nu - nu_tpd)

    # 5) Fit to cube-root model
    valid = np.isfinite(shifts) & (shifts > 1e-12)
    if np.count_nonzero(valid) < 5:
        return np.nan

    try:
        popt, _ = curve_fit(cbrt_model, eps_samples[valid], shifts[valid],
                            p0=[1.0], maxfev=10000)
        return popt[0]
    except RuntimeError:
        return np.nan


def test_nuisance_scaling_visualization():
    """
    Visualize the nuisance response for φ = π/2.
    Shows peak location vs nuisance step and checks for cbrt scaling.
    """
    phi = np.pi / 2
    kappa_tilde_c = 1.5

    # Get TPD location
    tpd = standard_tpd_locations(phi, kappa_tilde_c, left_tpd=True)
    dk0 = float(tpd.Delta_tilde_kappa)
    df0 = float(tpd.Delta_tilde_f)

    print(f"φ = π/2, κ̃_c = {kappa_tilde_c}")
    print(f"TPD location: Δ̃_κ = {dk0:.4f}, Δ̃_f = {df0:.4f}")

    # Nuisance direction
    nx, ny = nuisance_unit_vector(phi, dk0)
    print(f"Nuisance direction: ({nx:.4f}, {ny:.4f})")

    # Reference peak at TPD
    peaks_tpd = peak_location(1, 0, kappa_tilde_c, df0, dk0, phi)
    nu_tpd = float(np.mean(peaks_tpd))
    print(f"Peak at TPD: ν̃ = {nu_tpd:.6f}")
    print(f"Number of peaks at TPD: {len(peaks_tpd)}")

    # Scan along nuisance direction — MORE POINTS, LARGER RANGE
    n_pts = 5000
    eps_max = 1e-3  # larger range to see the path
    eps_range = np.linspace(-eps_max, eps_max, n_pts)
    peak_vals = []
    num_peaks_list = []
    dk_path = []
    df_path = []

    for eps in eps_range:
        dk = dk0 + eps * nx
        df = df0 + eps * ny
        dk_path.append(dk)
        df_path.append(df)

        peaks = peak_location(1, 0, kappa_tilde_c, df, dk, phi)

        if peaks is None or len(peaks) == 0:
            peak_vals.append(np.nan)
            num_peaks_list.append(0)
        else:
            peak_vals.append(float(peaks[0]) if len(peaks) == 1 else float(np.mean(peaks)))
            num_peaks_list.append(len(peaks))

    peak_vals = np.array(peak_vals)
    num_peaks_list = np.array(num_peaks_list)
    dk_path = np.array(dk_path)
    df_path = np.array(df_path)
    shifts = np.abs(peak_vals - nu_tpd)

    # Create figure
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # Plot 1: Peak location vs nuisance step (projected onto Δκ axis for interpretability)
    ax1 = axes[0, 0]
    ax1.plot(dk_path, peak_vals, 'b-', linewidth=2)
    ax1.axhline(nu_tpd, color='r', linestyle='--', label='TPD peak')
    ax1.axvline(dk0, color='gray', linestyle=':', alpha=0.5, label=f'Δ̃_κ at TPD = {dk0:.3f}')
    ax1.set_xlabel('Δ̃_κ (along nuisance path)')
    ax1.set_ylabel('Peak frequency ν̃')
    ax1.set_title(f'Peak vs Δ̃_κ along nuisance path (φ=π/2, κ̃_c={kappa_tilde_c})')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Plot 2: Number of peaks along path
    ax2 = axes[0, 1]
    ax2.plot(dk_path, num_peaks_list, 'g-', linewidth=2)
    ax2.axvline(dk0, color='gray', linestyle=':', alpha=0.5)
    ax2.set_xlabel('Δ̃_κ (along nuisance path)')
    ax2.set_ylabel('Number of peaks')
    ax2.set_title('Number of peaks along nuisance path')
    ax2.grid(True, alpha=0.3)

    # Plot 3: SEPARATE positive and negative eps
    ax3 = axes[1, 0]

    # Positive side (eps > 0)
    pos_mask = eps_range > 1e-8
    eps_pos = eps_range[pos_mask]
    shifts_pos = shifts[pos_mask]
    ax3.plot(eps_pos ** (1 / 3), shifts_pos, 'b.', markersize=4, alpha=0.7, label='ε > 0')

    # Negative side (eps < 0) — use |eps|
    neg_mask = eps_range < -1e-8
    eps_neg = np.abs(eps_range[neg_mask])
    shifts_neg = shifts[neg_mask]
    ax3.plot(eps_neg ** (1 / 3), shifts_neg, 'r.', markersize=4, alpha=0.7, label='ε < 0')

    ax3.set_xlabel('|ε|^(1/3)')
    ax3.set_ylabel('|ν̃ - ν̃_TPD|')
    ax3.set_title('Shift vs |ε|^(1/3) — SEPARATED by sign')
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    # Fit cbrt to each side
    if np.count_nonzero(np.isfinite(shifts_pos)) > 5:
        try:
            popt_pos, _ = curve_fit(cbrt_model, eps_pos, shifts_pos, p0=[1.0])
            print(f"Fitted cbrt coeff (ε > 0): b = {popt_pos[0]:.4f}")
            eps_fit = np.linspace(0, eps_max, 100)
            ax3.plot(eps_fit ** (1 / 3), cbrt_model(eps_fit, popt_pos[0]), 'b-',
                     linewidth=2, label=f'Fit ε>0: b={popt_pos[0]:.3f}')
        except Exception as e:
            print(f"Pos fit failed: {e}")

    if np.count_nonzero(np.isfinite(shifts_neg)) > 5:
        try:
            popt_neg, _ = curve_fit(cbrt_model, eps_neg, shifts_neg, p0=[1.0])
            print(f"Fitted cbrt coeff (ε < 0): b = {popt_neg[0]:.4f}")
            eps_fit = np.linspace(0, eps_max, 100)
            ax3.plot(eps_fit ** (1 / 3), cbrt_model(eps_fit, popt_neg[0]), 'r-',
                     linewidth=2, label=f'Fit ε<0: b={popt_neg[0]:.3f}')
        except Exception as e:
            print(f"Neg fit failed: {e}")

    ax3.legend()

    # Plot 4: Paths in parameter space — ZOOMED OUT
    ax4 = axes[1, 1]

    # Nuisance path (now visible with larger eps range)
    ax4.plot(dk_path, df_path, 'b-', linewidth=2, label='Nuisance path')

    # TPD point
    ax4.plot(dk0, df0, 'ro', markersize=12, label='TPD', zorder=5)

    # Sensing path (q=0 hyperbola) — extend range
    dk_sense = np.linspace(-2.0, -0.3, 200)
    dk_sense = dk_sense[dk_sense != 0]
    df_sense = 2 * np.sin(phi) / dk_sense
    ax4.plot(dk_sense, df_sense, 'm--', linewidth=2, label='Sensing path (q=0)')

    ax4.set_xlabel('Δ̃_κ')
    ax4.set_ylabel('Δ̃_f')
    ax4.set_title('Paths in parameter space')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    ax4.set_aspect('equal')

    plt.tight_layout()
    plt.savefig('nuisance_scaling_test_v2.png', dpi=150)
    plt.show()

    # Print summary
    print(f"\n=== SUMMARY ===")
    print(f"Nuisance direction makes angle {np.degrees(np.arctan2(ny, nx)):.1f}° with Δ̃_κ axis")
    print(f"Path spans Δ̃_κ from {dk_path.min():.3f} to {dk_path.max():.3f}")
    print(f"Path spans Δ̃_f from {df_path.min():.3f} to {df_path.max():.3f}")


def plot_nuisance_scaling_all_phi():
    """Plot nuisance scaling coefficient vs κ̃_c for φ = 0, π, π/2."""

    kappas = np.linspace(0.1, 2.8, 50)

    fig, ax = plt.subplots(figsize=(10, 6))

    # φ = 0 (closed form)
    b_phi0 = np.abs(kappas ** 2 - 4) ** (1 / 3) / 2.0
    ax.plot(kappas, b_phi0, 'b-', linewidth=2, label='φ = 0')

    # φ = π (closed form)
    b_phi_pi = (2 ** (1 / 3) * kappas ** (1 / 3) * (kappas ** 2 + 4) ** (1 / 6)) / 2.0
    ax.plot(kappas, b_phi_pi, color='purple', linewidth=2, label='φ = π')

    # φ = π/2 (numerical)
    b_phi_pi2 = []
    for kc in kappas:
        b = nuisance_scaling_coefficient_at_tpd(np.pi / 2, kc)
        b_phi_pi2.append(b)
    ax.plot(kappas, b_phi_pi2, 'g-', linewidth=2, label='φ = π/2')

    # Mark where φ=0 crosses zero (robust TPD at κ̃_c = 2)
    ax.axvline(2.0, color='blue', linestyle=':', alpha=0.5)
    ax.axhline(0, color='gray', linestyle='-', alpha=0.3)

    ax.set_xlabel('κ̃_c', fontsize=12)
    ax.set_ylabel('Nuisance coefficient b̃_cbrt^TPD', fontsize=12)
    ax.set_title('Nuisance scaling: ν̃₀ ≈ b · |δ|^(1/3)', fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('nuisance_scaling_vs_kappa.png', dpi=150)
    plt.show()

def min_split_over_J(phi, kappa_tilde_c, delta_tilde_f_uncertainty, delta_tilde_kappa_uncertainty, left_tpd=True):
    # Step 1 - Get the TPD location
    tpd_loc = standard_tpd_locations(phi, kappa_tilde_c, left_tpd=left_tpd)
    df0, dk0 = tpd_loc.Delta_tilde_f, tpd_loc.Delta_tilde_kappa

    # N = 10000
    # df_vals = np.random.uniform(df0 - delta_tilde_f_uncertainty,
    #                             df0 + delta_tilde_f_uncertainty, size=N)
    # dk_vals = np.random.uniform(dk0 - delta_tilde_kappa_uncertainty,
    #                             dk0 + delta_tilde_kappa_uncertainty, size=N)
    # samples = np.column_stack([df_vals, dk_vals])

    # corners: (df, dk) = ±(Δf_unc, Δk_unc) away from (df0, dk0)
    df_corners = np.array([df0 - delta_tilde_f_uncertainty,
                           df0 + delta_tilde_f_uncertainty])
    dk_corners = np.array([dk0 - delta_tilde_kappa_uncertainty,
                           dk0 + delta_tilde_kappa_uncertainty])

    df_grid, dk_grid = np.meshgrid(df_corners, dk_corners, indexing="xy")
    samples = np.column_stack([df_grid.ravel(), dk_grid.ravel()])   # shape (4, 2)


    # Step 4 - Calculate q for each sample
    q_vals = np.array([q_tilde(df, dk, kappa_tilde_c, phi) for df, dk in samples])

    # Step 5 - Take max absolute value of q
    max_q = np.max(np.abs(q_vals))

    # Step 6 - Return the minimum splitting estimate
    result = (3 / 2) * (2 ** (2 / 3)) * (max_q ** (1 / 3))
    return result

def calculate_max_response_tilde(strength, minimum_splitting):
    # Response is the derivative, a/ (2 *sqrt(eps))
    # But, we know that minimum splitting is y = a * sqrt(eps)
    # Solving, putting in y = minimum splitting, we get
    return strength ** 2 /  (2 * minimum_splitting)


# ---------------------------------------------------------------------
# Quick test driver (one frame): dist_2(TPD, instability) vs kappa_tilde_c
# ---------------------------------------------------------------------

def _phi_label(phi: float) -> str:
    if np.isclose(phi, np.pi):
        return "phi=pi"
    if np.isclose(phi, np.pi / 2.0):
        return "phi=pi/2"
    if np.isclose(phi, 0.0):
        return "phi=0"
    return "phi"


def test_distance_tpd_to_instability_plot(
    *,
    kappa_min: float = 0.0,
    kappa_max: float = 2.8,
    n_kappa: int = 300,
    N_grid: int = 241,
    lim: float = 5.0,
    left_tpd: bool = True,
) -> None:
    phi_set = (0.0, np.pi, np.pi / 2.0)
    kappas = np.linspace(float(kappa_min), float(kappa_max), int(n_kappa))

    plt.figure(figsize=(9, 5))
    for phi in phi_set:
        dists = np.array(
            [
                distance_between_instability_and_tpd(
                    phi,
                    kc,
                    left_tpd=left_tpd,
                    lim=lim,
                    N=N_grid,
                )
                for kc in kappas
            ],
            dtype=float,
        )
        plt.plot(kappas, dists, label=_phi_label(phi))

    plt.xlabel("kappa_tilde_c")
    plt.ylabel("dist_2(Instability-TPD)")
    plt.title("TPD to nearest instability boundary distance")
    plt.legend()
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    # ld = standard_tpd_locations(0, np.sqrt(8) - .00001)
    # ld2 = petermann_factor_at_tpd(ld, 0)
    # print(ld)
    # print(ld2)
    # ff2 = splitting_strength_at_tpd_arc(np.pi/2, 0.1)
    # ff3 = splitting_strength_at_tpd_arc_2(np.pi/2, 0.5, left_tpd=False)
    #
    # print('with arc length normalization, ff2 =', ff2)
    # print('with arc length normalization, ff3 =', ff3)
    #
    # fff = eigenvalue_splitting_strength(np.pi/2)
    # print(fff)
    #
    # delta_f_uncertainty = 1e-5
    # delta_tilde_f_uncertainty = delta_f_uncertainty / 1.045e-3
    #
    # ms = 1.045e-3 * min_split_over_J(0, 1.9906844595395117, delta_tilde_f_uncertainty, 0) * 1e9
    # print(ms)
    #
    # ms2 = 1.045e-3 * min_split_over_J(0, 0.7, delta_tilde_f_uncertainty, 0) * 1e9
    # print(ms2)
    #
    # ms3 = min_split_over_J(0, 1.9906844595395117, delta_tilde_f_uncertainty, 0)
    # print(ms3)
    #
    # ktc = 1.9906844595395117
    # pv = 0
    # delta_tilde_f_uncertainty = delta_f_uncertainty / 1.045e-3
    #
    # str = splitting_strength_at_tpd_arc(pv, ktc)
    # ms = min_split_over_J(pv, ktc, delta_tilde_f_uncertainty, 0)
    #
    # cmrt = calculate_max_response_tilde(ff3, ms3)
    # print(cmrt)
    #
    # test_distance_tpd_to_instability_plot()

    # Run the visualization test
    test_nuisance_scaling_visualization()

    # Plot comparison across phi values
    plot_nuisance_scaling_all_phi()