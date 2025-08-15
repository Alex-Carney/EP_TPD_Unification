import numpy as np

import matplotlib.pyplot as plt

from figures.figure_3_experiment.experiment_tpds import standard_tpd_locations, standard_ep_locations, TPDLocation
from fitting.peak_fitting import peak_location

from scipy.optimize import curve_fit

# def sqrt_model(dk, a, dk0):
#     return a * np.sqrt(dk - dk0)

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

if __name__ == "__main__":
    ld = standard_tpd_locations(0, np.sqrt(8) - .00001)
    ld2 = petermann_factor_at_tpd(ld, 0)
    print(ld)
    print(ld2)
    ff2 = splitting_strength_at_tpd_arc(np.pi/2, 0.1)
    ff3 = splitting_strength_at_tpd_arc_2(np.pi/2, 0.5, left_tpd=False)

    print('with arc length normalization, ff2 =', ff2)
    print('with arc length normalization, ff3 =', ff3)

    fff = eigenvalue_splitting_strength(np.pi/2)
    print(fff)

    delta_f_uncertainty = 1e-5
    delta_tilde_f_uncertainty = delta_f_uncertainty / 1.045e-3

    ms = 1.045e-3 * min_split_over_J(0, 1.9906844595395117, delta_tilde_f_uncertainty, 0) * 1e9
    print(ms)

    ms2 = 1.045e-3 * min_split_over_J(0, 0.7, delta_tilde_f_uncertainty, 0) * 1e9
    print(ms2)

    ms3 = min_split_over_J(0, 1.9906844595395117, delta_tilde_f_uncertainty, 0)
    print(ms3)

    ktc = 1.9906844595395117
    pv = 0
    delta_tilde_f_uncertainty = delta_f_uncertainty / 1.045e-3

    str = splitting_strength_at_tpd_arc(pv, ktc)
    ms = min_split_over_J(pv, ktc, delta_tilde_f_uncertainty, 0)

    cmrt = calculate_max_response_tilde(ff3, ms3)
    print(cmrt)
