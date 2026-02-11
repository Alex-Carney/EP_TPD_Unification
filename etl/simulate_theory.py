import numpy as np
import matplotlib.pyplot as plt

from etl.settings import ETLSettings
import fitting.peak_fitting as peaks
import fitting.transition_fitting as tfi


def simulate_theory(phi, df, dk, J_avg, fc_avg, kc_avg, settings: ETLSettings):
    """
    Returns theory arrays for the sweep implied by phi.

    Important coupled constraints enforced per-point:
      Delta_f = f_c - f_y, with f_c locked -> f_y = f_c - Delta_f
      Delta_kappa = (kappa_c - kappa_y)/2, with kappa_c locked -> kappa_y = kappa_c - 2*Delta_kappa
    """
    theory_df = []
    theory_dk = []
    theory_fy = []
    theory_ky = []
    theory_nu_plus = []
    theory_nu_minus = []

    if np.isclose(phi, np.pi):
        # DF is independent, DK locked to 0
        df_set = np.linspace(float(np.min(df)), float(np.max(df)), int(settings.THEORY_SIZE))
        dk_set = np.zeros_like(df_set)

    elif np.isclose(phi, 0.0):
        # DK is independent, DF locked to 0
        dk_set = np.linspace(float(np.min(dk)), float(np.max(dk)), int(settings.THEORY_SIZE))
        df_set = np.zeros_like(dk_set)

    elif np.isclose(phi, np.pi / 2):
        # Coupled sweep used in your existing code
        dk_set = np.linspace(float(np.min(dk)), float(np.max(dk)), int(settings.THEORY_SIZE))
        df_set = (2.0 * (J_avg ** 2)) / dk_set
    else:
        raise ValueError("simulate_theory: unsupported phi value")

    for df_val, dk_val in zip(df_set, dk_set):
        # Enforce coupled constraints
        fy_val = fc_avg - df_val
        ky_val = kc_avg - 2.0 * dk_val

        # Peak locations (these looked fine already)
        theory_peaks = peaks.peak_location(J_avg, fc_avg, kc_avg, df_val, dk_val, phi)
        if len(theory_peaks) == 2:
            hi = max(theory_peaks)
            lo = min(theory_peaks)
            theory_nu_plus.append(hi)
            theory_nu_minus.append(lo)
        else:
            theory_nu_plus.append(float("nan"))
            theory_nu_minus.append(theory_peaks[0])

        theory_df.append(df_val)
        theory_dk.append(dk_val)
        theory_fy.append(fy_val)
        theory_ky.append(ky_val)

    return (
        np.asarray(theory_df, dtype=float),
        np.asarray(theory_dk, dtype=float),
        np.asarray(theory_nu_plus, dtype=float),
        np.asarray(theory_nu_minus, dtype=float),
        np.asarray(theory_fy, dtype=float),
        np.asarray(theory_ky, dtype=float),
    )


def simulate_theory_noise_function(
    kappa_y,            # float or np.ndarray
    n_th,               # float or np.ndarray
    f_c: float,
    f_d,                # float or np.ndarray (here: peak location array)
    kappa_c: float,
    f_y,                # float or np.ndarray
    J: float,
    phi: float,
):
    """
    Vectorized numeric translation of the MATLAB symbolic expression.

    Supports arrays for f_d, f_y, kappa_y, n_th (they will broadcast).
    """
    i = 1j

    f_d_arr = np.asarray(f_d, dtype=float)
    f_y_arr = np.asarray(f_y, dtype=float)
    kappa_y_arr = np.asarray(kappa_y, dtype=float)
    n_th_arr = np.asarray(n_th, dtype=float)

    f_d_b, f_y_b, kappa_y_b, n_th_b = np.broadcast_arrays(f_d_arr, f_y_arr, kappa_y_arr, n_th_arr)

    den = (
        4.0 * f_c * f_d_b * i
        - 4.0 * f_c * f_y_b * i
        + 4.0 * f_d_b * f_y_b * i
        - 2.0 * f_c * kappa_y_b
        + 2.0 * f_d_b * kappa_c
        + 2.0 * f_d_b * kappa_y_b
        - 2.0 * f_y_b * kappa_c
        + kappa_c * kappa_y_b * i
        + 4.0 * (J ** 2) * np.exp(phi * i) * i
        - 4.0 * (f_d_b ** 2) * i
    )

    den_abs_sq = (np.abs(den) ** 2)

    term1 = (4.0 * kappa_y_b * ((2.0 * f_c - 2.0 * f_d_b) ** 2 + kappa_c ** 2)) / den_abs_sq
    term2 = (16.0 * (J ** 2) * kappa_c) / den_abs_sq

    out = kappa_y_b * n_th_b * (term1 + term2)

    # return scalar if truly scalar input
    if np.isscalar(f_d) and np.isscalar(f_y) and np.isscalar(kappa_y) and np.isscalar(n_th):
        return float(out)
    return out


def compute_theory_noise_from_peaks(
    theory_nu_plus,
    theory_nu_minus,
    *,
    kappa_y,    # float or np.ndarray
    n_th,       # float or np.ndarray
    f_c: float,
    kappa_c: float,
    f_y,        # float or np.ndarray
    J: float,
    phi: float,
):
    nu_plus = np.asarray(theory_nu_plus, dtype=float)
    nu_minus = np.asarray(theory_nu_minus, dtype=float)

    noise_plus = simulate_theory_noise_function(
        kappa_y=kappa_y,
        n_th=n_th,
        f_c=f_c,
        f_d=nu_plus,
        kappa_c=kappa_c,
        f_y=f_y,
        J=J,
        phi=phi,
    )

    noise_minus = simulate_theory_noise_function(
        kappa_y=kappa_y,
        n_th=n_th,
        f_c=f_c,
        f_d=nu_minus,
        kappa_c=kappa_c,
        f_y=f_y,
        J=J,
        phi=phi,
    )

    return noise_plus, noise_minus


def _overlay_transition_lines(ax, ep_loc, tpd_loc, instab_loc):
    if ep_loc is not None:
        ax.axvline(ep_loc, color="red", linestyle="--", label="EP Location")
    if tpd_loc is not None:
        ax.axvline(tpd_loc, color="cyan", linestyle="--", label="TPD Location")
    if instab_loc is not None:
        ax.axvline(instab_loc, color="lime", linestyle="--", label="Instability Location")

def _robust_ylim(y_list, *, low_q=1.0, high_q=99.0, pad_frac=0.08, min_span=1e-30):
    """
    Compute robust y-limits from quantiles across one or more series.

    - Removes NaN/inf
    - Uses [low_q, high_q] percentiles
    - Adds a small padding
    """
    ys = []
    for y in y_list:
        if y is None:
            continue
        arr = np.asarray(y, dtype=float).ravel()
        arr = arr[np.isfinite(arr)]
        if arr.size:
            ys.append(arr)

    if not ys:
        return None

    data = np.concatenate(ys)
    lo = np.percentile(data, low_q)
    hi = np.percentile(data, high_q)

    if not np.isfinite(lo) or not np.isfinite(hi):
        return None

    if hi - lo < min_span:
        mid = 0.5 * (hi + lo)
        lo = mid - 0.5 * min_span
        hi = mid + 0.5 * min_span

    pad = pad_frac * (hi - lo)
    return (lo - pad, hi + pad)


def _clip_for_plot(y, *, clip_q=99.5):
    """
    Optional: clip only for display so huge divergences do not blow up autoscaling.
    Keeps NaNs as NaN.
    """
    arr = np.asarray(y, dtype=float)
    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        return arr
    cap = np.percentile(finite, clip_q)
    out = arr.copy()
    mask = np.isfinite(out)
    out[mask] = np.minimum(out[mask], cap)
    return out

def run_theory_and_noise_for_phi(
    *,
    phi: float,
    df: np.ndarray,
    dk: np.ndarray,
    J_avg: float,
    fc_avg: float,
    kc_avg: float,
    n_th: float,
    settings: ETLSettings,
):
    # Transition overlay locations are expressed in the sweep coordinate for that phi
    ep_loc = tfi.EP_location(phi, J_avg)
    tpd_loc = tfi.TPD_location(phi, kc_avg, J_avg)
    instab_loc = tfi.instability_location(phi, kc_avg, J_avg)

    theory_df, theory_dk, nu_plus, nu_minus, theory_fy, theory_ky = simulate_theory(
        phi, df, dk, J_avg, fc_avg, kc_avg, settings
    )

    noise_plus, noise_minus = compute_theory_noise_from_peaks(
        nu_plus,
        nu_minus,
        kappa_y=theory_ky,
        n_th=n_th,
        f_c=fc_avg,
        kappa_c=kc_avg,
        f_y=theory_fy,
        J=J_avg,
        phi=phi,
    )

    return {
        "phi": phi,
        "ep_loc": ep_loc,
        "tpd_loc": tpd_loc,
        "instab_loc": instab_loc,
        "theory_df": theory_df,
        "theory_dk": theory_dk,
        "theory_fy": theory_fy,
        "theory_ky": theory_ky,
        "nu_plus": nu_plus,
        "nu_minus": nu_minus,
        "noise_plus": noise_plus,
        "noise_minus": noise_minus,
    }


def plot_peaks_and_noise(res, *, x_key: str, x_label: str, title: str):
    x = np.asarray(res[x_key], dtype=float)

    fig, (ax0, ax1) = plt.subplots(2, 1, sharex=True, figsize=(10, 8))

    ax0.plot(x, res["nu_plus"], label="Nu Plus")
    ax0.plot(x, res["nu_minus"], label="Nu Minus")
    ax0.set_ylabel("Peak Frequencies (Nu)")
    ax0.set_title(title)
    _overlay_transition_lines(ax0, res["ep_loc"], res["tpd_loc"], res["instab_loc"])
    ax0.legend()

    ax1.plot(x, abs(res["noise_plus"]), label="Noise Plus")
    ax1.plot(x, abs(res["noise_minus"]), label="Noise Minus")
    ax1.set_xlabel(x_label)
    ax1.set_ylabel("Noise")
    _overlay_transition_lines(ax1, res["ep_loc"], res["tpd_loc"], res["instab_loc"])
    ax1.legend()

    ax1.set_ylim([0, 10])

    if x_key == "theory_df":
        # Add a plot of Delta_f vs f_c - Delta_f/2
        ys = fc_avg - 0.5 * x
        ax0.plot(x, ys, color="gray", linestyle=":", label="f_c - Delta_f/2")
        ax0.legend()


    plt.tight_layout()
    plt.show()

def noise_at_tpd_for_kc(phi_val: float, kc_val: float, *, J: float, fc: float, n_th_val: float):
    """
    Returns (noise_plus, noise_minus) evaluated exactly at the TPD for a given kappa_c.

    Enforces your coupled constraints:
      Delta_f = f_c - f_y -> f_y = f_c - Delta_f
      Delta_kappa = (kappa_c - kappa_y)/2 -> kappa_y = kappa_c - 2*Delta_kappa
    """
    tpd_loc = tfi.TPD_location(phi_val, kc_val, J)

    if tpd_loc is None or not np.isfinite(tpd_loc):
        return float("nan"), float("nan")

    # Determine which coordinate the TPD lives in for this phi
    if np.isclose(phi_val, np.pi):
        # Along DF with DK = 0
        df_val = float(tpd_loc)
        dk_val = 0.0
    elif np.isclose(phi_val, 0.0):
        # Along DK with DF = 0
        df_val = 0.0
        dk_val = float(tpd_loc)
    else:
        # Not requested for this plot
        return float("nan"), float("nan")

    # Enforce coupled constraints at this point
    fy_val = fc - df_val
    ky_val = kc_val - 2.0 * dk_val

    # Peak locations at the TPD point
    peak_list = peaks.peak_location(J, fc, kc_val, df_val, dk_val, phi_val)
    if len(peak_list) == 2:
        nu_plus = float(max(peak_list))
        nu_minus = float(min(peak_list))
    else:
        # If only one peak reported, treat plus as nan and minus as that peak (same convention as above)
        nu_plus = float("nan")
        nu_minus = float(peak_list[0])

    # Evaluate noise at each peak frequency (f_d is the peak location)
    noise_plus = simulate_theory_noise_function(
        kappa_y=ky_val,
        n_th=n_th_val,
        f_c=fc,
        f_d=nu_plus,
        kappa_c=kc_val,
        f_y=fy_val,
        J=J,
        phi=phi_val,
    )

    noise_minus = simulate_theory_noise_function(
        kappa_y=ky_val,
        n_th=n_th_val,
        f_c=fc,
        f_d=nu_minus,
        kappa_c=kc_val,
        f_y=fy_val,
        J=J,
        phi=phi_val,
    )

    return noise_plus, noise_minus


if __name__ == "__main__":
    settings = ETLSettings()

    # Replace these with your actual averaged parameters
    J_avg = 1.0e6
    fc_avg = 6.0e9
    kc_avg = .67e6

    # Thermal occupancy (kept constant here)
    n_th = 1.0

    # These arrays are only used to set min/max for the theory grids
    df_data = np.linspace(-5e6, 5e6, 10)
    dk_data = np.linspace(-3e6, 3e6, 10)

    # Sweep A: phi = pi, Delta_kappa locked to 0, sweep Delta_f (implies f_y changes)
    res_df = run_theory_and_noise_for_phi(
        phi=float(np.pi),
        df=df_data,
        dk=np.zeros_like(df_data),
        J_avg=J_avg,
        fc_avg=fc_avg,
        kc_avg=kc_avg,
        n_th=n_th,
        settings=settings,
    )

    # Sweep B: phi = 0, Delta_f locked to 0, sweep Delta_kappa (implies kappa_y changes)
    res_dk = run_theory_and_noise_for_phi(
        phi=0.0,
        df=np.zeros_like(dk_data),
        dk=dk_data,
        J_avg=J_avg,
        fc_avg=fc_avg,
        kc_avg=kc_avg,
        n_th=n_th,
        settings=settings,
    )

    # Plot DF sweep (x axis is Delta_f)
    plot_peaks_and_noise(
        res_df,
        x_key="theory_df",
        x_label="Delta f (DF)",
        title="Phi = pi, Delta_kappa = 0, sweep Delta_f (f_y = f_c - Delta_f)",
    )

    # Plot DK sweep (x axis is Delta_kappa)
    plot_peaks_and_noise(
        res_dk,
        x_key="theory_dk",
        x_label="Delta kappa (DK)",
        title="Phi = 0, Delta_f = 0, sweep Delta_kappa (kappa_y = kappa_c - 2*Delta_kappa)",
    )

    # NEW PLOT

    # Choose a kappa_c sweep for this plot (use your real range)
    kc_sweep = np.linspace(0.5e6, 2.83e6, 10000)

    noise_tpd_pi_plus = np.empty_like(kc_sweep, dtype=float)
    noise_tpd_pi_minus = np.empty_like(kc_sweep, dtype=float)
    noise_tpd_0_plus = np.empty_like(kc_sweep, dtype=float)
    noise_tpd_0_minus = np.empty_like(kc_sweep, dtype=float)

    for idx, kc_val in enumerate(kc_sweep):
        a_plus, a_minus = noise_at_tpd_for_kc(np.pi, kc_val, J=J_avg, fc=fc_avg, n_th_val=n_th)
        b_plus, b_minus = noise_at_tpd_for_kc(0.0, kc_val, J=J_avg, fc=fc_avg, n_th_val=n_th)

        noise_tpd_pi_plus[idx] = float(a_plus) if np.isfinite(a_plus) else float("nan")
        noise_tpd_pi_minus[idx] = float(a_minus) if np.isfinite(a_minus) else float("nan")
        noise_tpd_0_plus[idx] = float(b_plus) if np.isfinite(b_plus) else float("nan")
        noise_tpd_0_minus[idx] = float(b_minus) if np.isfinite(b_minus) else float("nan")

    # Plot: two subplots, shared x (kappa_c), no vertical markers
    fig, (ax_top, ax_bot) = plt.subplots(2, 1, sharex=True, figsize=(10, 8))

    # Top: phi = pi
    ax_top.plot(kc_sweep, np.abs(noise_tpd_pi_plus), label="TPD Noise Plus")
    ax_top.plot(kc_sweep, np.abs(noise_tpd_pi_minus), label="TPD Noise Minus")
    ax_top.set_ylabel("Noise at TPD")
    ax_top.set_title("Noise at the TPD vs kappa_c (top: phi = pi)")
    ax_top.legend()

    # Bottom: phi = 0
    ax_bot.plot(kc_sweep, np.abs(noise_tpd_0_plus), label="TPD Noise Plus")
    ax_bot.plot(kc_sweep, np.abs(noise_tpd_0_minus), label="TPD Noise Minus")
    ax_bot.set_xlabel("kappa_c")
    ax_bot.set_ylabel("Noise at TPD")
    ax_bot.set_title("Noise at the TPD vs kappa_c (bottom: phi = 0)")
    ax_bot.legend()

    # Robust y-lims so divergences do not destroy visibility
    ylim_top = _robust_ylim([np.abs(noise_tpd_pi_plus), np.abs(noise_tpd_pi_minus)], low_q=1.0, high_q=99.0)
    if ylim_top is not None:
        ax_top.set_ylim(0.0, max(0.0, ylim_top[1]))

    ylim_bot = _robust_ylim([np.abs(noise_tpd_0_plus), np.abs(noise_tpd_0_minus)], low_q=1.0, high_q=99.0)
    if ylim_bot is not None:
        ax_bot.set_ylim(0.0, max(0.0, ylim_bot[1]))

    plt.tight_layout()
    plt.show()