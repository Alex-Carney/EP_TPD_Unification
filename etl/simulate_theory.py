import numpy as np
from etl.settings import ETLSettings
import peak_fitting as peaks


def simulate_theory(phi, df, dk, J_avg, fc_avg, kc_avg, settings: ETLSettings):
    # Add theory results
    theory_df, theory_dk, theory_results_nu_plus, theory_results_nu_minus = [], [], [], []
    if np.isclose(phi, np.pi):
        # DF is the independent variable
        df_theory_set = np.linspace(min(df), max(df), settings.THEORY_SIZE)
        for df_theoretical_val in df_theory_set:
            # ----- "Theory" plot, which uses all average parameters  ----------------
            theory_peaks = peaks.peak_location(
                J_avg, fc_avg, kc_avg, df_theoretical_val, 0, phi
            )
            if len(theory_peaks) == 2:
                hi, lo = max(theory_peaks), min(theory_peaks)
                theory_results_nu_plus.append(hi)
                theory_results_nu_minus.append(lo)
            else:
                theory_results_nu_plus.append(float('nan'))
                theory_results_nu_minus.append(theory_peaks[0])

            theory_df.append(df_theoretical_val)
            theory_dk.append(0)  # dk is zero for phi = pi
    if np.isclose(phi, 0):
        # DK is the independent variable
        dk_theory_set = np.linspace(min(dk), max(dk), settings.THEORY_SIZE)
        for dk_theoretical_val in dk_theory_set:
            # ----- "Theory" plot, which uses all average parameters  ----------------
            theory_peaks = peaks.peak_location(
                J_avg, fc_avg, kc_avg, 0, dk_theoretical_val, phi
            )
            if len(theory_peaks) == 2:
                hi, lo = max(theory_peaks), min(theory_peaks)
                theory_results_nu_plus.append(hi)
                theory_results_nu_minus.append(lo)
            else:
                theory_results_nu_plus.append(float('nan'))
                theory_results_nu_minus.append(theory_peaks[0])

            theory_dk.append(dk_theoretical_val)
            theory_df.append(0)

    if np.isclose(phi, np.pi / 2):
        dk_theory_set = np.linspace(min(dk), max(dk), settings.THEORY_SIZE)
        df_theory_set = (2 * J_avg ** 2) / dk_theory_set

        for dk_theoretical_val, df_theoretical_val in zip(dk_theory_set, df_theory_set):
            # ----- "Theory" plot, which uses all average parametrs ----------------
            theory_peaks = peaks.peak_location(
                J_avg, fc_avg, kc_avg, df_theoretical_val, dk_theoretical_val, phi
            )
            if len(theory_peaks) == 2:
                hi, lo = max(theory_peaks), min(theory_peaks)
                theory_results_nu_plus.append(hi)
                theory_results_nu_minus.append(lo)
            else:
                theory_results_nu_plus.append(float('nan'))
                theory_results_nu_minus.append(theory_peaks[0])

            theory_df.append(df_theoretical_val)
            theory_dk.append(dk_theoretical_val)

    return theory_df, theory_dk, theory_results_nu_plus, theory_results_nu_minus




