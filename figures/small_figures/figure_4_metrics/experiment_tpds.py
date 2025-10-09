from dataclasses import dataclass
from typing import Optional
import math
import numpy as np


@dataclass
class TPDLocation:
    Delta_tilde_f: Optional[float]
    Delta_tilde_kappa: Optional[float]
    Delta_tilde_f_unc: Optional[float] = None
    Delta_tilde_kappa_unc: Optional[float] = None

    @staticmethod
    def _fmt(val: Optional[float], unc: Optional[float]) -> str:
        """
        Format a value±uncertainty as 'value(d)', where d is the first
        significant digit of the uncertainty and matches the rounding
        of the value.  If unc is None or zero, just return the raw value.
        """
        if val is None:
            return "None"
        if not unc:                        # None or exactly 0
            return f"{val}"

        unc = abs(unc)

        # Exponent (base‑10) of the uncertainty’s first significant digit
        exp = int(math.floor(math.log10(unc)))
        first_digit = int(round(unc / (10 ** exp)))

        # Handle the rare case where rounding bumps 9.9… to 10
        if first_digit == 10:
            first_digit = 1
            exp += 1

        # Decimal places needed so value and uncertainty line up
        dec_places = max(-exp, 0)
        rounded_val = round(val, dec_places)

        if dec_places == 0:
            val_str = f"{int(round(rounded_val))}"
        else:
            val_str = f"{rounded_val:.{dec_places}f}"

        return f"{val_str}({first_digit})"

    def __str__(self):
        return (
            "TPDLocation("
            f"Delta_tilde_f={self._fmt(self.Delta_tilde_f, self.Delta_tilde_f_unc)}, "
            f"Delta_tilde_kappa={self._fmt(self.Delta_tilde_kappa, self.Delta_tilde_kappa_unc)})"
        )

def standard_ep_locations(phi, left_ep=True) -> TPDLocation:
    factor = -1 if left_ep else 1
    return TPDLocation(Delta_tilde_kappa =  factor * 2 * np.cos(phi/2), Delta_tilde_f = factor * 2 * np.sin(phi/2))

def standard_tpd_locations(phi, kappa_tilde_c, left_tpd=True, sigma_kappa_tilde_c=None) -> TPDLocation:
    if phi == 0:
        # check bounds - requires kappa_c^2 - 8 <= 0
        if kappa_tilde_c**2 - 8 > 0:
            print("Error: kappa_c^2 - 8 > 0")
            return TPDLocation(None, None)
        ker = 8 - kappa_tilde_c ** 2
        dtk_val = (kappa_tilde_c/2) - np.sqrt(ker) / 2 if left_tpd else (kappa_tilde_c/2) + np.sqrt(ker) / 2
        sgn = 1 if left_tpd else -1
        d_dkc = (kappa_tilde_c / 2 * np.sqrt(8 - kappa_tilde_c**2)) + sgn * 0.5

        return TPDLocation(Delta_tilde_f=0, Delta_tilde_kappa=dtk_val, Delta_tilde_f_unc=0, Delta_tilde_kappa_unc= np.abs(d_dkc * sigma_kappa_tilde_c) if sigma_kappa_tilde_c is not None else None)
    if phi == np.pi:
        # Always exists - no condition to check
        dtf_val = np.sqrt(kappa_tilde_c**2 + 4)
        d_dtf = kappa_tilde_c / np.sqrt(4 + kappa_tilde_c**2)
        return TPDLocation(Delta_tilde_f=dtf_val, Delta_tilde_kappa=0, Delta_tilde_kappa_unc=0, Delta_tilde_f_unc=np.abs(d_dtf * sigma_kappa_tilde_c) if sigma_kappa_tilde_c is not None else None)
    else:
        # For all other values, more complicated solution
        # Frist, find the roots of hte quartic
        possible_dks = np.roots([2, -2 * kappa_tilde_c, kappa_tilde_c **2 - 4 * np.cos(phi), 0, 4 * np.cos(phi)**2 - 4])
        # Throw out the complex roots - only get the real root with negative real part
        real_roots = possible_dks[np.abs(possible_dks.imag) < 1e-10].real
        x = min(real_roots) if left_tpd else max(real_roots)
        # Now, find the corresponding DF
        corresponding_df = -1 * (-kappa_tilde_c**2 * x + 2 * kappa_tilde_c * x**2 - 2 * x**3 + 4 * np.cos(phi) * x) / (2 * np.sin(phi))

        df_unc, dk_unc = None, None
        if sigma_kappa_tilde_c is not None:
            df_unc, dk_unc = _sigma_tpd_mc(phi, kappa_tilde_c, sigma_kappa_tilde_c, left_tpd)

        return TPDLocation(Delta_tilde_f=corresponding_df, Delta_tilde_kappa=x, Delta_tilde_f_unc=df_unc, Delta_tilde_kappa_unc=dk_unc)

def _sigma_tpd_mc(phi, kc, sigma_kc, left_tpd, N: int = 4000):
    rng = np.random.default_rng()
    kc_samples = rng.normal(kc, sigma_kc, N)
    dfs, dks = [], []
    for k in kc_samples:
        dfs.append(standard_tpd_locations(phi, k, left_tpd).Delta_tilde_f)
        dks.append(standard_tpd_locations(phi, k, left_tpd).Delta_tilde_kappa)
    dfs = np.asarray(dfs)
    dks = np.asarray(dks)
    return float(np.std(dfs, ddof=1)) if dfs.size else None, float(np.std(dks, ddof=1)) if dks.size else None


if __name__ == "__main__":
    ld = standard_tpd_locations(np.pi/2, 2)
    print(ld)
