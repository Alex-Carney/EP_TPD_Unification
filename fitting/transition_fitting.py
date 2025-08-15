import numpy as np


def EP_location(phi, J):
    if np.isclose(phi, 0):
        return -2 * J
    if np.isclose(phi, np.pi):
        return 2 * J
    if np.isclose(phi, np.pi / 2):
        return -np.sqrt(2) * J
    raise ValueError(
        f"EP location not defined for phi = {phi}. "
        "Valid values are 0, pi, and pi/2."
    )

def TPD_location(phi, kappa_c, J):
    if np.isclose(phi, 0):
        return (kappa_c - np.sqrt(8 * J ** 2 - kappa_c ** 2)) / 2
    if np.isclose(phi, np.pi):
        return np.sqrt(4 * J ** 2 + kappa_c ** 2)
    if np.isclose(phi, np.pi / 2):
        kappa_tilde_c = kappa_c / J
        full_poly_coeffs = [2, -2 * kappa_tilde_c, kappa_tilde_c ** 2, 0, -4]
        roots = np.roots(full_poly_coeffs)
        print(roots)
        real_roots = roots[np.abs(roots.imag) < 1e-10].real
        return real_roots[real_roots < 0] * J
    raise ValueError(
        f"TPD location not defined for phi = {phi}. "
        "Valid values are 0, pi, and pi/2."
    )

def instability_location(phi, kappa_c, J):
    if np.isclose(phi, 0):
        if kappa_c/J <= 2:
            return kappa_c
        else:
            kcj = kappa_c / J
            return ( (kcj**2 + 4) / (2 *kcj) ) * J
    if np.isclose(phi, np.pi):
        return np.sqrt(4 * J ** 2 - kappa_c ** 2)
    if np.isclose(phi, np.pi / 2):
        return None
    raise ValueError(
        f"Instability location not defined for phi = {phi}. "
        "Valid values are 0, pi, and pi/2."
    )
