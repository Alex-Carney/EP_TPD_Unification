import numpy as np

def phase_peak_theory_color_map(phi):
    if phi == 0.0:
        return "darkblue"
    elif np.isclose(phi, np.pi):
        return "purple"
    elif np.isclose(phi, np.pi/2):
        return "darkgreen"
    else:
        print(f'cannot get color for phi = {phi}')
