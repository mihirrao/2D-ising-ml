import numpy as np

def binder_cumulant(m: np.ndarray) -> float:
    # m is magnetization per spin or total magnetization (consistent scaling cancels)
    m2 = np.mean(m**2)
    m4 = np.mean(m**4)
    return float(1.0 - m4 / (3.0 * (m2**2 + 1e-12)))

def susceptibility(m: np.ndarray, T: float, N: int) -> float:
    # chi = (1/T) * N * ( <m^2> - <m>^2 ) where m is magnetization per spin
    mean = np.mean(m)
    mean2 = np.mean(m**2)
    return float((N / T) * (mean2 - mean**2))

def heat_capacity(E: np.ndarray, T: float) -> float:
    # C = ( <E^2> - <E>^2 ) / T^2
    mean = np.mean(E)
    mean2 = np.mean(E**2)
    return float((mean2 - mean**2) / (T**2 + 1e-12))