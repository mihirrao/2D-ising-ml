import numpy as np

def binder_cumulant(m: np.ndarray) -> float:
    m2 = np.mean(m**2)
    m4 = np.mean(m**4)
    return float(1.0 - m4 / (3.0 * (m2**2 + 1e-12)))

def susceptibility(m: np.ndarray, T: float, N: int) -> float:
    mean = np.mean(m)
    mean2 = np.mean(m**2)
    return float((N / T) * (mean2 - mean**2))

def heat_capacity(E: np.ndarray, T: float) -> float:
    mean = np.mean(E)
    mean2 = np.mean(E**2)
    return float((mean2 - mean**2) / (T**2 + 1e-12))