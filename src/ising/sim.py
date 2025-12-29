import numpy as np

def _neighbors_sum(spins: np.ndarray) -> np.ndarray:
    # periodic boundary conditions
    return (
        np.roll(spins, 1, axis=0) + np.roll(spins, -1, axis=0) +
        np.roll(spins, 1, axis=1) + np.roll(spins, -1, axis=1)
    )

def energy(spins: np.ndarray, J: float = 1.0, h: float = 0.0) -> float:
    # E = -J sum_{<ij>} s_i s_j - h sum_i s_i
    # using neighbor sum counts each bond twice -> divide by 2
    nn = _neighbors_sum(spins)
    E = -J * 0.5 * np.sum(spins * nn) - h * np.sum(spins)
    return float(E)

def magnetization(spins: np.ndarray) -> float:
    return float(np.sum(spins))

def metropolis_sweep(spins: np.ndarray, beta: float, J: float = 1.0, h: float = 0.0, rng=None) -> None:
    """
    One Metropolis sweep: attempt N = L*L single-spin flips.
    In-place updates.
    """
    if rng is None:
        rng = np.random.default_rng()
    L = spins.shape[0]
    N = L * L

    for _ in range(N):
        i = rng.integers(0, L)
        j = rng.integers(0, L)
        s = spins[i, j]

        # local field from neighbors + external h
        nb = spins[(i+1) % L, j] + spins[(i-1) % L, j] + spins[i, (j+1) % L] + spins[i, (j-1) % L]
        dE = 2.0 * s * (J * nb + h)

        if dE <= 0.0 or rng.random() < np.exp(-beta * dE):
            spins[i, j] = -s

def wolff_update(spins: np.ndarray, beta: float, J: float = 1.0, rng=None) -> int:
    """
    Wolff cluster update for h=0 (standard).
    Returns cluster size.
    """
    if rng is None:
        rng = np.random.default_rng()
    L = spins.shape[0]

    p_add = 1.0 - np.exp(-2.0 * beta * J)

    i0 = rng.integers(0, L)
    j0 = rng.integers(0, L)
    s0 = spins[i0, j0]

    stack = [(i0, j0)]
    cluster = np.zeros_like(spins, dtype=bool)
    cluster[i0, j0] = True

    while stack:
        i, j = stack.pop()
        # neighbors with periodic BC
        nbs = [((i+1) % L, j), ((i-1) % L, j), (i, (j+1) % L), (i, (j-1) % L)]
        for ni, nj in nbs:
            if not cluster[ni, nj] and spins[ni, nj] == s0:
                if rng.random() < p_add:
                    cluster[ni, nj] = True
                    stack.append((ni, nj))

    spins[cluster] *= -1
    return int(cluster.sum())

def make_lattice(L: int, rng=None) -> np.ndarray:
    if rng is None:
        rng = np.random.default_rng()
    spins = rng.choice([-1, 1], size=(L, L), replace=True).astype(np.int8)
    return spins

def sample_configs(
    L: int,
    T: float,
    n_therm: int,
    n_samples: int,
    steps_between: int,
    method: str = "wolff",
    J: float = 1.0,
    h: float = 0.0,
):
    rng = np.random.default_rng()
    beta = 1.0 / T
    spins = make_lattice(L, rng=rng)

    # thermalize
    if method == "metropolis":
        for _ in range(n_therm):
            metropolis_sweep(spins, beta=beta, J=J, h=h, rng=rng)
    elif method == "wolff":
        if h != 0.0:
            raise ValueError("Wolff implementation here assumes h=0.")
        for _ in range(n_therm):
            wolff_update(spins, beta=beta, J=J, rng=rng)
    else:
        raise ValueError(f"Unknown method: {method}")

    configs = np.zeros((n_samples, L, L), dtype=np.int8)
    Es = np.zeros(n_samples, dtype=np.float32)
    Ms = np.zeros(n_samples, dtype=np.float32)

    for k in range(n_samples):
        for _ in range(steps_between):
            if method == "metropolis":
                metropolis_sweep(spins, beta=beta, J=J, h=h, rng=rng)
            else:
                wolff_update(spins, beta=beta, J=J, rng=rng)

        configs[k] = spins
        Es[k] = energy(spins, J=J, h=h)
        Ms[k] = magnetization(spins)

    return configs, Es, Ms