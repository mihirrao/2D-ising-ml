import argparse
import numpy as np
from tqdm import tqdm
from src.ising.sim import sample_configs
from src.ising.io import save_npz

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--L", type=int, default=32)
    p.add_argument("--Tmin", type=float, default=1.5)
    p.add_argument("--Tmax", type=float, default=3.5)
    p.add_argument("--T_step", type=float, default=0.25,
                   help='Temperature increment for base temperatures')
    p.add_argument("--n_therm", type=int, default=200)
    p.add_argument("--n_samples", type=int, default=2000)
    p.add_argument("--steps_between", type=int, default=5)
    p.add_argument("--method", type=str, default="wolff", choices=["wolff", "metropolis"])
    p.add_argument("--out", type=str, default="data/raw/ising_L32.npz")
    p.add_argument("--Tc", type=float, default=2.269185314213022,
                   help='Critical temperature')
    p.add_argument("--Tc_window", type=float, default=0.5,
                   help='Window around Tc to add extra temperatures')
    p.add_argument("--Tc_step", type=float, default=0.1,
                   help='Temperature increment for temperatures near Tc')
    args = p.parse_args()

    # Generate base temperatures with clean increments
    Ts_base = []
    T = args.Tmin
    while T <= args.Tmax:
        Ts_base.append(round(T, 2))  # Round to 2 decimal places for clean values
        T += args.T_step
    
    Ts_base = np.array(Ts_base, dtype=np.float32)
    
    # Add extra temperatures near Tc with clean increments
    Tc = args.Tc
    Tc_min = max(args.Tmin, Tc - args.Tc_window)
    Tc_max = min(args.Tmax, Tc + args.Tc_window)
    
    # Generate clean temperatures around Tc
    Ts_near_Tc = []
    # Round Tc_min and Tc_max to nearest 0.1
    Tc_min_clean = round(Tc_min, 1)
    Tc_max_clean = round(Tc_max, 1)
    
    T = Tc_min_clean
    while T <= Tc_max_clean:
        # Only add if not already in base temperatures
        if not np.any(np.abs(Ts_base - T) < 0.01):
            Ts_near_Tc.append(round(T, 2))
        T += args.Tc_step
    
    Ts_near_Tc = np.array(Ts_near_Tc, dtype=np.float32)
    
    # Combine and remove duplicates, then sort
    Ts = np.unique(np.concatenate([Ts_base, Ts_near_Tc])).astype(np.float32)
    Ts = np.sort(Ts)
    
    print(f"Generated {len(Ts)} temperatures ({len(Ts_base)} base + {len(Ts_near_Tc)} near Tc)")
    print(f"Temperature range: [{Ts.min():.2f}, {Ts.max():.2f}]")
    print(f"Base temperatures: {Ts_base}")
    print(f"Temperatures near Tc ({Tc:.3f}): {Ts_near_Tc}")

    all_X, all_E, all_M, all_T = [], [], [], []

    for idx, T in enumerate(tqdm(Ts, desc="Temperatures")):
        X, E, M = sample_configs(
            L=args.L, T=float(T),
            n_therm=args.n_therm,
            n_samples=args.n_samples,
            steps_between=args.steps_between,
            method=args.method,
        )
        all_X.append(X)
        all_E.append(E)
        all_M.append(M)
        all_T.append(np.full((args.n_samples,), T, dtype=np.float32))

    X = np.concatenate(all_X, axis=0)          # (N, L, L)
    E = np.concatenate(all_E, axis=0)          # (N,)
    M = np.concatenate(all_M, axis=0)          # (N,)
    T = np.concatenate(all_T, axis=0)          # (N,)

    # derived labels: phase label using known Tc ≈ 2.269 for J=1, kB=1 (for convenience)
    y_phase = (T < args.Tc).astype(np.int64)

    # Store also per-spin values commonly used
    Nspin = args.L * args.L
    m = (M / Nspin).astype(np.float32)
    e = (E / Nspin).astype(np.float32)

    save_npz(args.out, X=X, E=E, M=M, T=T, y_phase=y_phase, m=m, e=e, L=args.L, Tc=args.Tc)
    print(f"Saved: {args.out}")
    print(f"N={X.shape[0]} configs, L={args.L}, T in [{T.min():.3f}, {T.max():.3f}]")

if __name__ == "__main__":
    main()