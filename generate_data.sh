python -m src.train.generate_data \
  --L 32 --Tmin 1.5 --Tmax 3.5 --T_step 0.25 \
  --Tc 2.269185314213022 --Tc_window 0.5 --Tc_step 0.1 \
  --n_therm 200 --n_samples 2000 --steps_between 5 \
  --method wolff --seed 0 \
  --out data/raw/ising_L32.npz