#!/bin/bash

# Activate virtual environment if it exists
if [ -d "venv" ]; then
    source venv/bin/activate
fi

python -m src.utils.visualize_dataset \
  --data data/raw/ising_L32.npz \
  --outdir results/figures \
  --n_configs 5 \

