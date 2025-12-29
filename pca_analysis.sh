#!/bin/bash

# Activate virtual environment if it exists
if [ -d "venv" ]; then
    source venv/bin/activate
fi

echo "=========================================="
echo "PCA Analysis"
echo "=========================================="

python -m src.eval.pca_analysis \
  --data data/raw/ising_L32.npz \
  --outdir results/figures \
  --n_components 50 \

echo ""
echo "=========================================="
echo "PCA analysis complete!"
echo "=========================================="

