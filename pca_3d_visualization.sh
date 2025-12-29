#!/bin/bash

# Activate virtual environment if it exists
if [ -d "venv" ]; then
    source venv/bin/activate
fi

echo "=========================================="
echo "PCA 3D Visualization"
echo "=========================================="

# Run PCA 3D visualization
echo ""
echo "Creating PCA 3D visualization..."
python -m src.eval.pca_3d_visualization \
  --data data/raw/ising_L32.npz \
  --outdir results/figures

echo ""
echo "=========================================="
echo "PCA 3D visualization complete!"
echo "=========================================="

