#!/bin/bash

# Activate virtual environment if it exists
if [ -d "venv" ]; then
    source venv/bin/activate
fi

echo "=========================================="
echo "VAE 3D Visualization"
echo "=========================================="

# Run VAE 3D visualization
echo ""
echo "Creating VAE 3D visualization..."
python -m src.eval.vae_3d_visualization \
  --checkpoint-dir results/checkpoints/vae \
  --data data/raw/ising_L32.npz \
  --outdir results/figures

echo ""
echo "=========================================="
echo "VAE 3D visualization complete!"
echo "=========================================="

