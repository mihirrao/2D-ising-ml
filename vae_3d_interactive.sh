#!/bin/bash

# Activate virtual environment if it exists
if [ -d "venv" ]; then
    source venv/bin/activate
fi

echo "=========================================="
echo "VAE 3D Interactive Visualization"
echo "=========================================="

# Run VAE 3D interactive visualization
echo ""
echo "Starting interactive 3D visualization..."
echo "  - Two windows will open (VAE Latent Dims and VAE PCA)"
echo "  - Use mouse to rotate, zoom, and pan"
echo "  - Close windows to exit"
echo ""
python -m src.eval.vae_3d_interactive \
  --checkpoint-dir results/checkpoints/vae \
  --data data/raw/ising_L32.npz

echo ""
echo "=========================================="
echo "Interactive visualization closed."
echo "=========================================="

