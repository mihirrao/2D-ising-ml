#!/bin/bash

# Activate virtual environment if it exists
if [ -d "venv" ]; then
    source venv/bin/activate
fi

echo "=========================================="
echo "VAE Post-Processing"
echo "=========================================="

# Remove all existing VAE figures
echo ""
echo "Removing existing VAE figures..."
rm -f results/figures/vae_*.png

# Run post-processing
echo ""
echo "Running VAE post-processing..."
python -m src.eval.vae_postprocess \
  --checkpoint-dir results/checkpoints/vae \
  --data data/raw/ising_L32.npz \
  --outdir results/figures

echo ""
echo "=========================================="
echo "VAE post-processing complete!"
echo "=========================================="

