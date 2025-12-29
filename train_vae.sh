#!/bin/bash

# Activate virtual environment if it exists
if [ -d "venv" ]; then
    source venv/bin/activate
fi

echo "=========================================="
echo "VAE Training"
echo "=========================================="

# Train VAE for latent dimensions 1 and 4
echo ""
echo "Training VAE for latent dimensions 1 and 4..."
python -m src.train.train_vae \
  --data data/raw/ising_L32.npz \
  --latent-dims 1 4 \
  --epochs 5 --batch 128 --lr 1e-3 \
  --beta 1.0 \
  --outdir results/checkpoints/vae \
  --early-stopping 5

echo ""
echo "=========================================="
echo "VAE training complete!"
echo "=========================================="

