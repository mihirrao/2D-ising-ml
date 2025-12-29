#!/bin/bash

# Activate virtual environment if it exists
if [ -d "venv" ]; then
    source venv/bin/activate
fi

echo "=========================================="
echo "CNN Post-Processing and Analysis"
echo "=========================================="

# 1. Grad-CAM analysis
echo ""
echo "Step 1: Grad-CAM analysis..."
python -m src.eval.gradcam_analysis \
  --checkpoint results/checkpoints/cnn/best.pt \
  --data data/raw/ising_L32.npz \
  --outdir results/figures \
  --n_temps 5 \
  --alpha 0.5 \

# 2. Linear probe analysis
echo ""
echo "Step 2: CNN linear probe analysis..."
python -m src.eval.cnn_linear_probe \
  --checkpoint results/checkpoints/cnn/best.pt \
  --data data/raw/ising_L32.npz \
  --outdir results/figures

echo ""
echo "=========================================="
echo "CNN post-processing complete!"
echo "=========================================="

