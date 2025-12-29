#!/bin/bash

# Activate virtual environment if it exists
if [ -d "venv" ]; then
    source venv/bin/activate
fi

echo "=========================================="
echo "CNN Linear Probe 3D Visualization"
echo "=========================================="

# Run CNN linear probe 3D visualization
echo ""
echo "Creating CNN linear probe 3D visualization..."
python -m src.eval.cnn_linear_probe_3d_visualization \
  --checkpoint results/checkpoints/cnn/best.pt \
  --data data/raw/ising_L32.npz \
  --outdir results/figures

echo ""
echo "=========================================="
echo "CNN linear probe 3D visualization complete!"
echo "=========================================="

