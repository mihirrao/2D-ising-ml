#!/bin/bash

# Activate virtual environment if it exists
if [ -d "venv" ]; then
    source venv/bin/activate
fi

echo "=========================================="
echo "CNN Linear Probe 3D Interactive Visualization"
echo "=========================================="

# Run CNN linear probe 3D interactive visualization
echo ""
echo "Starting interactive 3D visualization..."
echo "  - Window will open showing CNN PCA space"
echo "  - Use mouse to rotate, zoom, and pan"
echo "  - Close window to exit"
echo ""
python -m src.eval.cnn_linear_probe_3d_interactive \
  --checkpoint results/checkpoints/cnn/best.pt \
  --data data/raw/ising_L32.npz

echo ""
echo "=========================================="
echo "Interactive visualization closed."
echo "=========================================="

