#!/bin/bash

# Activate virtual environment if it exists
if [ -d "venv" ]; then
    source venv/bin/activate
fi

echo "=========================================="
echo "PCA 3D Interactive Visualization"
echo "=========================================="

# Run PCA 3D interactive visualization
echo ""
echo "Starting interactive 3D visualization..."
echo "  - Window will open showing PCA space"
echo "  - Use mouse to rotate, zoom, and pan"
echo "  - Close window to exit"
echo ""
python -m src.eval.pca_3d_interactive \
  --data data/raw/ising_L32.npz

echo ""
echo "=========================================="
echo "Interactive visualization closed."
echo "=========================================="

