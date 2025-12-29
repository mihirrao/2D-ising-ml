#!/bin/bash

# Activate virtual environment if it exists
if [ -d "venv" ]; then
    source venv/bin/activate
fi

echo "=========================================="
echo "CNN Training"
echo "=========================================="

# Train CNN
echo ""
echo "Training CNN..."
python -m src.train.train_cnn \
  --data data/raw/ising_L32.npz \
  --epochs 10 --batch 128 --lr 1e-3 \
  --outdir results/checkpoints/cnn \
  --early-stopping 3

echo ""
echo "=========================================="
echo "CNN training complete!"
echo "=========================================="

