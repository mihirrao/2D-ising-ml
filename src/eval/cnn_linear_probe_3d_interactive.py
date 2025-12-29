"""
CNN Linear Probe 3D Interactive Visualization: Interactive rotatable 3D plots of CNN embeddings.

This script creates interactive 3D scatter plots that can be rotated in real-time.
Use mouse to rotate, zoom, and pan the plots.
"""

import argparse
import numpy as np
import torch
import matplotlib
# Use interactive backend (not Agg) for live rotation
matplotlib.use('TkAgg')  # or 'Qt5Agg' depending on your system
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import os
import hashlib
from src.ising.io import load_npz
from src.models.cnn import IsingCNN


def main():
    p = argparse.ArgumentParser(description='CNN linear probe 3D interactive visualization')
    p.add_argument('--checkpoint', type=str, required=True,
                   help='Path to trained CNN checkpoint')
    p.add_argument('--data', type=str, default='data/raw/ising_L32.npz',
                   help='Path to NPZ data file')
    args = p.parse_args()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load data
    print(f"Loading data from {args.data}...")
    data = load_npz(args.data)
    X = data['X']
    y_phase = data['y_phase']  # phase labels (0=disordered, 1=ordered)
    L = int(data['L'])
    
    print(f"Loaded {len(X)} configurations at lattice size L={L}")
    
    # Check for cached embeddings
    cache_dir = os.path.join('results', 'cache')
    os.makedirs(cache_dir, exist_ok=True)
    
    # Create hash of checkpoint and data file paths for cache key
    checkpoint_hash = hashlib.md5(args.checkpoint.encode()).hexdigest()[:8]
    data_hash = hashlib.md5(args.data.encode()).hexdigest()[:8]
    cache_file = os.path.join(cache_dir, f'cnn_embeddings_{data_hash}_{checkpoint_hash}.npz')
    
    if os.path.exists(cache_file):
        print("Loading cached CNN embeddings...")
        cached = np.load(cache_file)
        embeddings = cached['embeddings']
        print(f"Loaded cached embeddings of shape: {embeddings.shape}")
    else:
        # Load model
        print(f"Loading model from {args.checkpoint}...")
        checkpoint = torch.load(args.checkpoint, map_location=device)
        model = IsingCNN(L=checkpoint['L']).to(device)
        model.load_state_dict(checkpoint['model'])
        model.eval()
        
        # Extract embeddings
        print("Extracting embeddings...")
        X_tensor = torch.from_numpy(X[:, None, :, :].astype(np.float32)).to(device)
        
        with torch.no_grad():
            # Extract embeddings h(x) = model.net(x) (before classification head)
            embeddings = model.net(X_tensor).cpu().numpy()
        
        print(f"Extracted embeddings of shape: {embeddings.shape}")
        
        # Save to cache
        print(f"Saving embeddings to cache: {cache_file}")
        np.savez(cache_file, embeddings=embeddings)
    
    # Filter out dimensions with NaN or Inf values before PCA
    print("Filtering out invalid dimensions...")
    valid_mask = np.ones(embeddings.shape[1], dtype=bool)
    for dim in range(embeddings.shape[1]):
        if np.any(np.isnan(embeddings[:, dim])) or np.any(np.isinf(embeddings[:, dim])):
            valid_mask[dim] = False
        # Also check for constant values (zero variance)
        elif np.std(embeddings[:, dim]) < 1e-9:
            valid_mask[dim] = False
    
    embeddings_clean = embeddings[:, valid_mask]
    n_valid = np.sum(valid_mask)
    print(f"Using {n_valid} valid dimensions out of {embeddings.shape[1]} total")
    
    if n_valid == 0:
        print("Error: No valid dimensions found (all have NaN/Inf/constant values)")
        return
    
    if n_valid < 3:
        print(f"Error: Need at least 3 valid dimensions for 3D plot, found {n_valid}")
        return
    
    # Run PCA on all valid embeddings
    print("\nRunning PCA on CNN embeddings...")
    scaler = StandardScaler()
    embeddings_scaled = scaler.fit_transform(embeddings_clean)
    pca = PCA(n_components=3)
    embeddings_pca = pca.fit_transform(embeddings_scaled)
    
    explained_variance = pca.explained_variance_ratio_
    print(f"  PC1 explains {explained_variance[0]:.2%} of variance")
    print(f"  PC2 explains {explained_variance[1]:.2%} of variance")
    print(f"  PC3 explains {explained_variance[2]:.2%} of variance")
    
    # Separate by phase label
    ordered_mask = y_phase == 1
    disordered_mask = y_phase == 0
    
    # Set publication-quality sans-serif style
    plt.rcParams.update({
        'font.size': 12,
        'axes.labelsize': 14,
        'axes.titlesize': 14,
        'xtick.labelsize': 12,
        'ytick.labelsize': 12,
        'legend.fontsize': 11,
        'font.family': 'sans-serif',
        'text.usetex': False,
    })
    
    # Get colors from RdYlBu_r colormap (same as other plots)
    cmap = plt.get_cmap('RdYlBu_r')
    blue_color = cmap(0.0)[:3]
    red_color = cmap(1.0)[:3]
    
    # Create interactive window
    print("\nCreating interactive 3D visualization...")
    print("  - Use mouse to rotate (click and drag)")
    print("  - Use scroll wheel to zoom")
    print("  - Close window to exit")
    
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    ax.scatter(embeddings_pca[ordered_mask, 0], 
               embeddings_pca[ordered_mask, 1], 
               embeddings_pca[ordered_mask, 2],
               marker='o', s=10, alpha=0.6, label='Ordered (T < Tc)',
               color=blue_color, edgecolors='none')
    ax.scatter(embeddings_pca[disordered_mask, 0], 
               embeddings_pca[disordered_mask, 1], 
               embeddings_pca[disordered_mask, 2],
               marker='o', s=10, alpha=0.6, label='Disordered (T ≥ Tc)',
               color=red_color, edgecolors='none')
    
    ax.set_xlabel(f'CNN PC1 ({explained_variance[0]:.2%} var)', fontsize=14, labelpad=10)
    ax.set_ylabel(f'CNN PC2 ({explained_variance[1]:.2%} var)', fontsize=14, labelpad=10)
    ax.set_zlabel(f'CNN PC3 ({explained_variance[2]:.2%} var)', fontsize=14, labelpad=15)
    ax.zaxis.set_rotate_label(True)
    ax.zaxis.labelpad = 15
    ax.set_title('CNN PCA (Interactive)', fontsize=16, pad=15)
    ax.legend(loc='upper left', fontsize=11, frameon=True, fancybox=False,
              edgecolor='black')
    ax.grid(True, alpha=0.3)
    
    # Show window
    plt.show()
    
    print("\nInteractive visualization complete!")


if __name__ == '__main__':
    main()

