import argparse
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import os
from src.ising.io import load_npz


def main():
    p = argparse.ArgumentParser(description='PCA 3D interactive visualization')
    p.add_argument('--data', type=str, default='data/raw/ising_L32.npz',
                   help='Path to NPZ data file')
    args = p.parse_args()
    
    print(f"Loading data from {args.data}...")
    data = load_npz(args.data)
    X = data['X']
    y_phase = data['y_phase']
    L = int(data['L'])
    
    print(f"Loaded {len(X)} configurations at lattice size L={L}")
    
    N = X.shape[0]
    X_flat = X.reshape(N, -1)
    
    print(f"Flattened configurations to shape: {X_flat.shape}")
    
    print("Standardizing data...")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_flat)
    
    print("Performing PCA with 3 components...")
    pca = PCA(n_components=3)
    X_pca = pca.fit_transform(X_scaled)
    
    explained_variance = pca.explained_variance_ratio_
    print(f"  PC1 explains {explained_variance[0]:.2%} of variance")
    print(f"  PC2 explains {explained_variance[1]:.2%} of variance")
    print(f"  PC3 explains {explained_variance[2]:.2%} of variance")
    
    ordered_mask = y_phase == 1
    disordered_mask = y_phase == 0
    
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
    
    cmap = plt.get_cmap('RdYlBu_r')
    blue_color = cmap(0.0)[:3]
    red_color = cmap(1.0)[:3]
    
    print("\nCreating interactive 3D visualization...")
    print("  - Use mouse to rotate (click and drag)")
    print("  - Use scroll wheel to zoom")
    print("  - Close window to exit")
    
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    ax.scatter(X_pca[ordered_mask, 0], 
               X_pca[ordered_mask, 1], 
               X_pca[ordered_mask, 2],
               marker='o', s=10, alpha=0.6, label='Ordered (T < Tc)',
               color=blue_color, edgecolors='none')
    ax.scatter(X_pca[disordered_mask, 0], 
               X_pca[disordered_mask, 1], 
               X_pca[disordered_mask, 2],
               marker='o', s=10, alpha=0.6, label='Disordered (T ≥ Tc)',
               color=red_color, edgecolors='none')
    
    ax.set_xlabel(f'PC1 ({explained_variance[0]:.2%} var)', fontsize=14, labelpad=10)
    ax.set_ylabel(f'PC2 ({explained_variance[1]:.2%} var)', fontsize=14, labelpad=10)
    ax.set_zlabel(f'PC3 ({explained_variance[2]:.2%} var)', fontsize=14, labelpad=15)
    ax.zaxis.set_rotate_label(True)
    ax.zaxis.labelpad = 15
    ax.set_title('PCA (Interactive)', fontsize=16, pad=15)
    ax.legend(loc='upper left', fontsize=11, frameon=True, fancybox=False,
              edgecolor='black')
    ax.grid(True, alpha=0.3)
    
    plt.show()
    
    print("\nInteractive visualization complete!")


if __name__ == '__main__':
    main()

