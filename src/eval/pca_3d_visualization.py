import argparse
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import os
from src.ising.io import load_npz


def main():
    p = argparse.ArgumentParser(description='PCA 3D visualization')
    p.add_argument('--data', type=str, default='data/raw/ising_L32.npz',
                   help='Path to NPZ data file')
    p.add_argument('--outdir', type=str, default='results/figures',
                   help='Output directory for figures')
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
    
    print("\nCreating 3D visualization with multiple rotated views...")
    fig = plt.figure(figsize=(24, 8))
    
    rotations = [0, 45, 90]
    
    for col_idx, azim in enumerate(rotations):
        ax = fig.add_subplot(1, 3, col_idx + 1, projection='3d')
        
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
        
        ax.set_xlabel(f'PC1 ({explained_variance[0]:.2%} var)', fontsize=12, labelpad=8)
        ax.set_ylabel(f'PC2 ({explained_variance[1]:.2%} var)', fontsize=12, labelpad=8)
        ax.set_zlabel(f'PC3 ({explained_variance[2]:.2%} var)', fontsize=12, labelpad=12)
        ax.zaxis.set_rotate_label(True)
        ax.zaxis.labelpad = 12
        
        if azim == 0:
            ax.set_title('PCA', fontsize=13, pad=10)
        else:
            ax.set_title(f'PCA (Rotated {azim}° about z-axis)', fontsize=13, pad=10)
        
        if col_idx == 0:
            ax.legend(loc='upper left', fontsize=10, frameon=True, fancybox=False,
                      edgecolor='black')
        ax.grid(True, alpha=0.3)
        ax.view_init(elev=20, azim=azim)
    
    plt.subplots_adjust(left=0.05, right=0.80, bottom=0.05, top=0.95, wspace=0.25)
    os.makedirs(args.outdir, exist_ok=True)
    output_path = os.path.join(args.outdir, 'pca_3d_visualization.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight', pad_inches=0.5,
                facecolor='white', edgecolor='none')
    plt.close()
    print(f"✓ Saved to: {output_path}")
    
    print("\nPCA 3D visualization complete!")


if __name__ == '__main__':
    main()

