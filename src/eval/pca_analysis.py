"""
PCA analysis of Ising model dataset.

This script performs Principal Component Analysis on spin configurations
and visualizes the results with respect to phase labels and magnetization.
"""

import argparse
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from scipy.stats import pearsonr
import os
from src.ising.io import load_npz


def main():
    p = argparse.ArgumentParser(description='PCA analysis of Ising dataset')
    p.add_argument('--data', type=str, default='data/raw/ising_L32.npz',
                   help='Path to NPZ data file')
    p.add_argument('--outdir', type=str, default='results/figures',
                   help='Output directory for figures')
    p.add_argument('--n_components', type=int, default=50,
                   help='Number of principal components to compute')
    args = p.parse_args()
    
    # Load data
    print(f"Loading data from {args.data}...")
    data = load_npz(args.data)
    X = data['X']  # spin configurations (N, L, L)
    T = data['T']  # temperatures
    y_phase = data['y_phase']  # phase labels (0=disordered, 1=ordered)
    m = data['m']  # magnetization per spin
    e = data['e']  # energy per spin
    L = int(data['L'])
    
    print(f"Loaded {len(X)} configurations at lattice size L={L}")
    
    # Flatten configurations for PCA: (N, L, L) -> (N, L*L)
    N = X.shape[0]
    X_flat = X.reshape(N, -1)  # (N, L*L)
    
    print(f"Flattened configurations to shape: {X_flat.shape}")
    
    # Standardize the data (mean=0, std=1)
    print("Standardizing data...")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_flat)
    
    # Perform PCA
    print(f"Performing PCA with {args.n_components} components...")
    pca = PCA(n_components=args.n_components)
    X_pca = pca.fit_transform(X_scaled)
    
    # Get explained variance
    explained_variance = pca.explained_variance_ratio_
    cumulative_variance = np.cumsum(explained_variance)
    
    print(f"First 5 PCs explain {cumulative_variance[4]:.2%} of variance")
    print(f"First 10 PCs explain {cumulative_variance[9]:.2%} of variance")
    
    # Compute correlation with magnetization for PC1
    print("Computing correlation with magnetization...")
    corr_pc1, pval_pc1 = pearsonr(X_pca[:, 0], m)
    r2_pc1 = corr_pc1 ** 2
    
    # Get colors from RdYlBu_r colormap (same as gradcam analysis)
    cmap = plt.get_cmap('RdYlBu_r')
    # RdYlBu_r is reversed: 0.0 = blue, 1.0 = red
    blue_color = cmap(0.0)[:3]  # Blue from colormap
    red_color = cmap(1.0)[:3]   # Red from colormap
    
    # Set publication-quality grayscale style
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
    
    # PC1 vs PC2 scatter plot with 3 panels
    print("Creating PC1 vs PC2 scatter plot (3 panels)...")
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    # Separate by phase label
    ordered_mask = y_phase == 1
    disordered_mask = y_phase == 0
    
    # Panel 1: Colored by phase (ordered/disordered)
    ax = axes[0]
    ax.scatter(X_pca[ordered_mask, 0], X_pca[ordered_mask, 1],
               marker='o', s=10, alpha=1.0, label='Ordered (T < Tc)',
               color=blue_color, edgecolors='none')
    ax.scatter(X_pca[disordered_mask, 0], X_pca[disordered_mask, 1],
               marker='o', s=10, alpha=1.0, label='Disordered (T ≥ Tc)',
               color=red_color, edgecolors='none')
    ax.legend(loc='best', fontsize=11, frameon=True, fancybox=False,
              edgecolor='black')
    ax.set_xlabel(f'PC1 ({explained_variance[0]:.2%} variance)', 
                  fontsize=14)
    ax.set_ylabel(f'PC2 ({explained_variance[1]:.2%} variance)', fontsize=14)
    ax.grid(True, alpha=0.3, color='gray', linestyle=':')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.tick_params(labelsize=12)
    
    # Panel 2: Colored by m value
    ax = axes[1]
    scatter = ax.scatter(X_pca[:, 0], X_pca[:, 1], c=m, marker='o', s=10, 
                        alpha=1.0, cmap='viridis', edgecolors='none')
    ax.set_xlabel(f'PC1 ({explained_variance[0]:.2%} variance)', 
                  fontsize=14)
    ax.set_ylabel(f'PC2 ({explained_variance[1]:.2%} variance)', fontsize=14)
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label('Magnetization (m)', fontsize=12)
    ax.grid(True, alpha=0.3, color='gray', linestyle=':')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.tick_params(labelsize=12)
    
    # Panel 3: Colored by E value
    ax = axes[2]
    scatter = ax.scatter(X_pca[:, 0], X_pca[:, 1], c=e, marker='o', s=10, 
                        alpha=1.0, cmap='viridis', edgecolors='none')
    ax.set_xlabel(f'PC1 ({explained_variance[0]:.2%} variance)', 
                  fontsize=14)
    ax.set_ylabel(f'PC2 ({explained_variance[1]:.2%} variance)', fontsize=14)
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label('Energy per spin (e)', fontsize=12)
    ax.grid(True, alpha=0.3, color='gray', linestyle=':')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.tick_params(labelsize=12)
    
    plt.tight_layout()
    os.makedirs(args.outdir, exist_ok=True)
    output_path = os.path.join(args.outdir, 'pca.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close()
    print(f"✓ Saved PCA plot to: {output_path}")
    
    # Separate plot: Magnetization vs PC1 (analogous to VAE latent1 vs m)
    print("\nCreating plot: Magnetization vs PC1...")
    fig, ax = plt.subplots(1, 1, figsize=(8, 6))
    
    ax.scatter(X_pca[:, 0], m, marker='o', s=10, alpha=1.0,
               label=f'PC1 vs m (R²={r2_pc1:.3f})',
               color='black', edgecolors='none')
    
    ax.set_xlabel('PC1', fontsize=14)
    ax.set_ylabel('Magnetization (m)', fontsize=14)
    ax.legend(loc='best', fontsize=11, frameon=True, fancybox=False,
              edgecolor='black')
    ax.grid(True, alpha=0.3, color='gray', linestyle=':')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.tick_params(labelsize=12)
    
    plt.tight_layout()
    output_path = os.path.join(args.outdir, 'pca_pc1_vs_m.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close()
    print(f"✓ Saved PC1 vs m plot to: {output_path}")
    
    # Print summary statistics
    print("\n" + "="*60)
    print("PCA Analysis Summary")
    print("="*60)
    print(f"Total variance explained by first {args.n_components} PCs: "
          f"{cumulative_variance[-1]:.2%}")
    print(f"PC1 explains {explained_variance[0]:.2%} of variance")
    print(f"PC2 explains {explained_variance[1]:.2%} of variance")
    print(f"PC3 explains {explained_variance[2]:.2%} of variance")
    print(f"\nCorrelation of m with PC1: r={corr_pc1:.4f}, R²={r2_pc1:.4f} (p={pval_pc1:.2e})")
    print("="*60)


if __name__ == '__main__':
    main()

