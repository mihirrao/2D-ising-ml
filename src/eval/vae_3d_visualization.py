"""
VAE 3D Visualization: Visualize VAE latent dimensions 1-3 in 3D space.

This script creates a 2-panel figure:
- Left panel: 3D scatter plot of VAE latent dimensions 1, 2, 3
- Right panel: 3D scatter plot of PCA applied to VAE latent dimensions 1-3
"""

import argparse
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import os
import hashlib
from src.ising.io import load_npz
from src.models.vae import IsingVAE


def main():
    p = argparse.ArgumentParser(description='VAE 3D visualization')
    p.add_argument('--checkpoint-dir', type=str, default='results/checkpoints/vae',
                   help='Directory containing VAE checkpoints')
    p.add_argument('--data', type=str, default='data/raw/ising_L32.npz',
                   help='Path to NPZ data file')
    p.add_argument('--outdir', type=str, default='results/figures',
                   help='Output directory for figures')
    args = p.parse_args()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load data
    print(f"Loading data from {args.data}...")
    data = load_npz(args.data)
    X = data['X']
    y_phase = data['y_phase']  # phase labels (0=disordered, 1=ordered)
    L = int(data['L'])
    
    # Use full dataset for visualizations
    X_full = X
    y_full = y_phase
    
    print(f"Using full dataset: {len(X_full)} samples")
    
    # Load zdim=4 model and encode data (with caching)
    checkpoint_path = os.path.join(args.checkpoint_dir, 'latent_4', 'best.pt')
    
    if not os.path.exists(checkpoint_path):
        print(f"Error: Checkpoint not found: {checkpoint_path}")
        return
    
    # Check for cached embeddings
    cache_dir = os.path.join(args.outdir, '..', 'cache')
    os.makedirs(cache_dir, exist_ok=True)
    
    # Create hash of checkpoint and data file paths for cache key
    checkpoint_hash = hashlib.md5(checkpoint_path.encode()).hexdigest()[:8]
    data_hash = hashlib.md5(args.data.encode()).hexdigest()[:8]
    cache_file = os.path.join(cache_dir, f'vae_embeddings_latent4_{data_hash}_{checkpoint_hash}.npz')
    
    if os.path.exists(cache_file):
        print("\nLoading cached VAE embeddings for latent_dim=4...")
        cached = np.load(cache_file)
        z_np = cached['embeddings']
        print(f"  Loaded {len(z_np)} cached embeddings")
        print(f"  Latent space shape: {z_np.shape}")
    else:
        print(f"\nLoading model with latent_dim=4...")
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model = IsingVAE(L=checkpoint['L'], latent_dim=checkpoint['latent_dim']).to(device)
        model.load_state_dict(checkpoint['model'])
        model.eval()
        
        X_tensor = torch.from_numpy(X_full[:, None, :, :].astype(np.float32)).to(device)
        
        print("  Extracting embeddings...")
        with torch.no_grad():
            mu, logvar = model.encode(X_tensor)
            z = model.reparameterize(mu, logvar)
            z_np = z.cpu().numpy()
        
        print(f"  Encoded {len(z_np)} samples")
        print(f"  Latent space shape: {z_np.shape}")
        
        # Save to cache
        print(f"  Saving embeddings to cache: {cache_file}")
        np.savez(cache_file, embeddings=z_np)
    
    # Extract dimensions 1-3 (indices 0, 1, 2)
    z_dims_1_3 = z_np[:, :3]  # First 3 dimensions
    
    # Separate by phase label
    ordered_mask = y_full == 1
    disordered_mask = y_full == 0
    
    # Run PCA on dimensions 1-3
    print("\nRunning PCA on VAE latent dimensions 1-3...")
    scaler = StandardScaler()
    z_dims_1_3_scaled = scaler.fit_transform(z_dims_1_3)
    pca = PCA(n_components=3)
    z_pca = pca.fit_transform(z_dims_1_3_scaled)
    
    explained_variance = pca.explained_variance_ratio_
    print(f"  PC1 explains {explained_variance[0]:.2%} of variance")
    print(f"  PC2 explains {explained_variance[1]:.2%} of variance")
    print(f"  PC3 explains {explained_variance[2]:.2%} of variance")
    
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
    
    # Create 2x3 grid: 2 rows (VAE dims, PCA) x 3 columns (original, rotated views)
    print("\nCreating 3D visualization with multiple rotated views...")
    fig = plt.figure(figsize=(24, 16))
    
    # Define rotation angles (azimuth around z-axis)
    rotations = [0, 45, 90]  # Original, 45°, 90°
    
    # Row 1: VAE latent dimensions 1-3 (with 3 different rotations)
    # Row 2: PCA of VAE latent dimensions 1-3 (with 3 different rotations)
    
    # Row 1: VAE latent dimensions 1-3
    for col_idx, azim in enumerate(rotations):
        ax1 = fig.add_subplot(2, 3, col_idx + 1, projection='3d')
        
        ax1.scatter(z_dims_1_3[ordered_mask, 0], 
                    z_dims_1_3[ordered_mask, 1], 
                    z_dims_1_3[ordered_mask, 2],
                    marker='o', s=10, alpha=0.6, label='Ordered (T < Tc)',
                    color=blue_color, edgecolors='none')
        ax1.scatter(z_dims_1_3[disordered_mask, 0], 
                    z_dims_1_3[disordered_mask, 1], 
                    z_dims_1_3[disordered_mask, 2],
                    marker='o', s=10, alpha=0.6, label='Disordered (T ≥ Tc)',
                    color=red_color, edgecolors='none')
        
        ax1.set_xlabel('VAE Dim 1 (z=4)', fontsize=12, labelpad=8)
        ax1.set_ylabel('VAE Dim 2 (z=4)', fontsize=12, labelpad=8)
        ax1.set_zlabel('VAE Dim 3 (z=4)', fontsize=12, labelpad=12)
        ax1.zaxis.set_rotate_label(True)
        ax1.zaxis.labelpad = 12
        
        # Add title indicating rotation
        if azim == 0:
            ax1.set_title('VAE Latent Dims 1-3', fontsize=13, pad=10)
        else:
            ax1.set_title(f'VAE Latent Dims 1-3 (Rotated {azim}° about z-axis)', fontsize=13, pad=10)
        
        # Only show legend in first column
        if col_idx == 0:
            ax1.legend(loc='upper left', fontsize=10, frameon=True, fancybox=False,
                      edgecolor='black')
        ax1.grid(True, alpha=0.3)
        ax1.view_init(elev=20, azim=azim)  # Set rotation
    
    # Row 2: PCA of VAE latent dimensions 1-3
    for col_idx, azim in enumerate(rotations):
        ax2 = fig.add_subplot(2, 3, col_idx + 4, projection='3d')
        
        ax2.scatter(z_pca[ordered_mask, 0], 
                    z_pca[ordered_mask, 1], 
                    z_pca[ordered_mask, 2],
                    marker='o', s=10, alpha=0.6, label='Ordered (T < Tc)',
                    color=blue_color, edgecolors='none')
        ax2.scatter(z_pca[disordered_mask, 0], 
                    z_pca[disordered_mask, 1], 
                    z_pca[disordered_mask, 2],
                    marker='o', s=10, alpha=0.6, label='Disordered (T ≥ Tc)',
                    color=red_color, edgecolors='none')
        
        ax2.set_xlabel(f'VAE PC1 ({explained_variance[0]:.2%} var)', fontsize=12, labelpad=8)
        ax2.set_ylabel(f'VAE PC2 ({explained_variance[1]:.2%} var)', fontsize=12, labelpad=8)
        ax2.set_zlabel(f'VAE PC3 ({explained_variance[2]:.2%} var)', fontsize=12, labelpad=12)
        ax2.zaxis.set_rotate_label(True)
        ax2.zaxis.labelpad = 12
        
        # Add title indicating rotation
        if azim == 0:
            ax2.set_title('VAE PCA', fontsize=13, pad=10)
        else:
            ax2.set_title(f'VAE PCA (Rotated {azim}° about z-axis)', fontsize=13, pad=10)
        
        # Only show legend in first column
        if col_idx == 0:
            ax2.legend(loc='upper left', fontsize=10, frameon=True, fancybox=False,
                      edgecolor='black')
        ax2.grid(True, alpha=0.3)
        ax2.view_init(elev=20, azim=azim)  # Set rotation
    
    # Don't use tight_layout for 3D plots as it can clip z-axis labels
    # Instead, adjust subplot parameters manually to give more space on the right for z-axis labels
    plt.subplots_adjust(left=0.05, right=0.75, bottom=0.05, top=0.95, hspace=-0.2, wspace=0.25)
    os.makedirs(args.outdir, exist_ok=True)
    output_path = os.path.join(args.outdir, 'vae_3d_visualization.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight', pad_inches=0.5,
                facecolor='white', edgecolor='none')
    plt.close()
    print(f"✓ Saved to: {output_path}")
    
    print("\nVAE 3D visualization complete!")


if __name__ == '__main__':
    main()

