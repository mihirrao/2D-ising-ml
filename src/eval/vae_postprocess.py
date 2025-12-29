"""
VAE post-processing: Create specific plots for latent dimension analysis.
- Plot 1: Latent dim 1 of zdim=1 vs m and latent dim 1 of zdim=4 vs m on same plot with R²
- Plot 2: Latent dim 1 vs 2 for zdim=4
"""

import argparse
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import os
import hashlib
from src.ising.io import load_npz
from src.models.vae import IsingVAE


def main():
    p = argparse.ArgumentParser(description='VAE post-processing analysis')
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
    T = data['T']
    y_phase = data['y_phase']
    m = data['m']
    L = int(data['L'])
    
    # Use full dataset for visualizations (not just test set)
    # Test set is only for performance metrics like loss/accuracy
    X_full = X
    m_full = m
    y_full = y_phase
    e_full = data['e']  # Energy per spin
    
    print(f"Using full dataset: {len(X_full)} samples")
    
    # Load models and encode data (with caching)
    z_data = {}
    cache_dir = os.path.join(args.outdir, '..', 'cache')
    os.makedirs(cache_dir, exist_ok=True)
    
    # Create hash of data file path for cache key
    data_hash = hashlib.md5(args.data.encode()).hexdigest()[:8]
    
    for latent_dim in [1, 4]:
        checkpoint_path = os.path.join(args.checkpoint_dir, f'latent_{latent_dim}', 'best.pt')
        
        if not os.path.exists(checkpoint_path):
            print(f"Warning: Checkpoint not found: {checkpoint_path}")
            continue
        
        # Create cache file path
        checkpoint_hash = hashlib.md5(checkpoint_path.encode()).hexdigest()[:8]
        cache_file = os.path.join(cache_dir, f'vae_embeddings_latent{latent_dim}_{data_hash}_{checkpoint_hash}.npz')
        
        # Check if cached embeddings exist
        if os.path.exists(cache_file):
            print(f"\nLoading cached embeddings for latent_dim={latent_dim}...")
            cached = np.load(cache_file)
            z_np = cached['embeddings']
            print(f"  Loaded {len(z_np)} cached embeddings")
        else:
            print(f"\nLoading model with latent_dim={latent_dim}...")
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
            
            # Save to cache
            print(f"  Saving embeddings to cache: {cache_file}")
            np.savez(cache_file, embeddings=z_np)
            print(f"  Encoded {len(z_np)} samples")
        
        z_data[latent_dim] = z_np
    
    os.makedirs(args.outdir, exist_ok=True)
    
    # Set publication-quality style (matching PCA analysis)
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
    
    # Plot 1: Magnetization vs each of the 4 latent dimensions for zdim=4 (2x2 plot)
    if 4 in z_data:
        print("\nCreating plot: Magnetization vs each of 4 latent dimensions for zdim=4...")
        
        z4 = z_data[4]
        fig, axes = plt.subplots(2, 2, figsize=(12, 12))
        axes = axes.flatten()
        
        for dim in range(4):
            ax = axes[dim]
            z4_dim = z4[:, dim]
            corr, pval = pearsonr(z4_dim, m_full)
            r2 = corr ** 2
            
            ax.scatter(z4_dim, m_full, marker='o', s=10, alpha=1.0,
                       label=f'Dim {dim+1} vs m (R²={r2:.3f})',
                       color='black', edgecolors='none')
            ax.set_xlabel(f'Latent Dimension {dim+1} (z=4)', fontsize=14)
            ax.set_ylabel('Magnetization (m)', fontsize=14)
            ax.legend(loc='best', fontsize=11, frameon=True, fancybox=False,
                      edgecolor='black')
            ax.grid(True, alpha=0.3, color='gray', linestyle=':')
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.tick_params(labelsize=12)
            print(f"  Dim {dim+1}: r={corr:.4f}, R²={r2:.4f}")
        
        plt.tight_layout()
        output_path = os.path.join(args.outdir, 'vae_latent_vs_m.png')
        plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
        plt.close()
        print(f"✓ Saved to: {output_path}")
    
        # Plot 2: All pairwise combinations of latent dimensions for zdim=4
    if 4 in z_data:
        print("\nCreating plot: All pairwise combinations of latent dimensions for zdim=4 (3 rows x 6 cols)...")

        from matplotlib.lines import Line2D

        z4 = z_data[4]
        ordered_mask = y_full == 1
        disordered_mask = y_full == 0

        # Colors
        cmap_phase = plt.get_cmap('RdYlBu_r')
        blue_color = cmap_phase(0.0)[:3]
        red_color = cmap_phase(1.0)[:3]

        pairs = [(0,1), (0,2), (0,3), (1,2), (1,3), (2,3)]
        col_titles = [f'Dim {i+1} vs Dim {j+1}' for (i, j) in pairs]

        fig, axes = plt.subplots(
            3, 6,
            figsize=(32, 14),
            constrained_layout=True
        )

        # Legend handles
        phase_handles = [
            Line2D([0], [0], marker='s', linestyle='None', markersize=14,
                   markerfacecolor=blue_color, markeredgecolor='black',
                   label='Ordered (T < Tc)'),
            Line2D([0], [0], marker='s', linestyle='None', markersize=14,
                   markerfacecolor=red_color, markeredgecolor='black',
                   label='Disordered (T ≥ Tc)')
        ]

        # ---------- ROW 1: PHASE ----------
        for col, (i, j) in enumerate(pairs):
            ax = axes[0, col]

            ax.scatter(z4[ordered_mask, i], z4[ordered_mask, j],
                       s=10, color=blue_color, alpha=0.9, edgecolors='none')
            ax.scatter(z4[disordered_mask, i], z4[disordered_mask, j],
                       s=10, color=red_color, alpha=0.9, edgecolors='none')

            ax.set_title(col_titles[col], fontsize=16, pad=10)
            ax.set_ylabel(f'Latent Dim {j+1} (z=4)', fontsize=14)

            ax.set_xlabel("")
            ax.set_xticklabels([])

            if col == 0:
                ax.legend(handles=phase_handles, loc='upper right',
                          frameon=True, edgecolor='black')

            ax.grid(True, alpha=0.25, linestyle=':')
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)

        # ---------- ROW 2: MAGNETIZATION ----------
        last_sc_m = None
        for col, (i, j) in enumerate(pairs):
            ax = axes[1, col]

            sc = ax.scatter(z4[:, i], z4[:, j],
                            c=m_full, cmap='viridis',
                            s=10, alpha=0.9, edgecolors='none')
            last_sc_m = sc

            ax.set_ylabel(f'Latent Dim {j+1} (z=4)', fontsize=14)
            ax.set_xlabel("")
            ax.set_xticklabels([])

            ax.grid(True, alpha=0.25, linestyle=':')
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)

        cbar_m = fig.colorbar(last_sc_m, ax=axes[1, :], fraction=0.02, pad=0.01)
        cbar_m.set_label("Magnetization (m)", fontsize=14)

        # ---------- ROW 3: ENERGY ----------
        last_sc_e = None
        for col, (i, j) in enumerate(pairs):
            ax = axes[2, col]

            sc = ax.scatter(z4[:, i], z4[:, j],
                            c=e_full, cmap='viridis',
                            s=10, alpha=0.9, edgecolors='none')
            last_sc_e = sc

            ax.set_ylabel(f'Latent Dim {j+1} (z=4)', fontsize=14)
            ax.set_xlabel(f'Latent Dim {i+1} (z=4)', fontsize=14)

            ax.grid(True, alpha=0.25, linestyle=':')
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)

        cbar_e = fig.colorbar(last_sc_e, ax=axes[2, :], fraction=0.02, pad=0.01)
        cbar_e.set_label("Energy per spin (e)", fontsize=14)

        output_path = os.path.join(args.outdir, 'vae_latent_space_latent4.png')
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"✓ Saved to: {output_path}")
        
        # Plot 3: PCA on VAE embeddings
        print("\nRunning PCA on VAE embeddings...")
        
        # Standardize and perform PCA
        scaler = StandardScaler()
        z4_scaled = scaler.fit_transform(z4)
        pca = PCA(n_components=min(4, z4.shape[1]))
        z4_pca = pca.fit_transform(z4_scaled)
        
        explained_variance = pca.explained_variance_ratio_
        print(f"  PC1 explains {explained_variance[0]:.2%} of variance")
        print(f"  PC2 explains {explained_variance[1]:.2%} of variance")
        
        # Compute correlation with magnetization for PC1
        corr_pc1, pval_pc1 = pearsonr(z4_pca[:, 0], m_full)
        r2_pc1 = corr_pc1 ** 2
        print(f"  PC1 vs m: r={corr_pc1:.4f}, R²={r2_pc1:.4f}")
        
        # Plot 3a: VAE PCA dim 1 vs m
        print("\nCreating plot: VAE PCA dim 1 vs m...")
        fig, ax = plt.subplots(1, 1, figsize=(8, 6))
        
        ax.scatter(z4_pca[:, 0], m_full, marker='o', s=10, alpha=1.0,
                   label=f'VAE PC1 vs m (R²={r2_pc1:.3f})',
                   color='black', edgecolors='none')
        
        ax.set_xlabel('VAE PC1', fontsize=14)
        ax.set_ylabel('Magnetization (m)', fontsize=14)
        ax.legend(loc='best', fontsize=11, frameon=True, fancybox=False,
                  edgecolor='black')
        ax.grid(True, alpha=0.3, color='gray', linestyle=':')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.tick_params(labelsize=12)
        
        plt.tight_layout()
        output_path = os.path.join(args.outdir, 'vae_pca_pc1_vs_m.png')
        plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
        plt.close()
        print(f"✓ Saved to: {output_path}")
        
        # Plot 3b: VAE PCA dim 1 vs dim 2 (3 panels: phase, m, E)
        print("\nCreating PCA-style plot: VAE PCA dim 1 vs dim 2 (3 panels)...")
        
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        
        # Panel 1: Colored by phase (ordered/disordered)
        ax = axes[0]
        ax.scatter(z4_pca[ordered_mask, 0], z4_pca[ordered_mask, 1],
                   marker='o', s=10, alpha=1.0, label='Ordered (T < Tc)',
                   color=blue_color, edgecolors='none')
        ax.scatter(z4_pca[disordered_mask, 0], z4_pca[disordered_mask, 1],
                   marker='o', s=10, alpha=1.0, label='Disordered (T ≥ Tc)',
                   color=red_color, edgecolors='none')
        ax.set_xlabel(f'VAE PC1 ({explained_variance[0]:.2%} variance)', fontsize=14)
        ax.set_ylabel(f'VAE PC2 ({explained_variance[1]:.2%} variance)', fontsize=14)
        ax.legend(loc='best', fontsize=11, frameon=True, fancybox=False,
                  edgecolor='black')
        ax.grid(True, alpha=0.3, color='gray', linestyle=':')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.tick_params(labelsize=12)
        
        # Panel 2: Colored by m value
        ax = axes[1]
        scatter = ax.scatter(z4_pca[:, 0], z4_pca[:, 1], c=m_full, marker='o', s=10, 
                            alpha=1.0, cmap='viridis', edgecolors='none')
        ax.set_xlabel(f'VAE PC1 ({explained_variance[0]:.2%} variance)', fontsize=14)
        ax.set_ylabel(f'VAE PC2 ({explained_variance[1]:.2%} variance)', fontsize=14)
        cbar = plt.colorbar(scatter, ax=ax)
        cbar.set_label('Magnetization (m)', fontsize=12)
        ax.grid(True, alpha=0.3, color='gray', linestyle=':')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.tick_params(labelsize=12)
        
        # Panel 3: Colored by E value
        ax = axes[2]
        scatter = ax.scatter(z4_pca[:, 0], z4_pca[:, 1], c=e_full, marker='o', s=10, 
                            alpha=1.0, cmap='viridis', edgecolors='none')
        ax.set_xlabel(f'VAE PC1 ({explained_variance[0]:.2%} variance)', fontsize=14)
        ax.set_ylabel(f'VAE PC2 ({explained_variance[1]:.2%} variance)', fontsize=14)
        cbar = plt.colorbar(scatter, ax=ax)
        cbar.set_label('Energy per spin (e)', fontsize=12)
        ax.grid(True, alpha=0.3, color='gray', linestyle=':')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.tick_params(labelsize=12)
        
        plt.tight_layout()
        output_path = os.path.join(args.outdir, 'vae_pca.png')
        plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
        plt.close()
        print(f"✓ Saved to: {output_path}")
    
    print("\nVAE post-processing complete!")


if __name__ == '__main__':
    main()

