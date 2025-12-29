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
from src.models.cnn import IsingCNN


def main():
    p = argparse.ArgumentParser(description='CNN linear probe analysis')
    p.add_argument('--checkpoint', type=str, required=True,
                   help='Path to trained CNN checkpoint')
    p.add_argument('--data', type=str, default='data/raw/ising_L32.npz',
                   help='Path to NPZ data file')
    p.add_argument('--outdir', type=str, default='results/figures',
                   help='Output directory for figures')
    args = p.parse_args()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print(f"Loading data from {args.data}...")
    data = load_npz(args.data)
    X = data['X']
    m = data['m']
    y_phase = data['y_phase']
    e = data['e']
    L = int(data['L'])
    
    print(f"Loaded {len(X)} configurations at lattice size L={L}")
    
    cache_dir = os.path.join(args.outdir, '..', 'cache')
    os.makedirs(cache_dir, exist_ok=True)
    
    checkpoint_hash = hashlib.md5(args.checkpoint.encode()).hexdigest()[:8]
    data_hash = hashlib.md5(args.data.encode()).hexdigest()[:8]
    cache_file = os.path.join(cache_dir, f'cnn_embeddings_{data_hash}_{checkpoint_hash}.npz')
    
    if os.path.exists(cache_file):
        print("Loading cached CNN embeddings...")
        cached = np.load(cache_file)
        embeddings = cached['embeddings']
        print(f"Loaded cached embeddings of shape: {embeddings.shape}")
    else:
        print(f"Loading model from {args.checkpoint}...")
        checkpoint = torch.load(args.checkpoint, map_location=device)
        model = IsingCNN(L=checkpoint['L']).to(device)
        model.load_state_dict(checkpoint['model'])
        model.eval()
        
        print("Extracting embeddings...")
        X_tensor = torch.from_numpy(X[:, None, :, :].astype(np.float32)).to(device)
        
        with torch.no_grad():
            embeddings = model.net(X_tensor).cpu().numpy()
        
        print(f"Extracted embeddings of shape: {embeddings.shape}")
        
        print(f"Saving embeddings to cache: {cache_file}")
        np.savez(cache_file, embeddings=embeddings)
    
    print("Filtering out invalid dimensions...")
    valid_mask = np.ones(embeddings.shape[1], dtype=bool)
    for dim in range(embeddings.shape[1]):
        if np.any(np.isnan(embeddings[:, dim])) or np.any(np.isinf(embeddings[:, dim])):
            valid_mask[dim] = False
        elif np.std(embeddings[:, dim]) < 1e-9:
            valid_mask[dim] = False
    
    embeddings_clean = embeddings[:, valid_mask]
    n_valid = np.sum(valid_mask)
    print(f"Using {n_valid} valid dimensions out of {embeddings.shape[1]} total")
    
    if n_valid == 0:
        print("Error: No valid dimensions found (all have NaN/Inf/constant values)")
        return
    
    if n_valid < 2:
        print(f"Error: Need at least 2 valid dimensions for PC1 vs PC2 plot, found {n_valid}")
        return
    
    print("\nRunning PCA on CNN embeddings...")
    scaler = StandardScaler()
    embeddings_scaled = scaler.fit_transform(embeddings_clean)
    pca = PCA(n_components=2)
    embeddings_pca = pca.fit_transform(embeddings_scaled)
    
    explained_variance = pca.explained_variance_ratio_
    print(f"  PC1 explains {explained_variance[0]:.2%} of variance")
    print(f"  PC2 explains {explained_variance[1]:.2%} of variance")
    
    corr_pc1, pval_pc1 = pearsonr(embeddings_pca[:, 0], m)
    r2_pc1 = corr_pc1 ** 2
    print(f"  PC1 vs m: r={corr_pc1:.4f}, R²={r2_pc1:.4f}")
    
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
    
    ordered_mask = y_phase == 1
    disordered_mask = y_phase == 0
    
    print("\nCreating PCA visualization: CNN PC1 vs PC2 (3 panels)...")
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    ax = axes[0]
    ax.scatter(embeddings_pca[ordered_mask, 0], embeddings_pca[ordered_mask, 1],
               marker='o', s=10, alpha=1.0, label='Ordered (T < Tc)',
               color=blue_color, edgecolors='none')
    ax.scatter(embeddings_pca[disordered_mask, 0], embeddings_pca[disordered_mask, 1],
               marker='o', s=10, alpha=1.0, label='Disordered (T ≥ Tc)',
               color=red_color, edgecolors='none')
    ax.set_xlabel(f'CNN PC1 ({explained_variance[0]:.2%} variance)', fontsize=14)
    ax.set_ylabel(f'CNN PC2 ({explained_variance[1]:.2%} variance)', fontsize=14)
    ax.legend(loc='best', fontsize=11, frameon=True, fancybox=False,
              edgecolor='black')
    ax.grid(True, alpha=0.3, color='gray', linestyle=':')
    ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.tick_params(labelsize=12)
    
    ax = axes[1]
    scatter = ax.scatter(embeddings_pca[:, 0], embeddings_pca[:, 1], c=m, marker='o', s=10,
                        alpha=1.0, cmap='viridis', edgecolors='none')
    ax.set_xlabel(f'CNN PC1 ({explained_variance[0]:.2%} variance)', fontsize=14)
    ax.set_ylabel(f'CNN PC2 ({explained_variance[1]:.2%} variance)', fontsize=14)
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label('Magnetization (m)', fontsize=12)
    ax.grid(True, alpha=0.3, color='gray', linestyle=':')
    ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.tick_params(labelsize=12)
    
    ax = axes[2]
    scatter = ax.scatter(embeddings_pca[:, 0], embeddings_pca[:, 1], c=e, marker='o', s=10,
                        alpha=1.0, cmap='viridis', edgecolors='none')
    ax.set_xlabel(f'CNN PC1 ({explained_variance[0]:.2%} variance)', fontsize=14)
    ax.set_ylabel(f'CNN PC2 ({explained_variance[1]:.2%} variance)', fontsize=14)
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label('Energy per spin (e)', fontsize=12)
    ax.grid(True, alpha=0.3, color='gray', linestyle=':')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.tick_params(labelsize=12)
    
    plt.tight_layout()
    os.makedirs(args.outdir, exist_ok=True)
    output_path = os.path.join(args.outdir, 'cnn_linear_probe.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close()
    print(f"✓ Saved to: {output_path}")
    
    print("\nCNN linear probe analysis complete!")


if __name__ == '__main__':
    main()

