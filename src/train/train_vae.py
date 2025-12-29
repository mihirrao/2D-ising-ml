"""
Train Variational Autoencoder for Ising model configurations.

Trains VAE with different latent dimensions and generates analysis plots.
"""

import argparse
import os
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from src.ising.io import load_npz
from src.models.vae import IsingVAE


class NPZDataset(Dataset):
    def __init__(self, npz_path: str, split: str, seed: int = 0):
        d = load_npz(npz_path)
        X = d["X"].astype(np.float32)  # (-1,+1)

        # shuffle + split
        rng = np.random.default_rng(seed)
        idx = np.arange(len(X))
        rng.shuffle(idx)
        n = len(idx)
        n_train = int(0.8 * n)
        n_val = int(0.1 * n)
        if split == "train":
            sel = idx[:n_train]
        elif split == "val":
            sel = idx[n_train:n_train+n_val]
        elif split == "test":
            sel = idx[n_train+n_val:]
        else:
            raise ValueError(split)

        self.X = X[sel]
        self.L = int(d["L"])
        self.T = d["T"][sel] if "T" in d else None
        self.y_phase = d["y_phase"][sel] if "y_phase" in d else None

    def __len__(self): return len(self.X)

    def __getitem__(self, i):
        x = self.X[i][None, :, :]   # (1, L, L)
        return torch.from_numpy(x)


def vae_loss(recon, x, mu, logvar, beta=1.0):
    """VAE loss: reconstruction + KL divergence."""
    # Reconstruction loss (MSE)
    recon_loss = F.mse_loss(recon, x, reduction='sum')
    
    # KL divergence
    kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    
    return recon_loss + beta * kl_loss, recon_loss, kl_loss


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", type=str, required=True)
    p.add_argument("--latent-dims", type=int, nargs='+', default=[1, 4],
                   help='List of latent dimensions to train')
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--batch", type=int, default=128)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--beta", type=float, default=1.0,
                   help='Weight for KL divergence term')
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--outdir", type=str, default="results/checkpoints/vae")
    p.add_argument("--early-stopping", type=int, default=5,
                   help="Early stopping patience (0 to disable)")
    args = p.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load data once
    train_ds = NPZDataset(args.data, "train", seed=args.seed)
    val_ds = NPZDataset(args.data, "val", seed=args.seed)
    test_ds = NPZDataset(args.data, "test", seed=args.seed)
    
    train_dl = DataLoader(train_ds, batch_size=args.batch, shuffle=True, num_workers=0)
    val_dl = DataLoader(val_ds, batch_size=args.batch, shuffle=False, num_workers=0)
    
    # Store results for all latent dimensions
    all_results = {}
    
    for latent_dim in args.latent_dims:
        print(f"\n{'='*60}")
        print(f"Training VAE with latent_dim={latent_dim}")
        print(f"{'='*60}")
        
        model = IsingVAE(L=train_ds.L, latent_dim=latent_dim).to(device)
        opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
        
        # Track metrics
        train_losses = []
        val_losses = []
        train_recon_losses = []
        train_kl_losses = []
        val_recon_losses = []
        val_kl_losses = []
        
        best_val_loss = float('inf')
        patience_counter = 0
        
        for ep in range(1, args.epochs + 1):
            # Training
            model.train()
            train_loss_sum = 0.0
            train_recon_sum = 0.0
            train_kl_sum = 0.0
            train_count = 0
            
            pbar = tqdm(train_dl, desc=f"epoch {ep} train (latent={latent_dim})")
            for x in pbar:
                x = x.to(device)
                recon, mu, logvar, z = model(x)
                loss, recon_loss, kl_loss = vae_loss(recon, x, mu, logvar, beta=args.beta)
                
                opt.zero_grad()
                loss.backward()
                opt.step()
                
                batch_size = len(x)
                train_loss_sum += float(loss) * batch_size
                train_recon_sum += float(recon_loss) * batch_size
                train_kl_sum += float(kl_loss) * batch_size
                train_count += batch_size
                pbar.set_postfix(loss=float(loss)/batch_size)
            
            avg_train_loss = train_loss_sum / train_count
            avg_train_recon = train_recon_sum / train_count
            avg_train_kl = train_kl_sum / train_count
            train_losses.append(avg_train_loss)
            train_recon_losses.append(avg_train_recon)
            train_kl_losses.append(avg_train_kl)
            
            # Validation
            model.eval()
            val_loss_sum = 0.0
            val_recon_sum = 0.0
            val_kl_sum = 0.0
            val_count = 0
            
            with torch.no_grad():
                for x in val_dl:
                    x = x.to(device)
                    recon, mu, logvar, z = model(x)
                    loss, recon_loss, kl_loss = vae_loss(recon, x, mu, logvar, beta=args.beta)
                    
                    batch_size = len(x)
                    val_loss_sum += float(loss) * batch_size
                    val_recon_sum += float(recon_loss) * batch_size
                    val_kl_sum += float(kl_loss) * batch_size
                    val_count += batch_size
            
            avg_val_loss = val_loss_sum / val_count
            avg_val_recon = val_recon_sum / val_count
            avg_val_kl = val_kl_sum / val_count
            val_losses.append(avg_val_loss)
            val_recon_losses.append(avg_val_recon)
            val_kl_losses.append(avg_val_kl)
            
            print(f"Epoch {ep}: train_loss={avg_train_loss:.4f} (recon={avg_train_recon:.4f}, kl={avg_train_kl:.4f}), "
                  f"val_loss={avg_val_loss:.4f} (recon={avg_val_recon:.4f}, kl={avg_val_kl:.4f})")
            
            # Save best model
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                model_dir = os.path.join(args.outdir, f"latent_{latent_dim}")
                os.makedirs(model_dir, exist_ok=True)
                torch.save({
                    "model": model.state_dict(),
                    "L": train_ds.L,
                    "latent_dim": latent_dim
                }, os.path.join(model_dir, "best.pt"))
                patience_counter = 0
                print(f"  → New best model saved (validation_loss={avg_val_loss:.4f})")
            else:
                patience_counter += 1
            
            # Early stopping
            if args.early_stopping > 0 and patience_counter >= args.early_stopping:
                print(f"\nEarly stopping triggered after {ep} epochs (patience={args.early_stopping})")
                break
        
        # Store results
        all_results[latent_dim] = {
            'train_losses': train_losses,
            'val_losses': val_losses,
            'train_recon_losses': train_recon_losses,
            'train_kl_losses': train_kl_losses,
            'val_recon_losses': val_recon_losses,
            'val_kl_losses': val_kl_losses,
            'best_val_loss': best_val_loss
        }
        
        # Evaluate best model on validation set per temperature
        print("Evaluating best model on validation set per temperature...")
        best_model_path = os.path.join(args.outdir, f"latent_{latent_dim}", "best.pt")
        best_checkpoint = torch.load(best_model_path, map_location=device)
        best_model = IsingVAE(L=best_checkpoint['L'], latent_dim=best_checkpoint['latent_dim']).to(device)
        best_model.load_state_dict(best_checkpoint['model'])
        best_model.eval()
        
        # Load full validation data with temperatures
        data_full = load_npz(args.data)
        X_full = data_full['X'].astype(np.float32)
        T_full = data_full['T']
        Tc = float(data_full['Tc'])
        
        # Get validation indices (same split as used during training)
        rng = np.random.default_rng(args.seed)
        idx = np.arange(len(X_full))
        rng.shuffle(idx)
        n = len(idx)
        n_train = int(0.8 * n)
        n_val = int(0.1 * n)
        val_indices = idx[n_train:n_train+n_val]
        
        X_val = X_full[val_indices]
        T_val = T_full[val_indices]
        
        # Compute loss per temperature
        T_unique = np.sort(np.unique(T_val))
        val_losses_temp = []
        T_plot = []
        
        with torch.no_grad():
            for temp in T_unique:
                mask = T_val == temp
                if np.sum(mask) > 0:
                    X_temp = X_val[mask]
                    X_tensor = torch.from_numpy(X_temp[:, None, :, :].astype(np.float32)).to(device)
                    
                    recon, mu, logvar, z = best_model(X_tensor)
                    loss, _, _ = vae_loss(recon, X_tensor, mu, logvar, beta=args.beta)
                    
                    avg_loss = float(loss) / len(X_temp)
                    val_losses_temp.append(avg_loss)
                    T_plot.append(temp)
        
        val_losses_temp = np.array(val_losses_temp)
        T_plot = np.array(T_plot)
        
        # Plot loss curves for this latent dimension (3 panels)
        # Use the same model_dir that was created during training
        model_dir = os.path.join(args.outdir, f"latent_{latent_dim}")
        os.makedirs(model_dir, exist_ok=True)
        plot_path = os.path.join(model_dir, "loss_curves.png")
        
        plt.rcParams.update({
            'font.size': 12,
            'axes.labelsize': 14,
            'axes.titlesize': 14,
            'xtick.labelsize': 12,
            'ytick.labelsize': 12,
            'legend.fontsize': 12,
            'font.family': 'serif',
            'text.usetex': False,
        })
        
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 5))
        
        epochs = range(1, len(train_losses) + 1)
        
        # Panel 1: Total Loss
        ax1.plot(epochs, train_losses, 'k-', label='Training Loss', linewidth=2.0)
        ax1.plot(epochs, val_losses, 'k--', label='Validation Loss', linewidth=2.0, dashes=(5, 3))
        ax1.set_xlabel('Epoch', fontsize=14)
        ax1.set_ylabel('Loss', fontsize=14)
        ax1.legend(frameon=True, fancybox=False, edgecolor='black', fontsize=12)
        ax1.grid(True, alpha=0.3, color='gray', linestyle=':')
        ax1.spines['top'].set_visible(False)
        ax1.spines['right'].set_visible(False)
        
        # Panel 2: Reconstruction Loss
        ax2.plot(epochs, train_recon_losses, 'k-', label='Train Recon', linewidth=2.0)
        ax2.plot(epochs, val_recon_losses, 'k--', label='Val Recon', linewidth=2.0, dashes=(5, 3))
        ax2.set_xlabel('Epoch', fontsize=14)
        ax2.set_ylabel('Reconstruction Loss', fontsize=14)
        ax2.legend(frameon=True, fancybox=False, edgecolor='black', fontsize=12)
        ax2.grid(True, alpha=0.3, color='gray', linestyle=':')
        ax2.spines['top'].set_visible(False)
        ax2.spines['right'].set_visible(False)
        
        # Panel 3: KL Loss
        ax3.plot(epochs, train_kl_losses, 'k-', label='Train KL', linewidth=2.0)
        ax3.plot(epochs, val_kl_losses, 'k--', label='Val KL', linewidth=2.0, dashes=(5, 3))
        ax3.set_xlabel('Epoch', fontsize=14)
        ax3.set_ylabel('KL Divergence', fontsize=14)
        ax3.legend(frameon=True, fancybox=False, edgecolor='black', fontsize=12)
        ax3.grid(True, alpha=0.3, color='gray', linestyle=':')
        ax3.spines['top'].set_visible(False)
        ax3.spines['right'].set_visible(False)
        
        plt.tight_layout()
        plt.savefig(plot_path, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        print(f"Loss curves saved to {plot_path}")
        
        # Separate figure: Validation Loss vs Temperature
        loss_vs_temp_path = os.path.join(model_dir, "loss_vs_temp.png")
        fig, ax = plt.subplots(1, 1, figsize=(8, 6))
        
        ax.plot(T_plot, val_losses_temp, 'o-', color='black', markersize=5,
                linewidth=2.0, markerfacecolor='white', markeredgewidth=1.5,
                markeredgecolor='black', label='Validation Loss')
        ax.axvline(x=Tc, color='gray', linestyle='--', linewidth=1.5, alpha=0.7,
                  label=f'Tc={Tc:.3f}')
        ax.set_xlabel('Temperature (T)', fontsize=14)
        ax.set_ylabel('Validation Loss', fontsize=14)
        ax.legend(loc='best', fontsize=11, frameon=True, fancybox=False,
                 edgecolor='black')
        ax.grid(True, alpha=0.3, color='gray', linestyle=':')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.tick_params(labelsize=12)
        
        plt.tight_layout()
        plt.savefig(loss_vs_temp_path, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        print(f"Loss vs temperature saved to {loss_vs_temp_path}")
    
    print(f"\nTraining complete for all latent dimensions!")
    for latent_dim in args.latent_dims:
        print(f"  latent_dim={latent_dim}: best_val_loss={all_results[latent_dim]['best_val_loss']:.4f}")


if __name__ == "__main__":
    main()

