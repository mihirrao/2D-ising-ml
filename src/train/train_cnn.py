import argparse, os
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt
from src.ising.io import load_npz
from src.models.cnn import IsingCNN

class NPZDataset(Dataset):
    def __init__(self, npz_path: str, split: str):
        d = load_npz(npz_path)
        X = d["X"].astype(np.float32)
        y = d["y_phase"].astype(np.int64)

        rng = np.random.default_rng()
        rng = np.random.default_rng()
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
        self.y = y[sel]
        self.L = int(d["L"])

    def __len__(self): return len(self.X)

    def __getitem__(self, i):
        x = self.X[i][None, :, :]
        return torch.from_numpy(x), torch.tensor(self.y[i])

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", type=str, required=True)
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--batch", type=int, default=128)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--outdir", type=str, default="results/checkpoints/cnn")
    p.add_argument("--early-stopping", type=int, default=0, 
                   help="Early stopping patience (0 to disable)")
    args = p.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    train_ds = NPZDataset(args.data, "train")
    val_ds = NPZDataset(args.data, "val")

    train_dl = DataLoader(train_ds, batch_size=args.batch, shuffle=True, num_workers=0)
    val_dl = DataLoader(val_ds, batch_size=args.batch, shuffle=False, num_workers=0)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = IsingCNN(L=train_ds.L).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)

    train_losses = []
    val_losses = []
    val_accs = []
    
    best_val_loss = float('inf')
    best_val_acc = 0.0
    patience_counter = 0
    
    for ep in range(1, args.epochs + 1):
        model.train()
        train_loss_sum = 0.0
        train_count = 0
        pbar = tqdm(train_dl, desc=f"epoch {ep} train")
        for x, y in pbar:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            loss = F.cross_entropy(logits, y)
            opt.zero_grad()
            loss.backward()
            opt.step()
            
            train_loss_sum += float(loss) * len(y)
            train_count += len(y)
            pbar.set_postfix(loss=float(loss))
        
        avg_train_loss = train_loss_sum / train_count
        train_losses.append(avg_train_loss)

        model.eval()
        val_loss_sum = 0.0
        correct, total = 0, 0
        with torch.no_grad():
            for x, y in val_dl:
                x, y = x.to(device), y.to(device)
                logits = model(x)
                loss = F.cross_entropy(logits, y)
                val_loss_sum += float(loss) * len(y)
                
                pred = logits.argmax(dim=-1)
                correct += int((pred == y).sum())
                total += int(y.numel())
        
        avg_val_loss = val_loss_sum / total
        val_acc = correct / total
        val_losses.append(avg_val_loss)
        val_accs.append(val_acc)
        
        print(f"Epoch {ep}: training_loss={avg_train_loss:.4f}, validation_loss={avg_val_loss:.4f}, validation_accuracy={val_acc:.4f}")

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_val_acc = val_acc
            torch.save({"model": model.state_dict(), "L": train_ds.L}, os.path.join(args.outdir, "best.pt"))
            patience_counter = 0
            print(f"  → New best model saved (validation_loss={avg_val_loss:.4f})")
        else:
            patience_counter += 1
        
        if args.early_stopping > 0 and patience_counter >= args.early_stopping:
            print(f"\nEarly stopping triggered after {ep} epochs (patience={args.early_stopping})")
            break
    
    print("\nComputing accuracy vs temperature on validation set...")
    data = load_npz(args.data)
    X_full = data['X']
    T_full = data['T']
    y_full = data['y_phase']
    Tc = float(data['Tc'])
    
    rng = np.random.default_rng()
    idx = np.arange(len(X_full))
    rng.shuffle(idx)
    n = len(idx)
    n_train = int(0.8 * n)
    n_val = int(0.1 * n)
    val_indices = idx[n_train:n_train+n_val]
    
    X_val = X_full[val_indices]
    T_val = T_full[val_indices]
    y_val = y_full[val_indices]
    
    model.eval()
    X_tensor = torch.from_numpy(X_val[:, None, :, :].astype(np.float32)).to(device)
    
    with torch.no_grad():
        logits = model(X_tensor)
        predictions = logits.argmax(dim=1).cpu().numpy()
        correct = (predictions == y_val).astype(int)
    
    T_unique = np.sort(np.unique(T_val))
    accuracies_temp = []
    T_plot = []
    
    for temp in T_unique:
        mask = T_val == temp
        if np.sum(mask) > 0:
            temp_correct = correct[mask].sum()
            temp_total = mask.sum()
            temp_acc = temp_correct / temp_total
            accuracies_temp.append(temp_acc)
            T_plot.append(temp)
    
    accuracies_temp = np.array(accuracies_temp)
    T_plot = np.array(T_plot)
    
    plot_path = os.path.join(args.outdir, "loss_curves.png")
    
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
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    epochs = range(1, len(train_losses) + 1)
    ax1.plot(epochs, train_losses, 'k-', label='Training Loss', linewidth=2.0)
    ax1.plot(epochs, val_losses, 'k--', label='Validation Loss', linewidth=2.0, dashes=(5, 3))
    ax1.set_xlabel('Epoch', fontsize=14)
    ax1.set_ylabel('Loss', fontsize=14)
    ax1.legend(frameon=True, fancybox=False, edgecolor='black', fontsize=12)
    ax1.grid(True, alpha=0.3, color='gray', linestyle=':')
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    
    ax2.plot(T_plot, accuracies_temp, 'o-', color='black', markersize=5,
             linewidth=2.0, markerfacecolor='white', markeredgewidth=1.5,
             markeredgecolor='black', label='Validation Accuracy')
    ax2.axvline(x=Tc, color='gray', linestyle='--', linewidth=1.5, alpha=0.7,
                label=f'Tc={Tc:.3f}')
    ax2.set_xlabel('Temperature (T)', fontsize=14)
    ax2.set_ylabel('Accuracy', fontsize=14)
    ax2.legend(loc='best', fontsize=11, frameon=True, fancybox=False, 
               edgecolor='black')
    ax2.grid(True, alpha=0.3, color='gray', linestyle=':')
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    ax2.set_ylim([0, 1.05])
    ax2.tick_params(labelsize=12)
    
    plt.tight_layout()
    plt.savefig(plot_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"\nLoss curves saved to {plot_path}")
    
    print(f"Best model: validation_loss={best_val_loss:.4f}, validation_accuracy={best_val_acc:.4f}")
    print(f"Model saved to {args.outdir}/best.pt")

if __name__ == "__main__":
    main()