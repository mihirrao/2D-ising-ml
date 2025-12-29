import argparse
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import os
from src.ising.io import load_npz
from src.models.cnn import IsingCNN


class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        
        self.target_layer.register_forward_hook(self.save_activation)
        self.target_layer.register_backward_hook(self.save_gradient)
    
    def save_activation(self, module, input, output):
        self.activations = output.detach()
    
    def save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()
    
    def __call__(self, x, class_idx=None):
        self.model.eval()
        output = self.model(x)
        
        if class_idx is None:
            class_idx = output.argmax(dim=1).item()
        
        self.model.zero_grad()
        one_hot = torch.zeros_like(output)
        one_hot[0, class_idx] = 1
        output.backward(gradient=one_hot, retain_graph=True)
        
        weights = torch.mean(self.gradients, dim=(2, 3), keepdim=True)
        cam = torch.sum(weights * self.activations, dim=1, keepdim=True)
        cam = F.relu(cam)
        
        cam = cam.squeeze().cpu().numpy()
        if cam.max() > 0:
            cam = (cam - cam.min()) / (cam.max() - cam.min())
        
        return cam, class_idx


def overlay_gradcam(image, cam, alpha=0.5):
    from scipy.ndimage import zoom
    if cam.shape != image.shape:
        zoom_factors = (image.shape[0] / cam.shape[0], 
                       image.shape[1] / cam.shape[1])
        cam = zoom(cam, zoom_factors, order=1)
    
    image_gray = (image + 1) / 2.0
    rgb_image = np.stack([image_gray] * 3, axis=-1)
    
    cmap = plt.get_cmap('RdYlBu_r')
    heatmap = cmap(cam)[:, :, :3]
    
    overlay = alpha * heatmap + (1 - alpha) * rgb_image
    
    return overlay


def main():
    p = argparse.ArgumentParser(description='Grad-CAM analysis for Ising CNN')
    p.add_argument('--checkpoint', type=str, default='results/checkpoints/cnn/best.pt',
                   help='Path to trained model checkpoint')
    p.add_argument('--data', type=str, default='data/raw/ising_L32.npz',
                   help='Path to NPZ data file')
    p.add_argument('--outdir', type=str, default='results/figures',
                   help='Output directory for figure')
    p.add_argument('--n_temps', type=int, default=5,
                   help='Number of temperatures to analyze')
    p.add_argument('--alpha', type=float, default=0.5,
                   help='Overlay transparency (0=only heatmap, 1=only image)')
    args = p.parse_args()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    print(f"Loading data from {args.data}...")
    data = load_npz(args.data)
    X = data['X']
    T = data['T']
    y_phase = data['y_phase']
    Tc = float(data['Tc'])
    L = int(data['L'])
    
    print(f"Loaded {len(X)} configurations at lattice size L={L}")
    
    rng = np.random.default_rng()
    idx = np.arange(len(X))
    rng.shuffle(idx)
    n = len(idx)
    n_train = int(0.8 * n)
    n_val = int(0.1 * n)
    test_indices = idx[n_train+n_val:]
    
    X = X[test_indices]
    T = T[test_indices]
    y_phase = y_phase[test_indices]
    
    print(f"Using {len(X)} test set samples for GradCAM visualization")
    
    print(f"Loading model from {args.checkpoint}...")
    checkpoint = torch.load(args.checkpoint, map_location=device)
    model = IsingCNN(L=checkpoint['L']).to(device)
    model.load_state_dict(checkpoint['model'])
    model.eval()
    print("Model loaded successfully")
    
    target_layer = model.net[6]
    
    gradcam = GradCAM(model, target_layer)
    
    T_unique = np.sort(np.unique(T))
    T_min = T_unique.min()
    T_max = T_unique.max()
    
    if args.n_temps == 3:
        clean_targets = [T_min, Tc, T_max]
    elif args.n_temps == 4:
        clean_targets = [T_min, (T_min + Tc) / 2, Tc, T_max]
    elif args.n_temps == 5:
        if T_min <= 1.5 and T_max >= 3.5:
            clean_targets = [1.5, 2.0, 2.25, 2.5, 3.5]
        else:
            clean_targets = [T_min, (T_min + Tc) / 2, Tc, (Tc + T_max) / 2, T_max]
    else:
        if T_min <= 1.5 and T_max >= 3.5:
            clean_targets = []
            t = 1.5
            while t <= 3.5 and len(clean_targets) < args.n_temps:
                clean_targets.append(t)
                t += 0.25
            clean_targets = clean_targets[:args.n_temps]
        else:
            step = (T_max - T_min) / (args.n_temps - 1)
            step = round(step * 4) / 4
            clean_targets = [round(T_min + i * step, 2) for i in range(args.n_temps)]
    
    selected_Ts = []
    for target in clean_targets:
        closest_idx = np.argmin(np.abs(T_unique - target))
        selected_Ts.append(T_unique[closest_idx])
    
    selected_Ts = np.sort(np.unique(selected_Ts))
    
    if len(selected_Ts) < args.n_temps:
        additional_targets = [1.75, 2.25, 2.75, 3.25] if T_min <= 1.5 and T_max >= 3.5 else []
        for target in additional_targets:
            if len(selected_Ts) >= args.n_temps:
                break
            if T_min <= target <= T_max:
                closest_idx = np.argmin(np.abs(T_unique - target))
                if T_unique[closest_idx] not in selected_Ts:
                    selected_Ts = np.append(selected_Ts, T_unique[closest_idx])
        selected_Ts = np.sort(selected_Ts)[:args.n_temps]
    
    selected_configs = []
    selected_labels = []
    cams = []
    predictions = []
    
    print("\nGenerating Grad-CAM visualizations...")
    for temp in selected_Ts:
        indices = np.where(T == temp)[0]
        idx = np.random.choice(indices)
        
        config = X[idx]
        label = y_phase[idx]
        
        x_tensor = torch.from_numpy(config[None, None, :, :].astype(np.float32)).to(device)
        
        with torch.no_grad():
            pred_logits = model(x_tensor)
        pred_class = pred_logits.argmax(dim=1).item()
        
        cam, _ = gradcam(x_tensor, class_idx=pred_class)
        
        selected_configs.append(config)
        selected_labels.append(label)
        cams.append(cam)
        predictions.append(pred_class)
        
        phase_name = "Ordered" if pred_class == 1 else "Disordered"
        correct = "✓" if pred_class == label else "✗"
        print(f"  T={temp:.2f}: Predicted={phase_name} {correct}")
    
    print("\nCreating visualization...")
    
    plt.rcParams.update({
        'font.size': 11,
        'axes.labelsize': 12,
        'axes.titlesize': 11,
        'xtick.labelsize': 10,
        'ytick.labelsize': 10,
        'font.family': 'sans-serif',
        'text.usetex': False,
    })
    
    fig, axes = plt.subplots(1, args.n_temps, figsize=(3 * args.n_temps, 4.5), 
                             squeeze=False)
    axes = axes[0]  # Flatten to 1D array
    
    for i, (config, cam, temp, pred, label) in enumerate(
            zip(selected_configs, cams, selected_Ts, predictions, selected_labels)):
        
        ax = axes[i]
        
        true_phase = "Ordered" if label == 1 else "Disordered"
        pred_phase = "Ordered" if pred == 1 else "Disordered"
        
        overlay = overlay_gradcam(config, cam, alpha=args.alpha)
        ax.imshow(overlay, interpolation='nearest')
        ax.set_xticks([])
        ax.set_yticks([])
        
        title_lines = [
            f"T={temp:.2f}",
            f"True: {true_phase}",
            f"Pred: {pred_phase}"
        ]
        title = "\n".join(title_lines)
        ax.set_title(title, fontsize=10, fontweight='bold', pad=10, linespacing=1.3)
        
        for spine in ax.spines.values():
            spine.set_edgecolor('black')
            spine.set_linewidth(1.5)
    
    plt.tight_layout(rect=[0, 0, 0.90, 1])
    
    bbox_first = axes[0].get_position()
    
    cax = fig.add_axes([0.92, bbox_first.y0, 0.015, bbox_first.height])
    cmap = plt.get_cmap('RdYlBu_r')
    norm = plt.Normalize(vmin=0, vmax=1)
    cb = plt.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), 
                     cax=cax, orientation='vertical')
    cb.set_label('Activation Intensity', fontsize=11, rotation=90, 
                labelpad=10)
    cb.ax.tick_params(labelsize=10)
    
    os.makedirs(args.outdir, exist_ok=True)
    output_path = os.path.join(args.outdir, 'gradcam_analysis.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight', 
                facecolor='white', edgecolor='none')
    plt.close()
    
    print(f"\n✓ Grad-CAM analysis saved to: {output_path}")
    print(f"  Figure size: {3 * args.n_temps}×4.5 inches at 300 DPI")
    print(f"  Visualization: Grad-CAM overlays with predictions")
    print(f"  Color scale: Blue (low) → Yellow → Red (high activation)")
    print(f"  Check marks indicate correct predictions, X indicates incorrect")


if __name__ == '__main__':
    main()

