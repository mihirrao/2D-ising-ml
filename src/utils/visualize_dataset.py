"""
Generate publication-quality figure showcasing the Ising model phase transition.

This script creates a single-panel figure with:
- Sample spin configurations at the top, aligned with their temperatures
- Energy and magnetization vs temperature on dual y-axes below
"""

import argparse
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import os
from src.ising.io import load_npz


def plot_configuration(ax, config, title, fontsize=12):
    """Plot a single spin configuration in grayscale."""
    ax.imshow(config, cmap='gray', vmin=-1, vmax=1, interpolation='nearest')
    ax.set_title(title, fontsize=fontsize, fontweight='bold', pad=8)
    ax.set_xticks([])
    ax.set_yticks([])
    # Add thin border
    for spine in ax.spines.values():
        spine.set_edgecolor('black')
        spine.set_linewidth(1.0)


def plot_observable_vs_temp(ax, T_unique, obs_mean, obs_std, ylabel, Tc, 
                            marker='o', markersize=5):
    """Plot observable vs temperature with error bars."""
    ax.errorbar(T_unique, obs_mean, yerr=obs_std, 
                fmt=marker, color='black', markersize=markersize,
                linewidth=1.5, capsize=3, capthick=1.5,
                elinewidth=1.5, markerfacecolor='white', 
                markeredgewidth=1.5, markeredgecolor='black')
    
    # Add vertical line at critical temperature
    ymin, ymax = ax.get_ylim()
    ax.axvline(Tc, color='gray', linestyle='--', linewidth=2.0, 
               alpha=0.7, zorder=1)
    ax.text(Tc, ymax * 0.95, r'$T_c$', fontsize=13, 
            ha='center', va='top', fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white', 
                     edgecolor='gray', linewidth=1.5))
    
    ax.set_xlabel('Temperature (T)', fontsize=14)
    ax.set_ylabel(ylabel, fontsize=14)
    ax.grid(True, alpha=0.3, color='gray', linestyle=':', linewidth=0.8)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.tick_params(labelsize=12)


def main():
    p = argparse.ArgumentParser(description='Visualize Ising model dataset')
    p.add_argument('--data', type=str, default='data/raw/ising_L32.npz',
                   help='Path to NPZ data file')
    p.add_argument('--outdir', type=str, default='results/figures',
                   help='Output directory for figure')
    p.add_argument('--n_configs', type=int, default=5,
                   help='Number of sample configurations to show')
    p.add_argument('--seed', type=int, default=42,
                   help='Random seed for reproducibility')
    args = p.parse_args()
    
    # Load data
    print(f"Loading data from {args.data}...")
    data = load_npz(args.data)
    X = data['X']  # spin configurations
    T = data['T']  # temperatures
    E = data['e']  # energy per spin
    M = data['m']  # magnetization per spin
    Tc = float(data['Tc'])  # critical temperature
    L = int(data['L'])  # lattice size
    
    print(f"Loaded {len(X)} configurations at lattice size L={L}")
    print(f"Temperature range: [{T.min():.2f}, {T.max():.2f}]")
    print(f"Critical temperature: Tc={Tc:.3f}")
    
    # Get unique temperatures and compute statistics
    T_unique = np.sort(np.unique(T))
    E_mean = np.array([E[T == t].mean() for t in T_unique])
    E_std = np.array([E[T == t].std() for t in T_unique])
    M_abs = np.abs(M)
    M_mean = np.array([M_abs[T == t].mean() for t in T_unique])
    M_std = np.array([M_abs[T == t].std() for t in T_unique])
    
    # Select clean temperature values for display (like 1.5, 2.0, 2.5, 3.0, 3.5)
    T_min = T_unique.min()
    T_max = T_unique.max()
    
    # Define clean target temperatures (nice round numbers)
    # For standard range 1.5-3.5, use: 1.5, 2.0, 2.5, 3.0, 3.5
    if T_min <= 1.5 and T_max >= 3.5:
        if args.n_configs == 3:
            clean_targets = [1.5, 2.5, 3.5]
        elif args.n_configs == 4:
            clean_targets = [1.5, 2.0, 2.5, 3.5]
        elif args.n_configs == 5:
            clean_targets = [1.5, 2.0, 2.5, 3.0, 3.5]
        else:
            # Use evenly spaced clean values
            clean_targets = [1.5, 2.0, 2.5, 3.0, 3.5][:args.n_configs]
    else:
        # For custom ranges, generate clean values
        if args.n_configs == 3:
            clean_targets = [T_min, (T_min + T_max) / 2, T_max]
        else:
            # Round to nearest 0.5
            step = (T_max - T_min) / (args.n_configs - 1)
            step = round(step * 2) / 2  # Round to 0.5
            clean_targets = [round(T_min + i * step, 1) for i in range(args.n_configs)]
    
    # Find closest actual temperatures to clean targets
    selected_Ts = []
    for target in clean_targets:
        closest_idx = np.argmin(np.abs(T_unique - target))
        selected_Ts.append(T_unique[closest_idx])
    
    selected_Ts = np.sort(np.unique(selected_Ts))
    
    # If we don't have enough unique values, fill with additional clean targets
    if len(selected_Ts) < args.n_configs:
        # Try additional clean values
        additional_targets = [1.75, 2.25, 2.75, 3.25] if T_min <= 1.5 and T_max >= 3.5 else []
        for target in additional_targets:
            if len(selected_Ts) >= args.n_configs:
                break
            if T_min <= target <= T_max:
                closest_idx = np.argmin(np.abs(T_unique - target))
                if T_unique[closest_idx] not in selected_Ts:
                    selected_Ts = np.append(selected_Ts, T_unique[closest_idx])
        selected_Ts = np.sort(selected_Ts)[:args.n_configs]
    
    # For each selected temperature, pick a random configuration
    np.random.seed(args.seed)
    selected_configs = []
    for temp in selected_Ts:
        indices = np.where(T == temp)[0]
        idx = np.random.choice(indices)
        selected_configs.append(X[idx])
    
    # Set publication-quality style
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
    
    # Create figure with custom layout
    fig = plt.figure(figsize=(14, 9))
    
    # Main plot area - leave space at top for taller configurations
    ax_combined = fig.add_axes([0.08, 0.12, 0.82, 0.45])
    
    # Plot Energy on left y-axis
    ax_energy = ax_combined
    # Adjust marker size based on number of points
    n_temps = len(T_unique)
    marker_size = max(3, 5 - int(n_temps / 20))  # Smaller markers for more points
    line1 = ax_energy.errorbar(T_unique, E_mean, yerr=E_std, 
                fmt='o', color='black', markersize=marker_size,
                linewidth=1.5, capsize=2, capthick=1.5,
                elinewidth=1.5, markerfacecolor='white', 
                markeredgewidth=1.5, markeredgecolor='black',
                label='Energy per Spin (e)')
    
    ax_energy.set_xlabel('Temperature (T)', fontsize=14)
    ax_energy.set_ylabel('Energy per Spin (e)', fontsize=14, color='black')
    ax_energy.tick_params(axis='y', labelcolor='black', labelsize=12)
    ax_energy.tick_params(axis='x', labelsize=12)
    ax_energy.spines['top'].set_visible(False)
    
    # Set y-axis limits with padding to prevent cutting off points
    y_energy_range = E_mean.max() - E_mean.min()
    y_energy_padding = 0.05 * y_energy_range
    ax_energy.set_ylim(E_mean.min() - y_energy_padding, E_mean.max() + y_energy_padding)
    
    # Plot Magnetization on right y-axis
    ax_mag = ax_energy.twinx()
    marker_size_x = max(4, 7 - int(n_temps / 20))  # Smaller markers for more points
    line2 = ax_mag.errorbar(T_unique, M_mean, yerr=M_std, 
                fmt='x', color='black', markersize=marker_size_x,
                linewidth=1.5, capsize=2, capthick=1.5,
                elinewidth=1.5, markeredgewidth=1.5,
                label='Absolute Magnetization per Spin (|m|)')
    
    ax_mag.set_ylabel('Absolute Magnetization per Spin (|m|)', fontsize=14, color='black')
    ax_mag.tick_params(axis='y', labelcolor='black', labelsize=12)
    ax_mag.spines['top'].set_visible(False)
    
    # Set y-axis limits with padding to prevent cutting off points
    y_mag_range = M_mean.max() - M_mean.min()
    y_mag_padding = 0.05 * y_mag_range
    ax_mag.set_ylim(M_mean.min() - y_mag_padding, M_mean.max() + y_mag_padding)
    
    # Add padding to x-axis to prevent cutting off edge points
    T_range = T_unique.max() - T_unique.min()
    padding = 0.03 * T_range
    ax_energy.set_xlim(T_unique.min() - padding, T_unique.max() + padding)
    
    # Select clean temperature values for x-axis ticks
    # Use nice round numbers like 1.5, 2.0, 2.5, 3.0, 3.5
    T_min_plot = T_unique.min()
    T_max_plot = T_unique.max()
    
    # Generate clean target ticks
    if n_temps > 25:
        n_ticks = 8
    elif n_temps > 15:
        n_ticks = 10
    else:
        n_ticks = min(12, n_temps)
    
    # Create clean target values (round to 0.5 or 1.0)
    tick_targets = []
    if T_min_plot <= 1.5 and T_max_plot >= 3.5:
        # Standard range: use 0.5 increments
        tick_targets = [1.5, 2.0, 2.5, 3.0, 3.5]
        # Add more if needed
        if n_ticks > 5:
            # Add intermediate values
            tick_targets = [1.5, 1.75, 2.0, 2.25, 2.5, 2.75, 3.0, 3.25, 3.5][:n_ticks]
    else:
        # Custom range: generate clean values
        step = (T_max_plot - T_min_plot) / (n_ticks - 1)
        # Round step to nearest 0.1
        step = round(step * 10) / 10
        tick_targets = [round(T_min_plot + i * step, 1) for i in range(n_ticks)]
    
    # Find closest actual temperatures to clean targets
    tick_values = []
    for target in tick_targets:
        if T_min_plot <= target <= T_max_plot:
            closest_idx = np.argmin(np.abs(T_unique - target))
            tick_values.append(T_unique[closest_idx])
    
    # Remove duplicates and sort
    tick_values = np.sort(np.unique(tick_values))
    
    # Set ticks and format labels
    ax_energy.set_xticks(tick_values)
    # Format to show 1 decimal place for clean values, 2 for others
    tick_labels = []
    for t in tick_values:
        if abs(t - round(t, 1)) < 0.01:  # Close to 0.1 precision
            tick_labels.append(f'{t:.1f}')
        else:
            tick_labels.append(f'{t:.2f}')
    ax_energy.set_xticklabels(tick_labels)
    
    # Add vertical line at critical temperature (after setting all limits)
    ymin_e, ymax_e = ax_energy.get_ylim()
    ax_energy.axvline(Tc, color='gray', linestyle='--', linewidth=2.0, 
                     alpha=0.7, zorder=1)
    ax_energy.text(Tc, ymin_e + (ymax_e - ymin_e) * 0.05, r'$T_c$', fontsize=13, 
                  ha='center', va='bottom', fontweight='bold',
                  bbox=dict(boxstyle='round,pad=0.3', facecolor='white', 
                           edgecolor='gray', linewidth=1.5))
    
    ax_energy.grid(True, alpha=0.3, color='gray', linestyle=':', linewidth=0.8)
    
    # Combined legend - position in center right to avoid data overlap
    lines = [line1, line2]
    labels = [l.get_label() for l in lines]
    ax_energy.legend(lines, labels, loc='center right', frameon=True, 
                    fancybox=False, edgecolor='black', fontsize=11)
    
    # Add spin configurations above the plot, aligned with temperature axis
    config_fig_width = 0.165  # Width in figure coordinates (larger, close but not touching)
    config_fig_height = config_fig_width * 1.5  # Rectangular aspect ratio (1.5x taller)
    config_y_position = 0.65  # Position configurations at top (adjusted for taller configs)
    
    # Get the position of the main plot
    bbox_plot = ax_combined.get_position()
    
    # Use the actual axis limits for alignment (including padding)
    xlim = ax_combined.get_xlim()
    
    # Add configurations at their corresponding temperatures
    for i, (config, temp) in enumerate(zip(selected_configs, selected_Ts)):
        # Calculate horizontal position based on temperature using actual axis limits
        temp_fraction = (temp - xlim[0]) / (xlim[1] - xlim[0])
        center_x = bbox_plot.x0 + temp_fraction * bbox_plot.width
        
        # Create inset axes for each configuration
        ax_inset = fig.add_axes([center_x - config_fig_width/2, 
                                 config_y_position, 
                                 config_fig_width, 
                                 config_fig_height])
        ax_inset.imshow(config, cmap='gray', vmin=-1, vmax=1, interpolation='nearest')
        ax_inset.set_xticks([])
        ax_inset.set_yticks([])
        
        # Add border
        for spine in ax_inset.spines.values():
            spine.set_edgecolor('black')
            spine.set_linewidth(1.5)
        
        # Determine phase label
        if temp < Tc * 0.95:
            phase = "Ordered"
        elif temp > Tc * 1.05:
            phase = "Disordered"
        else:
            phase = "Critical"
        
        # Add label above configuration (format to 2 decimal places)
        ax_inset.text(0.5, 1.05, f"T={temp:.2f}", 
                     transform=ax_inset.transAxes, fontsize=10,
                     ha='center', va='bottom', fontweight='bold')
        
        # Add subtle connecting line from configuration to plot
        line_start_y = config_y_position - 0.01  # Just below the config
        line_end_y = bbox_plot.y0 + bbox_plot.height + 0.01  # Just above the plot
        fig.add_artist(plt.Line2D([center_x, center_x], 
                                   [line_start_y, line_end_y],
                                   transform=fig.transFigure,
                                   color='gray', linewidth=1.0, 
                                   alpha=0.4, linestyle='-'))
    
    # Save figure
    os.makedirs(args.outdir, exist_ok=True)
    output_path = os.path.join(args.outdir, 'phase_transition.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight', 
                facecolor='white', edgecolor='none')
    plt.close()
    
    print(f"\n✓ Figure saved to: {output_path}")
    print(f"  Figure size: 14×9 inches at 300 DPI")
    print(f"  Format: Grayscale, publication-quality")
    print(f"  Layout: Sample configurations above plot, aligned with temperature axis")
    print(f"          Energy & Magnetization on dual y-axes")


if __name__ == '__main__':
    main()

