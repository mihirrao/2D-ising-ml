# Latent Manifolds of the 2D Ising Model

This repository presents a comprehensive analysis of latent representations learned from 2D Ising model spin configurations using Principal Component Analysis (PCA), Variational Autoencoders (VAE), and Convolutional Neural Networks (CNN). The study explores how different dimensionality reduction techniques capture the phase transition and physical properties of the Ising model.

## Dataset

### Generation

The dataset consists of 2D Ising model spin configurations generated using Monte Carlo simulations. Key parameters:

- **Lattice size**: 32×32 spins
- **Temperature range**: 1.5 to 3.5 (in units of J/k_B)
- **Critical temperature**: Tc ≈ 2.269
- **Base temperature spacing**: 0.25 (20 base temperatures)
- **Enhanced sampling near Tc**: Additional 10 temperatures with 0.1 spacing within ±0.5 of Tc
- **Total temperatures**: 30 distinct temperature points
- **Samples per temperature**: 2000 configurations
- **Total dataset size**: 60,000 spin configurations

### Simulation Details

Configurations are generated using the **Wolff cluster algorithm**, which efficiently samples near the critical point. The simulation parameters are:

- **Thermalization steps**: 200 Monte Carlo steps
- **Sampling interval**: 5 steps between saved configurations
- **Boundary conditions**: Periodic

Temperature-dependent parameters are used to enhance physical realism:
- Lower temperatures (ordered regime): Increased thermalization and sampling intervals to capture larger spin clusters
- Higher temperatures (disordered regime): Standard parameters to capture higher entropy configurations

### Dataset Properties

The dataset captures the phase transition at Tc ≈ 2.269:

- **Ordered phase (T < Tc)**: Configurations exhibit large spin clusters with high magnetization (|m| ≈ 1)
- **Disordered phase (T ≥ Tc)**: Configurations show random spin patterns with low magnetization (|m| ≈ 0)

Each configuration is stored as a 32×32 array of spin values (-1 or +1), along with:
- Temperature (T)
- Phase label (ordered/disordered based on T < Tc)
- Magnetization per spin (m)
- Energy per spin (e)

The dataset is split into train/validation/test sets (80/10/10) with a fixed random seed for reproducibility.

![Phase Transition](results/figures/phase_transition.png)

*Figure 1: Phase transition visualization showing sample spin configurations at different temperatures (top) and the corresponding energy and magnetization vs temperature (bottom). The vertical dashed line indicates the critical temperature Tc ≈ 2.269.*

## Principal Component Analysis (PCA)

### Methodology

PCA is applied directly to the flattened 32×32 = 1024-dimensional spin configurations. The data is standardized (zero mean, unit variance) before computing principal components.

### Results

**Variance Explained**:
- PC1 captures the dominant variance in the dataset
- PC1 shows strong correlation with magnetization (R² values typically > 0.8)
- The first few principal components capture the phase transition structure

**Visualizations**:

1. **PC1 vs PC2 (3-panel figure)**:
   - Panel 1: Colored by phase (ordered/disordered) - reveals clear separation between phases
   - Panel 2: Colored by magnetization - shows smooth gradient from high to low |m|
   - Panel 3: Colored by energy per spin - correlates with temperature and phase

![PCA PC1 vs PC2](results/figures/pca.png)

*Figure 2: Principal Component Analysis of Ising configurations. Left: PC1 vs PC2 colored by phase (ordered/disordered). Middle: Colored by magnetization. Right: Colored by energy per spin. PC1 captures the dominant variance and is highly correlated with the order parameter.*

2. **Magnetization vs PC1**:
   - Direct correlation plot showing linear relationship between PC1 and magnetization
   - R² value quantifies the strength of correlation
   - Demonstrates that PC1 primarily captures the order parameter

![PCA PC1 vs Magnetization](results/figures/pca_pc1_vs_m.png)

*Figure 3: Magnetization vs PC1 correlation plot. The strong linear relationship (high R²) demonstrates that PC1 primarily captures the order parameter of the phase transition.*

3. **3D Visualization**:
   - PC1, PC2, PC3 scatter plot with three rotated views (0°, 45°, 90°)
   - Colored by phase to visualize the 3D manifold structure
   - Interactive version available for exploration

![PCA 3D Visualization](results/figures/pca_3d_visualization.png)

*Figure 4: 3D visualization of PCA showing PC1, PC2, and PC3 with three rotated views. The clear separation between ordered (blue) and disordered (red) phases is visible in the 3D manifold structure.*

### Key Findings

- PCA successfully identifies the phase transition as the primary source of variance
- PC1 is highly correlated with magnetization, the order parameter of the Ising model
- The 2D projection (PC1 vs PC2) shows clear separation between ordered and disordered phases
- Higher-order PCs capture additional structure but with diminishing variance

## Variational Autoencoder (VAE)

### Architecture

The VAE uses a convolutional encoder-decoder architecture:

**Encoder**:
- 3×3 convolutions: 1→32→64→128 channels
- Average pooling (2×2) after each block
- Fully connected layers map to latent space (μ, log σ²)

**Decoder**:
- Fully connected layer expands latent vector
- Transposed convolutions: 128→64→32→1 channels
- Tanh activation ensures output in [-1, 1] range

**Training**:
- Latent dimensions: z = 1 and z = 4
- Loss: Reconstruction (MSE) + β·KL divergence (β = 1.0)
- Optimizer: AdamW (lr = 1e-3)
- Batch size: 128
- Epochs: 5 (with early stopping)

### Results

**Training Curves**:
- Three-panel figure showing:
  - Total loss (reconstruction + KL)
  - Reconstruction loss
  - KL divergence loss
- Separate validation loss vs temperature plot shows performance across the phase transition

**Latent Space Analysis**:

1. **Magnetization vs Latent Dimensions**:
   - For z=1: Single latent dimension vs magnetization with R²
   - For z=4: All 4 latent dimensions vs magnetization (2×2 grid)
   - Quantifies how well each dimension captures the order parameter

![VAE Latent vs Magnetization](results/figures/vae_latent_vs_m.png)

*Figure 5: Correlation between VAE latent dimensions and magnetization. Top-left: z=1 model showing strong correlation. Remaining panels: All 4 dimensions of z=4 model, each with R² values quantifying the relationship with the order parameter.*

2. **Latent Space Visualization**:
   - 3 rows × 6 columns grid showing all 6 pairwise combinations of 4 latent dimensions
   - Each row has 3 panels:
     - Colored by phase (ordered/disordered)
     - Colored by magnetization (viridis colormap)
     - Colored by energy per spin (viridis colormap)
   - Reveals how different latent dimensions capture different aspects of the data

![VAE Latent Space](results/figures/vae_latent_space_latent4.png)

*Figure 6: Comprehensive visualization of VAE latent space for z=4 model. Each row shows a pairwise combination of latent dimensions, with three coloring schemes: phase (left), magnetization (middle), and energy (right). This reveals how different dimensions capture complementary information about the phase transition.*

3. **VAE PCA**:
   - PCA applied to VAE latent embeddings (z=4)
   - PC1 vs PC2 colored by phase, magnetization, and energy
   - Magnetization vs VAE PC1 correlation plot
   - Shows that VAE learns a structured latent space that can be further compressed

![VAE PCA](results/figures/vae_pca.png)

*Figure 7: PCA applied to VAE latent embeddings (z=4). The learned latent space can be further compressed while retaining phase information, demonstrating the structured nature of the VAE representation.*

![VAE PCA PC1 vs Magnetization](results/figures/vae_pca_pc1_vs_m.png)

*Figure 8: Magnetization vs VAE PC1 correlation. Even after applying PCA to the VAE latent space, the first principal component maintains strong correlation with the order parameter.*

4. **3D Visualization**:
   - VAE latent dimensions 1-3 in 3D space
   - PCA of VAE latent dimensions 1-3
   - Two rows × three columns showing rotated views
   - Interactive version available for exploration

![VAE 3D Visualization](results/figures/vae_3d_visualization.png)

*Figure 9: 3D visualization of VAE latent space. Top row: Direct VAE latent dimensions 1-3. Bottom row: PCA of VAE latent dimensions 1-3. Three rotated views (0°, 45°, 90°) reveal the 3D manifold structure of the learned representation.*

### Key Findings

- VAE learns a compact latent representation that captures the phase transition
- For z=1, the single latent dimension shows strong correlation with magnetization
- For z=4, different dimensions capture complementary information
- The learned latent space is structured and interpretable
- VAE PCA reveals that the latent space can be further compressed while retaining phase information

## Convolutional Neural Network (CNN)

### Architecture

The CNN uses a feature extractor followed by a classification head:

**Feature Extractor** (`net`):
- 3×3 convolutions: 1→32→64→128 channels
- Average pooling (2×2) after each block
- Flattened to feature vector

**Classification Head**:
- Fully connected: feature_dim → 256 → 2 (ordered/disordered)

**Training**:
- Task: Binary phase classification (ordered vs disordered)
- Loss: Cross-entropy
- Optimizer: AdamW (lr = 1e-3)
- Batch size: 128
- Epochs: 10

### Results

**Training Curves**:
- Two-panel figure:
  - Training and validation loss vs epoch
  - Validation accuracy vs temperature (shows performance across phase transition)

**Grad-CAM Analysis**:
- Gradient-weighted Class Activation Mapping visualization
- Shows which spatial regions the CNN focuses on for classification
- Examples from test set at different temperatures (near and far from Tc)
- Reveals that the model learns to identify spin clusters and boundaries

![Grad-CAM Analysis](results/figures/gradcam_analysis.png)

*Figure 10: Grad-CAM visualization showing which spatial regions the CNN focuses on for phase classification. The heatmaps reveal that the model learns to identify spin clusters and phase boundaries, with stronger attention near the critical temperature.*

**Linear Probe Analysis**:
- Extracts embeddings h(x) from the feature extractor (before classification head)
- Computes Pearson correlation (R²) between each embedding dimension and magnetization
- Identifies top 3 dimensions with highest R²
- Shows that learned features correlate with physical order parameter

![CNN Linear Probe](results/figures/cnn_linear_probe.png)

*Figure 11: CNN linear probe analysis. Top row: Top 3 embedding dimensions vs magnetization, showing strong correlations (R² values). Bottom row: PCA of CNN embeddings (PC1 vs PC2) colored by phase, magnetization, and energy. The supervised learning task produces embeddings that capture the order parameter without explicit supervision.*

**3D Visualization**:
- PCA of CNN embeddings (PC1, PC2, PC3) in 3D space
- Three rotated views (0°, 45°, 90°)
- Colored by phase to visualize the learned manifold
- Interactive version available for exploration

![CNN 3D Visualization](results/figures/cnn_linear_probe_3d_visualization.png)

*Figure 12: 3D visualization of PCA applied to CNN embeddings. Three rotated views show the learned manifold structure, with clear separation between ordered (blue) and disordered (red) phases in the embedding space.*

### Key Findings

- CNN achieves high accuracy on phase classification task
- Learned embeddings h(x) show strong correlation with magnetization
- The feature extractor learns representations that capture the order parameter
- Grad-CAM reveals attention to spin clusters and phase boundaries
- CNN embeddings form a structured manifold similar to PCA and VAE

## Comparison and Discussion

### Common Patterns

All three methods (PCA, VAE, CNN) reveal similar underlying structure:

1. **Phase separation**: Clear distinction between ordered and disordered phases in low-dimensional projections
2. **Order parameter correlation**: Strong correlation between first principal component/latent dimension and magnetization
3. **Manifold structure**: The data lies on a low-dimensional manifold that can be effectively visualized in 2D or 3D

### Differences

1. **PCA**: Linear, unsupervised, directly on raw configurations
   - Most interpretable (PC1 = magnetization)
   - No learning required
   - Limited to linear transformations

2. **VAE**: Nonlinear, unsupervised, learned compression
   - Learns compact, generative representation
   - Can sample new configurations
   - Latent space is structured and interpretable

3. **CNN**: Nonlinear, supervised, task-specific
   - Optimized for classification
   - Learns discriminative features
   - Embeddings capture order parameter despite no explicit supervision

### Physical Interpretation

The success of all three methods in capturing the phase transition demonstrates that:

- The phase transition is the dominant source of structure in the data
- The order parameter (magnetization) is encoded in the first few dimensions of any reasonable representation
- Machine learning methods can discover physical order parameters without explicit supervision
- The Ising model provides a rich testbed for understanding representation learning in physical systems

## Reproducibility

All scripts are provided for full reproducibility:

### Data Generation
```bash
bash generate_data.sh
```

### PCA Analysis
```bash
bash pca_analysis.sh
```

### VAE Training and Analysis
```bash
bash train_vae.sh        # Training
bash vae_postprocess.sh  # Analysis
```

### CNN Training and Analysis
```bash
bash train_cnn.sh        # Training
bash cnn_postprocess.sh # Analysis
```

### 3D Visualizations
```bash
bash pca_3d_visualization.sh
bash vae_3d_visualization.sh
bash cnn_linear_probe_3d_visualization.sh
```

### Interactive 3D Plots
```bash
bash pca_3d_interactive.sh
bash vae_3d_interactive.sh
bash cnn_linear_probe_3d_interactive.sh
```

All results are cached to speed up repeated analysis runs. Figures are saved to `results/figures/` and checkpoints to `results/checkpoints/`.

## Setup

```bash
python3 -m venv venv
source venv/bin/activate
pip install -U pip
pip install -r requirements.txt
```

## Citation

If you use this code or findings, please cite appropriately.
