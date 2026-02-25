import numpy as np

# --- Core Parameters (Pure FSEP, no fitting) ---
D = 2.4739          # Hausdorff dimension of 3D Apollonian sphere packing
layers = 5          # Number of recursion layers before observational cutoff
num_samples = 5000  # Samples for averaging beta
np.random.seed(42)  # Reproducible

# --- Function to compute cumulative hidden fraction ---
def compute_dark_fraction(beta_min, beta_max, sampling='uniform', label=""):
    if sampling == 'log-uniform':
        log_min = np.log(beta_min)
        log_max = np.log(beta_max)
        log_samples = np.random.uniform(log_min, log_max, num_samples)
        beta_samples = np.exp(log_samples)
    else:
        beta_samples = np.random.uniform(beta_min, beta_max, num_samples)
    
    beta_avg = np.mean(beta_samples)
    f_hidden_single = 1 - beta_avg ** (3 - D)
    
    f_total_dark = 0.0
    remaining_vis = 1.0
    print(f"\n{label} ({sampling} sampling)")
    print("=" * 70)
    print(f"β range: [{beta_min:.0e}, {beta_max:.2e}]")
    print(f"Mean β: {beta_avg:.4e}")
    print(f"Single-layer hidden efficiency: {f_hidden_single:.4f}")
    print("-" * 40)
    print("Layer | Layer Hidden | Remaining Visible | Cumulative Dark")
    print("-" * 40)
    
    for i in range(layers):
        layer_dark = remaining_vis * f_hidden_single
        f_total_dark += layer_dark
        remaining_vis *= (1 - f_hidden_single)
        print(f"{i+1:5d} | {layer_dark:12.4f} | {remaining_vis:17.4f} | {f_total_dark:15.4f}")
    
    print("-" * 40)
    print(f"TOTAL DARK FRACTION AFTER {layers} LAYERS: {f_total_dark:.1%}")
    print(f"Remaining visible (baryonic?): {remaining_vis:.1e}")
    print(f"Physics takeaway: {'Near-total dark dominance' if f_total_dark > 0.999 else 'Significant hidden buildup'}")
    print("=" * 70)

# --- 1. Planck scale to human hair diameter (~10^{-35} m to ~10^{-4} m) ---
compute_dark_fraction(1e-35, 1e-4, sampling='uniform', label="1. Planck → Human Hair (Uniform)")
compute_dark_fraction(1e-35, 1e-4, sampling='log-uniform', label="1. Planck → Human Hair (Log-Uniform)")

# --- 2. Star scale to galaxy scale (~10^8–10^9 m to ~10^{20–21} m) ---
compute_dark_fraction(1e-6, 0.1, sampling='uniform', label="2. Star → Galaxy (Uniform)")
compute_dark_fraction(1e-6, 0.1, sampling='log-uniform', label="2. Star → Galaxy (Log-Uniform)")

# --- 3. Galaxy scale to visible universe (~10^{21} m to ~4×10^{26} m) ---
compute_dark_fraction(1e-5, 0.01, sampling='uniform', label="3. Galaxy → Visible Universe (Uniform)")
compute_dark_fraction(1e-5, 0.01, sampling='log-uniform', label="3. Galaxy → Visible Universe (Log-Uniform)")

# --- Your original "Milky Way hierarchy / 2 Planck layers" example for reference ---
beta_min_orig = 1e-3
beta_max_orig = 0.01 + 1e-3 * np.exp(3)  # ≈0.0301
compute_dark_fraction(beta_min_orig, beta_max_orig, sampling='uniform', label="Original: Milky Way / 2 Planck Layers")