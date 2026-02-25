import numpy as np

np.random.seed(42)

# ===============================================================
# Core Parameters (Pure FSEP, no fitting)
# ===============================================================

D      = 2.4739   # Hausdorff dimension of 3D Apollonian sphere packing
alpha  = 3 - D    # ≈ 0.526
layers = 5        # Recursion layers before observational cutoff
num_samples = 5000

# ===============================================================
# Core Functions
# ===============================================================

def dark_fraction_from_beta(beta):
    """Single-layer dark fraction from packing ratio beta."""
    return 1 - beta ** alpha


def compute_dark_fraction(beta_min, beta_max, sampling='uniform', label=""):
    """Multi-layer cumulative dark fraction over beta range."""
    if sampling == 'log-uniform':
        log_samples = np.random.uniform(np.log(beta_min), np.log(beta_max), num_samples)
        beta_samples = np.exp(log_samples)
    else:
        beta_samples = np.random.uniform(beta_min, beta_max, num_samples)

    beta_avg         = np.mean(beta_samples)
    f_hidden_single  = dark_fraction_from_beta(beta_avg)

    f_total_dark  = 0.0
    remaining_vis = 1.0

    print(f"\n{label} ({sampling} sampling)")
    print("=" * 70)
    print(f"beta range : [{beta_min:.0e}, {beta_max:.2e}]")
    print(f"Mean beta  : {beta_avg:.4e}")
    print(f"Single-layer hidden efficiency: {f_hidden_single:.4f}")
    print("-" * 70)
    print(f"{'Layer':>5} | {'Layer Hidden':>12} | {'Remaining Visible':>17} | {'Cumulative Dark':>15}")
    print("-" * 70)

    for i in range(layers):
        layer_dark    = remaining_vis * f_hidden_single
        f_total_dark += layer_dark
        remaining_vis *= (1 - f_hidden_single)
        print(f"{i+1:5d} | {layer_dark:12.4f} | {remaining_vis:17.4f} | {f_total_dark:15.4f}")

    print("-" * 70)
    print(f"TOTAL DARK FRACTION AFTER {layers} LAYERS: {f_total_dark:.1%}")
    print(f"Remaining visible (baryonic)  : {remaining_vis:.1e}")
    print("=" * 70)
    return f_total_dark, beta_avg


# ===============================================================
# Scale Hierarchy Runs
# ===============================================================

print("\n" + "#" * 70)
print("# FSEP — SCALE HIERARCHY DARK FRACTIONS")
print("#" * 70)

# 1. Planck scale to human hair diameter
compute_dark_fraction(1e-35, 1e-4, sampling='uniform',     label="1. Planck → Human Hair (Uniform)")
compute_dark_fraction(1e-35, 1e-4, sampling='log-uniform', label="1. Planck → Human Hair (Log-Uniform)")

# 2. Star scale to galaxy scale
compute_dark_fraction(1e-6, 0.1, sampling='uniform',     label="2. Star → Galaxy (Uniform)")
compute_dark_fraction(1e-6, 0.1, sampling='log-uniform', label="2. Star → Galaxy (Log-Uniform)")

# 3. Galaxy scale to visible universe
compute_dark_fraction(1e-5, 0.01, sampling='uniform',     label="3. Galaxy → Visible Universe (Uniform)")
compute_dark_fraction(1e-5, 0.01, sampling='log-uniform', label="3. Galaxy → Visible Universe (Log-Uniform)")

# 4. Original Milky Way / 2 Planck layers reference
beta_min_orig = 1e-3
beta_max_orig = 0.01 + 1e-3 * np.exp(3)
compute_dark_fraction(beta_min_orig, beta_max_orig, sampling='uniform',
                      label="4. Original: Milky Way / 2 Planck Layers")


# ===============================================================
# Merger Dark Fraction Drop — First Principles
# ===============================================================
#
# f_dark = 1 - beta^alpha,  alpha = 3 - D ≈ 0.526
#
# Merger physically expands the outer halo (core contracts, halo puffs).
# If beta tracks outer occupancy scale, beta increases → f_dark decreases.
#
# k = beta scaling factor from merger
#   k_virial   = 2^(1/3) ≈ 1.26   (pure M∝R^3 mass doubling)
#   k_realistic = 1.37             (upper range of observed halo expansion)
#   k_strong    = 1.5              (strong redistribution)
#
# Drop = f_dark(beta) - f_dark(k * beta)
#      = beta^alpha * (k^alpha - 1)
# ===============================================================

print("\n\n" + "#" * 70)
print("# MERGER DARK FRACTION DROP")
print("#" * 70)

beta_pre = 0.05   # representative single-galaxy packing ratio

k_values = [
    ("Pure virial  (k = 2^1/3 ≈ 1.26)",  2 ** (1/3)),
    ("Realistic    (k = 1.37)",            1.37),
    ("Strong       (k = 1.50)",            1.50),
    ("Very strong  (k = 2.00)",            2.00),
]

print(f"\nBase beta (pre-merger) = {beta_pre}")
print(f"alpha = 3 - D = {alpha:.4f}\n")
print(f"{'Scenario':<38} | {'k':>5} | {'beta_pre':>8} | {'beta_post':>9} | "
      f"{'f_pre':>6} | {'f_post':>6} | {'DROP':>6}")
print("-" * 90)

for label, k in k_values:
    beta_post = beta_pre * k
    f_pre     = dark_fraction_from_beta(beta_pre)
    f_post    = dark_fraction_from_beta(beta_post)
    drop      = f_pre - f_post
    print(f"{label:<38} | {k:5.3f} | {beta_pre:8.4f} | {beta_post:9.4f} | "
          f"{f_pre:6.4f} | {f_post:6.4f} | {drop*100:5.2f}%")

# --- Analytic drop formula ---
print("\n--- Analytic formula ---")
print("Drop = beta^alpha * (k^alpha - 1)")
print(f"With beta={beta_pre}, alpha={alpha:.4f}:")
for label, k in k_values:
    drop_analytic = beta_pre ** alpha * (k ** alpha - 1)
    print(f"  {label:<38}  drop = {drop_analytic*100:.2f}%")


# ===============================================================
# Sensitivity: drop vs beta_pre for fixed k=1.37
# ===============================================================

print("\n\n--- Sensitivity: drop vs beta_pre (k=1.37) ---")
print(f"{'beta_pre':>10} | {'f_dark_pre':>10} | {'f_dark_post':>11} | {'drop':>6}")
print("-" * 48)
k_fixed = 1.37
for b in [0.01, 0.02, 0.05, 0.10, 0.20, 0.30]:
    fp  = dark_fraction_from_beta(b)
    fpo = dark_fraction_from_beta(b * k_fixed)
    d   = fp - fpo
    print(f"{b:10.3f} | {fp:10.4f} | {fpo:11.4f} | {d*100:5.2f}%")


# ===============================================================
# SINGLE-LAYER CONSTRAINED VERSION
# ===============================================================
#
# The observable universe (and observer) can resolve exactly ONE
# fractal layer — either one step down or one step up, not both,
# not compounding. So the recursion loop is physically inadmissible.
#
# Dark fraction is simply:
#   f_dark = 1 - beta^(3-D)
#
# One evaluation. No accumulation. No remaining_vis loop.
# ===============================================================

print("\n\n" + "#" * 70)
print("# SINGLE-LAYER CONSTRAINED (physically admissible)")
print("# One recursion level only — no compounding")
print("#" * 70)

def dark_fraction_single_layer(beta_min, beta_max, sampling='uniform', label=""):
    if sampling == 'log-uniform':
        log_samples = np.random.uniform(np.log(beta_min), np.log(beta_max), num_samples)
        beta_samples = np.exp(log_samples)
    else:
        beta_samples = np.random.uniform(beta_min, beta_max, num_samples)

    beta_avg = np.mean(beta_samples)
    f_dark   = dark_fraction_from_beta(beta_avg)

    print(f"\n{label} ({sampling} sampling)")
    print("=" * 70)
    print(f"beta range : [{beta_min:.0e}, {beta_max:.2e}]")
    print(f"Mean beta  : {beta_avg:.4e}")
    print(f"f_dark     : {f_dark:.4f}  ({f_dark:.1%})")
    print(f"f_visible  : {1-f_dark:.4f}  ({1-f_dark:.1%})")
    print("=" * 70)
    return f_dark, beta_avg

# Same scale ranges as before, now single layer
dark_fraction_single_layer(1e-35, 1e-4,  sampling='log-uniform', label="Planck → Human Hair")
dark_fraction_single_layer(1e-6,  0.1,   sampling='log-uniform', label="Star → Galaxy")
dark_fraction_single_layer(1e-5,  0.01,  sampling='log-uniform', label="Galaxy → Visible Universe")
dark_fraction_single_layer(beta_min_orig, beta_max_orig, sampling='uniform', label="Original: Milky Way / 2 Planck Layers")

# --- Single-layer merger drop ---
print("\n\n--- Single-layer merger drop (same as before, already was 1 layer) ---")
print(f"beta_pre = {beta_pre},  alpha = {alpha:.4f}\n")
print(f"{'Scenario':<38} | {'k':>5} | {'f_pre':>6} | {'f_post':>6} | {'DROP':>6}")
print("-" * 65)
for label, k in k_values:
    beta_post = beta_pre * k
    f_pre     = dark_fraction_from_beta(beta_pre)
    f_post    = dark_fraction_from_beta(beta_post)
    drop      = f_pre - f_post
    print(f"{label:<38} | {k:5.3f} | {f_pre:6.4f} | {f_post:6.4f} | {drop*100:5.2f}%")

print("\nNote: merger drop was already single-layer. The multi-layer section")
print("above was only the scale hierarchy runs that change under this constraint.")