import numpy as np
import matplotlib.pyplot as plt

np.random.seed(42)

# ===============================================================
# Functions (same as yours, jitter turned off for stability)
# ===============================================================

def approximate_soddy(c1, r1, c2, r2, c3, r3, c4, r4):
    r1 = float(r1)
    r2 = float(r2)
    r3 = float(r3)
    r4 = float(r4)
    
    k = np.array([1.0/r1, 1.0/r2, 1.0/r3, 1.0/r4])
    k = np.maximum(k, 1e-8)
    
    k_sum = np.sum(k)
    cross_sum = np.sum(k[:, None] * k[None, :] ) - np.sum(k**2)
    k_cross = 2 * np.sqrt(np.maximum(0.0, cross_sum))
    
    k_new1 = k_sum + k_cross
    k_new2 = k_sum - k_cross
    
    weights = k / np.sum(k)
    centers = np.vstack([c1, c2, c3, c4])
    center_avg = np.dot(weights, centers)
    
    jitter_scale = 0.0  # <--- turned off for reproducibility
    c_new1 = center_avg + jitter_scale * np.random.randn(3)
    c_new2 = center_avg + jitter_scale * np.random.randn(3)
    
    r_new1 = 1.0 / k_new1 if k_new1 > 1e-6 else 1000.0
    r_new2 = 1.0 / k_new2 if k_new2 > 1e-6 else 1000.0
    
    return (c_new1, float(r_new1)), (c_new2, float(r_new2))

def generate_shallow_spheres(separation=20.0, max_depth=2, min_r=0.008, max_spheres=400):
    spheres = []
    
    r_gal = 1.0
    c1 = np.array([-separation/2, 0.0, 0.0])
    c2 = np.array([ separation/2, 0.0, 0.0])
    spheres.append(np.array([c1[0], c1[1], c1[2], float(r_gal)]))
    spheres.append(np.array([c2[0], c2[1], c2[2], float(r_gal)]))
    
    r_seed = 0.25
    spheres.append(np.array([0.0,  0.35, 0.0, float(r_seed)]))
    spheres.append(np.array([0.0, -0.35, 0.0, float(r_seed)]))
    
    to_process = [(0,1,2,3)]
    
    level = 0
    while to_process and len(spheres) < max_spheres and level < max_depth:
        new_quads = []
        for quad in to_process:
            idxs = list(quad)
            cs = [spheres[i][:3] for i in idxs]
            rs = [float(spheres[i][3]) for i in idxs]
            
            sol1, sol2 = approximate_soddy(cs[0], rs[0], cs[1], rs[1], cs[2], rs[2], cs[3], rs[3])
            for c_new, r_new in [sol1, sol2]:
                if not (min_r < r_new < 20.0):
                    continue
                
                dists = [np.linalg.norm(c_new - s[:3]) for s in spheres]
                min_sep = min(d - s[3] - r_new for d, s in zip(dists, spheres))
                if min_sep < -0.02:
                    continue
                
                spheres.append(np.array([c_new[0], c_new[1], c_new[2], float(r_new)]))
                new_idx = len(spheres) - 1
                
                new_quads.append((0, 1, 2, new_idx))
                new_quads.append((0, 1, 3, new_idx))
                if len(new_quads) > 40:
                    break
        
        to_process = new_quads
        level += 1
    
    return np.array(spheres)

def estimate_crossing_fraction(spheres, n_lines=180000):
    if len(spheres) < 4:
        return 0.0, 0.0
    
    c_g1 = spheres[0, :3]
    c_g2 = spheres[1, :3]
    direction = c_g2 - c_g1
    dist = np.linalg.norm(direction)
    unit_dir = direction / dist
    
    total_inside = 0.0
    total_length = 0.0
    hit_count = 0
    
    for _ in range(n_lines):
        perp1 = np.cross(unit_dir, np.array([0.,0.,1.]))
        norm = np.linalg.norm(perp1)
        if norm < 1e-4:
            perp1 = np.cross(unit_dir, np.array([1.,0.,0.]))
            norm = np.linalg.norm(perp1)
        perp1 /= norm
        perp2 = np.cross(unit_dir, perp1)
        perp2 /= np.linalg.norm(perp2)
        
        offset_mag = np.random.uniform(0, 0.5)
        phi = np.random.uniform(0, 2*np.pi)
        offset = offset_mag * (np.cos(phi) * perp1 + np.sin(phi) * perp2)
        
        start = c_g1 + 0.1 * direction + offset
        end   = c_g2   - 0.1 * direction + offset
        seg_len = np.linalg.norm(end - start)
        if seg_len < 1e-6: continue
        dir_vec = (end - start) / seg_len
        
        length_inside = 0.0
        hit_micro = False
        
        for sph in spheres[2:]:
            c = sph[:3]
            r = float(sph[3])
            oc = c - start
            proj = np.dot(oc, dir_vec)
            d2 = np.dot(oc, oc) - proj**2
            if d2 >= r**2: continue
            delta = np.sqrt(max(0.0, r**2 - d2))
            t1 = proj - delta
            t2 = proj + delta
            enter = max(t1, 0.0)
            ex = min(t2, seg_len)
            if enter < ex:
                length_inside += ex - enter
                hit_micro = True
        
        total_inside += length_inside
        total_length += seg_len
        if hit_micro:
            hit_count += 1
    
    frac_length = total_inside / total_length if total_length > 0 else 0.0
    frac_hit = hit_count / n_lines if n_lines > 0 else 0.0
    
    return frac_length, frac_hit

# ===============================================================
# Main: multi-run stats + plots
# ===============================================================

print("=== Fractal Tennis α proxy — multi-run statistics & plots ===\n")

# Configuration
separations = np.linspace(19.5, 21.5, 11)   # 18 to 23, 11 points
n_runs_per_sep = 8                         # more runs = better mean/std
n_lines = 180000                           # high for low noise

results = []  # (sep, mean_frac, std_frac, inv_mean)

for sep in separations:
    frac_lengths = []
    for run_idx in range(n_runs_per_sep):
        spheres = generate_shallow_spheres(
            separation=sep,
            max_depth=2,
            min_r=0.008,
            max_spheres=400
        )
        frac, _ = estimate_crossing_fraction(spheres, n_lines=n_lines)
        frac_lengths.append(frac)
        print(f"  run {run_idx+1}/{n_runs_per_sep} at sep={sep:.2f}: frac={frac:.6f}")
    
    mean_frac = np.mean(frac_lengths)
    std_frac = np.std(frac_lengths)
    inv_mean = 1.0 / mean_frac if mean_frac > 1e-10 else np.inf
    
    results.append((sep, mean_frac, std_frac, inv_mean))
    
    print(f"sep = {sep:.2f} | mean frac = {mean_frac:.6f} ± {std_frac:.6f} | 1/mean = {inv_mean:.2f}")
    print("-" * 70)

# Plot 1: Chord fraction with error bars
seps = [r[0] for r in results]
means = [r[1] for r in results]
stds = [r[2] for r in results]

plt.figure(figsize=(10, 6))
plt.errorbar(seps, means, yerr=stds, fmt='o-', capsize=5, color='blue', label='Mean chord fraction ± std')
plt.axhline(1/137.036, color='red', linestyle='--', linewidth=1.5, label='1/137.036 ≈ 0.007299')
plt.xlabel('Separation (units)')
plt.ylabel('Chord-length filling fraction')
plt.title(f'Chord fraction vs Separation (max_depth=2, {n_runs_per_sep} runs each, {n_lines:,} lines)')
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()
plt.savefig('chord_fraction_vs_sep.png', dpi=150)
print("\nPlot saved: chord_fraction_vs_sep.png")

# Plot 2: Inverse
invs = [r[3] for r in results]

plt.figure(figsize=(10, 6))
plt.plot(seps, invs, 'o-', color='darkgreen', label='1 / mean fraction')
plt.axhline(137.036, color='red', linestyle='--', linewidth=1.5, label='137.036')
plt.xlabel('Separation (units)')
plt.ylabel('Inverse fraction (proxy for 1/α)')
plt.title('Inverse chord fraction vs Separation')
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()
plt.savefig('inverse_vs_sep.png', dpi=150)
print("Plot saved: inverse_vs_sep.png")