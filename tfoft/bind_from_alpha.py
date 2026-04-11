import numpy as np
import matplotlib.pyplot as plt

# ====================== SEMF ======================
def semf_binding_per_nucleon(A: int, Z: int) -> float:
    a_v = 15.8; a_s = 18.3; a_c = 0.714; a_a = 23.2; a_p = 12.0
    B = (a_v * A - a_s * A**(2/3) - a_c * Z * (Z - 1) * A**(-1/3)
         - a_a * (A - 2 * Z)**2 / A)
    if A % 2 == 0 and Z % 2 == 0:
        B += a_p / A**0.5
    elif A % 2 == 0:
        B -= a_p / A**0.5
    return B / A

# ====================== PURE ASSPSC RECURSIVE LOG GENERATION (no Strassen) ======================
def build_asspsc_sequence(n=300):
    """
    Pure recursive Soddy-gasket progression — NO Strassen gain or overhead.
    Normal behavior only.
    """
    base_stages = np.array([0.104, 0.85*np.sqrt(2), 2.457, 7.235208, 7.235208 - 2.457])
    deltas = np.diff(base_stages)
    seq = base_stages.tolist()
    current = base_stages[-1]
    cycle = 1
    scale_factor = 1.0 / np.e

    while len(seq) < n:
        if cycle == 5:
            protons_done = len(seq) - 1
            protons_next_start = len(seq)
            print(f"Hit recursion cycle 5: proton count already done: {protons_done}, "
                  f"proton count about to do: {protons_next_start} to {protons_next_start + 2}")
            iron_check = abs(protons_done - 26) <= 5
            print(f"  Iron proton count (26) check: within 4 or 5? {iron_check} (diff = {abs(protons_done - 26)})")

        # Pure normal scaling — no gain or overhead modification
        scaled_deltas = deltas[0:].copy() * (scale_factor ** cycle)

        for d in scaled_deltas:
            current += d
            seq.append(current)

        cycle += 1

    return np.array(seq[:n])


# Build the pure sequence
ASSPSC_SEQ = build_asspsc_sequence(300)

# ====================== DATA ======================
measured_data = [
    (1, 1, 0.000), (2, 1, 1.112), (3, 2, 2.572), (4, 2, 7.074),
    (6, 3, 5.332), (7, 3, 5.606), (9, 4, 6.462), (10, 5, 6.475),
    (11, 5, 6.928), (12, 6, 7.680), (13, 6, 7.470), (14, 7, 7.476),
    (15, 7, 7.700), (16, 8, 7.976), (17, 8, 7.751), (18, 8, 7.767),
    (19, 9, 7.779), (20, 10, 8.032), (24, 12, 8.260), (28, 14, 8.448),
    (32, 16, 8.493), (40, 20, 8.551), (56, 26, 8.792), (62, 28, 8.800),
    (90, 40, 8.710), (120, 50, 8.500), (144, 62, 8.250),
    (208, 82, 7.867), (238, 92, 7.570),
]

N_values = np.array([a for a, z, b in measured_data])
B_meas = np.array([b for a, z, b in measured_data])
B_semf = np.array([semf_binding_per_nucleon(int(a), int(z)) for a, z, b in measured_data])

# Pure ASSPSC list — no post-processing
B_asspsc = ASSPSC_SEQ[N_values - 1]

# ====================== SODDY GAIN POINTS ======================
soddy_indices = [0] + list(range(4, len(ASSPSC_SEQ), 4))
valid_idx = [i for i in soddy_indices if i < len(N_values)]
soddy_n = N_values[valid_idx]
soddy_y = ASSPSC_SEQ[valid_idx]

# ====================== PLOT ======================
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(13, 9), sharex=True, height_ratios=[3, 1])

ax1.scatter(N_values, B_meas, color='black', s=60, label='Measured')
ax1.plot(N_values, B_semf, '--', color='blue', linewidth=2, label='SEMF')
ax1.plot(N_values, B_asspsc, 'o-', color='green', linewidth=1.5, label='ASSPSC (pure recursive list)')

# Red circles around Soddy gain points
ax1.scatter(soddy_n, soddy_y, s=300, facecolors='none', edgecolors='red',
            linewidth=3.5, zorder=10, label='Soddy Gain Points (idx 0, 4, 8, ...)')

# ====================== STRASSEN REGION MARKERS ======================
# Vertical lines for the regions you want to observe
ax1.axvline(x=23, color='red', linestyle='-', linewidth=2.5, alpha=0.85, label='Z=23 (Strassen region start)')
ax1.axvline(x=27, color='red', linestyle='-', linewidth=2.5, alpha=0.85, label='Z=27 (Strassen region end)')

# Double lines for the next block
ax1.axvline(x=46, color='red', linestyle='--', linewidth=2.0, alpha=0.7)
ax1.axvline(x=54, color='red', linestyle='--', linewidth=2.0, alpha=0.7, label='Next block 46–54')

ax1.set_xlabel('Nucleon Number N')
ax1.set_ylabel('Binding Energy per Nucleon (MeV)')
ax1.set_title('Binding Energy Models Comparison\n(PURE ASSPSC — Strassen regions marked for observation)')
ax1.grid(True, alpha=0.3)
ax1.legend()

# Error plot
errors = B_meas - B_asspsc
ax2.plot(N_values, errors, 'o-', color='green', label='Error (Measured − ASSPSC)')
ax2.axhline(0, color='gray', linestyle='--')
ax2.set_xlabel('Nucleon Number N')
ax2.set_ylabel('Error (MeV/nucleon)')
ax2.grid(True, alpha=0.3)
ax2.legend()



plt.tight_layout()
plt.show()

# ====================== OUTPUT ======================
print("Pure ASSPSC sequence (no Strassen gain/overhead) generated.")
print("Vertical markers added at 23/27 and 46/54 for visual reference.")
print(f"Sequence length: {len(ASSPSC_SEQ)}")