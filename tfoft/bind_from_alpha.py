import numpy as np
import matplotlib.pyplot as plt

# ====================== TFOFT CONSTANTS ======================
T_SPHERE = 4 * np.pi**3 + np.pi**2 + np.pi
DELTA_TWO_BALL = 1.9
LAMBDA = 1.0
MU = 1.17e-4
SIGMA = 8.0

# ====================== ORIGINAL MODEL ======================
def fractal_seed(Z: int) -> float:
    if Z == 1:   return 0.0
    elif Z == 2: return np.sqrt(2)
    elif Z == 3: return np.pi
    elif Z == 4: return 7.0
    elif Z == 5: return 6.1
    else:        return 0.0

def saving(Z: int) -> float:
    if Z <= 1:
        return 0.0
    base = DELTA_TWO_BALL
    num_pairs = max(0.0, (Z - 2) / 2.0)
    per_pair_max = 1284.0
    per_pair = per_pair_max * (1 - np.exp(-num_pairs / SIGMA))
    return base + num_pairs * per_pair * 0.85

def overhead(Z: int, A: int) -> float:
    base = LAMBDA * Z + MU * Z**2
    coulomb_k = 0.714
    coulomb = coulomb_k * Z * (Z - 1) / (A ** (1.0/3.0))
    return base + coulomb

def tfoft_binding_per_nucleon(Z: int, A: int) -> float:
    if Z <= 5:
        raw = fractal_seed(Z)
        S_seed = 6.462 / (2 * 3.5)
        return raw * S_seed
    else:
        net = saving(Z) - overhead(Z, A)
        comp_per_nucleon = net / Z
        SCALE = 8.792 / ((saving(26) - overhead(26, 56)) / 26)
        return comp_per_nucleon * SCALE

# ====================== LOG-FRACTAL SEED SYSTEM ======================
def build_log_fractal_sequence(n=300):
    seed = [0.0, np.sqrt(2), np.pi, 7.0, 6.1]
    base = [x for x in seed if x > 0]

    seq = []

    # first layer
    for x in base:
        seq.append(np.log(x) + 6.1)

    # recursive cascade
    while len(seq) < n:
        prev = seq[-1]
        prev2 = seq[-2] if len(seq) > 1 else prev

        new_val = prev + np.log(prev2)
        seq.append(new_val)

    # todo, need to scale to mev

    return np.array(seq[:n])

def fractal_binding_per_nucleon(Z: int) -> float:
    seq = build_log_fractal_sequence(300)
    idx = max(0, Z - 2)

    scale = 8.792 / seq[24]
    return seq[min(idx, len(seq)-1)] * scale

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

# ====================== PLOT ======================
A_values = np.array([a for a, z, b in measured_data])
Z_values = np.array([z for a, z, b in measured_data])
B_meas = np.array([b for a, z, b in measured_data])

B_tfoft = np.array([tfoft_binding_per_nucleon(int(z), int(a)) for z, a in zip(Z_values, A_values)])
B_semf = np.array([semf_binding_per_nucleon(int(a), int(z)) for a, z in zip(A_values, Z_values)])
B_frac = np.array([fractal_binding_per_nucleon(int(z)) for z in Z_values])

fig, ax = plt.subplots(figsize=(11, 6))

ax.scatter(A_values, B_meas, color='black', s=60, label='Measured')
ax.plot(A_values, B_tfoft, 'o-', color='red', label='TFOFT')
ax.plot(A_values, B_semf, '--', color='blue', label='SEMF')
ax.plot(A_values, B_frac, 'o-', color='green', label='Log-Fractal Seed Cascade')

ax.set_xlabel('Mass Number A')
ax.set_ylabel('Binding Energy per Nucleon (MeV)')
ax.set_title('Binding Energy Models Comparison')
ax.grid(True, alpha=0.3)
ax.legend()

plt.tight_layout()
plt.show()

# ====================== OUTPUT ======================
print("Fractal system added: log-seed recursive cascade active")