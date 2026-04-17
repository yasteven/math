import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import differential_evolution
import matplotlib.animation as animation
import json, os, sys

os.makedirs("./outputs", exist_ok=True)

# ===================== LOAD JSON =====================
json_path = sys.argv[1] if len(sys.argv) > 1 else "spectral_lines.json"

with open(json_path, "r") as f:
    data = json.load(f)

def extract(element):
    lines = data[element]["lines"]
    ln = np.array([l["ln_frequency"] for l in lines])
    inten = np.array([l["intensity"] for l in lines])
    labels = [l["label"] for l in lines]
    return ln, inten, labels

H_ln, H_int, H_labels = extract("H")
He_ln, He_int, He_labels = extract("He")

print(f"H lines: {len(H_ln)} | He lines: {len(He_ln)}")

# ===================== EMBEDDING =====================
def embed_complex(ln_freqs, intensities):
    x = (ln_freqs - ln_freqs.min()) / (ln_freqs.max() - ln_freqs.min())
    y = np.log(intensities + 1e-9)
    y = (y - y.min()) / (y.max() - y.min())
    return x + 1j*y

h_complex = embed_complex(H_ln, H_int)
he_complex = embed_complex(He_ln, He_int)

# ===================== GEOMETRY =====================
def sphere_dist(z1, z2):
    return 2*np.arctan2(abs(z1-z2), 1 + abs(z1)*abs(z2))

def su2_apply(z, p):
    a = complex(p[0], p[1])
    b = complex(p[2], p[3])
    norm = np.sqrt(abs(a)**2 + abs(b)**2) + 1e-12
    a /= norm; b /= norm
    return (a*z + b) / (-np.conj(b)*z + np.conj(a))

def mobius_apply(z, p):
    a = complex(p[0], p[1])
    b = complex(p[2], p[3])
    c = complex(p[4], p[5])
    d = complex(p[6], p[7])
    det = a*d - b*c
    if abs(det) > 1e-12:
        s = np.sqrt(1/abs(det))
        a*=s; b*=s; c*=s; d*=s
    return (a*z + b) / (c*z + d + 1e-12)

# ===================== LOSSES =====================
def loss(mapped):
    err = 0
    for m in mapped:
        d = np.array([sphere_dist(m, q) for q in he_complex])
        w = np.exp(-d/0.1)
        err += np.sum(w*d)/np.sum(w)
    return err / len(mapped)

def loss_su2(p):
    mapped = np.array([su2_apply(z, p) for z in h_complex])
    return loss(mapped)

def loss_gen(p):
    mapped = np.array([mobius_apply(z, p) for z in h_complex])
    return loss(mapped)

# ===================== OPTIMIZE =====================
print("Optimizing SU(2)...")
res_su2 = differential_evolution(loss_su2, [(-2,2)]*4, maxiter=120)
p_su2 = res_su2.x

print("Optimizing general Möbius...")
res_gen = differential_evolution(loss_gen, [(-2,2)]*8, maxiter=120)
p_gen = res_gen.x

# ===================== ANALYSIS =====================
def per_line_error(mapped):
    errs = []
    for m in mapped:
        d = np.array([sphere_dist(m, q) for q in he_complex])
        errs.append(np.min(d))
    return np.array(errs)

mapped_su2 = np.array([su2_apply(z, p_su2) for z in h_complex])
mapped_gen = np.array([mobius_apply(z, p_gen) for z in h_complex])

err_su2 = per_line_error(mapped_su2)
err_gen = per_line_error(mapped_gen)

# ===================== PRINT =====================
a = complex(p_su2[0], p_su2[1])
b = complex(p_su2[2], p_su2[3])
norm = np.sqrt(abs(a)**2 + abs(b)**2)
a/=norm; b/=norm

print("\n=== SU(2) ===")
print("a =", a)
print("b =", b)
print("mean =", err_su2.mean())
print("max  =", err_su2.max())

print("\n=== GENERAL ===")
print("mean =", err_gen.mean())
print("max  =", err_gen.max())

# ===================== IDENTIFY DROPPED =====================
threshold = np.percentile(err_su2, 75)  # top ~25% = outliers
dropped_idx = np.where(err_su2 > threshold)[0]

print("\nLikely dropped H lines:")
for i in dropped_idx:
    print(f"{i}: {H_labels[i]} | err={err_su2[i]:.4f}")

# ===================== ERROR PLOT =====================
plt.figure(figsize=(8,4))
plt.plot(err_su2, 'o-', label="SU(2)")
plt.plot(err_gen, 'x--', label="General")
plt.axhline(threshold, color='red', linestyle=':', label="drop threshold")
plt.title("Per-line error (all H lines)")
plt.xlabel("H index")
plt.ylabel("spherical error")
plt.legend()
plt.savefig("./outputs/per_line_error.png", dpi=150)
plt.close()

# ===================== OVERLAY =====================
plt.figure(figsize=(6,6))
plt.scatter(he_complex.real, he_complex.imag, label="He")
plt.scatter(mapped_su2.real, mapped_su2.imag, label="SU2")
plt.legend()
plt.title("Final alignment")
plt.savefig("./outputs/final_overlay.png", dpi=150)
plt.close()

# ===================== ANIMATION =====================
def interp_su2(p, t):
    a = complex(p[0]*t, p[1]*t)
    b = complex(p[2]*t, p[3]*t)
    norm = np.sqrt(abs(a)**2 + abs(b)**2) + 1e-12
    return [a.real/norm, a.imag/norm, b.real/norm, b.imag/norm]

fig, ax = plt.subplots()

def update(frame):
    ax.clear()
    t = frame / 100
    p_t = interp_su2(p_su2, t)
    mapped = np.array([su2_apply(z, p_t) for z in h_complex])
    
    ax.scatter(he_complex.real, he_complex.imag, label="He")
    ax.scatter(mapped.real, mapped.imag, label="H→SU2")
    ax.set_title(f"t={t:.2f}")
    ax.legend()
    ax.set_xlim(-0.5,1.5)
    ax.set_ylim(-0.5,1.5)

ani = animation.FuncAnimation(fig, update, frames=100, interval=50)
ani.save("./outputs/su2_animation.gif", dpi=120)

print("\nSaved to ./outputs/")