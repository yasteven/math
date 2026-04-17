import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import differential_evolution
import matplotlib.animation as animation
import os

os.makedirs("./outputs", exist_ok=True)

# ===================== LOAD YOUR DATA HERE =====================
# Replace with your processed ln_freq + intensity arrays
# Example placeholders:
H_ln = np.array([1,2,3,4,5,6])
H_int = np.array([10,50,20,80,30,60])

He_ln = np.array([1.2,2.1,3.2,4.1,5.2,6.1])
He_int = np.array([15,55,25,85,35,65])

# ===================== EMBEDDING =====================
def embed_complex(ln_freqs, intensities):
    x = (ln_freqs - ln_freqs.min()) / (ln_freqs.max() - ln_freqs.min())
    y = np.log(intensities + 1e-9)
    y = (y - y.min()) / (y.max() - y.min())
    return x + 1j * y

h_complex = embed_complex(H_ln, H_int)
he_complex = embed_complex(He_ln, He_int)

# ===================== GEOMETRY =====================
def mobius_apply(z, p):
    a = complex(p[0], p[1])
    b = complex(p[2], p[3])
    c = complex(p[4], p[5])
    d = complex(p[6], p[7])
    return (a*z + b) / (c*z + d + 1e-9)

def su2_apply(z, p):
    # SU(2): a,b with constraint |a|^2 + |b|^2 = 1
    a = complex(p[0], p[1])
    b = complex(p[2], p[3])
    norm = np.sqrt(abs(a)**2 + abs(b)**2) + 1e-9
    a /= norm
    b /= norm
    return (a*z + b) / (-np.conj(b)*z + np.conj(a))

def sphere_dist(z1, z2):
    return 2*np.arctan2(abs(z1-z2), 1 + abs(z1)*abs(z2))

# ===================== LOSS WITH LINE DROPPING =====================
def loss_general(p):
    mapped = np.array([mobius_apply(z, p) for z in h_complex])
    err = 0
    for m in mapped:
        d = np.array([sphere_dist(m, q) for q in he_complex])
        w = np.exp(-d/0.1)
        err += np.sum(w*d)/np.sum(w)
    return err / len(mapped)

def loss_su2(p):
    mapped = np.array([su2_apply(z, p) for z in h_complex])
    err = 0
    for m in mapped:
        d = np.array([sphere_dist(m, q) for q in he_complex])
        w = np.exp(-d/0.1)
        err += np.sum(w*d)/np.sum(w)
    return err / len(mapped)

# ===================== OPTIMIZATION =====================
print("Optimizing general Möbius...")
res_general = differential_evolution(loss_general, [(-2,2)]*8, maxiter=80)
p_general = res_general.x

print("Optimizing SU(2) rotation...")
res_su2 = differential_evolution(loss_su2, [(-2,2)]*4, maxiter=80)
p_su2 = res_su2.x

# ===================== DROP LINE DETECTION =====================
def detect_drops(mapped):
    keep = []
    for m in mapped:
        d = np.array([sphere_dist(m, q) for q in he_complex])
        keep.append(np.min(d) < 0.05)
    keep = np.array(keep)
    return np.where(~keep)[0]

mapped_general = np.array([mobius_apply(z, p_general) for z in h_complex])
mapped_su2 = np.array([su2_apply(z, p_su2) for z in h_complex])

drops_general = detect_drops(mapped_general)
drops_su2 = detect_drops(mapped_su2)

print("Dropped (general):", drops_general)
print("Dropped (SU2):", drops_su2)

# ===================== ANIMATION =====================
def interp(p, t):
    return p * t

fig, ax = plt.subplots()

def update(frame, mode="general"):
    ax.clear()
    t = frame / 100
    
    if mode == "general":
        p = interp(p_general, t)
        mapped = np.array([mobius_apply(z, p) for z in h_complex])
    else:
        p = interp(p_su2, t)
        mapped = np.array([su2_apply(z, p) for z in h_complex])
    
    ax.scatter(he_complex.real, he_complex.imag, label="He", alpha=0.7)
    ax.scatter(mapped.real, mapped.imag, label="H mapped", alpha=0.7)
    ax.set_title(f"{mode} t={t:.2f}")
    ax.legend()
    ax.set_xlim(-1,2)
    ax.set_ylim(-1,2)

ani = animation.FuncAnimation(fig, lambda f: update(f, "general"), frames=100)
ani.save("./outputs/mobius_general.gif", dpi=120)

ani2 = animation.FuncAnimation(fig, lambda f: update(f, "su2"), frames=100)
ani2.save("./outputs/mobius_su2.gif", dpi=120)

print("Saved animations to ./outputs/")