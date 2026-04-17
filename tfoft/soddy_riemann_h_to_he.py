import json
import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as plt

# -----------------------
# Load + embed
# -----------------------
def load_complex(json_path, element):
    with open(json_path) as f:
        data = json.load(f)

    lines = data[element]["lines"]

    x = np.array([l["ln_frequency"] for l in lines])
    y = np.array([l["intensity"] for l in lines])

    y = np.log(y + 1e-9)
    alpha = 0.5

    z = x + 1j * alpha * y
    return z

# -----------------------
# Möbius
# -----------------------
def mobius(z, a, b, c, d):
    return (a*z + b) / (c*z + d + 1e-9)

# -----------------------
# Loss
# -----------------------
def loss(params, z1, z2):
    a, b, c, d = params
    fz = mobius(z1, a, b, c, d)
    return np.mean(np.abs(fz - z2)**2)

# -----------------------
# Fit
# -----------------------
def fit_mobius(zH, zHe):
    p0 = [1, 0, 0, 1]

    res = minimize(loss, p0, args=(zH, zHe), method='Nelder-Mead')
    return res.x, res.fun

# -----------------------
# Sphere projection
# -----------------------
def stereographic(z):
    x = np.real(z)
    y = np.imag(z)
    r2 = x*x + y*y

    X = 2*x/(1+r2)
    Y = 2*y/(1+r2)
    Z = (r2 - 1)/(1+r2)

    return np.stack([X,Y,Z], axis=1)

# -----------------------
# Plot
# -----------------------
def plot_sphere(H, He, Hmap):
    from mpl_toolkits.mplot3d import Axes3D

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    SH = stereographic(H)
    SHe = stereographic(He)
    SHm = stereographic(Hmap)

    ax.scatter(*SH.T, label="H")
    ax.scatter(*SHe.T, label="He")
    ax.scatter(*SHm.T, label="Mapped H")

    ax.legend()
    plt.show()

# -----------------------
# Main
# -----------------------
if __name__ == "__main__":
    import sys

    path = sys.argv[1]

    zH = load_complex(path, "H")
    zHe = load_complex(path, "He")

    # match lengths
    n = min(len(zH), len(zHe))
    zH = zH[:n]
    zHe = zHe[:n]

    params, err = fit_mobius(zH, zHe)

    print("Mobius params:", params)
    print("MSE:", err)

    zH_map = mobius(zH, *params)

    errors = np.abs(zH_map - zHe)

    plt.plot(errors, 'o-')
    plt.title("Per-line error")
    plt.show()

    plot_sphere(zH, zHe, zH_map)