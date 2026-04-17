import json
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from matplotlib.animation import FuncAnimation

# ---------------------------
# Load data
# ---------------------------
def load_lines(json_path, element="He"):
    with open(json_path, "r") as f:
        data = json.load(f)
    lines = data[element]["lines"]
    x = np.array([l["ln_frequency"] for l in lines])
    y = np.array([l["intensity"] for l in lines])
    labels = [l["label"] for l in lines]
    return x, y, labels

# ---------------------------
# SU(2)-like periodic model
# ---------------------------
def su2_model(x, A, B, C, D):
    return A * np.sin(B * x + C)**2 + D

# ---------------------------
# Möbius model (iterated)
# ---------------------------
def mobius_iter(x, a, b, c, d, n_iter):
    z = x.copy()
    for _ in range(int(n_iter)):
        z = (a * z + b) / (c * z + d + 1e-9)
    return z

def mobius_model(x, a, b, c, d, scale, offset):
    return scale * mobius_iter(x, a, b, c, d, 3) + offset

# ---------------------------
# Fit + metrics
# ---------------------------
def fit_model(model, x, y, p0):
    params, _ = curve_fit(model, x, y, p0=p0, maxfev=20000)
    y_fit = model(x, *params)
    rmse = np.sqrt(np.mean((y - y_fit)**2))
    return params, y_fit, rmse

# ---------------------------
# Plotting
# ---------------------------
def plot_results(x, y, y_su2, y_mob, labels):
    fig, axs = plt.subplots(2, 1, figsize=(10, 8))

    # Fit comparison
    axs[0].scatter(x, y, label="Data")
    axs[0].plot(x, y_su2, label="SU(2) fit")
    axs[0].plot(x, y_mob, label="Mobius fit")
    axs[0].legend()
    axs[0].set_title("Fits")

    # Error per line
    err_su2 = y - y_su2
    err_mob = y - y_mob

    axs[1].plot(err_su2, 'o-', label="SU(2) error")
    axs[1].plot(err_mob, 'x-', label="Mobius error")
    axs[1].legend()
    axs[1].set_title("Error per spectral line")

    plt.tight_layout()
    plt.show()

# ---------------------------
# Möbius animation
# ---------------------------
def animate_mobius(x, a, b, c, d):
    fig, ax = plt.subplots()
    z = x.copy()
    scat = ax.scatter(range(len(z)), z)

    def update(frame):
        nonlocal z
        z = (a * z + b) / (c * z + d + 1e-9)
        scat.set_offsets(np.c_[range(len(z)), z])
        ax.set_title(f"Iteration {frame}")
        return scat,

    anim = FuncAnimation(fig, update, frames=20, interval=300)
    plt.show()

# ---------------------------
# Main
# ---------------------------
if __name__ == "__main__":
    import sys

    json_path = sys.argv[1]  # pass path as argument
    element = sys.argv[2] if len(sys.argv) > 2 else "He"

    x, y, labels = load_lines(json_path, element)

    # SU(2) fit
    su2_p0 = [max(y), 1.0, 0.0, min(y)]
    su2_params, y_su2, su2_rmse = fit_model(su2_model, x, y, su2_p0)

    print("\nSU(2) params:", su2_params)
    print("SU(2) RMSE:", su2_rmse)

    # Möbius fit
    mob_p0 = [1, 1, 0.1, 1, 1, 0]
    mob_params, y_mob, mob_rmse = fit_model(mobius_model, x, y, mob_p0)

    print("\nMobius params:", mob_params)
    print("Mobius RMSE:", mob_rmse)

    # Plot
    plot_results(x, y, y_su2, y_mob, labels)

    # Animate Möbius
    animate_mobius(x, *mob_params[:4])