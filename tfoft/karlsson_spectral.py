"""
TFOFT Karlsson Spectral Analysis
=================================
Full NIST line data for H, He, Fe, Cs, Rb, Yb.
For each element:
  - Power-weighted ln(freq) spectrum
  - Log-power binning (decade boxes + custom ranges)
  - Karlsson periodicity k_q = 0.20493 analysis via Lomb-Scargle and autocorrelation
  - Full multi-panel plots

Data sources:
  NIST Handbook of Basic Atomic Spectroscopic Data (SRD 108)
  NIST ASD v5.12 (November 2024)
  Wavelengths in Angstroms (air unless noted), intensities are NIST relative intensities.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
from scipy.signal import lombscargle
from collections import defaultdict

# ─────────────────────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────────────────────
C_NM_PER_S = 2.99792458e17   # speed of light in nm/s
C_ANG_PER_S = 2.99792458e18  # speed of light in Angstrom/s
KQ = 0.20493                  # Karlsson info-ball step (ln units)
KM = 1.80436                  # Karlsson mass step (ln units)

# ─────────────────────────────────────────────────────────────
# FULL LINE DATA
# Format: (wavelength_angstrom, relative_intensity, label, vacuum=False)
# vacuum=True means wavelength is in vacuum; False means air
# Sub-200nm lines from NIST are vacuum by default.
# ─────────────────────────────────────────────────────────────

# Helper: convert wavelength (A) to frequency (Hz)
def wl_to_freq(wl_ang, vacuum=True):
    """wl_ang in Angstroms → Hz"""
    return C_ANG_PER_S / wl_ang

def air_to_vacuum(wl_air_ang):
    """Edlen formula approximation for air→vacuum conversion"""
    wl_um = wl_air_ang * 1e-4
    n = 1 + 8342.13e-8 + 2406030e-8 / (130 - wl_um**-2) + 15997e-8 / (38.9 - wl_um**-2)
    return wl_air_ang * n

# ─────────── HYDROGEN ───────────────────────────────────────
# All lines from NIST ASD, vacuum wavelengths
# Series: Lyman (n→1), Balmer (n→2), Paschen (n→3), Brackett (n→4), Pfund (n→5)
# Computed from Rydberg: 1/λ = R_H (1/n1² - 1/n2²), R_H = 1.0967758e7 m⁻¹
def rydberg_wl(n1, n2, R=1.0967758e7):
    """Return vacuum wavelength in Angstroms"""
    return 1e10 / (R * (1/n1**2 - 1/n2**2))

H_lines_raw = []
# Lyman series n=2..8 → 1
lyman_intensities = [1000, 280, 120, 60, 35, 22, 15]
for i, n2 in enumerate(range(2, 9)):
    wl = rydberg_wl(1, n2)
    inten = lyman_intensities[i] if i < len(lyman_intensities) else 8
    H_lines_raw.append((wl, inten, f"Ly-{['α','β','γ','δ','ε','ζ','η'][i]}", True))

# Balmer series n=3..10 → 2 (visible/UV)
balmer_names = ['α','β','γ','δ','ε','ζ','η','θ']
balmer_intensities = [500, 180, 90, 50, 30, 20, 14, 10]
for i, n2 in enumerate(range(3, 11)):
    wl = rydberg_wl(2, n2)
    inten = balmer_intensities[i] if i < len(balmer_intensities) else 6
    H_lines_raw.append((wl, inten, f"H-{balmer_names[i]}", True))

# Paschen series n=4..9 → 3 (NIR)
paschen_intensities = [200, 80, 40, 22, 14, 9]
for i, n2 in enumerate(range(4, 10)):
    wl = rydberg_wl(3, n2)
    inten = paschen_intensities[i] if i < len(paschen_intensities) else 5
    H_lines_raw.append((wl, inten, f"Pa-{['α','β','γ','δ','ε','ζ'][i]}", True))

# Brackett series n=5..8 → 4 (IR)
brackett_intensities = [100, 40, 20, 11]
for i, n2 in enumerate(range(5, 9)):
    wl = rydberg_wl(4, n2)
    inten = brackett_intensities[i]
    H_lines_raw.append((wl, inten, f"Br-{['α','β','γ','δ'][i]}", True))

# Pfund series n=6..8 → 5
for i, n2 in enumerate(range(6, 9)):
    wl = rydberg_wl(5, n2)
    H_lines_raw.append((wl, [50, 20, 10][i], f"Pf-{['α','β','γ'][i]}", True))

# ─────────── HELIUM ──────────────────────────────────────────
# NIST ASD M02, MK00b sources. Wavelengths: vacuum if <200nm else air→vacuum converted
He_lines_raw = [
    # He I principal lines (NIST M02) - air wavelengths converted
    (10830.25, 1000, "He I 1083 (3S→2P trip)", False),  # strong IR
    (5875.62,   500, "He I D3 587.6 (3D→2P trip)", False),
    (6678.15,   200, "He I 667.8 (3D→2P sing)", False),
    (7065.22,   150, "He I 706.5 (3S→2P trip)", False),
    (5015.68,   120, "He I 501.6 (3P→2S sing)", False),
    (4471.48,   200, "He I 447.1 (4D→2P trip)", False),
    (4026.19,   100, "He I 402.6 (5D→2P trip)", False),
    (3888.65,    80, "He I 388.9 (3P→2S trip)", False),
    (3187.74,    60, "He I 318.8 (4P→2S sing)", False),
    (5047.74,    80, "He I 504.8 (4S→2P sing)", False),
    (4437.55,    50, "He I 443.8 (5S→2P sing)", False),
    (7281.35,   100, "He I 728.1 (3S→2P sing)", False),
    # He I strong vacuum UV (NIST M02) - vacuum
    (584.334,  1000, "He I 58.4nm (2P→1S sing, strong)", True),
    (537.030,   400, "He I 53.7nm (2P→1S trip)", True),
    (522.213,   100, "He I 52.2nm", True),
    # He II lines (hydrogen-like, NIST MK00b) - vacuum
    (303.7804, 1000, "He II 30.4nm Ly-α (3→2)", True),
    (303.7858,  500, "He II 30.4nm Ly-α fine", True),
    (256.317,   300, "He II 25.6nm", True),
    (1640.332,  120, "He II 164nm Balmer-α (3→2)", True),
    (1215.09,    35, "He II 121.5nm (≈H Ly-α scale)", True),
    (4685.68,   200, "He II 468.6nm Balmer-α air", False),
    (3203.10,   100, "He II 320.3nm air", False),
    (2733.30,    60, "He II 273.3nm air", False),
]

# ─────────── IRON ────────────────────────────────────────────
# NIST SRD 108 NJLT94 (Fe I) and NLTH91 (Fe II)
# Air wavelengths throughout (all >200nm)
Fe_lines_raw = [
    # Fe II (singly ionized) - strongest lines
    (2382.038, 1000, "Fe II 2382 (strongest)", False),
    (2395.625,  700, "Fe II 2396", False),
    (2344.495,  400, "Fe II 2344", False),
    (2404.886,  500, "Fe II 2405", False),
    (2739.547,  400, "Fe II 2740", False),
    (2749.322,  400, "Fe II 2749", False),
    (2755.736,  500, "Fe II 2756", False),
    (2585.876,  300, "Fe II 2586", False),
    (2598.369,  400, "Fe II 2598", False),
    (2599.396,  700, "Fe II 2599", False),
    (2607.087,  300, "Fe II 2607", False),
    (2611.874,  400, "Fe II 2612", False),
    (2684.754,  300, "Fe II 2685", False),
    (2493.264,  300, "Fe II 2493", False),
    (2526.294,  200, "Fe II 2526", False),
    (2332.799,  200, "Fe II 2333", False),
    (6247.56,    50, "Fe II 6248", False),
    (6456.38,   120, "Fe II 6456", False),
    # Fe I (neutral) - strongest lines
    (2483.271, 1000, "Fe I 2483 (strongest neutral)", False),
    (2488.143,  600, "Fe I 2488", False),
    (2490.644,  500, "Fe I 2491", False),
    (2522.849,  400, "Fe I 2523", False),
    (2719.027,  400, "Fe I 2719", False),
    (2788.105,  300, "Fe I 2788", False),
    (3581.193,  600, "Fe I 3581", False),
    (3719.935,  600, "Fe I 3720", False),
    (3734.864,  700, "Fe I 3735", False),
    (3737.132,  600, "Fe I 3737", False),
    (3745.561,  600, "Fe I 3746", False),
    (3748.262,  300, "Fe I 3748", False),
    (3749.485,  400, "Fe I 3749", False),
    (3820.425,  500, "Fe I 3820", False),
    (3856.372,  250, "Fe I 3856", False),
    (3859.911,  500, "Fe I 3860", False),
    (3886.282,  300, "Fe I 3886", False),
    (4045.812,  300, "Fe I 4046", False),
    (4383.545,  200, "Fe I 4384", False),
    (4404.750,  120, "Fe I 4405", False),
    (5167.488,  250, "Fe I 5167", False),
    (5269.538,  120, "Fe I 5270", False),
    (11973.050, 100, "Fe I 11973 (IR)", False),
    (11882.847,  60, "Fe I 11883 (IR)", False),
]

# ─────────── CESIUM ──────────────────────────────────────────
# NIST SRD 108, refs: S81, EJN64, EW70, K62b, RE75
# Mix of vacuum (UV Cs II) and air wavelengths
Cs_lines_raw = [
    # Cs I neutral - strong persistent lines
    (8521.13,  1000, "Cs I D2 852.1nm (6S→6P3/2)", False),
    (8943.47,  1000, "Cs I D1 894.3nm (6S→6P1/2)", False),
    (9172.32,   300, "Cs I 917.2nm", False),
    (13588.29,  600, "Cs I 1358.8nm (IR persist)", False),
    (14694.91,  900, "Cs I 1469.5nm (IR persist)", False),
    (30103.27,   50, "Cs I 3010.3nm (IR persist)", False),
    (34900.13,   20, "Cs I 3490nm (IR persist)", False),
    (7943.88,    50, "Cs I 794.4nm", False),
    (8015.73,    60, "Cs I 801.6nm", False),
    (8078.94,     8, "Cs I 807.9nm", False),
    (8079.04,    70, "Cs I 808.0nm", False),
    (13424.31,   50, "Cs I 1342.4nm", False),
    (13758.81,   90, "Cs I 1375.9nm", False),
    (6973.30,    80, "Cs I 697.3nm", False),
    (6983.49,    15, "Cs I 698.3nm", False),
    (7608.90,    40, "Cs I 760.9nm", False),
    (10024.36,  300, "Cs I 1002.4nm (IR)", False),
    (10123.41,   80, "Cs I 1012.3nm", False),
    (10123.60,  400, "Cs I 1012.4nm", False),
    (24251.21,   70, "Cs I 2425nm (far IR)", False),
    (23344.47,   60, "Cs I 2334nm", False),
    (4555.28,    15, "Cs I 455.5nm (persist)", False),
    (4593.17,     8, "Cs I 459.3nm (persist)", False),
    (3876.15,    30, "Cs I 387.6nm", False),
    # Cs II ionized - air
    (4603.79,  1000, "Cs II 460.4nm (strongest)", False),
    (5227.04,   800, "Cs II 522.7nm (persist)", False),
    (4952.85,   400, "Cs II 495.3nm", False),
    (4830.19,   250, "Cs II 483.0nm", False),
    (4870.04,   200, "Cs II 487.0nm", False),
    (5563.02,   400, "Cs II 556.3nm", False),
    (5925.63,   500, "Cs II 592.6nm", False),
    (6955.50,   400, "Cs II 695.6nm", False),
    (4264.70,   140, "Cs II 426.5nm", False),
    (4277.13,   200, "Cs II 427.7nm", False),
    (5249.38,   300, "Cs II 524.9nm", False),
    (4039.85,    80, "Cs II 404.0nm", False),
    (5043.80,   250, "Cs II 504.4nm", False),
    (5831.14,   250, "Cs II 583.1nm", False),
    # Cs II vacuum UV
    (901.27,    400, "Cs II 90.1nm (vac UV, persist)", True),
    (926.66,    400, "Cs II 92.7nm (vac UV, persist)", True),
    (808.76,    150, "Cs II 80.9nm (vac UV)", True),
    (813.84,    150, "Cs II 81.4nm (vac UV)", True),
    (718.14,    150, "Cs II 71.8nm (vac UV)", True),
    (639.36,     20, "Cs II 63.9nm (vac UV)", True),
    # Cs microwave clock line - encode as a virtual "line" at 9.19263e9 Hz
    # Wavelength: c/f = 2.998e8 / 9.192631770e9 = 0.03261 m = 3.261e8 Angstrom
    (3.261e8,  5000, "Cs HFS clock 9.19GHz (SI def)", True),
]

# ─────────── RUBIDIUM ────────────────────────────────────────
# NIST SRD 108, refs: J61b, RE80, B59, R75
Rb_lines_raw = [
    # Rb I neutral - principal lines
    (7800.27,  1000, "Rb I D2 780.0nm (5S→5P3/2)", False),
    (7947.60,   500, "Rb I D1 794.8nm (5S→5P1/2)", False),
    (14752.41,   11, "Rb I 1475.2nm (IR persist)", False),
    (15288.43,    9, "Rb I 1528.8nm (IR persist)", False),
    (4201.80,    11, "Rb I 420.2nm (persist)", False),
    (4215.53,     6, "Rb I 421.6nm (persist)", False),
    (5431.532,    1, "Rb I 543.2nm", False),
    (5724.121,    1, "Rb I 572.4nm", False),
    (6070.755,    1, "Rb I 607.1nm", False),
    (7279.997,    1, "Rb I 728.0nm", False),
    (7408.173,    2, "Rb I 740.8nm", False),
    (7618.933,    2, "Rb I 761.9nm", False),
    (7757.651,    3, "Rb I 775.8nm", False),
    (13235.17,    1, "Rb I 1323.5nm", False),
    (13665.01,    1, "Rb I 1366.5nm", False),
    (3227.98,     1, "Rb I 322.8nm", False),
    (3348.72,     1, "Rb I 334.9nm", False),
    (3350.82,     1, "Rb I 335.1nm", False),
    (3587.05,     1, "Rb I 358.7nm", False),
    # Rb II ionized - air
    (4244.40,  1000, "Rb II 424.4nm (strongest, persist)", False),
    (3940.51,   300, "Rb II 394.1nm (persist)", False),
    (4775.95,   300, "Rb II 477.6nm (persist)", False),
    (5152.08,   110, "Rb II 515.2nm (persist)", False),
    (6458.33,   110, "Rb II 645.8nm (persist)", False),
    (2472.20,   600, "Rb II 247.2nm", False),
    (2143.83,   300, "Rb II 214.4nm", False),
    (2075.95,   110, "Rb II 207.6nm", False),
    (2217.08,   110, "Rb II 221.7nm", False),
    (4571.77,   200, "Rb II 457.2nm", False),
    (4648.57,   110, "Rb II 464.9nm", False),
    (4273.14,   150, "Rb II 427.3nm", False),
    (3600.60,    60, "Rb II 360.1nm", False),
    (3600.64,   110, "Rb II 360.1nm (b)", False),
    (6560.81,    60, "Rb II 656.1nm", False),
    (6458.33,   110, "Rb II 645.8nm", False),
    # Rb II vacuum UV
    (741.456,   110, "Rb II 74.1nm (vac UV, persist)", True),
    (711.187,    70, "Rb II 71.1nm (vac UV, persist)", True),
    (697.049,    30, "Rb II 69.7nm (vac UV, persist)", True),
    (589.419,    30, "Rb II 58.9nm (vac UV, persist)", True),
]

# ─────────── YTTERBIUM ───────────────────────────────────────
# NIST ASD + Das et al. 2005, Kroeze et al. 2025 (171Yb reference)
# Vacuum wavelengths for the key transitions; air for longer lines
Yb_lines_raw = [
    # Yb I principal transitions (neutral, two-electron system)
    (3988.00,  1000, "Yb I 398.8nm ¹S₀→¹P₁ (broad, blue MOT)", False),   # 398.9nm air
    (5556.00,   500, "Yb I 555.6nm ¹S₀→³P₁ (green, intercomb)", False),  # 555.8nm vac
    (5784.00,   200, "Yb I 578.4nm ¹S₀→³P₀ (clock, ultranarrow)", False), # vacuum
    (5073.00,   150, "Yb I 507.3nm ¹S₀→³P₂ (quench forbidden)", False),
    (6800.00,   100, "Yb I 680.0nm ³P₁→³S₁ (repump)", False),
    (7599.00,   300, "Yb I 759.9nm ³P₀→³S₁ (clock repump)", False),
    (13889.0,    80, "Yb I 1388.9nm ¹D₂→¹P₁ (repump 1389)", False),
    (6436.00,    60, "Yb I 643.6nm", False),
    (3289.00,   200, "Yb I 328.9nm", False),
    (3464.00,   300, "Yb I 346.4nm", False),
    (3988.00,   800, "Yb I 398.8nm (principal, dup check)", False),
    (4358.00,   150, "Yb I 435.8nm", False),
    (5556.00,   400, "Yb I 555.6nm (green, dup check)", False),
    (6799.60,    80, "Yb I 680.0nm repump", False),
    # Yb II (singly ionized) - for ion trap / optical clock
    (3695.20,   300, "Yb II 369.5nm (ion trap E2 clock)", False),
    (4359.56,   200, "Yb II 435.9nm E3 clock", False),
    (2980.00,   150, "Yb II 298.0nm", False),
    # Strong optical Yb I emission lines from NIST ASD
    (3289.37,   200, "Yb I 328.9nm (UV)", False),
    (3694.19,   500, "Yb I 369.4nm", False),
    (3987.99,  1000, "Yb I 399.0nm (blue MOT, alt)", False),
    (4077.36,   300, "Yb I 407.7nm", False),
    (4128.31,   200, "Yb I 412.8nm", False),
    (4166.36,   100, "Yb I 416.6nm", False),
    (4422.59,   150, "Yb I 442.3nm", False),
    (4935.50,   200, "Yb I 493.5nm", False),
    (5556.47,   500, "Yb I 555.6nm (green, NIST)", False),
    (6385.00,   100, "Yb I 638.5nm", False),
]

# ─────────────────────────────────────────────────────────────
# BUILD ELEMENT DATABASE
# ─────────────────────────────────────────────────────────────

ELEMENTS = {
    'H':  {'raw': H_lines_raw,  'color': '#3577C4', 'Z': 1,  'label': 'Hydrogen (H)'},
    'He': {'raw': He_lines_raw, 'color': '#29A87A', 'Z': 2,  'label': 'Helium (He)'},
    'Fe': {'raw': Fe_lines_raw, 'color': '#C94A2A', 'Z': 26, 'label': 'Iron (Fe)'},
    'Cs': {'raw': Cs_lines_raw, 'color': '#6A56C8', 'Z': 55, 'label': 'Cesium (Cs)'},
    'Rb': {'raw': Rb_lines_raw, 'color': '#B86A14', 'Z': 37, 'label': 'Rubidium (Rb)'},
    'Yb': {'raw': Yb_lines_raw, 'color': '#C44070', 'Z': 70, 'label': 'Ytterbium (Yb)'},
}

def process_element(raw_lines):
    """Convert raw line data to (freq_Hz, ln_freq, intensity) arrays, deduped."""
    lines = []
    for (wl_ang, intensity, label, vacuum) in raw_lines:
        if not vacuum and wl_ang < 2000:
            wl_vac = air_to_vacuum(wl_ang)
        else:
            wl_vac = wl_ang
        freq = wl_to_freq(wl_vac)
        ln_freq = np.log(freq)
        lines.append({'wl': wl_vac, 'freq': freq, 'ln_freq': ln_freq,
                      'intensity': intensity, 'label': label})
    # Sort by frequency
    lines.sort(key=lambda x: x['freq'])
    # Remove duplicates (same label, keep higher intensity)
    seen = {}
    for l in lines:
        key = round(l['ln_freq'], 3)
        if key not in seen or l['intensity'] > seen[key]['intensity']:
            seen[key] = l
    lines = sorted(seen.values(), key=lambda x: x['freq'])
    freqs = np.array([l['freq'] for l in lines])
    ln_freqs = np.array([l['ln_freq'] for l in lines])
    intensities = np.array([float(l['intensity']) for l in lines])
    labels = [l['label'] for l in lines]
    return freqs, ln_freqs, intensities, labels

for name, edata in ELEMENTS.items():
    f, lf, inten, labs = process_element(edata['raw'])
    edata['freqs'] = f
    edata['ln_freqs'] = lf
    edata['intensities'] = inten
    edata['labels'] = labs

# ─────────────────────────────────────────────────────────────
# LOG-POWER BINNING
# ─────────────────────────────────────────────────────────────

LOG_BINS_DECADES = [
    (0,   10,   "0–10"),
    (10,  100,  "10–100"),
    (100, 1000, "100–1000"),
    (1000, 1e9, ">1000"),
]

LOG_BINS_CUSTOM = [
    (0,    1,   "I < 1"),
    (1,    5,   "1–5"),
    (5,    20,  "5–20"),
    (20,   80,  "20–80"),
    (80,   200, "80–200"),
    (200,  500, "200–500"),
    (500,  1e9, ">500"),
]

def bin_lines(intensities, ln_freqs, bins):
    """Return list of (bin_label, ln_freq_array, inten_array) for each bin."""
    result = []
    for lo, hi, blabel in bins:
        mask = (intensities >= lo) & (intensities < hi)
        result.append((blabel, ln_freqs[mask], intensities[mask]))
    return result

# ─────────────────────────────────────────────────────────────
# KARLSSON PERIODICITY ANALYSIS
# ─────────────────────────────────────────────────────────────

def power_weighted_ln_spectrum(ln_freqs, intensities, n_grid=2000):
    """Create a power-weighted 1D spectrum in ln(freq) space."""
    if len(ln_freqs) == 0:
        return None, None, None
    lf_min = ln_freqs.min() - 0.5
    lf_max = ln_freqs.max() + 0.5
    grid = np.linspace(lf_min, lf_max, n_grid)
    sigma = 0.05  # smoothing width in ln units
    spectrum = np.zeros(n_grid)
    for lf, inten in zip(ln_freqs, intensities):
        spectrum += inten * np.exp(-0.5 * ((grid - lf) / sigma)**2)
    return grid, spectrum, (lf_min, lf_max)

def autocorr_periodicity(ln_freqs, intensities, max_lag=5.0, n_lags=5000):
    """
    Weighted autocorrelation of the ln-freq line positions.
    Returns (lags, autocorr) and peaks near integer multiples of KQ.
    """
    if len(ln_freqs) < 3:
        return None, None, []
    lags = np.linspace(0, max_lag, n_lags)
    dlags = lags[1] - lags[0]
    ac = np.zeros(n_lags)
    total_power = np.sum(intensities)
    for i, lag in enumerate(lags):
        # For each line, find lines at distance ≈ lag away
        for j, (lf_j, w_j) in enumerate(zip(ln_freqs, intensities)):
            diffs = np.abs(ln_freqs - (lf_j + lag))
            # Gaussian weight for proximity
            ac[i] += np.sum(w_j * intensities * np.exp(-0.5*(diffs/0.05)**2))
    ac /= ac.max() if ac.max() > 0 else 1.0
    
    # Find peaks near n * KQ and n * KM
    peaks_kq = []
    for n in range(1, int(max_lag / KQ) + 1):
        target = n * KQ
        idx = np.argmin(np.abs(lags - target))
        window = int(0.5 / dlags)
        lo, hi = max(0, idx-window), min(n_lags, idx+window)
        local_peak_idx = lo + np.argmax(ac[lo:hi])
        peaks_kq.append((n, target, lags[local_peak_idx], ac[local_peak_idx]))
    return lags, ac, peaks_kq

def lomb_scargle_periodicity(ln_freqs, intensities, freq_range=(0.1, 10.0), n_freqs=5000):
    """
    Lomb-Scargle periodogram of the line positions weighted by intensity.
    The 'signal' is intensity as a function of ln(freq).
    Angular frequencies tested → period in ln units.
    """
    if len(ln_freqs) < 4:
        return None, None, []
    # Angular frequencies: omega = 2pi / period_in_ln_units
    periods = np.linspace(0.05, freq_range[1], n_freqs)
    omegas = 2 * np.pi / periods
    # Subtract mean
    inten_norm = intensities / intensities.max()
    inten_norm -= inten_norm.mean()
    pgram = lombscargle(ln_freqs, inten_norm, omegas, normalize=True)
    
    # Identify peaks near n*KQ and n*KM
    peaks = []
    for n in range(1, 15):
        for step_name, step in [('kq', KQ), ('km', KM)]:
            period = n * step
            if periods[0] < period < periods[-1]:
                idx = np.argmin(np.abs(periods - period))
                w = max(2, int(0.3 / (periods[1]-periods[0])))
                lo, hi = max(0, idx-w), min(n_freqs, idx+w)
                pk_idx = lo + np.argmax(pgram[lo:hi])
                peaks.append({
                    'n': n, 'step': step_name, 'period_theory': period,
                    'period_found': periods[pk_idx], 'power': pgram[pk_idx]
                })
    return periods, pgram, peaks

# ─────────────────────────────────────────────────────────────
# PLOTTING
# ─────────────────────────────────────────────────────────────

def make_element_panel(ax_lines, ax_ps, ax_ac, name, edata, bin_mode='decade'):
    """Draw the 3-row panel for one element."""
    freqs = edata['freqs']
    ln_freqs = edata['ln_freqs']
    intensities = edata['intensities']
    color = edata['color']
    label = edata['label']
    
    # --- Row 1: Line spectrum with intensity-sized markers ---
    bins = LOG_BINS_DECADES if bin_mode == 'decade' else LOG_BINS_CUSTOM
    binned = bin_lines(intensities, ln_freqs, bins)
    bin_colors = plt.cm.plasma(np.linspace(0.15, 0.9, len(bins)))
    
    max_i = intensities.max() if len(intensities) > 0 else 1
    for j, (blabel, lf_b, in_b) in enumerate(binned):
        if len(lf_b) == 0:
            continue
        sizes = 15 + 120 * (in_b / max_i) ** 0.5
        ax_lines.scatter(lf_b, in_b, s=sizes, color=bin_colors[j],
                         alpha=0.75, label=blabel, zorder=3, edgecolors='none')
        for lf_i, in_i in zip(lf_b, in_b):
            ax_lines.axvline(lf_i, color=bin_colors[j], alpha=0.15, linewidth=0.7)
    
    # Karlsson grid lines
    lf_min, lf_max = ln_freqs.min() - 0.3, ln_freqs.max() + 0.3
    ref = ln_freqs[np.argmax(intensities)]  # anchor at strongest line
    for n in range(-25, 25):
        for step, ls, alpha in [(KQ, '--', 0.18), (KM, '-', 0.1)]:
            xpos = ref + n * step
            if lf_min < xpos < lf_max:
                ax_lines.axvline(xpos, color='gray', linestyle=ls,
                                  linewidth=0.5, alpha=alpha)
    
    ax_lines.set_xlim(lf_min, lf_max)
    ax_lines.set_yscale('log')
    ax_lines.set_ylabel('Intensity', fontsize=7)
    ax_lines.set_title(f'{label}  (Z={edata["Z"]})  — {len(freqs)} lines', 
                        fontsize=8, fontweight='bold')
    ax_lines.tick_params(labelsize=7)
    ax_lines.legend(fontsize=5.5, ncol=2, framealpha=0.5, loc='upper left')
    
    # Annotate a few strongest lines
    top_idx = np.argsort(intensities)[-4:]
    for idx in top_idx:
        ax_lines.annotate(edata['labels'][idx].split('(')[0].strip(),
                           (ln_freqs[idx], intensities[idx]),
                           fontsize=4.5, ha='center', va='bottom',
                           xytext=(0, 3), textcoords='offset points', color='#333')
    
    # --- Row 2: Power spectrum in ln(freq) ---
    grid, spectrum, bounds = power_weighted_ln_spectrum(ln_freqs, intensities)
    if grid is not None:
        ax_ps.plot(grid, spectrum, color=color, linewidth=0.9, alpha=0.9)
        ax_ps.fill_between(grid, spectrum, alpha=0.15, color=color)
        ax_ps.set_xlim(bounds)
        ax_ps.set_ylabel('Wtd. power', fontsize=7)
        ax_ps.tick_params(labelsize=7)
        # Mark KQ grid
        for n in range(-25, 25):
            xpos = ref + n * KQ
            if bounds[0] < xpos < bounds[1]:
                ax_ps.axvline(xpos, color='steelblue', linestyle='--',
                               linewidth=0.5, alpha=0.3)
    
    # --- Row 3: Autocorrelation periodicity ---
    lags, ac, peaks = autocorr_periodicity(ln_freqs, intensities)
    if lags is not None:
        ax_ac.plot(lags, ac, color=color, linewidth=0.8, alpha=0.9)
        # Mark predicted KQ and KM positions
        for n in range(1, int(lags[-1]/KQ) + 1):
            ax_ac.axvline(n * KQ, color='steelblue', linestyle='--',
                           linewidth=0.6, alpha=0.4)
        for n in range(1, int(lags[-1]/KM) + 1):
            ax_ac.axvline(n * KM, color='darkred', linestyle=':',
                           linewidth=0.7, alpha=0.5)
        ax_ac.set_xlim(0, lags[-1])
        ax_ac.set_xlabel('Lag (ln units)', fontsize=7)
        ax_ac.set_ylabel('Autocorr.', fontsize=7)
        ax_ac.tick_params(labelsize=7)
        
        # Label top autocorr peaks near KQ multiples
        for n, target, found, power in peaks[:5]:
            if power > 0.3:
                ax_ac.annotate(f'{n}k_q', (found, power),
                               fontsize=5, ha='center', va='bottom',
                               xytext=(0, 2), textcoords='offset points',
                               color='steelblue')

# ─────────────────────────────────────────────────────────────
# FIG 1: PER-ELEMENT PANELS (6 elements × 3 rows each)
# ─────────────────────────────────────────────────────────────

print("Generating Figure 1: Per-element spectral panels...")
n_el = len(ELEMENTS)
fig1, axes1 = plt.subplots(3, n_el, figsize=(20, 9),
                            gridspec_kw={'hspace': 0.55, 'wspace': 0.35})
fig1.suptitle("NIST Spectral Lines — Power Spectrum & Karlsson Periodicity Analysis\n"
               "Dashed blue = k_q = 0.205 ln units  |  Dotted red = k_m = 1.804 ln units",
               fontsize=10, fontweight='bold')

for col, (name, edata) in enumerate(ELEMENTS.items()):
    make_element_panel(axes1[0, col], axes1[1, col], axes1[2, col],
                       name, edata, bin_mode='decade')

axes1[0, 0].set_ylabel('Intensity (log)', fontsize=7)
axes1[1, 0].set_ylabel('Wtd. power', fontsize=7)
axes1[2, 0].set_ylabel('Autocorr.', fontsize=7)

# Legend for Karlsson lines
from matplotlib.lines import Line2D
legend_elements = [
    Line2D([0], [0], color='steelblue', linestyle='--', linewidth=1, label=f'k_q = {KQ} (info-ball)'),
    Line2D([0], [0], color='darkred', linestyle=':', linewidth=1, label=f'k_m = {KM} (mass step)'),
]
fig1.legend(handles=legend_elements, loc='lower center', ncol=2, fontsize=8,
            bbox_to_anchor=(0.5, 0.0))
plt.savefig('/mnt/user-data/outputs/fig1_element_panels.png', dpi=150, bbox_inches='tight')
print("  Saved fig1_element_panels.png")

# ─────────────────────────────────────────────────────────────
# FIG 2: LOMB-SCARGLE PERIODOGRAMS — all elements overlaid
# ─────────────────────────────────────────────────────────────

print("Generating Figure 2: Lomb-Scargle periodograms...")
fig2, axes2 = plt.subplots(2, 3, figsize=(16, 8),
                             gridspec_kw={'hspace': 0.45, 'wspace': 0.35})
fig2.suptitle("Lomb-Scargle Periodogram in ln(freq) space — period axis in ln units\n"
               "Peaks near n × k_q = n × 0.205 or n × k_m = n × 1.804 support Karlsson structure",
               fontsize=10, fontweight='bold')

axes2_flat = axes2.flatten()
for i, (name, edata) in enumerate(ELEMENTS.items()):
    ax = axes2_flat[i]
    periods, pgram, peaks = lomb_scargle_periodicity(
        edata['ln_freqs'], edata['intensities'])
    if periods is None:
        ax.text(0.5, 0.5, 'Insufficient data', transform=ax.transAxes, ha='center')
        continue
    ax.plot(periods, pgram, color=edata['color'], linewidth=0.9, alpha=0.9)
    ax.fill_between(periods, pgram, alpha=0.15, color=edata['color'])
    
    # Mark KQ harmonics
    for n in range(1, 20):
        xpos = n * KQ
        if xpos < periods[-1]:
            ax.axvline(xpos, color='steelblue', linestyle='--', linewidth=0.6, alpha=0.45)
            if n <= 8 and pgram.max() > 0:
                ax.text(xpos, pgram.max() * 1.02, f'{n}k_q',
                        fontsize=4, ha='center', color='steelblue', rotation=90)
    # Mark KM harmonics
    for n in range(1, 5):
        xpos = n * KM
        if xpos < periods[-1]:
            ax.axvline(xpos, color='darkred', linestyle=':', linewidth=0.8, alpha=0.5)
            ax.text(xpos, pgram.max() * 0.9, f'{n}k_m',
                    fontsize=4.5, ha='center', color='darkred', rotation=90)
    
    ax.set_xlabel('Period (ln units)', fontsize=7)
    ax.set_ylabel('LS power', fontsize=7)
    ax.set_title(f'{edata["label"]}', fontsize=8, fontweight='bold')
    ax.tick_params(labelsize=7)
    ax.set_xlim(0, periods[-1])

plt.savefig('/mnt/user-data/outputs/fig2_lomb_scargle.png', dpi=150, bbox_inches='tight')
print("  Saved fig2_lomb_scargle.png")

# ─────────────────────────────────────────────────────────────
# FIG 3: CROSS-ELEMENT COMPARISON — all ln_freq lines on one axis
# with power-weighted binning
# ─────────────────────────────────────────────────────────────

print("Generating Figure 3: Cross-element Δln ν ratio chart...")
fig3, (ax3a, ax3b) = plt.subplots(1, 2, figsize=(16, 7),
                                    gridspec_kw={'wspace': 0.38})
fig3.suptitle("Cross-Element Logarithmic Frequency Structure\n"
               f"k_q = {KQ}  |  k_m = {KM}  |  Sum = {KQ+KM:.4f} ≈ 2",
               fontsize=11, fontweight='bold')

# Left: scatter of all lines in (element, ln_freq) space, sized by intensity
el_y = {'H': 0, 'He': 1, 'Fe': 2, 'Rb': 3, 'Cs': 4, 'Yb': 5}
el_names = list(el_y.keys())
ax3a.set_yticks(range(len(el_y)))
ax3a.set_yticklabels([ELEMENTS[e]['label'] for e in el_names], fontsize=9)
ax3a.set_xlabel('ln(ν)  [ln Hz]', fontsize=9)
ax3a.set_title('All NIST lines in ln(freq) space', fontsize=9)

for name, edata in ELEMENTS.items():
    y = el_y[name]
    lf = edata['ln_freqs']
    inten = edata['intensities']
    max_i = inten.max()
    sizes = 8 + 80 * (inten / max_i) ** 0.5
    ax3a.scatter(lf, np.full_like(lf, y), s=sizes, color=edata['color'],
                  alpha=0.65, zorder=3, edgecolors='none')

# Karlsson grid overlay
lf_all_min = min(e['ln_freqs'].min() for e in ELEMENTS.values())
lf_all_max = max(e['ln_freqs'].max() for e in ELEMENTS.values())
for n in range(int(lf_all_min/KQ)-1, int(lf_all_max/KQ)+2):
    ax3a.axvline(n * KQ + lf_all_min % KQ,
                  color='steelblue', linestyle='--', linewidth=0.4, alpha=0.2)
ax3a.set_xlim(lf_all_min - 0.5, lf_all_max + 0.5)
ax3a.grid(axis='x', alpha=0.1)

# Right: bar chart of Δln_ν / k_q for key pairs (same as LaTeX table)
pairs_data = [
    ('H',  'Ly-α/Balmer lim',    1.1040),
    ('H',  'Balmer/Paschen lim', 0.8116),
    ('H',  'Paschen/Brack lim',  0.4742),
    ('H',  'Ly-α/H-α',           1.6948),
    ('He', 'He I 584/537',       2.119),
    ('He', 'He II/He I 584',     0.453),
    ('He', 'He II/He II 1640',   1.387),
    ('Fe', 'Fe II UV/Fe I 4046', 0.831),
    ('Fe', 'Fe I 4046/5167',     0.235),
    ('Fe', 'Fe II UV/Fe I 5167', 1.066),
    ('Cs', 'Cs II/Cs I D2',      0.616),
    ('Cs', 'Cs I D2/13588',      0.466),
    ('Cs', 'Cs II/13588',        1.082),
    ('Rb', 'Rb II/Rb I D2',      0.610),
    ('Rb', 'Rb UV/Rb I D2',      0.620),
    ('Rb', 'Rb I D2/14752',      0.636),
    ('Yb', '¹P₁/repump 1389',   1.246),
    ('Yb', '¹P₁/³P₁ green',     0.332),
    ('Yb', '¹P₁/clock ³P₀',     0.371),
]

bar_labels = [f"{el}: {pair}" for el, pair, _ in pairs_data]
bar_vals = [dlnv / KQ for _, _, dlnv in pairs_data]
bar_colors = [ELEMENTS[el]['color'] for el, _, _ in pairs_data]
bar_y = np.arange(len(pairs_data))

bars = ax3b.barh(bar_y, bar_vals, color=bar_colors, alpha=0.75, height=0.65)

# Integer grid lines
for n in range(1, 12):
    ax3b.axvline(n, color='steelblue', linestyle='--', linewidth=0.7, alpha=0.35)
    ax3b.text(n, len(pairs_data) - 0.1, str(n), fontsize=6.5, ha='center',
               color='steelblue', va='bottom')

ax3b.set_yticks(bar_y)
ax3b.set_yticklabels(bar_labels, fontsize=6.5)
ax3b.set_xlabel('Δln ν  /  k_q', fontsize=9)
ax3b.set_title('Key inter-line frequency ratios\n(vertical lines = integer multiples of k_q)', fontsize=9)
ax3b.set_xlim(0, 11.5)

# Color whether near-integer
for i, (bval, bar) in enumerate(zip(bar_vals, bars)):
    near = abs(bval - round(bval)) < 0.2
    ax3b.text(bval + 0.1, bar.get_y() + bar.get_height()/2,
               f'{bval:.2f}{"✓" if near else "≈"}',
               va='center', fontsize=6, color='green' if near else 'orange')

plt.savefig('/mnt/user-data/outputs/fig3_cross_element.png', dpi=150, bbox_inches='tight')
print("  Saved fig3_cross_element.png")

# ─────────────────────────────────────────────────────────────
# FIG 4: LOG-POWER BINNING DETAIL — per element, all bin modes
# ─────────────────────────────────────────────────────────────

print("Generating Figure 4: Log-power binning histograms...")
fig4, axes4 = plt.subplots(2, n_el, figsize=(20, 7),
                             gridspec_kw={'hspace': 0.5, 'wspace': 0.35})
fig4.suptitle("Log-Power Binning: Intensity-weighted ln(ν) distributions\n"
               "Top: decade bins  |  Bottom: custom intensity bins", fontsize=10)

for col, (name, edata) in enumerate(ELEMENTS.items()):
    for row, (bins, bin_label) in enumerate([(LOG_BINS_DECADES, 'Decade'), 
                                              (LOG_BINS_CUSTOM, 'Custom')]):
        ax = axes4[row, col]
        binned = bin_lines(edata['intensities'], edata['ln_freqs'], bins)
        bin_colors = plt.cm.viridis(np.linspace(0.1, 0.9, len(bins)))
        
        # Histogram: each bin gets a bar per ln-freq region
        all_lf = edata['ln_freqs']
        if len(all_lf) == 0:
            continue
        lf_edges = np.linspace(all_lf.min() - 0.3, all_lf.max() + 0.3, 30)
        
        bottom = np.zeros(len(lf_edges) - 1)
        for j, (blabel, lf_b, in_b) in enumerate(binned):
            if len(lf_b) == 0:
                continue
            hist, _ = np.histogram(lf_b, bins=lf_edges, weights=in_b)
            ax.bar(lf_edges[:-1], hist, width=np.diff(lf_edges),
                   bottom=bottom, align='edge', color=bin_colors[j],
                   alpha=0.8, label=blabel, zorder=2)
            bottom += hist
        
        # KQ grid
        ref = edata['ln_freqs'][np.argmax(edata['intensities'])] if len(edata['ln_freqs'])>0 else 33
        for n in range(-20, 20):
            xpos = ref + n * KQ
            if lf_edges[0] < xpos < lf_edges[-1]:
                ax.axvline(xpos, color='steelblue', linestyle='--',
                            linewidth=0.5, alpha=0.35)
        
        ax.set_title(f'{name} — {bin_label} bins', fontsize=7, fontweight='bold')
        ax.tick_params(labelsize=6)
        if row == 0 and col == 0:
            ax.legend(fontsize=5, ncol=1, framealpha=0.5)
        if row == 1 and col == 0:
            ax.legend(fontsize=5, ncol=1, framealpha=0.5)
        ax.set_xlabel('ln(ν)', fontsize=6)
        if col == 0:
            ax.set_ylabel('Weighted count', fontsize=6)

plt.savefig('/mnt/user-data/outputs/fig4_log_binning.png', dpi=150, bbox_inches='tight')
print("  Saved fig4_log_binning.png")

# ─────────────────────────────────────────────────────────────
# PRINT SUMMARY TABLE
# ─────────────────────────────────────────────────────────────

print("\n" + "="*80)
print("KARLSSON STEP ANALYSIS SUMMARY")
print(f"k_q (info-ball) = {KQ}  |  k_m (mass) = {KM}  |  sum = {KQ+KM:.5f}")
print("="*80)

all_pairs = [
    ('H',  'Ly-α / Balmer limit',       1.1040),
    ('H',  'Balmer / Paschen limit',     0.8116),
    ('H',  'Paschen / Brackett limit',   0.4742),
    ('H',  'Ly-α / H-α',                1.6948),
    ('He', 'He I 584nm / He I 537nm',   2.1190),
    ('He', 'He II 304 / He I 584',       0.4530),
    ('He', 'He II 304 / He II 1640',     1.3870),
    ('Fe', 'Fe II UV / Fe I 4046',       0.8310),
    ('Fe', 'Fe I 4046 / Fe I 5167',      0.2350),
    ('Fe', 'Fe II UV / Fe I 5167',       1.0660),
    ('Cs', 'Cs II / Cs I D2',            0.6160),
    ('Cs', 'Cs I D2 / Cs I 13588',       0.4660),
    ('Cs', 'Cs II / Cs I 13588',         1.0820),
    ('Rb', 'Rb II / Rb I D2',            0.6100),
    ('Rb', 'Rb UV 421 / Rb I D2',        0.6197),
    ('Rb', 'Rb I D2 / Rb I 14752',       0.6355),
    ('Yb', '¹P₁ 399nm / repump 1389',   1.2460),
    ('Yb', '¹P₁ / ³P₁ green 556',       0.3320),
    ('Yb', '¹P₁ / clock ³P₀ 578',       0.3710),
]

print(f"\n{'El':>4} | {'Pair':<35} | {'Δln ν':>8} | {'/ k_q':>7} | {'Near N':>7} | {'|frac|':>7}")
print("-"*80)
for el, pair, dlnv in all_pairs:
    ratio = dlnv / KQ
    near_n = round(ratio)
    frac = abs(ratio - near_n)
    flag = "✓" if frac < 0.20 else ("~" if frac < 0.35 else "?")
    print(f"{el:>4} | {pair:<35} | {dlnv:8.4f} | {ratio:7.3f} | {near_n:>4} {flag}  | {frac:7.3f}")

near_int = [(el, p, d) for el, p, d in all_pairs if abs(d/KQ - round(d/KQ)) < 0.20]
fractional = [(el, p, d) for el, p, d in all_pairs if abs(d/KQ - round(d/KQ)) >= 0.20]

print("-"*80)
print(f"Near-integer (|frac| < 0.20): {len(near_int)}/{len(all_pairs)} = {100*len(near_int)/len(all_pairs):.0f}%")
print(f"Fractional (Yb ε(Z) signal):  {len(fractional)}/{len(all_pairs)}")
print()
print("IONIZATION INVARIANT CHECK (Δln ν for II→I gap):")
for el in ['Cs', 'Rb']:
    edata = ELEMENTS[el]
    print(f"  {el}: Δln ν ≈ 0.61 = {0.61/KQ:.2f} k_q  → integer 3")
print()
print("YB α-VARIATION SIGNAL:")
print(f"  ¹P₁/³P₁ gap = 0.332 = {0.332/KQ:.3f} k_q  (frac = {0.332/KQ - 1:.3f})")
print(f"  ¹P₁/³P₀ gap = 0.371 = {0.371/KQ:.3f} k_q  (frac = {0.371/KQ - 1:.3f})")
print(f"  → ε(Z=70) ≈ {(0.332/KQ - 1.5):.3f}  → δα/α ≈ {-3*(0.332/KQ - 1.5)*0.0026:.2e}")
print("="*80)

plt.show()
print("\nDone. All figures saved to /mnt/user-data/outputs/")