import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.constants import pi
import urllib.request
import os
import re

print("=== SPARC FRACTAL SCALE FACTOR ANALYSIS - GALAXY-SPECIFIC DRAG ===")
print("     (drag modulated by luminosity proxy - anchored ~31%)     \n")

# ────────────────────────────────────────────────────────────────
# 1. DOWNLOAD THE MRT FILE
# ────────────────────────────────────────────────────────────────
url = "https://astroweb.cwru.edu/SPARC/SPARC_Lelli2016c.mrt"
local_mrt = "SPARC_Lelli2016c.mrt"

print(f"Downloading: {url}")
try:
    urllib.request.urlretrieve(url, local_mrt)
    print(f"Saved to: {local_mrt}")
except Exception as e:
    print(f"Download failed: {e}")
    exit(1)

# ────────────────────────────────────────────────────────────────
# 2. PARSE MRT → skip header, locate real data
# ────────────────────────────────────────────────────────────────
with open(local_mrt, 'r', encoding='utf-8') as f:
    raw_text = f.read()

lines = raw_text.splitlines()

data_start_idx = -1
for i, line in enumerate(lines):
    s = line.strip()
    if s and not s.startswith(('Title:', 'Authors:', 'Table:', 'Byte-by-byte', 'Note', '   Bytes', '   1-', '-----')):
        fields = re.split(r'\s+', s)
        if len(fields) >= 12 and any(f.replace('.', '').isdigit() for f in fields[2:7]):
            data_start_idx = i
            break

if data_start_idx == -1:
    print("Could not find data start. First 40 lines:")
    print('\n'.join(lines[:40]))
    exit(1)

print(f"Data starts ~ line {data_start_idx + 1}")

header = [
    'Galaxy', 'T', 'D', 'e_D', 'f_D', 'Inc', 'e_Inc',
    'L[3.6]', 'e_L[3.6]', 'Reff', 'SBeff', 'Rdisk', 'SBdisk',
    'MHI', 'RHI', 'Vflat', 'e_Vflat', 'Q', 'Ref.'
]

data_rows = []
for line in lines[data_start_idx:]:
    s = line.rstrip()
    if not s or s.startswith(('!', '#')):
        continue
    fields = re.split(r'\s{2,}|\s+(?=\d)', s.strip())
    fields = [f.strip() for f in fields if f.strip()]

    if len(fields) > len(header):
        ref_start = 18
        if len(fields) >= ref_start:
            fields[ref_start-1:] = [' '.join(fields[ref_start-1:])]
            fields = fields[:ref_start]

    if len(fields) >= 17:  # up to e_Vflat
        data_rows.append(fields[:len(header)])

print(f"Parsed {len(data_rows)} candidate rows")

df = pd.DataFrame(data_rows, columns=header)

# Convert numeric columns
numeric = ['T', 'D', 'e_D', 'f_D', 'Inc', 'e_Inc', 'L[3.6]', 'e_L[3.6]',
           'Reff', 'SBeff', 'Rdisk', 'SBdisk', 'MHI', 'RHI',
           'Vflat', 'e_Vflat', 'Q']

for col in numeric:
    if col in df.columns:
        df[col] = df[col].astype(str).str.replace(r'[a-zA-Z<]', '', regex=True)
        df[col] = pd.to_numeric(df[col], errors='coerce')

# ────────────────────────────────────────────────────────────────
# 3. FILTER TO VALID GALAXIES
# ────────────────────────────────────────────────────────────────
r_outer_col = 'Rlast' if 'Rlast' in df.columns else 'RHI'
df = df.dropna(subset=['Vflat', 'D', r_outer_col])
df = df[(df['Vflat'] > 5) & (df[r_outer_col] > 0.2)]

print(f"Kept {len(df)} galaxies with Vflat > 5 km/s and {r_outer_col} > 0.2 kpc")

if len(df) < 20:
    print("Too few valid galaxies — check parsing.")
    exit(1)

# ────────────────────────────────────────────────────────────────
# 4. COMPUTE ORBITAL FREQUENCY
# ────────────────────────────────────────────────────────────────
km_s_to_m_s = 1000
kpc_to_m    = 3.08568e19

df['R_outer_kpc'] = df[r_outer_col]
df['f_gal_Hz']    = (df['Vflat'] * km_s_to_m_s) / (2 * pi * df['R_outer_kpc'] * kpc_to_m)

print("\nSample frequencies (first 6):")
print(df[['Galaxy', 'D', 'Vflat', r_outer_col, 'f_gal_Hz']].head(6).round(4))

# ────────────────────────────────────────────────────────────────
# 5. GALAXY-SPECIFIC DRAG (FSEP-inspired, luminosity proxy)
# ────────────────────────────────────────────────────────────────
# Protect against invalid L[3.6]
df['L36_safe'] = df['L[3.6]'].clip(lower=1e-4)

# Normalize log-luminosity around median
logL = np.log10(df['L36_safe'])
logL_median = logL.median()
logL_norm = logL - logL_median   # + for brighter than median

# Drag: base 0.31, lower for bright (baryon-dominated), higher for faint (DM-dominated)
df['drag'] = 0.31 * (1 - 0.9 * logL_norm)   # stronger inverse scaling

# Realistic bounds (roughly 18–52% range seen in SPARC decompositions)
df['drag'] = df['drag'].clip(lower=0.18, upper=0.52)

print("\n=== Drag statistics (galaxy-specific) ===")
print(df['drag'].describe().round(3))

print("\nExamples - lowest drag (brightest/most baryon-dominated):")
print(df[['Galaxy', 'L[3.6]', 'Vflat', 'drag']].sort_values('drag').head(8).round(3))

print("\nExamples - highest drag (faintest/most DM-dominated):")
print(df[['Galaxy', 'L[3.6]', 'Vflat', 'drag']].sort_values('drag', ascending=False).head(8).round(3))

# ────────────────────────────────────────────────────────────────
# 6. HYDROGEN LINES & SCALE FACTORS
# ────────────────────────────────────────────────────────────────
h_lines = {
    'H_alpha':       4.568e14,
    'H_beta':        6.167e14,
    'H_gamma':       6.907e14,
    'H_delta':       7.309e14,
    'H_epsilon':     7.551e14,
    'H_zeta':        7.709e14,
    'H_eta':         7.816e14,
    'H_theta':       7.894e14,
    'H_iota':        7.951e14,
    'Lyman_alpha':   2.466e15,
    'Paschen_alpha': 1.599e14,
    'Paschen_beta':  2.338e14
}

for name, freq in h_lines.items():
    df[f'scale_{name}'] = freq / (df['f_gal_Hz'] * df['drag'])
    df[f'log10_{name}'] = np.log10(df[f'scale_{name}'])

# ────────────────────────────────────────────────────────────────
# 7. OUTPUT TABLE & CLUSTERING
# ────────────────────────────────────────────────────────────────
key_cols = ['Galaxy', 'D', 'Vflat', 'drag'] + [f'log10_{n}' for n in h_lines]

print("\n=== FIRST 20 GALAXIES WITH LOG10 SCALE FACTORS ===")
print(df[key_cols].round(3).head(20).to_string(index=False))

print("\n=== CLUSTERING SUMMARY (±0.25 around nearest integer) ===")
for name in h_lines:
    col = f'log10_{name}'
    frac = ((df[col] - np.round(df[col])).abs() <= 0.25).mean()
    print(f"{name:14} : {frac:5.1%}   (n={len(df):3d})")

# ────────────────────────────────────────────────────────────────
# 8. HISTOGRAMS — safe & dynamic range
# ────────────────────────────────────────────────────────────────
plot_lines = ['H_alpha', 'Lyman_alpha', 'Paschen_alpha']

plt.figure(figsize=(15, 5))

for i, name in enumerate(plot_lines, 1):
    col = f'log10_{name}'
    vals = df[col].dropna()
    vals = vals[np.isfinite(vals)]
    
    plt.subplot(1, 3, i)
    if len(vals) < 5:
        plt.text(0.5, 0.5, "Too few finite values", ha='center', va='center')
        continue
    
    plt.hist(vals, bins=40, color='cornflowerblue', edgecolor='navy', alpha=0.85)
    plt.title(name.replace('_', ' '))
    plt.xlabel('log₁₀ (scale factor)')
    plt.ylabel('Number of galaxies')

    vmin, vmax = vals.min(), vals.max()
    k_start = int(np.floor(vmin)) - 1
    k_end   = int(np.ceil(vmax)) + 2
    for k in range(k_start, k_end):
        plt.axvline(k, color='firebrick', ls='--', lw=1.0, alpha=0.7)
        plt.text(k + 0.08, plt.ylim()[1]*0.88, f'10^{k}', color='firebrick',
                 fontsize=9, rotation=90, va='top')

plt.tight_layout()
plt.show()

# ────────────────────────────────────────────────────────────────
# 9. CLEANUP
# ────────────────────────────────────────────────────────────────
if os.path.exists(local_mrt):
    os.remove(local_mrt)
    print(f"\nCleaned up: {local_mrt}")

print("\nScript finished. Galaxy-specific drag based on L[3.6] luminosity proxy.")