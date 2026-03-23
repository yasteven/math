"""
Apollonian Sphere Packing — Tetrahedral Seed  (fixed)

Bug fixes vs previous version:
  1. Sphere deduplication  — the Apollonian reflection group can reach the same
     geometric sphere via two different pocket paths (especially the seed spheres
     at depth 5+).  We now maintain sphere_map {(k_rounded, cx, cy, cz) -> idx}
     so duplicates reuse the existing sphere's index for child-pocket spawning
     without appending a second copy.
  2. Correct volume fraction  — previously excluded the 4 seed tetra-spheres
     from the numerator and included un-deduped duplicates.  Now: numerator =
     every positive-curvature sphere once, denominator = outer sphere volume.
  3. Per-depth stats now unaffected by duplicate seed spheres showing up late.
  4. Plotting uses z=0 cross-sections (circles) — much cleaner than 3-D
     wireframes for visualising the packing structure.
"""

import numpy as np
from itertools import combinations
from collections import deque
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches


# ---------------------------------------------------------------------------
# Geometry
# ---------------------------------------------------------------------------

def soddy_center(pocket_spheres: list[dict], r_new: float):
    """Linear solve for the center of a sphere of radius r_new tangent to all
    four spheres in pocket_spheres.  Returns ndarray or None."""
    cs = [np.array([float(x) for x in s["center"]]) for s in pocket_spheres]
    rs = [float(s["radius"])    for s in pocket_spheres]
    ks = [float(s["curvature"]) for s in pocket_spheres]
    r_new = float(r_new)

    # Squared tangency distance: (rᵢ+r)² for external, (rᵢ−r)² for internal
    d2 = [(rs[i] + r_new)**2 if ks[i] > 0 else (rs[i] - r_new)**2
          for i in range(4)]

    # Subtract eqn-0 from eqns 1,2,3 → 3×3 linear system
    A = np.stack([2.0*(cs[i] - cs[0]) for i in range(1, 4)])
    b = np.array([(d2[0] - cs[0]@cs[0]) - (d2[i] - cs[i]@cs[i])
                  for i in range(1, 4)], dtype=float)

    if abs(np.linalg.det(A)) < 1e-12:
        return None
    try:
        c = np.linalg.solve(A, b)
    except np.linalg.LinAlgError:
        return None

    residual = abs((c - cs[0])@(c - cs[0]) - d2[0])
    if residual > 1e-6 * (1.0 + d2[0]):
        return None
    return c


def _key(curvature, center) -> tuple:
    """Deduplication key — round to 4 decimal places."""
    return (round(float(curvature), 4),) + tuple(round(float(x), 4) for x in center)


# ---------------------------------------------------------------------------
# Packing BFS
# ---------------------------------------------------------------------------

def apollonian_sphere_packing(max_depth: int = 5) -> list[dict]:
    """
    Returns a list of sphere dicts:
        {"center": ndarray(3), "radius": float, "curvature": float, "depth": int}

    depth 0 = 4 seed unit-spheres + enclosing outer sphere
    depth 1 = first Soddy fill, etc.
    """

    # ── seed tetrahedron (edge = 2, so adjacent spheres are tangent) ──
    # One vertex points straight up (+z).  Top sphere center z ≈ 1.225 > r=1
    # so it does NOT cross the z=0 slice plane; the bottom 3 do, giving the
    # expected triangle-of-3 + inner Soddy view when sliced at z=0.
    R = np.sqrt(3.0 / 2.0)   # circumradius for edge-2 regular tetrahedron
    tet = np.array([
        [ 0,              0,    R   ],   # apex — above z=0 plane
        [ 2/np.sqrt(3),   0,   -R/3 ],  # base triangle
        [-1/np.sqrt(3),   1,   -R/3 ],
        [-1/np.sqrt(3),  -1,   -R/3 ],
    ], dtype=float)
    spheres: list[dict] = [
        {"center": v.copy(), "radius": 1.0, "curvature": 1.0, "depth": 0}
        for v in tet
    ]

    # outer / enclosing Soddy sphere  (k = 2 − √6 < 0)
    k_outer = 2.0 - np.sqrt(6.0)
    r_outer = 1.0 / abs(k_outer)
    spheres.append({"center": np.zeros(3), "radius": r_outer,
                    "curvature": k_outer, "depth": 0})

    # ── FIX 1: sphere deduplication map ──────────────────────────────────
    sphere_map: dict[tuple, int] = {_key(s["curvature"], s["center"]): i
                                    for i, s in enumerate(spheres)}

    seen_pockets: set[frozenset] = set()
    queue: deque = deque()

    for combo in combinations(range(5), 4):
        parent_idx = next(j for j in range(5) if j not in combo)
        fs = frozenset(combo)
        seen_pockets.add(fs)
        queue.append((list(combo), 1, spheres[parent_idx]["curvature"]))

    while queue:
        pocket_idx, depth, k_parent = queue.popleft()
        if depth > max_depth:
            continue

        pocket_sph = [spheres[i] for i in pocket_idx]
        k_new = sum(s["curvature"] for s in pocket_sph) - k_parent  # Descartes reflection

        if k_new < 1e-9:
            continue

        r_new  = 1.0 / k_new
        c_new  = soddy_center(pocket_sph, r_new)
        if c_new is None:
            continue

        key = _key(k_new, c_new)
        if key in sphere_map:
            # Sphere already exists — reuse its index so child pockets are
            # correctly connected, but don't append a duplicate.
            new_idx = sphere_map[key]
        else:
            new_idx = len(spheres)
            spheres.append({"center": c_new, "radius": r_new,
                             "curvature": k_new, "depth": depth})
            sphere_map[key] = new_idx

        # Spawn child pockets (even from a pre-existing sphere)
        for skip in range(4):
            child_idx = sorted([pocket_idx[j] for j in range(4) if j != skip] + [new_idx])
            fs = frozenset(child_idx)
            if fs not in seen_pockets:
                seen_pockets.add(fs)
                k_removed = pocket_sph[skip]["curvature"]
                queue.append((child_idx, depth + 1, k_removed))

    return spheres


# ---------------------------------------------------------------------------
# Analysis
# ---------------------------------------------------------------------------

def _vol(r: float) -> float:
    return (4.0/3.0) * np.pi * float(r)**3

def analyze(spheres: list[dict]) -> None:
    import statistics as _st

    outer    = next(s for s in spheres if s["curvature"] < 0)
    interior = [s for s in spheres if s["curvature"] > 0]
    v_outer  = _vol(outer["radius"])
    max_d    = max(s["depth"] for s in spheres)

    W = 80
    print("─" * W)
    print(f" Apollonian sphere packing  —  {len(spheres)} unique spheres total")
    print(f" Outer (enclosing) sphere radius : {outer['radius']:.7f}")
    print(f" Interior spheres                : {len(interior)}")
    print("─" * W)

    # collect layers first so we loop once
    layers = {}
    for d in range(max_d + 1):
        layer = [s for s in interior if s["depth"] == d]
        if layer:
            layers[d] = layer

    # ── volume summary ─────────────────────────────────────────────────────────
    print(f"  {'depth':>5}  {'n':>6}  {'cum.fill':>9}  {'new vol':>8}  {'of remain':>10}")
    print(f"  {'─'*5}  {'─'*6}  {'─'*9}  {'─'*8}  {'─'*10}")
    cumvol = 0.0
    for d, layer in layers.items():
        dv           = sum(_vol(s["radius"]) for s in layer)
        empty_before = v_outer - cumvol
        cumvol      += dv
        print(f"  {d:>5}  {len(layer):>6}  "
              f"{100*cumvol/v_outer:>8.4f}%  "
              f"{100*dv/v_outer:>7.4f}%  "
              f"{100*(dv/empty_before):>9.4f}%")
    print(f"  {'─'*5}  {'─'*6}  {'─'*9}  {'─'*8}  {'─'*10}")
    print(f"  Final filled : {100*cumvol/v_outer:.4f}%   "
          f"Empty remaining : {100*(v_outer-cumvol)/v_outer:.4f}%")

    # ── per-depth radius stats ─────────────────────────────────────────────────
    print()
    print(f"  Radius statistics per depth")
    print(f"  {'─'*72}")
    print(f"  {'depth':>5}  {'n':>5}  {'min':>10}  {'mean':>10}  "
          f"{'median':>10}  {'mode':>10}  {'max':>10}")
    print(f"  {'─'*5}  {'─'*5}  {'─'*10}  {'─'*10}  "
          f"{'─'*10}  {'─'*10}  {'─'*10}")
    for d, layer in layers.items():
        radii   = sorted(float(s["radius"]) for s in layer)
        n       = len(radii)
        rmin    = radii[0]
        rmax    = radii[-1]
        rmean   = sum(radii) / n
        rmed    = _st.median(radii)
        rounded = [round(r, 6) for r in radii]
        rmode   = _st.mode(rounded)
        print(f"  {d:>5}  {n:>5}  {rmin:>10.6f}  {rmean:>10.6f}  "
              f"{rmed:>10.6f}  {rmode:>10.6f}  {rmax:>10.6f}")
    print(f"  {'─'*5}  {'─'*5}  {'─'*10}  {'─'*10}  "
          f"{'─'*10}  {'─'*10}  {'─'*10}")
    print()


# ---------------------------------------------------------------------------
# Plotting — z = 0 cross-section
# ---------------------------------------------------------------------------
# Slicing a sphere of radius r centred at (cx,cy,cz) with the plane z=0
# gives a circle of radius  r_slice = √(r²−cz²)  centred at (cx,cy)
# — but only when |cz| < r.

DEPTH_COLOURS = [
    "#ffffff",   # 0  seed — white
    "#ffe44d",   # 1  yellow
    "#ff8c42",   # 2  orange
    "#e84855",   # 3  red
    "#7b2d8b",   # 4  purple
    "#2196f3",   # 5  blue
    "#00bcd4",   # 6  cyan
]


def _best_z(spheres_at_depth: list[dict]) -> float:
    """
    Find the z-slice that maximises the number of sphere cross-sections visible.
    We sweep z in 200 steps over the bounding range and pick the z that
    intersects the most spheres.  Ties broken by total cross-sectional area.
    """
    zs  = np.array([float(s["center"][2]) for s in spheres_at_depth])
    rs  = np.array([float(s["radius"])    for s in spheres_at_depth])
    z_lo, z_hi = (zs - rs).min(), (zs + rs).max()
    candidates = np.linspace(z_lo + 1e-4, z_hi - 1e-4, 400)

    best_z, best_score = 0.0, -1.0
    for zc in candidates:
        dz2  = (zc - zs) ** 2
        mask = dz2 < rs ** 2
        score = float(mask.sum()) + float((rs[mask] ** 2 - dz2[mask]).sum()) * 1e-6
        if score > best_score:
            best_score, best_z = score, float(zc)
    return best_z


def _draw_slice(ax, spheres: list[dict], z_cut: float,
                r_outer: float, max_d: int,
                depth_limit: int) -> None:
    """Draw all sphere cross-sections at height z_cut onto ax."""
    # Outer boundary circle at this z
    dz_outer = z_cut          # outer sphere is centred at origin
    if abs(dz_outer) < r_outer:
        r_outer_slice = np.sqrt(r_outer**2 - dz_outer**2)
        ax.add_patch(plt.Circle((0, 0), r_outer_slice, fill=False,
                                edgecolor="white", lw=1.5, linestyle="--", zorder=5))

    # All inner sphere cross-sections at z_cut
    visible = [s for s in spheres
               if s["curvature"] > 0 and s["depth"] <= depth_limit]
    for s in sorted(visible, key=lambda x: -float(x["radius"])):
        cx, cy, cz = (float(x) for x in s["center"])
        r = float(s["radius"])
        dz = z_cut - cz
        if dz * dz >= r * r:
            continue
        r_slice = np.sqrt(r**2 - dz**2)
        depth   = s["depth"]
        colour  = DEPTH_COLOURS[depth] if depth < len(DEPTH_COLOURS) else "#aaaaaa"
        alpha   = 0.85 if r > 0.5 else (0.70 if r > 0.1 else 0.55)
        ax.add_patch(plt.Circle((cx, cy), r_slice,
                                color=colour, alpha=alpha, lw=0, zorder=3))

    ax.set_xlim(-1.15*r_outer, 1.15*r_outer)
    ax.set_ylim(-1.15*r_outer, 1.15*r_outer)
    ax.set_aspect("equal")
    ax.set_facecolor("#0d0d0d")
    ax.axis("off")


def plot_cross_section(spheres: list[dict], depth_limit: int | None = None,
                       filename: str | None = None) -> None:
    """
    Render a two-panel figure per depth_limit:
      LEFT  — fixed z=0 slice (for reference)
      RIGHT — optimal-z slice for the NEW spheres added at this depth
              (maximises the number of depth=depth_limit spheres visible)

    This makes it obvious that z=0 is often a poor cut and shows what the
    correct cross-section looks like.
    """
    outer   = next(s for s in spheres if s["curvature"] < 0)
    r_outer = float(outer["radius"])
    max_d   = max(s["depth"] for s in spheres)
    if depth_limit is None:
        depth_limit = max_d

    # Find optimal z for the new layer
    new_layer = [s for s in spheres if s["depth"] == depth_limit and s["curvature"] > 0]
    z_opt = _best_z(new_layer) if new_layer else 0.0

    fig, axes = plt.subplots(1, 2, figsize=(14, 7))
    fig.patch.set_facecolor("#0d0d0d")

    for ax, (z_cut, label) in zip(axes, [
        (0.0,  f"z = 0  (fixed reference)"),
        (z_opt, f"z = {z_opt:.3f}  (optimal for depth {depth_limit})"),
    ]):
        _draw_slice(ax, spheres, z_cut, r_outer, max_d, depth_limit)
        n_vis = sum(1 for s in new_layer
                    if (z_cut - float(s["center"][2]))**2 < s["radius"]**2)
        ax.set_title(f"{label}\n{n_vis}/{len(new_layer)} depth-{depth_limit} spheres visible",
                     color="white", fontsize=9, pad=6)

    # Shared legend
    handles = []
    for d in range(depth_limit + 1):
        count = sum(1 for s in spheres if s["depth"] == d and s["curvature"] > 0)
        if count:
            col = DEPTH_COLOURS[d] if d < len(DEPTH_COLOURS) else "#aaaaaa"
            handles.append(mpatches.Patch(color=col, alpha=0.85,
                                          label=f"depth {d}  ({count} spheres)"))
    fig.legend(handles=handles, loc="lower center", ncol=len(handles),
               fontsize=8, facecolor="#1a1a1a", edgecolor="#555555",
               labelcolor="white", bbox_to_anchor=(0.5, 0.01))

    plt.suptitle(f"Apollonian Sphere Packing  (depth ≤ {depth_limit})",
                 color="white", fontsize=11, y=1.01)
    plt.tight_layout()

    if filename:
        plt.savefig(filename, dpi=150, bbox_inches="tight",
                    facecolor=fig.get_facecolor())
        print(f"  saved → {filename}")
    plt.close()


# ---------------------------------------------------------------------------
# HTML 3-D viewer
# ---------------------------------------------------------------------------

def generate_html(spheres: list[dict], filename: str = "apollonian_3d.html") -> None:
    """
    Write a self-contained HTML file that renders the packing interactively
    using Three.js (loaded from CDN).  The sphere data is embedded as JSON.
    Controls: left-drag orbit, right-drag pan, scroll zoom, depth toggles.
    """
    import json as _json

    inner = [s for s in spheres if s["curvature"] > 0]
    outer = next(s for s in spheres if s["curvature"] < 0)
    max_d = max(s["depth"] for s in inner)

    def _fmt(s):
        c = s["center"]
        return {
            "x": round(float(c[0]), 6), "y": round(float(c[1]), 6),
            "z": round(float(c[2]), 6), "r": round(float(s["radius"]), 6),
            "d": int(s["depth"]),
        }

    sphere_json = _json.dumps([_fmt(s) for s in inner])
    outer_json  = _json.dumps(_fmt(outer))
    depth_count_js = _json.dumps({
        str(d): sum(1 for s in inner if s["depth"] == d)
        for d in range(max_d + 1)
    })

    html = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>Apollonian Sphere Packing — 3D</title>
<style>
  *, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }
  :root {
    --bg: #080a0e; --panel: rgba(10,14,20,0.93); --border: #1e2530;
    --text: #c8d4e0; --muted: #4a5870;
  }
  html, body { width:100%; height:100%; overflow:hidden; background:var(--bg);
    font-family: 'Courier New', monospace; }
  #c { position:absolute; inset:0; }

  #ui {
    position:absolute; top:16px; left:16px; width:220px;
    background:var(--panel); border:1px solid var(--border);
    backdrop-filter:blur(10px); padding:16px; user-select:none;
  }
  h1 { font-size:11px; font-weight:700; letter-spacing:.15em;
       text-transform:uppercase; color:var(--text); margin-bottom:3px; }
  .sub { font-size:9px; color:var(--muted); letter-spacing:.07em; margin-bottom:14px; }
  .lbl { font-size:9px; letter-spacing:.1em; text-transform:uppercase;
         color:var(--muted); margin-bottom:7px; }
  .sep { border:none; border-top:1px solid var(--border); margin:12px 0; }

  .drow {
    display:flex; align-items:center; gap:8px; margin-bottom:6px;
    cursor:pointer; padding:2px 0;
  }
  .drow:hover .dlbl { color:var(--text); }
  .sw { width:10px; height:10px; border-radius:50%; flex-shrink:0; }
  .dlbl { font-size:10px; color:var(--muted); flex:1; transition:color .12s; }
  .drow.on .dlbl { color:var(--text); }
  .chk { width:13px; height:13px; border:1px solid var(--border);
         display:flex; align-items:center; justify-content:center;
         font-size:8px; color:transparent; flex-shrink:0; }
  .drow.on .chk { border-color:var(--text); color:var(--text); }

  .orow { display:flex; justify-content:space-between; align-items:center; margin-bottom:8px; }
  .olbl { font-size:10px; color:var(--muted); }
  .pills { display:flex; border:1px solid var(--border); overflow:hidden; }
  .pills button {
    background:none; border:none; border-right:1px solid var(--border);
    padding:3px 8px; font-family:inherit; font-size:9px;
    color:var(--muted); cursor:pointer; transition:background .12s,color .12s;
  }
  .pills button:last-child { border-right:none; }
  .pills button.on { background:var(--border); color:var(--text); }

  .srow { margin-bottom:8px; }
  .srow .slbl { font-size:10px; color:var(--muted);
                display:flex; justify-content:space-between; margin-bottom:3px; }
  input[type=range] {
    -webkit-appearance:none; width:100%; height:2px;
    background:var(--border); outline:none; cursor:pointer;
  }
  input[type=range]::-webkit-slider-thumb {
    -webkit-appearance:none; width:9px; height:9px;
    border-radius:50%; background:var(--text); cursor:pointer;
  }

  #info { position:absolute; bottom:16px; left:16px;
          font-size:9px; color:var(--muted); line-height:1.9; }
  #hint { position:absolute; bottom:16px; right:16px;
          font-size:9px; color:var(--muted); text-align:right; line-height:1.9; }
</style>
</head>
<body>
<div id="c"></div>

<div id="ui">
  <h1>Apollonian Packing</h1>
  <div class="sub">Tetrahedral seed &mdash; 3D viewer</div>
  <div class="lbl">Depths</div>
  <div id="dc"></div>
  <hr class="sep">
  <div class="lbl">Render</div>
  <div class="orow">
    <span class="olbl">Mode</span>
    <div class="pills" id="mp">
      <button class="on" data-v="solid">solid</button>
      <button data-v="wire">wire</button>
      <button data-v="both">both</button>
    </div>
  </div>
  <div class="orow">
    <span class="olbl">Outer sphere</span>
    <div class="pills" id="op">
      <button class="on" data-v="1">show</button>
      <button data-v="0">hide</button>
    </div>
  </div>
  <div class="srow">
    <div class="slbl"><span>Opacity</span><span id="ov">0.75</span></div>
    <input type="range" id="osl" min="5" max="100" value="75">
  </div>
  <div class="srow">
    <div class="slbl"><span>Max segs (large&rarr;small)</span><span id="sv">24</span></div>
    <input type="range" id="ssl" min="4" max="48" value="24" step="2">
  </div>
</div>

<div id="info"></div>
<div id="hint">left-drag: orbit<br>right-drag: pan<br>scroll: zoom</div>

<script src="https://cdnjs.cloudflare.com/ajax/libs/three.js/r134/three.min.js"></script>
<script>
// ── embedded data ────────────────────────────────────────────────────────────
const SPHERES = """ + sphere_json + r""";
const OUTER   = """ + outer_json  + r""";
const DCOUNTS = """ + depth_count_js + r""";
const MAX_D   = Math.max(...SPHERES.map(s => s.d));
const COLS    = [0xe8e8f0,0xf5d040,0xf07830,0xe83050,0x9040d0,0x2090f0,0x20d0c0];

// ── state ────────────────────────────────────────────────────────────────────
const vis   = Object.fromEntries(Array.from({length:MAX_D+1},(_,i)=>[i,true]));
let mode    = 'solid';
let showOut = true;
let opacity = 0.75;
let maxSegs = 24;  // segments for the largest sphere; smaller ones scale down

// ── renderer ─────────────────────────────────────────────────────────────────
const renderer = new THREE.WebGLRenderer({antialias:true});
renderer.setPixelRatio(Math.min(devicePixelRatio, 2));
renderer.setSize(innerWidth, innerHeight);
renderer.setClearColor(0x080a0e);
document.getElementById('c').appendChild(renderer.domElement);

const scene  = new THREE.Scene();
const camera = new THREE.PerspectiveCamera(42, innerWidth/innerHeight, 0.001, 200);

// ── lights ───────────────────────────────────────────────────────────────────
scene.add(new THREE.AmbientLight(0xffffff, 0.45));
const dl = new THREE.DirectionalLight(0xffffff, 0.9);
dl.position.set(4, 7, 5); scene.add(dl);
const dl2 = new THREE.DirectionalLight(0x6688cc, 0.35);
dl2.position.set(-4,-3,-4); scene.add(dl2);

// ── mesh pool ─────────────────────────────────────────────────────────────────
// Segments scale with sqrt(r / r_max) so large spheres are smooth and tiny
// ones stay cheap.  Geometries are cached by segment count to share across
// spheres of the same LOD bucket.
let groups  = {};
let geoPool = {};   // segs -> SphereGeometry
const R_MAX = Math.max(...SPHERES.map(s=>s.r));

function getGeo(s) {
  const n = Math.max(4, Math.round(maxSegs * Math.sqrt(s.r / R_MAX)));
  if (!geoPool[n]) geoPool[n] = new THREE.SphereGeometry(1, n, n);
  return geoPool[n];
}

function buildAll() {
  // dispose old meshes (but keep geo pool — geometries are reused)
  Object.values(groups).forEach(g=>[...g.s,...g.w].forEach(m=>{
    scene.remove(m); m.material.dispose();
  }));
  groups = {};

  for (let d = 0; d <= MAX_D; d++) {
    groups[d] = {s:[], w:[]};
    const col  = COLS[d] ?? 0xaaaaaa;
    SPHERES.filter(s=>s.d===d).forEach(s => {
      const geo = getGeo(s);
      const sm = new THREE.Mesh(geo,
        new THREE.MeshPhongMaterial({color:col,transparent:true,opacity,shininess:55}));
      sm.position.set(s.x,s.y,s.z);
      sm.scale.setScalar(s.r);
      scene.add(sm); groups[d].s.push(sm);

      const wm = new THREE.Mesh(geo,
        new THREE.MeshBasicMaterial({color:col,wireframe:true,transparent:true,opacity:0.22}));
      wm.position.set(s.x,s.y,s.z);
      wm.scale.setScalar(s.r);
      scene.add(wm); groups[d].w.push(wm);
    });
  }

  // outer sphere — always full segments
  groups['o'] = {s:[], w:[]};
  const outerGeo = new THREE.SphereGeometry(1, maxSegs, maxSegs);
  const om = new THREE.Mesh(outerGeo,
    new THREE.MeshPhongMaterial({color:0x223355,transparent:true,opacity:0.07,side:THREE.BackSide}));
  om.position.set(OUTER.x,OUTER.y,OUTER.z);
  om.scale.setScalar(OUTER.r); scene.add(om); groups['o'].s.push(om);

  syncVis();
}

function rebuildGeos() {
  // Dispose cached geos so getGeo() rebuilds them at new maxSegs
  Object.values(geoPool).forEach(g=>g.dispose());
  geoPool = {};
  buildAll();
}

function syncVis() {
  for (let d = 0; d <= MAX_D; d++) {
    const g = groups[d]; if (!g) continue;
    const on = vis[d];
    g.s.forEach(m=>{ m.visible = on && (mode==='solid'||mode==='both'); m.material.opacity=opacity; });
    g.w.forEach(m=>{ m.visible = on && (mode==='wire' ||mode==='both'); });
  }
  const og = groups['o']; if (og) og.s.forEach(m=>m.visible=showOut);
  updateInfo();
}

function updateInfo() {
  const n = SPHERES.filter(s=>vis[s.d]).length;
  document.getElementById('info').innerHTML =
    `${n} / ${SPHERES.length} spheres<br>` +
    Array.from({length:MAX_D+1},(_,d)=>`d${d}:${DCOUNTS[d]}`).join(' &middot; ');
}

// ── camera / orbit ────────────────────────────────────────────────────────────
let sph = {r:7, theta:0.4, phi:1.2};
let panV = new THREE.Vector3();

function camUpdate() {
  camera.position.set(
    panV.x + sph.r * Math.sin(sph.phi) * Math.sin(sph.theta),
    panV.y + sph.r * Math.cos(sph.phi),
    panV.z + sph.r * Math.sin(sph.phi) * Math.cos(sph.theta));
  camera.lookAt(panV);
}
camUpdate();

let drag=false, rDrag=false, lx=0, ly=0;
const el = renderer.domElement;
el.addEventListener('contextmenu',e=>e.preventDefault());
el.addEventListener('mousedown',e=>{drag=true;rDrag=e.button===2;lx=e.clientX;ly=e.clientY;});
window.addEventListener('mouseup',()=>drag=false);
window.addEventListener('mousemove',e=>{
  if(!drag) return;
  const dx=e.clientX-lx, dy=e.clientY-ly; lx=e.clientX; ly=e.clientY;
  if(rDrag){
    const fwd=new THREE.Vector3(panV.x-camera.position.x,0,panV.z-camera.position.z).normalize();
    const right=new THREE.Vector3().crossVectors(new THREE.Vector3(0,1,0),fwd).normalize();
    panV.addScaledVector(right,-dx*0.004);
    panV.y+=dy*0.004;
  } else {
    sph.theta-=dx*0.005;
    sph.phi=Math.max(0.05,Math.min(Math.PI-0.05,sph.phi-dy*0.005));
  }
  camUpdate();
});
el.addEventListener('wheel',e=>{
  sph.r=Math.max(0.5,Math.min(40,sph.r+e.deltaY*0.004)); camUpdate();
},{passive:true});

window.addEventListener('resize',()=>{
  renderer.setSize(innerWidth,innerHeight);
  camera.aspect=innerWidth/innerHeight;
  camera.updateProjectionMatrix();
});

// ── UI ────────────────────────────────────────────────────────────────────────
const dc = document.getElementById('dc');
for(let d=0;d<=MAX_D;d++){
  const col='#'+(COLS[d]??0xaaaaaa).toString(16).padStart(6,'0');
  const n = DCOUNTS[d];
  const row=document.createElement('div');
  row.className='drow on'; row.dataset.d=d;
  row.innerHTML=`<div class="sw" style="background:${col}"></div>
    <span class="dlbl">depth ${d} <span style="color:#2a3a4a">(${n})</span></span>
    <div class="chk">&#10003;</div>`;
  row.addEventListener('click',()=>{
    vis[d]=!vis[d];
    row.classList.toggle('on',vis[d]);
    row.querySelector('.chk').style.color=vis[d]?'':'transparent';
    syncVis();
  });
  dc.appendChild(row);
}

function pills(id, cb) {
  document.getElementById(id).addEventListener('click', e=>{
    const b=e.target.closest('button'); if(!b) return;
    document.querySelectorAll(`#${id} button`).forEach(x=>x.classList.remove('on'));
    b.classList.add('on'); cb(b.dataset.v);
  });
}
pills('mp', v=>{ mode=v; syncVis(); });
pills('op', v=>{ showOut=v==='1'; syncVis(); });

document.getElementById('osl').addEventListener('input',e=>{
  opacity=e.target.value/100;
  document.getElementById('ov').textContent=opacity.toFixed(2);
  for(let d=0;d<=MAX_D;d++) groups[d]?.s.forEach(m=>m.material.opacity=opacity);
});
document.getElementById('ssl').addEventListener('input',e=>{
  maxSegs=+e.target.value;
  document.getElementById('sv').textContent=maxSegs;
  rebuildGeos();
});

// ── init + loop ───────────────────────────────────────────────────────────────
buildAll();
(function loop(){ requestAnimationFrame(loop); renderer.render(scene,camera); })();
</script>
</body>
</html>"""

    with open(filename, "w") as f:
        f.write(html)
    print(f"Saved 3D viewer → {filename}  ({len(spheres)} spheres, depth 0–{max_d})")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import sys
    MAX = int(sys.argv[1]) if len(sys.argv) > 1 else 5
    print(f"Computing packing to depth {MAX} …")
    spheres = apollonian_sphere_packing(max_depth=MAX)
    analyze(spheres)

    print("Generating cross-section plots …")
    for d in range(MAX + 1):
        plot_cross_section(spheres, depth_limit=d,
                           filename=f"cross_section_depth_{d}.png")

    generate_html(spheres, filename="apollonian_3d.html")
    print("Done.")