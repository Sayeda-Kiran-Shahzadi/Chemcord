"""Label every tmQM complex with a coordination geometry, from its 3D structure.

Reads the tmQM xyz files (plain or .gz), and for each complex

  1. locates the metal (the one transition-metal atom; its position in the file varies),
  2. finds the donor atoms: every atom closer to the metal than the sum of the
     two covalent radii + BOND_TOLERANCE.  Hydrogens only count when the metal
     is their nearest neighbour (true hydrides, not agostic C-H contacts),
  3. collapses haptic ligands (donor atoms bonded to each other, e.g. Cp, arenes,
     eta2-alkenes) to ONE pseudo-donor at their centroid, so a Cp ring is one
     vertex of the coordination polyhedron, as is usual in structural chemistry,
  4. measures every metal-donor / metal-donor angle and compares the sorted
     angle list with those of ideal polyhedra (linear, trigonal planar,
     tetrahedral, square planar, square pyramidal, trigonal bipyramidal,
     octahedral, trigonal prismatic, pentagonal bipyramidal, ...).  The best
     match is the label; if even the best match is far off the complex is
     labelled "irregular (CN=n)".

Only the Python standard library is used.

Usage
    python label_geometry.py                      # looks for data/tmQM_X*_xyz(.gz)
    python label_geometry.py file1.gz file2.gz    # explicit files
    python label_geometry.py -j 4 ...             # number of worker processes

Output: data/geometry_labels.csv.gz
        csd_code, metal, n_donor_atoms, eff_cn, geometry, rmsd_deg, max_angle_deg
"""
import csv
import glob
import gzip
import math
import sys
from multiprocessing import Pool
from pathlib import Path

ROOT = Path(__file__).parent
DATA = ROOT / "data"
OUT = DATA / "geometry_labels.csv.gz"

BOND_TOLERANCE = 0.45      # Angstrom added to the sum of covalent radii (CSD convention)
HAPTIC_TOLERANCE = 0.45    # same rule for "two donor atoms are bonded to each other"
IRREGULAR_RMSD = 22.0      # degrees; best-match RMSD above this -> "irregular"
PRISM_MAX_ANGLE = 150.0    # CN=6: a trans angle above this is not a trigonal prism (ideal: 136)
BENT_ANGLE = 150.0         # CN=2: angle below this is "bent", else "linear"

# Cordero et al. 2008 covalent radii (Angstrom); high-spin value for Mn/Fe/Co averaged.
RADII = {
    "H": 0.31, "Li": 1.28, "Be": 0.96, "B": 0.84, "C": 0.73, "N": 0.71, "O": 0.66, "F": 0.57,
    "Na": 1.66, "Mg": 1.41, "Al": 1.21, "Si": 1.11, "P": 1.07, "S": 1.05, "Cl": 1.02,
    "K": 2.03, "Ca": 1.76, "Ga": 1.22, "Ge": 1.20, "As": 1.19, "Se": 1.20, "Br": 1.20,
    "Rb": 2.20, "Sr": 1.95, "In": 1.42, "Sn": 1.39, "Sb": 1.39, "Te": 1.38, "I": 1.39,
    "Cs": 2.44, "Ba": 2.15, "Tl": 1.45, "Pb": 1.46, "Bi": 1.48,
    "Sc": 1.70, "Ti": 1.60, "V": 1.53, "Cr": 1.39, "Mn": 1.50, "Fe": 1.42, "Co": 1.38,
    "Ni": 1.24, "Cu": 1.32, "Zn": 1.22,
    "Y": 1.90, "Zr": 1.75, "Nb": 1.64, "Mo": 1.54, "Tc": 1.47, "Ru": 1.46, "Rh": 1.42,
    "Pd": 1.39, "Ag": 1.45, "Cd": 1.44,
    "La": 2.07, "Hf": 1.75, "Ta": 1.70, "W": 1.62, "Re": 1.51, "Os": 1.44, "Ir": 1.41,
    "Pt": 1.36, "Au": 1.36, "Hg": 1.32,
}
DEFAULT_RADIUS = 1.50

# The transition metals (plus La) that tmQM complexes are built around.  The metal is
# NOT always the first atom in the xyz file, so it is located by element.
METALS = {"Sc", "Ti", "V", "Cr", "Mn", "Fe", "Co", "Ni", "Cu", "Zn",
          "Y", "Zr", "Nb", "Mo", "Tc", "Ru", "Rh", "Pd", "Ag", "Cd",
          "La", "Hf", "Ta", "W", "Re", "Os", "Ir", "Pt", "Au", "Hg"}


# --------------------------------------------------------------------------
# small vector helpers (no numpy)
# --------------------------------------------------------------------------
def _sub(a, b):
    return (a[0] - b[0], a[1] - b[1], a[2] - b[2])


def _norm(a):
    return math.sqrt(a[0] * a[0] + a[1] * a[1] + a[2] * a[2])


def _unit(a):
    n = _norm(a)
    return (a[0] / n, a[1] / n, a[2] / n)


def _angle(u, v):
    c = u[0] * v[0] + u[1] * v[1] + u[2] * v[2]
    return math.degrees(math.acos(max(-1.0, min(1.0, c))))


def sorted_angles(vectors):
    """All pairwise angles (degrees) between unit vectors, sorted ascending."""
    out = []
    for i in range(len(vectors)):
        for j in range(i + 1, len(vectors)):
            out.append(_angle(vectors[i], vectors[j]))
    out.sort()
    return out


def _polar(theta_deg, phi_deg):
    t, p = math.radians(theta_deg), math.radians(phi_deg)
    return (math.sin(t) * math.cos(p), math.sin(t) * math.sin(p), math.cos(t))


# --------------------------------------------------------------------------
# ideal polyhedra, keyed by coordination number
# --------------------------------------------------------------------------
def _ring(theta, n, offset=0.0):
    return [_polar(theta, offset + k * 360.0 / n) for k in range(n)]


def _build_references():
    z = (0.0, 0.0, 1.0)
    mz = (0.0, 0.0, -1.0)
    a = 1 / math.sqrt(3)
    ref = {
        3: {
            "trigonal planar": _ring(90, 3),
            "trigonal pyramidal": _ring(68, 3),         # ~100 deg N-M-N
            "T-shaped": [(1, 0, 0), (-1, 0, 0), (0, 1, 0)],
        },
        4: {
            "tetrahedral": [(a, a, a), (a, -a, -a), (-a, a, -a), (-a, -a, a)],
            "square planar": _ring(90, 4),
            "seesaw": [z, mz, _polar(90, 60), _polar(90, -60)],
            # eta-ring on top, three legs below (CpM(CO)3, (arene)M(CO)3 ...): L-M-L ~83, ring-M-L ~130
            "piano stool": [z] + _ring(130, 3),
        },
        5: {
            "trigonal bipyramidal": [z, mz] + _ring(90, 3),
            "square pyramidal": [z] + _ring(100, 4),    # apical-basal 100 deg, as in real SP complexes
        },
        6: {
            "octahedral": [z, mz] + _ring(90, 4),
            "trigonal prismatic": _ring(49.1, 3) + _ring(130.9, 3),
        },
        7: {
            "pentagonal bipyramidal": [z, mz] + _ring(90, 5),
            "capped octahedral": [z, mz] + _ring(90, 4) + [_unit((1, 1, 1))],
        },
        8: {
            "square antiprismatic": _ring(57.0, 4) + _ring(123.0, 4, 45.0),
            "cubic": _ring(54.74, 4) + _ring(125.26, 4),
        },
    }
    return {cn: {name: sorted_angles(v) for name, v in shapes.items()} for cn, shapes in ref.items()}


REFERENCES = _build_references()


def _rmsd(a, b):
    return math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b)) / len(a))


def classify_vectors(vectors, has_haptic=False):
    """vectors: list of unit vectors metal -> (pseudo-)donor.
    has_haptic: at least one vertex is a ring/pi-ligand centroid.
    Returns (geometry, rmsd, max_angle)."""
    cn = len(vectors)
    if cn == 0:
        return "no donors", 0.0, 0.0
    if cn == 1:
        return "monocoordinate", 0.0, 0.0
    angles = sorted_angles(vectors)
    max_angle = angles[-1]
    if cn == 2:
        return ("linear" if max_angle >= BENT_ANGLE else "bent"), 180.0 - max_angle, max_angle
    if cn >= 9:
        return "high coordination (CN>=9)", 0.0, max_angle
    best, best_r = None, 1e9
    for name, ref in REFERENCES[cn].items():
        if name == "piano stool" and not has_haptic:
            continue
        if name == "trigonal prismatic" and max_angle >= PRISM_MAX_ANGLE:
            continue
        r = _rmsd(angles, ref)
        if r < best_r:
            best, best_r = name, r
    if best_r > IRREGULAR_RMSD:
        return f"irregular (CN={cn})", best_r, max_angle
    return best, best_r, max_angle


# --------------------------------------------------------------------------
# one complex: atoms -> donor vectors -> label
# --------------------------------------------------------------------------
def find_metal(symbols):
    """Index of the transition-metal atom, or None if there is not exactly one."""
    idx = [i for i, s in enumerate(symbols) if s in METALS]
    return idx[0] if len(idx) == 1 else None


def find_donors(symbols, coords, m):
    """Indices of atoms bonded to the metal (atom m), by covalent-radius cutoff."""
    metal = coords[m]
    r_m = RADII.get(symbols[m], DEFAULT_RADIUS)
    donors = []
    for i in range(len(symbols)):
        if i == m:
            continue
        sym = symbols[i]
        d = _norm(_sub(coords[i], metal))
        if d > r_m + RADII.get(sym, DEFAULT_RADIUS) + BOND_TOLERANCE:
            continue
        if sym == "H":
            # hydride only if the metal is the closest atom to this H
            ok = True
            for j in range(len(symbols)):
                if j != i and _norm(_sub(coords[j], coords[i])) < d:
                    ok = False
                    break
            if not ok:
                continue
        donors.append(i)
    return donors


def group_haptic(donors, symbols, coords):
    """Union donor atoms that are bonded to each other; each group = one ligand vertex."""
    parent = {i: i for i in donors}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for a in range(len(donors)):
        for b in range(a + 1, len(donors)):
            i, j = donors[a], donors[b]
            lim = RADII.get(symbols[i], DEFAULT_RADIUS) + RADII.get(symbols[j], DEFAULT_RADIUS) + HAPTIC_TOLERANCE
            if _norm(_sub(coords[i], coords[j])) < lim:
                parent[find(i)] = find(j)
    groups = {}
    for i in donors:
        groups.setdefault(find(i), []).append(i)
    return list(groups.values())


def label_complex(symbols, coords):
    """Returns (metal, n_donor_atoms, eff_cn, geometry, rmsd, max_angle)."""
    m = find_metal(symbols)
    if m is None:
        return None, 0, 0, "no single metal", 0.0, 0.0
    donors = find_donors(symbols, coords, m)
    groups = group_haptic(donors, symbols, coords)
    metal = coords[m]
    vecs = []
    for g in groups:
        cx = sum(coords[i][0] for i in g) / len(g)
        cy = sum(coords[i][1] for i in g) / len(g)
        cz = sum(coords[i][2] for i in g) / len(g)
        vecs.append(_unit(_sub((cx, cy, cz), metal)))
    geom, rmsd, max_angle = classify_vectors(vecs, has_haptic=any(len(g) > 1 for g in groups))
    return symbols[m], len(donors), len(groups), geom, rmsd, max_angle


# --------------------------------------------------------------------------
# file reading
# --------------------------------------------------------------------------
def _open(path):
    return gzip.open(path, "rt") if str(path).endswith(".gz") else open(path, "rt")


def iter_complexes(path):
    """Yield (csd_code, symbols, coords) for each complex in a tmQM xyz file."""
    with _open(path) as fh:
        while True:
            line = fh.readline()
            if not line:
                return
            line = line.strip()
            if not line:
                continue
            n = int(line)
            header = fh.readline()
            code = header.split("|")[0].split("=")[1].strip()
            symbols, coords = [], []
            for _ in range(n):
                p = fh.readline().split()
                symbols.append(p[0])
                coords.append((float(p[1]), float(p[2]), float(p[3])))
            yield code, symbols, coords


def process_file(path):
    rows = []
    for code, symbols, coords in iter_complexes(path):
        metal, n_atoms, cn, geom, rmsd, mx = label_complex(symbols, coords)
        rows.append((code, metal, n_atoms, cn, geom, round(rmsd, 2), round(mx, 1)))
    print(f"  {Path(path).name}: {len(rows):,} complexes", flush=True)
    return rows


def find_inputs(args):
    if args:
        return [Path(a) for a in args]
    found = []
    for base in (DATA, ROOT):
        found += [Path(p) for p in sorted(glob.glob(str(base / "tmQM_X*_xyz*")))]
    if not found:
        raise FileNotFoundError("No tmQM_X*_xyz(.gz) files found in data/. Pass the paths as arguments.")
    return found


def main(argv):
    jobs = 3
    args = []
    it = iter(argv)
    for a in it:
        if a == "-j":
            jobs = int(next(it))
        else:
            args.append(a)
    files = find_inputs(args)
    print(f"Labelling geometries from {len(files)} file(s) ...")
    if jobs > 1 and len(files) > 1:
        with Pool(min(jobs, len(files))) as pool:
            parts = pool.map(process_file, files)
    else:
        parts = [process_file(f) for f in files]
    rows = [r for part in parts for r in part]
    DATA.mkdir(exist_ok=True)
    with gzip.open(OUT, "wt", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["csd_code", "metal", "n_donor_atoms", "eff_cn", "geometry", "rmsd_deg", "max_angle_deg"])
        w.writerows(rows)
    from collections import Counter
    c = Counter(r[4] for r in rows)
    print(f"\nWrote {len(rows):,} labels to {OUT}")
    for k, v in c.most_common():
        print(f"  {k:32s} {v:7,d}  {v / len(rows):6.1%}")


if __name__ == "__main__":
    main(sys.argv[1:])
