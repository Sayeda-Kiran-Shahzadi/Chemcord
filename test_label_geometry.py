"""Self-check for label_geometry.py: build ideal polyhedra (metal + donors 2.0 A away),
add noise, and make sure each is labelled correctly.  Run:  python test_label_geometry.py"""
import random

import label_geometry as L

random.seed(0)


def make(vectors, noise_deg=0.0, haptic_ring=False):
    symbols, coords = ["Fe"], [(0.0, 0.0, 0.0)]
    for v in vectors:
        x, y, z = (c + random.gauss(0, noise_deg * 0.0175) for c in v)   # ~noise_deg in direction
        n = (x * x + y * y + z * z) ** 0.5
        symbols.append("N")
        coords.append((2.0 * x / n, 2.0 * y / n, 2.0 * z / n))
    return symbols, coords


bad = 0
import math
ideal = {}
a = 1 / math.sqrt(3)
z, mz = (0.0, 0.0, 1.0), (0.0, 0.0, -1.0)
ideal.update({
    "linear": [z, mz], "bent": [L._polar(55, 0), L._polar(55, 180)],
    "trigonal planar": L._ring(90, 3), "T-shaped": [(1, 0, 0), (-1, 0, 0), (0, 1, 0)],
    "tetrahedral": [(a, a, a), (a, -a, -a), (-a, a, -a), (-a, -a, a)], "square planar": L._ring(90, 4),
    "seesaw": [z, mz, L._polar(90, 60), L._polar(90, -60)],
    "trigonal bipyramidal": [z, mz] + L._ring(90, 3), "square pyramidal": [z] + L._ring(100, 4),
    "octahedral": [z, mz] + L._ring(90, 4), "trigonal prismatic": L._ring(49.1, 3) + L._ring(130.9, 3),
    "pentagonal bipyramidal": [z, mz] + L._ring(90, 5),
    "square antiprismatic": L._ring(57.0, 4) + L._ring(123.0, 4, 45.0),
})
for name, vecs in ideal.items():
    for noise in (0.0, 0.08):          # perfect and ~5 degree scatter
        sym, xyz = make(vecs, noise)
        got = L.label_complex(sym, xyz)[3]
        ok = got == name
        bad += not ok
        print(f"{'ok ' if ok else 'BAD'} {name:24s} noise={noise:<4} -> {got}")
print("\nFAILED" if bad else "\nall good")
raise SystemExit(1 if bad else 0)
