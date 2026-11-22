"""Tests for the crystal-field physics in ligandfieldmodel.py.

Run with:  python test_ligandfield.py      (prints "all good" when everything passes)

What is checked
  * every orbital-energy pattern sums to zero (barycentre rule)
  * high-spin electron counts in t2g/eg and e/t2 for d1-d10
  * CFSE is exactly 0 for d0, high-spin d5 and d10 (all geometries)
  * CFAE reproduces the classic Basolo-Pearson values (in units of Dq = Delta_oct / 10)
  * pairing energy is a cost (net stabilisation = -(CFSE + P * extra pairs))
  * inert / labile labels for well-known complexes
"""
import json
from pathlib import Path

import ligandfieldmodel as L

ROOT = Path(__file__).parent
CLASSES = json.load(open(ROOT / "data" / "dataset_meta.json", encoding="utf-8"))["classes"]


def lig(name, n):
    return L.mixed_ligand({name: n}, CLASSES)


CN, H2O, NH3 = lig("Cyanide (CN-)", 6), lig("Water (H2O)", 6), lig("Ammonia (NH3)", 6)
close = lambda a, b, tol=0.02: abs(a - b) <= tol
ok = lambda msg: print("ok ", msg)

# 1) barycentre rule -------------------------------------------------------------------------
for name, e in list(L.GEOMETRY_ORBITALS.items()) + [("pentagonal_bipyramid", L.PENTAGONAL_BIPYRAMID)]:
    assert abs(sum(e)) < 2e-3, (name, sum(e))
ok("all orbital patterns sum to zero")

# 2) high-spin occupancies ---------------------------------------------------------------------
OCT_HS = {1: (1, 0), 2: (2, 0), 3: (3, 0), 4: (3, 1), 5: (3, 2), 6: (4, 2), 7: (5, 2), 8: (6, 2), 9: (6, 3), 10: (6, 4)}
TET_HS = {1: (1, 0), 2: (2, 0), 3: (2, 1), 4: (2, 2), 5: (2, 3), 6: (3, 3), 7: (4, 3), 8: (4, 4), 9: (4, 5), 10: (4, 6)}
for d in range(1, 11):
    o = L.fill_high_spin(L.GEOMETRY_ORBITALS["octahedral"], d)
    assert (sum(o[:3]), sum(o[3:])) == OCT_HS[d], ("oct", d, o)
    t = L.fill_high_spin(L.GEOMETRY_ORBITALS["tetrahedral"], d)
    assert (sum(t[:2]), sum(t[2:])) == TET_HS[d], ("tet", d, t)
ok("high-spin t2g/eg and e/t2 occupancies for d1-d10")

# 3) CFSE = 0 for d0, high-spin d5, d10 ----------------------------------------------------------
assert abs(L.calculate_complex("Sc", 3, "octahedral", H2O)["cfse"]) < 1e-9
assert abs(L.calculate_complex("Mn", 2, "octahedral", H2O)["cfse"]) < 1e-9          # d5 high spin
for geom in L.GEOMETRIES:
    assert abs(L.calculate_complex("Zn", 2, geom, H2O)["cfse"]) < 1e-9, geom           # d10
assert abs(L.calculate_complex("Mn", 2, "tetrahedral", H2O)["cfse"]) < 1e-9          # d5 tetrahedral
ok("CFSE is 0 for d0, high-spin d5 and d10 in every geometry")

# 4) CFAE vs textbook (Dq units) ------------------------------------------------------------------
def cfae_dq(metal, ox, ligand):
    r = L.calculate_complex(metal, ox, "octahedral", ligand)
    return r["cfae_sn1"] / r["delta_oct"] * 10, r["cfae_sn2"] / r["delta_oct"] * 10, r
sn1, sn2, r = cfae_dq("Cr", 3, NH3)
assert close(sn1, 2.00) and close(sn2, 4.26), (sn1, sn2)                                # d3
sn1, sn2, r = cfae_dq("Fe", 2, CN)
assert r["spin"] == "low" and close(sn1, 4.00) and close(sn2, 8.52), (sn1, sn2)         # low-spin d6
sn1, sn2, r = cfae_dq("Mn", 2, H2O)
assert r["spin"] == "high" and close(sn1, 0.0) and close(sn2, 0.0)                      # high-spin d5
sn1, sn2, r = cfae_dq("Cr", 2, H2O)
assert close(sn1, -3.14), sn1                                                              # high-spin d4 (Jahn-Teller)
ok("CFAE: d3 2.00/4.26, low-spin d6 4.00/8.52, high-spin d5 0, high-spin d4 -3.14 (Dq)")
assert L.calculate_complex("Fe", 2, "tetrahedral", CN)["cfae"] is None
ok("CFAE is only defined for octahedral complexes")

# 5) pairing is a cost ---------------------------------------------------------------------------
r = L.calculate_complex("Fe", 2, "octahedral", CN)                                         # low-spin d6: 2 extra pairs
assert r["extra_pairs"] == 2
assert close(r["net_stabilization"], -r["cfse"] - 2 * r["pairing_energy"], 1e-9)
assert r["net_stabilization"] < -r["cfse"]
ok("net stabilisation = -(CFSE + P * extra pairs); pairing lowers the net stabilisation")

# 6) kinetic labels for well-known complexes ----------------------------------------------------
def kin(metal, ox, geom, ligand):
    return L.calculate_complex(metal, ox, geom, ligand)["kinetic"]
inert = [("Cr", 3, "octahedral", NH3), ("Cr", 3, "octahedral", H2O), ("Co", 3, "octahedral", NH3),
         ("Fe", 2, "octahedral", CN), ("Rh", 3, "octahedral", NH3), ("Ir", 3, "octahedral", NH3),
         ("Ru", 2, "octahedral", NH3), ("Pt", 2, "square_planar", NH3)]
labile = [("Fe", 2, "octahedral", H2O), ("Fe", 3, "octahedral", H2O), ("Mn", 2, "octahedral", H2O),
          ("Co", 2, "octahedral", H2O), ("Cu", 2, "octahedral", H2O), ("Zn", 2, "octahedral", H2O),
          ("Ti", 3, "octahedral", H2O), ("Ni", 2, "square_planar", CN), ("Zn", 2, "tetrahedral", NH3)]
for m, ox, g, lg in inert:
    assert kin(m, ox, g, lg) == "Inert", (m, ox, g, kin(m, ox, g, lg))
for m, ox, g, lg in labile:
    assert kin(m, ox, g, lg) == "Labile", (m, ox, g, kin(m, ox, g, lg))
assert kin("Ni", 2, "octahedral", H2O) == "Moderate"            # d8 cap (documented limitation)
ok(f"kinetic labels: {len(inert)} inert, {len(labile)} labile, Ni(II) aqua moderate")

# 7) geometry Delta values ----------------------------------------------------------------------
f = L.GEOMETRY_DELTA_FACTOR
assert close(f["octahedral"], 1.0, 1e-9) and close(f["tetrahedral"], 4 / 9, 1e-3)
assert close(f["square_planar"], 1.742, 2e-3) and close(f["square_pyramidal"], 1.371, 2e-3)
assert L.GEOMETRIES == ["octahedral", "square_pyramidal", "square_planar", "tetrahedral"]
ok("geometry Delta factors derived from the orbital patterns")
# 8) the refit against experimental Delta values (fit/reference_delta.csv) ---------------------------------
import csv, statistics
rows = list(csv.DictReader(open(ROOT / "fit" / "reference_delta.csv", encoding="utf-8")))
SMAP = {"H2O": "Water (H2O)", "NH3": "Ammonia (NH3)", "CN": "Cyanide (CN-)", "Cl": "Chloride (Cl-)", "Br": "Bromide (Br-)", "F": "Fluoride (F-)"}
def lg(code): return {"strength": L.NAMED_STRENGTH[SMAP[code]], "formula": code, "sites": 1}
fit_rows = [r for r in rows if r["geometry"] == "octahedral" and r["flag"] == "ok" and not r["note"].startswith("in-plane")
            and r["ligand"] in SMAP and int(r["oxidation"]) in (2, 3, 4)]
errs = [abs(L.to_cm1(L.base_values(r["metal"], int(r["oxidation"]), lg(r["ligand"]))[1]) - float(r["delta_cm1"])) / float(r["delta_cm1"]) for r in fit_rows]
assert len(fit_rows) >= 45 and statistics.median(errs) < 0.10, (len(fit_rows), statistics.median(errs))
ok(f"Delta in cm-1 reproduces {len(fit_rows)} experimental octahedral complexes: median error {100 * statistics.median(errs):.1f}% (< 10%)")
spin = {("Co", 3, "H2O"): "low", ("Co", 3, "NH3"): "low", ("Co", 3, "F"): "high", ("Fe", 2, "H2O"): "high", ("Fe", 2, "CN"): "low",
        ("Fe", 3, "F"): "high", ("Fe", 3, "H2O"): "high", ("Fe", 3, "CN"): "low", ("Mn", 2, "H2O"): "high", ("Mn", 3, "CN"): "low",
        ("Mn", 3, "H2O"): "high", ("Co", 2, "H2O"): "high", ("Co", 2, "NH3"): "high"}
for (m, ox, code), want in spin.items():
    got = L.calculate_complex(m, ox, "octahedral", lg(code))["spin"]
    assert got == want, (m, ox, code, got, want)
ok("spin states of 13 classic octahedral complexes (incl. low-spin [Co(H2O)6]3+, high-spin [CoF6]3-)")
assert L.oxidation_factor(0) == L.oxidation_factor(1) == L.oxidation_factor(2)          # flat below +2 (documented assumption)
assert L.oxidation_factor(3) > L.oxidation_factor(2) and L.oxidation_factor(4) > L.oxidation_factor(3) and L.oxidation_factor(6) > L.oxidation_factor(4)
assert abs(L.oxidation_factor(3) / L.oxidation_factor(2) - L.OX_G[3]) < 1e-9
ok("oxidation factor: flat below +2, increasing above, M(III)/M(II) = fitted value")
assert "inside" in L.delta_confidence(["Water (H2O)"], 3) and "OUTSIDE" in L.delta_confidence(["Carbonyl (CO)"], 0)
ok("coverage statement distinguishes fitted and extrapolated selections")
print("\nall good")
