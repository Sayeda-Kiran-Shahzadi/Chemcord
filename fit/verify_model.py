"""Check the ligand-field model (as shipped in ligandfieldmodel.py) against the experimental table."""
import csv, statistics as st, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
import ligandfieldmodel as L

rows = list(csv.DictReader(open(ROOT / "fit" / "reference_delta.csv", encoding="utf-8")))
smap = {"H2O": "Water (H2O)", "NH3": "Ammonia (NH3)", "CN": "Cyanide (CN-)", "Cl": "Chloride (Cl-)", "Br": "Bromide (Br-)",
        "F": "Fluoride (F-)", "CO": "Carbonyl (CO)", "I": "Iodide (I-)"}
def lig(code): return {"strength": L.NAMED_STRENGTH[smap[code]], "formula": code, "sites": 1}
def model_cm1(r, geom=None):
    _, d_oct, _ = L.base_values(r["metal"], int(r["oxidation"]), lig(r["ligand"]))
    return L.to_cm1(d_oct * (L.GEOMETRY_DELTA_FACTOR["tetrahedral"] if r["geometry"] == "tetrahedral" else 1.0))
def err(rs):
    e = [abs(model_cm1(r) - float(r["delta_cm1"])) / float(r["delta_cm1"]) for r in rs]
    return f"n={len(e):2d} median {100*st.median(e):4.1f}%  mean {100*st.mean(e):4.1f}%  max {100*max(e):3.0f}%"
FIT = ("H2O", "NH3", "CN", "F", "Cl", "Br")
ok = lambda r: r["flag"] == "ok" and r["ligand"] in smap
fitset = [r for r in rows if ok(r) and r["geometry"] == "octahedral" and r["ligand"] in FIT and not r["note"].startswith("in-plane") and int(r["oxidation"]) in (2, 3, 4)]
print("octahedral, the 50 complexes used in the fit        :", err(fitset))
print("   3d metals                                       :", err([r for r in fitset if L.METAL_ROW[r["metal"]] == 1]))
print("   4d/5d metals                                    :", err([r for r in fitset if L.METAL_ROW[r["metal"]] > 1]))
print("tetrahedral (4/9 rule, NOT fitted)                  :", err([r for r in rows if ok(r) and r["geometry"] == "tetrahedral" and r["ligand"] in ("Cl", "Br", "NH3") and int(r["oxidation"]) in (2, 3, 4)]))
print("Jahn-Teller in-plane values (NOT fitted)            :", err([r for r in rows if ok(r) and r["note"].startswith("in-plane") and r["ligand"] in FIT and int(r["oxidation"]) in (2, 3, 4) and r["geometry"] == "octahedral"]))
co = [r for r in rows if r["ligand"] == "CO" and r["flag"] == "ok"][0]
print(f"Cr(CO)6 (oxidation 0, ONE data point, CO not fitted): observed {co['delta_cm1']}, model {model_cm1(co):.0f} cm-1")

# spin states reported with the data (Figgis & Hitchman Table 8.3 footnotes), octahedral d4-d7
KEY = {("Mn", 3), ("Mn", 2), ("Fe", 3), ("Fe", 2), ("Co", 3), ("Co", 2), ("Cr", 2)}
def book_spin(r):
    key = (r["metal"], int(r["oxidation"]))
    if key not in KEY or r["geometry"] != "octahedral": return None
    if key == ("Co", 3): return "high" if "high-spin" in r["note"] else "low"
    return "low" if "low-spin" in r["note"] else "high"
bad, n = [], 0
for r in rows:
    bs = book_spin(r)
    if bs is None or r["ligand"] not in FIT or r["flag"].startswith("UNCERTAIN"): continue
    n += 1
    ms = L.calculate_complex(r["metal"], int(r["oxidation"]), "octahedral", lig(r["ligand"]))["spin"]
    if ms != bs: bad.append((r["complex"], f"model {ms}", f"book {bs}"))
print(f"spin state vs the book: {n - len(bad)} of {n} agree", bad or "")
