"""Refit the ligand-field model's constants to the experimental Delta table.

Model (octahedral):   Delta_cm1 = K * s_L * R_row * 1.7 * g_ox          (1.7 = old factor at ox 2, so units stay comparable)
Fitted:  K (cm-1 per model unit), strengths s_L of NH3, CN, F, Cl, Br (water fixed at 1.05),
         row factors R_4d, R_5d (R_3d = 1), oxidation factors g(+3), g(+4) relative to +2.
Not constrained by data (kept at their old values and flagged): CO, I, hydride/phosphine/... strengths, oxidation states 0 and 1.
Then, from the spin states reported with the data, a pairing energy P(d) for d4-d7.

Reads  reference_delta.csv  (built by build_table.py);   writes  fitted_constants.json
"""
import csv, json, sys, statistics as st
import numpy as np
from scipy.optimize import least_squares
sys.path.insert(0, ".")
sys.path.insert(0, "..")                 # project root (for ligandfieldmodel.METAL_ROW)
import legacy_constants as OLD       # the constants BEFORE the refit
from ligandfieldmodel import METAL_ROW
import figgis_tables as FG

rows_all = list(csv.DictReader(open("reference_delta.csv", encoding="utf-8")))
FIT_LIG = ["H2O", "NH3", "CN", "F", "Cl", "Br"]
NAMES = ["K", "R2", "R3", "g3", "g4", "NH3", "CN", "F", "Cl", "Br"]
OLD_STRENGTH = OLD.STRENGTH
ROW = METAL_ROW

def usable(r):
    return (r["geometry"] == "octahedral" and r["flag"] == "ok" and not r["note"].startswith("in-plane")
            and r["ligand"] in FIT_LIG and int(r["oxidation"]) in (2, 3, 4))
fit_rows = [r for r in rows_all if usable(r)]
print(f"fitting {len(fit_rows)} octahedral complexes (clean, not Jahn-Teller in-plane values, ligands {FIT_LIG}, oxidation +2..+4)")
obs = lambda rs: np.array([float(r["delta_cm1"]) for r in rs])

def predict(theta, r):
    K, R2, R3, g3, g4, sN, sC, sF, sCl, sBr = np.exp(theta)
    s = {"H2O": 1.05, "NH3": sN, "CN": sC, "F": sF, "Cl": sCl, "Br": sBr}[r["ligand"]]
    R = {1: 1.0, 2: R2, 3: R3}[ROW[r["metal"]]]
    g = {2: 1.0, 3: g3, 4: g4}[int(r["oxidation"])]
    return K * s * R * 1.7 * g

THETA0 = np.log([5300, 1.45, 1.75, 2.05 / 1.7, 2.4 / 1.7, 1.35, 2.45, 0.85, 0.78, 0.72])
def fit(rs):
    y = np.log(obs(rs))
    res = least_squares(lambda th: np.array([np.log(predict(th, r)) for r in rs]) - y, THETA0, loss="soft_l1", f_scale=0.15)
    return res.x

def old_units(r):
    s = OLD_STRENGTH[r["ligand"]]; return s * OLD.ROW_FACTOR[ROW[r["metal"]]] * (1 + OLD.OX_SLOPE * int(r["oxidation"]))

def stats(err, label):
    e = np.abs(err); print(f"  {label:44s} n={len(e):2d}  median {100*np.median(e):4.1f}%  mean {100*e.mean():4.1f}%  90th pct {100*np.percentile(e,90):4.0f}%  max {100*e.max():4.0f}%"); return float(np.median(e))

theta = fit(fit_rows)
print("\n=== 1) fitted constants (all data)")
vals = dict(zip(NAMES, np.exp(theta)))
for n in NAMES: print(f"  {n:4s} = {vals[n]:8.3f}")
rel_new = (np.array([predict(theta, r) for r in fit_rows]) - obs(fit_rows)) / obs(fit_rows)
K_old = st.median(float(r["delta_cm1"]) / old_units(r) for r in fit_rows)
rel_old = (np.array([K_old * old_units(r) for r in fit_rows]) - obs(fit_rows)) / obs(fit_rows)
print("\n=== 2) fit quality on the data it was fitted to")
stats(rel_old, "OLD constants (one scale factor)"); stats(rel_new, "REFITTED constants")

print("\n=== 3) honest check: cross-validation (the fit never sees the complex it predicts)")
loo_new, loo_old = [], []
for i, r in enumerate(fit_rows):
    tr = fit_rows[:i] + fit_rows[i + 1:]
    th = fit(tr); loo_new.append((predict(th, r) - float(r["delta_cm1"])) / float(r["delta_cm1"]))
    k = st.median(float(x["delta_cm1"]) / old_units(x) for x in tr); loo_old.append((k * old_units(r) - float(r["delta_cm1"])) / float(r["delta_cm1"]))
m_old = stats(np.array(loo_old), "leave-one-complex-out, OLD constants"); m_new = stats(np.array(loo_new), "leave-one-complex-out, REFITTED")
metals = sorted({r["metal"] for r in fit_rows}); lomo_new, lomo_old, lomo_lab = [], [], []
for m in metals:
    tr = [r for r in fit_rows if r["metal"] != m]; te = [r for r in fit_rows if r["metal"] == m]
    try: th = fit(tr)
    except Exception: continue
    k = st.median(float(x["delta_cm1"]) / old_units(x) for x in tr)
    for r in te:
        lomo_new.append((predict(th, r) - float(r["delta_cm1"])) / float(r["delta_cm1"])); lomo_old.append((k * old_units(r) - float(r["delta_cm1"])) / float(r["delta_cm1"]))
stats(np.array(lomo_old), "leave-one-METAL-out, OLD constants"); stats(np.array(lomo_new), "leave-one-METAL-out, REFITTED")
d3 = [i for i, r in enumerate(fit_rows) if ROW[r["metal"]] == 1]; d45 = [i for i, r in enumerate(fit_rows) if ROW[r["metal"]] > 1]
for lab, idx in (("3d", d3), ("4d/5d", d45)):
    stats(np.array(loo_old)[idx], f"  LOO OLD, {lab}"); stats(np.array(loo_new)[idx], f"  LOO REFITTED, {lab}")

print("\n=== 4) how well is each constant determined? (300 bootstrap refits, 5th-95th percentile)")
rng = np.random.RandomState(0); boots = []
for _ in range(300):
    samp = [fit_rows[j] for j in rng.randint(0, len(fit_rows), len(fit_rows))]
    try: boots.append(np.exp(fit(samp)))
    except Exception: pass
boots = np.array(boots); ci = {}
for j, n in enumerate(NAMES):
    lo, hi = np.percentile(boots[:, j], [5, 95]); ci[n] = (float(lo), float(hi))
    cnt = sum(1 for r in fit_rows if (n in ("NH3","CN","F","Cl","Br") and r["ligand"] == n) or (n == "R2" and ROW[r["metal"]] == 2) or (n == "R3" and ROW[r["metal"]] == 3) or (n == "g3" and r["oxidation"] == "3") or (n == "g4" and r["oxidation"] == "4"))
    print(f"  {n:4s} {vals[n]:7.3f}   [{lo:7.3f}, {hi:7.3f}]   (data points behind it: {cnt if n != 'K' else len(fit_rows)})")

print("\n=== 5) data NOT used in the fit")
def report(rs, label, pred):
    e = np.array([(pred(r) - float(r["delta_cm1"])) / float(r["delta_cm1"]) for r in rs])
    if len(e): stats(e, label)
K = vals["K"]
def unit_new(r, ox_override=None):
    return predict(theta, r) / K
inpl = [r for r in rows_all if r["note"].startswith("in-plane") and r["geometry"] == "octahedral" and r["ligand"] in FIT_LIG and int(r["oxidation"]) in (2, 3, 4)]
report(inpl, "Jahn-Teller in-plane values (Cr2+, Mn3+, Cu2+)", lambda r: predict(theta, r) if int(r["oxidation"]) in (2, 3, 4) else np.nan)
tet = [r for r in rows_all if r["geometry"] == "tetrahedral" and r["ligand"] in ("Cl", "Br", "NH3") and r["flag"] == "ok" and int(r["oxidation"]) in (2, 3, 4)]
report(tet, "tetrahedral, 4/9 x octahedral (new)", lambda r: predict(theta, r) * 4 / 9)
report(tet, "tetrahedral, 4/9 x octahedral (OLD, own scale)", lambda r: K_old * old_units(r) * 4 / 9)
print("  tetrahedral detail (observed vs 4/9 rule, new constants):", [(r["complex"], r["delta_cm1"], round(predict(theta, r) * 4 / 9)) for r in tet])
other = [r for r in rows_all if r["geometry"] == "octahedral" and r["flag"] == "ok" and r["ligand"] == "CO"]
print("  Cr(CO)6 (CO strength NOT fitted: one data point):", [(r["complex"], r["delta_cm1"], round(K_old * OLD_STRENGTH.get(r["ligand"], 2.55) * 1.0 * 1 * 1) ) for r in other], "-> old model gives about", round(K_old * 2.55), "cm-1")

print("\n=== 6) spin states from the data  ->  pairing energy P(d) (cm-1)")
SP = {("Mn", 3): "d4", ("Mn", 2): "d5", ("Fe", 3): "d5", ("Fe", 2): "d6", ("Co", 3): "d6", ("Co", 2): "d7", ("Cr", 2): "d4"}
def book_spin(r):
    key = (r["metal"], int(r["oxidation"]))
    if key not in SP or r["geometry"] != "octahedral": return None
    if key == ("Co", 3): return "high" if "high-spin" in r["note"] else "low"
    return "low" if "low-spin" in r["note"] else "high"
bounds = {}
for r in rows_all:
    bs = book_spin(r)
    if bs is None or r["ligand"] not in FIT_LIG or r["flag"].startswith("UNCERTAIN"): continue
    if int(r["oxidation"]) not in (2, 3, 4): continue
    d = int(SP[(r["metal"], int(r["oxidation"]))][1]); val = predict(theta, r)
    b = bounds.setdefault(d, {"high": [], "low": []}); b[bs].append((val, r["complex"]))
bcrit = {}
for name, (cfg, B) in {r[0]: (r[1], r[3]) for r in FG.T85}.items():
    d = int(cfg[1]) if cfg[1].isdigit() else None
    if d in FG.CROSSOVER and name.startswith("[") and "printed" not in cfg and name in {r["complex"] for r in rows_all if r["geometry"] == "octahedral"}:
        bcrit.setdefault(d, []).append(FG.CROSSOVER[d] * B)
P = {}
for d in (4, 5, 6, 7):
    b = bounds.get(d, {"high": [], "low": []}); lo = max((v for v, _ in b["high"]), default=None); up = min((v for v, _ in b["low"]), default=None)
    bc = st.median(bcrit[d]) if d in bcrit else None
    if lo and up and lo < up: p, how = (lo * up) ** 0.5, "geometric mean of the interval allowed by the spin data"
    elif lo and up: p, how = None, f"CONFLICT: highest high-spin {lo:.0f} >= lowest low-spin {up:.0f}"
    elif lo: p, how = max(bc or 0, lo * 1.15), "only a lower bound from spin data; book crossover (Delta/B x B) used"
    else: p, how = None, "no data"
    P[d] = p
    print(f"  d{d}: highest high-spin Delta {lo and round(lo)}  lowest low-spin Delta {up and round(up)}  book crossover Delta/B*B {bc and round(bc)}  -> P = {p and round(p)}  ({how})")
P_mean = float(np.mean([v for v in P.values() if v]))
print(f"  P for other d-counts (only used for energy bookkeeping): mean = {P_mean:.0f} cm-1;  old model: {2.45*K_old:.0f} cm-1")
for d, b in bounds.items():
    print(f"    d{d} high-spin: {[(n, round(v)) for v, n in b['high']][:6]} | low-spin: {[(n, round(v)) for v, n in b['low']][:6]}")

out = {"K_cm1_per_unit": float(K), "row_factor": {"1": 1.0, "2": float(vals["R2"]), "3": float(vals["R3"])},
       "ox_relative_to_2": {"2": 1.0, "3": float(vals["g3"]), "4": float(vals["g4"])},
       "strength": {k: float(vals[k]) for k in ("NH3", "CN", "F", "Cl", "Br")}, "water": 1.05,
       "pairing_cm1": {str(d): (float(v) if v else None) for d, v in P.items()}, "pairing_other_cm1": P_mean,
       "bootstrap_5_95": ci, "n_fit": len(fit_rows),
       "cv": {"loo_median_old": m_old, "loo_median_new": m_new}}
json.dump(out, open("fitted_constants.json", "w"), indent=2)
print("\nwrote fitted_constants.json")
