"""Build the processed training table from the ORIGINAL tmQM_y dataset.

tmQM_y is a ';'-separated file with one row per complex:
  CSD_code; Electronic_E; Dispersion_E; Dipole_M; Metal_q; HL_Gap;
  HOMO_Energy; LUMO_Energy; Polarizability; CSD_years; SMILES
(energies in Hartree, SMILES with dative metal-ligand bonds).

This script
  1. reads it (default path: data/tmQM_y.csv, or pass a path),
  2. parses every SMILES (no RDKit needed) -> metal, formal metal charge
     (used as oxidation state) and the ligands bound to the metal,
  3. classifies ligands into named classes (Chloride, Carbonyl, bidentate
     N,N-donor, Cp-type ...), keeping classes with >= MIN_CLASS_COUNT uses,
  4. writes
       data/tmqm_processed.csv.gz   (one row per usable complex)
       data/dataset_meta.json       (ligand classes, valid oxidation states
                                     per metal, stability thresholds)

Usage:  python prepare_dataset.py [path/to/tmQM_y.csv]
"""
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from statistics import median

import pandas as pd

from ligandfieldmodel import METAL_GROUP, class_strength, class_type, max_oxidation
from tmqm_features import extract

ROOT = Path(__file__).parent
DATA = ROOT / "data"
HARTREE_TO_EV = 27.211386
MIN_CLASS_COUNT = 300     # ligand classes rarer than this become "Other ligand"
MIN_OX_SUPPORT = 25       # oxidation states offered in the app need this many complexes
OTHER = "Other ligand"


def find_input():
    if len(sys.argv) > 1:
        return Path(sys.argv[1])
    for name in ("tmQM_y.csv", "tmQM_y(1).csv", "tmQM_y__1_.csv"):
        for base in (DATA, ROOT):
            if (base / name).exists():
                return base / name
    raise FileNotFoundError("tmQM_y.csv not found. Put it in data/ or pass its path as an argument.")


def main():
    src = find_input()
    print(f"Reading {src} ...")
    raw = pd.read_csv(src, sep=";")
    raw.columns = [c.strip() for c in raw.columns]
    n_raw = len(raw)
    raw = raw.dropna(subset=["SMILES"])
    print(f"  {n_raw} complexes, {len(raw)} have a SMILES string")

    parsed = []
    for rec in raw.itertuples(index=False):
        try:
            r = extract(rec.SMILES)
        except Exception:
            continue
        if r is None or not r["ligands"] or r["metal"] not in METAL_GROUP:
            continue                      # multi-metal / unparsable / unknown metal
        if not (0 <= r["charge"] <= max_oxidation(r["metal"])):
            continue                      # keep chemically sensible oxidation states
        parsed.append((rec, r))
    print(f"  {len(parsed)} single-metal complexes with chemically sensible oxidation state kept")

    # ---- ligand class vocabulary ----------------------------------------
    cls_count = Counter(l[0] for _, r in parsed for l in r["ligands"])
    vocab = [c for c, n in cls_count.most_common() if n >= MIN_CLASS_COUNT]
    obs = defaultdict(lambda: {"sites": [], "donors": Counter()})
    for _, r in parsed:
        for name, dent, sites, donors in r["ligands"]:
            key = name if name in vocab else OTHER
            obs[key]["sites"].append(sites)
            obs[key]["donors"][tuple(sorted(donors))] += 1
    classes = {}
    for name in vocab + [OTHER]:
        o = obs[name]
        common_donors = Counter(o["donors"].most_common(1)[0][0])
        classes[name] = {
            "count": len(o["sites"]), "sites": max(1, round(median(o["sites"]))),
            "donors": dict(common_donors), "strength": class_strength(name, tuple(common_donors.elements())),
            "type": class_type(name),
        }

    # ---- processed rows -----------------------------------------------------
    rows = []
    for rec, r in parsed:
        c = Counter(l[0] if l[0] in vocab else OTHER for l in r["ligands"])
        rows.append({
            "csd_code": rec.CSD_code, "metal": r["metal"], "oxidation_state": r["charge"],
            "cn_sites": sum(l[2] for l in r["ligands"]), "n_ligands": len(r["ligands"]),
            "signature": "|".join(f"{k}::{v}" for k, v in sorted(c.items())),
            "hl_gap_eV": rec.HL_Gap * HARTREE_TO_EV, "metal_q": rec.Metal_q,
            "homo_eV": rec.HOMO_Energy * HARTREE_TO_EV, "lumo_eV": rec.LUMO_Energy * HARTREE_TO_EV,
        })
    df = pd.DataFrame(rows)
    DATA.mkdir(exist_ok=True)
    df.to_csv(DATA / "tmqm_processed.csv.gz", index=False, float_format="%.5f")

    ox_support = df.groupby(["metal", "oxidation_state"]).size()
    metal_ox = {m: sorted(int(o) for (mm, o), n in ox_support.items() if mm == m and n >= MIN_OX_SUPPORT)
                for m in sorted(df.metal.unique())}
    q1, q2 = df.hl_gap_eV.quantile([1 / 3, 2 / 3])
    meta = {
        "source_file": src.name, "n_raw": int(n_raw), "n_used": int(len(df)),
        "classes": classes, "vocab": vocab, "metal_oxidation_states": metal_ox,
        "gap_thresholds_eV": [round(float(q1), 3), round(float(q2), 3)],
    }
    (DATA / "dataset_meta.json").write_text(json.dumps(meta, indent=1, ensure_ascii=False), encoding="utf-8")
    other_share = (df.signature.str.contains(OTHER)).mean()
    print(f"  {len(vocab)} named ligand classes (+ '{OTHER}', present in {other_share:.0%} of complexes)")
    print(f"  stability thresholds (HOMO-LUMO gap, eV): Low < {q1:.2f} <= Moderate < {q2:.2f} <= High")
    print(f"Wrote {DATA / 'tmqm_processed.csv.gz'} and {DATA / 'dataset_meta.json'}")


if __name__ == "__main__":
    main()
