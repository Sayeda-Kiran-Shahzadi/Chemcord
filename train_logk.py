"""Train the log K1 model used by the app's "Binding strength (log K1)" tool.

INPUT  data/logk_raw/ must contain the six files published with
         Karunaratne, Zahariev, Perez Garcia, J. Chem. Inf. Model. 2025, 65, 11532
       (download them from the authors' repository, see the README):
         train_input.csv  train_features.csv
         test_input.csv   test_features.csv
         validation_input.csv  validation_features.csv   <- the paper's EXTERNAL test set
OUTPUT models/logk_bundle.joblib  (model + metal table + lookup of measured values + error estimates)

How the error estimates shown in the app are obtained
  1. A model is trained on the train split only.
  2. Every row of the paper's random TEST split is put in one of three groups:
       exact_complex  the same metal ion + ligand pair is in the training data (repeat measurement)
       known_ligand   the ligand is in the training data, but not with this metal ion
       new_ligand     the ligand is not in the training data
     and the mean absolute error (MAE, in log units) is measured for each group.
  3. For "new_ligand" the rows of the paper's external set (40 ligands that are absent from
     training by construction) are pooled with the test rows of that group.
  4. The final model is then trained on ALL rows.

Run with:  python train_logk.py        (about 2-3 minutes)
"""
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, r2_score

import logk

ROOT = Path(sys.executable).parent if getattr(sys, "frozen", False) else Path(__file__).parent
RAW = ROOT / "data" / "logk_raw"
OUT = ROOT / "models" / "logk_bundle.joblib"

SMILES_COLS = ["cmplx_smiles_neutral_std", "SMILES"]
Y_COLS = ["Exp LgK1", "Experimental Lg K1"]
METAL_RE = re.compile(r"\[([A-Z][a-z]?)([+-]\d*)\]$")


def _first(df, candidates, what):
    for c in candidates:
        if c in df.columns:
            return c
    raise ValueError(f"Could not find the {what} column; looked for {candidates}. Found: {list(df.columns)[:8]}...")


def load_raw(raw_dir=RAW):
    frames = []
    for split in ("train", "test", "validation"):
        fi, ff = raw_dir / f"{split}_input.csv", raw_dir / f"{split}_features.csv"
        if not fi.exists() or not ff.exists():
            raise FileNotFoundError(f"{fi.name} / {ff.name} not found in {raw_dir}. See the docstring of train_logk.py.")
        inp, feat = pd.read_csv(fi), pd.read_csv(ff)
        if len(inp) != len(feat):
            raise ValueError(f"{fi.name} and {ff.name} have different numbers of rows.")
        smi, y = _first(inp, SMILES_COLS, "SMILES"), _first(inp, Y_COLS, "log K")
        part = pd.concat([inp[[smi, y]].rename(columns={smi: "smiles", y: "logK"}).reset_index(drop=True),
                          feat.reset_index(drop=True)], axis=1)
        part["split"] = split
        frames.append(part)
    df = pd.concat(frames, ignore_index=True)

    def split_smiles(s):
        parts = s.split(".")
        m = METAL_RE.match(parts[-1])
        if not m:
            raise ValueError(f"Cannot read the metal ion at the end of SMILES: {s}")
        sign = 1 if m.group(2)[0] == "+" else -1
        charge = sign * (int(m.group(2)[1:]) if len(m.group(2)) > 1 else 1)
        return ".".join(parts[:-1]), m.group(1), charge

    parsed = df.smiles.map(split_smiles)
    df["ligand_raw"] = [p[0] for p in parsed]
    df["symbol"] = [p[1] for p in parsed]
    df["charge"] = [p[2] for p in parsed]
    df["ion"] = [logk.ion_key(s, c) for s, c in zip(df.symbol, df.charge)]
    return df


def build_matrix(df, metal_table, electrolytes, ligand_vecs):
    metal = np.vstack([metal_table[k] for k in df.ion])
    cond = df[logk.CONDITION_COLUMNS].values.astype(np.float32)          # NaN allowed (the external set has none)
    lig = np.vstack([ligand_vecs[s] for s in df.ligand_raw])
    return np.hstack([metal, cond, lig]).astype(np.float32)


def new_model():
    return HistGradientBoostingRegressor(max_iter=500, learning_rate=0.07, max_leaf_nodes=63,
                                         min_samples_leaf=15, l2_regularization=1.0, random_state=0)


def train(verbose=True):
    log = print if verbose else (lambda *a, **k: None)
    if not logk.HAVE_RDKIT:
        raise RuntimeError("RDKit is required: pip install rdkit")
    df = load_raw()
    log(f"Loaded {len(df):,} measurements ({df.ligand_raw.nunique():,} ligands, {df.ion.nunique()} metal ions/H+).")

    # metal table: identical numbers for every row of an (element, charge) pair (checked below)
    cols = logk.METAL_FEATURES + ["Metal_charge"]
    chk = df.groupby("ion")[cols].nunique(dropna=False)
    if (chk > 1).any().any():
        raise ValueError("Metal descriptors are not a pure function of (element, charge) in these files.")
    first = df.groupby("ion")[cols].first()
    metal_table = {k: first.loc[k].values.astype(np.float32) for k in first.index}
    ions = []
    for k, row in first.iterrows():
        sym = df.loc[df.ion == k, "symbol"].iloc[0]
        ch = int(row["Metal_charge"])
        ions.append({"key": k, "symbol": sym, "z": int(row["Atomic_number"]), "charge": ch,
                     "label": logk.ion_label(sym, ch), "n_rows": int((df.ion == k).sum())})
    ions.sort(key=lambda i: (i["z"], i["charge"]))

    # electrolytes present in the data
    pairs = df[logk.CONDITION_COLUMNS[2:]].dropna().drop_duplicates().values
    electrolytes = {}
    for mw, dens in pairs:
        name = next((n for n, m in logk.ELECTROLYTE_MASSES.items() if abs(m - mw) < 0.02), f"MW {mw:.1f}")
        electrolytes[name] = (float(mw), float(dens))

    # ligand vectors, built through the SAME code path the app uses
    log("Computing ligand features ...")
    ligand_vecs, canon_of, elements = {}, {}, set()
    for raw in df.ligand_raw.unique():
        mol, canon = logk.parse_ligand(raw)
        ligand_vecs[raw], canon_of[raw] = logk.ligand_vector(mol), canon
        elements |= {a.GetSymbol() for a in mol.GetAtoms()}
    df["canon"] = df.ligand_raw.map(canon_of)

    X = build_matrix(df, metal_table, electrolytes, ligand_vecs)
    y = df.logK.values
    assert X.shape[1] == logk.N_FEATURES, (X.shape, logk.N_FEATURES)
    tr, te, va = (np.where(df.split == s)[0] for s in ("train", "test", "validation"))

    # ---- honest evaluation: train split only --------------------------------------------------
    log("Evaluating (model trained on the train split only) ...")
    m0 = new_model().fit(X[tr], y[tr])
    train_pairs, train_ligs = set(df.smiles.iloc[tr]), set(df.canon.iloc[tr])
    sub = df.iloc[te]
    cat = np.where(sub.smiles.isin(train_pairs), "exact_complex",
                   np.where(sub.canon.isin(train_ligs), "known_ligand", "new_ligand"))
    pred_te, pred_va = m0.predict(X[te]), m0.predict(X[va])
    err_te = np.abs(pred_te - y[te])
    err = {c: float(err_te[cat == c].mean()) for c in ("exact_complex", "known_ligand") if (cat == c).any()}
    new_errs = np.concatenate([err_te[cat == "new_ligand"], np.abs(pred_va - y[va])])
    err["new_ligand"] = float(new_errs.mean())
    metrics = {
        "error": err,
        "n_by_category": {c: int((cat == c).sum()) for c in ("exact_complex", "known_ligand", "new_ligand")},
        "test_random_split": {"mae": float(mean_absolute_error(y[te], pred_te)), "r2": float(r2_score(y[te], pred_te)), "n": int(len(te))},
        "external": {"mae": float(mean_absolute_error(y[va], pred_va)), "r2": float(r2_score(y[va], pred_va)), "n": int(len(va))},
        "paper_best_model_external": {"mae": 0.834, "r2": 0.942, "note": "D-MPNN (Chemprop) M4, Karunaratne et al. 2025"},
        "n_rows": int(len(df)), "n_ligands": int(df.canon.nunique()), "n_ions": int(len(ions)),
    }
    log(f"  random TEST split : MAE {metrics['test_random_split']['mae']:.3f}  R2 {metrics['test_random_split']['r2']:.3f}  (leaky: same ligands as training)")
    log(f"  EXTERNAL set      : MAE {metrics['external']['mae']:.3f}  R2 {metrics['external']['r2']:.3f}  (40 unseen ligands)")
    for c, n in metrics["n_by_category"].items():
        log(f"  {c:14s} MAE {err[c]:.3f}   (n={n})")

    # ---- final model on everything ------------------------------------------------------------
    log("Training the final model on all rows ...")
    final = new_model().fit(X, y)
    met = df[df.symbol != "H"]
    lookup = (met.groupby(["ligand_raw", "ion"]).logK.agg(n="size", median="median", min="min", max="max").reset_index())
    bundle = {
        "model": final, "metal_table": metal_table, "ions": ions, "electrolytes": electrolytes,
        "metrics": metrics, "ligand_elements": sorted(elements), "ligand_raw": sorted(df.ligand_raw.unique()),
        "lookup": lookup,
        "condition_range": {"Temp": (float(df.Temp.min()), float(df.Temp.max())),
                            "Concentration of Medium": (float(df["Concentration of Medium"].min()),
                                                        float(df["Concentration of Medium"].max()))},
    }
    OUT.parent.mkdir(exist_ok=True)
    logk.joblib.dump(bundle, OUT, compress=3)
    log(f"Saved {OUT}  ({OUT.stat().st_size / 1e6:.1f} MB)")
    return metrics


if __name__ == "__main__":
    train()
