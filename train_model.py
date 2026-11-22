"""Train the AI models from the processed tmQM_y table.

Run `python prepare_dataset.py` first. Inputs for every model are the same
features the app can build from a user selection: metal, oxidation state and
a mixture of ligand classes (see featurize.py). Targets come straight from
tmQM_y's DFT data:

  1. gap_model        -- regresses the HOMO-LUMO gap (eV)  [HL_Gap]
  2. stability_model  -- classifies the gap into Low / Moderate / High
                         (terciles of the whole dataset)
  3. charge_model     -- regresses the DFT metal partial charge [Metal_q]

HOMO-LUMO gap is used as the electronic-stability descriptor because tmQM
has no thermodynamic stability constants. Geometry is NOT a tmQM column, so
it is assigned by rule in ligandfieldmodel.py rather than learned.

Writes models/*.joblib, models/metrics.json
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.metrics import accuracy_score, mean_absolute_error, r2_score, f1_score
from sklearn.model_selection import train_test_split

from featurize import feature_row

try:
    import joblib
except ImportError:  # pragma: no cover
    import pickle

    class _J:
        dump = staticmethod(lambda o, p: pickle.dump(o, open(p, "wb")))
        load = staticmethod(lambda p: pickle.load(open(p, "rb")))
    joblib = _J()

ROOT = Path(__file__).parent
DATA = ROOT / "data"
MODELS = ROOT / "models"
MODELS.mkdir(exist_ok=True)

meta = json.loads((DATA / "dataset_meta.json").read_text(encoding="utf-8"))
classes, vocab = meta["classes"], meta["vocab"] + ["Other ligand"]
lo, hi = meta["gap_thresholds_eV"]

df = pd.read_csv(DATA / "tmqm_processed.csv.gz")


def parse_sig(sig):
    return {k: int(v) for k, v in (p.rsplit("::", 1) for p in sig.split("|"))}


print(f"Building features for {len(df):,} complexes ...")
X = pd.DataFrame([feature_row(m, int(o), parse_sig(s), classes, vocab)
                  for m, o, s in zip(df.metal, df.oxidation_state, df.signature)])
joblib.dump(list(X.columns), MODELS / "feature_columns.joblib")

y_gap = df.hl_gap_eV.values
y_cls = np.where(y_gap < lo, "Low", np.where(y_gap < hi, "Moderate", "High"))
y_q = df.metal_q.values

# One shared split so every model is evaluated on the same held-out complexes.
idx_tr, idx_te = train_test_split(np.arange(len(df)), test_size=0.2, random_state=42, stratify=y_cls)
Xtr, Xte = X.iloc[idx_tr], X.iloc[idx_te]
metrics = {"n_train": int(len(idx_tr)), "n_test": int(len(idx_te))}

reg_kw = dict(max_iter=400, learning_rate=0.08, max_leaf_nodes=48, min_samples_leaf=20,
              l2_regularization=1.0, random_state=42)

# --- 1) HOMO-LUMO gap -------------------------------------------------------
gap_model = HistGradientBoostingRegressor(**reg_kw).fit(Xtr, y_gap[idx_tr])
p = gap_model.predict(Xte)
metrics["gap"] = {"MAE_eV": float(mean_absolute_error(y_gap[idx_te], p)),
                  "R2": float(r2_score(y_gap[idx_te], p)),
                  "baseline_MAE_eV(mean predictor)": float(mean_absolute_error(y_gap[idx_te], np.full(len(idx_te), y_gap[idx_tr].mean())))}
joblib.dump(gap_model, MODELS / "gap_model.joblib")

# --- 2) Stability class -----------------------------------------------------
stab = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.08, max_leaf_nodes=48,
                                      min_samples_leaf=20, l2_regularization=1.0, random_state=42)
stab.fit(Xtr, y_cls[idx_tr])
pc = stab.predict(Xte)
metrics["stability"] = {"accuracy": float(accuracy_score(y_cls[idx_te], pc)),
                        "macro_F1": float(f1_score(y_cls[idx_te], pc, average="macro")),
                        "majority_baseline_accuracy": float(pd.Series(y_cls[idx_tr]).value_counts(normalize=True).iloc[0])}
joblib.dump(stab, MODELS / "stability_model.joblib")

# --- 3) Metal partial charge ------------------------------------------------
q_model = HistGradientBoostingRegressor(**reg_kw).fit(Xtr, y_q[idx_tr])
pq = q_model.predict(Xte)
metrics["metal_charge"] = {"MAE": float(mean_absolute_error(y_q[idx_te], pq)), "R2": float(r2_score(y_q[idx_te], pq))}
joblib.dump(q_model, MODELS / "charge_model.joblib")

(MODELS / "metrics.json").write_text(json.dumps(metrics, indent=1), encoding="utf-8")
print(json.dumps(metrics, indent=1))
print(f"\nSaved models to {MODELS}/")
