"""Train the geometry classifier.

Labels come from the 3D structures (label_geometry.py -> data/geometry_labels.csv.gz);
inputs are exactly the features the app can build from a user selection
(metal, oxidation state, ligand classes -- see featurize.py), so the model can
predict a geometry for complexes that have no structure.

Writes models/geometry_model.joblib and models/geometry_metrics.json.
Reports accuracy, macro-F1, the old coordination-number/d-count rule on the same
held-out complexes, and a confidence-vs-accuracy table.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import accuracy_score, f1_score, log_loss
from sklearn.model_selection import train_test_split

from featurize import feature_row
from geometry_classes import to_class
from ligandfieldmodel import geometry_from_coordination

try:
    import joblib
except ImportError:  # pragma: no cover
    import pickle

    class _J:
        dump = staticmethod(lambda o, p: pickle.dump(o, open(p, "wb")))
        load = staticmethod(lambda p: pickle.load(open(p, "rb")))
    joblib = _J()

ROOT = Path(__file__).parent
DATA, MODELS = ROOT / "data", ROOT / "models"
MODELS.mkdir(exist_ok=True)

meta = json.loads((DATA / "dataset_meta.json").read_text(encoding="utf-8"))
classes, vocab = meta["classes"], meta["vocab"] + ["Other ligand"]
cols = joblib.load(MODELS / "feature_columns.joblib")

df = pd.read_csv(DATA / "tmqm_processed.csv.gz")
lab = pd.read_csv(DATA / "geometry_labels.csv.gz")[["csd_code", "geometry"]]
df = df.merge(lab, on="csd_code", how="inner")
df["geo_class"] = df.geometry.map(to_class)
df = df[df.geo_class.notna()].reset_index(drop=True)
print(f"{len(df):,} complexes with a structure-derived geometry label")


def parse_sig(sig):
    return {k: int(v) for k, v in (p.rsplit("::", 1) for p in sig.split("|"))}


X = pd.DataFrame([feature_row(m, int(o), parse_sig(s), classes, vocab)
                  for m, o, s in zip(df.metal, df.oxidation_state, df.signature)]).reindex(columns=cols, fill_value=0)
y = df.geo_class.values

idx_tr, idx_te = train_test_split(np.arange(len(df)), test_size=0.2, random_state=42, stratify=y)
Xtr, Xte, ytr, yte = X.iloc[idx_tr], X.iloc[idx_te], y[idx_tr], y[idx_te]

clf = HistGradientBoostingClassifier(max_iter=250, learning_rate=0.08, max_leaf_nodes=48, min_samples_leaf=20,
                                     l2_regularization=1.0, early_stopping=True, validation_fraction=0.1,
                                     n_iter_no_change=15, random_state=42)
clf.fit(Xtr, ytr)
proba = clf.predict_proba(Xte)
pred = clf.classes_[proba.argmax(1)]
conf = proba.max(1)

# ---- baseline 1: the old rule (CN + d count) ------------------------------------------
def old_rule(row):
    g, _ = geometry_from_coordination(int(row.d_electron_count), row.mean_field_strength, int(row.cn_sites), int(row.row))
    return g.replace("_", " ") if g else "(not modelled)"


old = np.array([old_rule(r) for r in Xte.itertuples()])
# ---- baseline 2: always the most common class ------------------------------------------
maj = pd.Series(ytr).value_counts().index[0]

metrics = {
    "n_train": int(len(idx_tr)), "n_test": int(len(idx_te)), "classes": list(clf.classes_),
    "accuracy": float(accuracy_score(yte, pred)),
    "macro_F1": float(f1_score(yte, pred, average="macro")),
    "log_loss": float(log_loss(yte, proba, labels=clf.classes_)),
    "old_rule_accuracy": float(accuracy_score(yte, old)),
    "old_rule_macro_F1": float(f1_score(yte, old, average="macro")),
    "majority_baseline_accuracy": float(np.mean(yte == maj)),
    "n_iter": int(clf.n_iter_),
}
# fairer to the old rule: only complexes whose true geometry is one of the four it can output
four = np.isin(yte, ["octahedral", "tetrahedral", "square planar", "square pyramidal"])
metrics["four_geometry_subset"] = {"n_test": int(four.sum()),
                                   "classifier_accuracy": float(np.mean(pred[four] == yte[four])),
                                   "old_rule_accuracy": float(np.mean(old[four] == yte[four]))}
# does the confidence mean something?  accuracy inside confidence bins
bins = [0, .4, .5, .6, .7, .8, .9, 1.0001]
table = []
for lo, hi in zip(bins[:-1], bins[1:]):
    m = (conf >= lo) & (conf < hi)
    if m.any():
        table.append({"confidence": f"{lo:.0%}-{min(hi, 1):.0%}", "n": int(m.sum()),
                      "accuracy": float(np.mean(pred[m] == yte[m]))})
metrics["confidence_vs_accuracy"] = table
metrics["per_class_recall"] = {
    c: {"n": int((yte == c).sum()), "recall": float(np.mean(pred[yte == c] == c))} for c in clf.classes_ if (yte == c).any()}

joblib.dump(clf, MODELS / "geometry_model.joblib")
(MODELS / "geometry_metrics.json").write_text(json.dumps(metrics, indent=1), encoding="utf-8")
print(json.dumps({k: v for k, v in metrics.items() if k != "per_class_recall"}, indent=1))
for c, v in metrics["per_class_recall"].items():
    print(f"  {c:26s} n={v['n']:6d}  recall={v['recall']:.2f}")
print(f"\nSaved {MODELS / 'geometry_model.joblib'}")
