"""Grouped-split evaluation of the geometry classifier.

The app's headline accuracy uses a random 80/20 split, where most test complexes
share their exact (metal, oxidation state, ligand set) input with a training
complex.  This script re-measures accuracy when whole groups are held out:

  random                 stratified random split (as in train_geometry.py)
  held-out exact combos  no (metal, oxidation state, ligand set) shared with training
  held-out ligand sets   no ligand set shared with training (any metal)
  held-out metals        no metal shared with training

For each split it reports the gradient-boosting model, a lookup-table baseline
(majority geometry for the exact combination, falling back to ligand set, then
metal + oxidation state, then the overall majority), the hand-set coordination
rule used before the model, and the majority class.

Usage:
    python evaluate_geometry_grouped.py                # all splits, ~10 min each
    python evaluate_geometry_grouped.py --sample 6000 --max-iter 30   # quick smoke test

Results are printed and written to results/geometry_grouped_splits.csv
(or --out).  Single split, seed 42: no confidence intervals.
"""
import argparse, json, sys, time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import GroupShuffleSplit, train_test_split

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from featurize import feature_row                      # noqa: E402
from geometry_classes import to_class                  # noqa: E402
from ligandfieldmodel import geometry_from_coordination  # noqa: E402


def parse_sig(s):
    return {k: int(v) for k, v in (p.rsplit("::", 1) for p in s.split("|"))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sample", type=int, default=0, help="use only N random complexes (smoke test)")
    ap.add_argument("--max-iter", type=int, default=250)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default=str(HERE / "results" / "geometry_grouped_splits.csv"))
    args = ap.parse_args()

    meta = json.load(open(HERE / "data" / "dataset_meta.json"))
    classes, vocab = meta["classes"], meta["vocab"] + ["Other ligand"]
    cols = joblib.load(HERE / "models" / "feature_columns.joblib")

    df = pd.read_csv(HERE / "data" / "tmqm_processed.csv.gz")
    lab = pd.read_csv(HERE / "data" / "geometry_labels.csv.gz")[["csd_code", "geometry"]]
    df = df.merge(lab, on="csd_code")
    df["y"] = df.geometry.map(to_class)
    df = df[df.y.notna()].reset_index(drop=True)
    if args.sample:
        df = df.sample(args.sample, random_state=args.seed).reset_index(drop=True)

    X = pd.DataFrame([feature_row(m, int(o), parse_sig(s), classes, vocab)
                      for m, o, s in zip(df.metal, df.oxidation_state, df.signature)]
                     ).reindex(columns=cols, fill_value=0)
    y = df.y.values
    metal = df.metal.values
    mo = (df.metal.astype(str) + "|" + df.oxidation_state.astype(str)).values
    sig = df.signature.values
    key = (mo + "|" + df.signature.astype(str)).values

    def old_rule(r):
        g, _ = geometry_from_coordination(int(r.d_electron_count), r.mean_field_strength,
                                          int(r.cn_sites), int(r.row))
        return g.replace("_", " ") if g else "(not modelled)"

    def lookup_pred(tr, te):
        glob = pd.Series(y[tr]).value_counts().index[0]
        tabs = [pd.Series(y[tr]).groupby(a[tr]).agg(lambda s: s.value_counts().index[0])
                for a in (key, sig, mo)]
        out = []
        for i in te:
            p = None
            for a, t in zip((key, sig, mo), tabs):
                if a[i] in t.index:
                    p = t[a[i]]
                    break
            out.append(p if p is not None else glob)
        return np.array(out)

    splits = {}
    stratify = y if pd.Series(y).value_counts().min() >= 2 else None
    splits["random (as in app)"] = train_test_split(np.arange(len(df)), test_size=0.2,
                                                    random_state=args.seed, stratify=stratify)
    for name, g in [("held-out exact combos (metal+ox+ligands)", key),
                    ("held-out ligand sets (any metal)", sig),
                    ("held-out metals", metal)]:
        splits[name] = next(GroupShuffleSplit(1, test_size=0.2, random_state=args.seed).split(X, y, g))

    rows = []
    for name, (tr, te) in splits.items():
        t0 = time.time()
        clf = HistGradientBoostingClassifier(
            max_iter=args.max_iter, learning_rate=0.08, max_leaf_nodes=48, min_samples_leaf=20,
            l2_regularization=1.0, early_stopping=True, validation_fraction=0.1,
            n_iter_no_change=15, random_state=args.seed)
        clf.fit(X.iloc[tr], y[tr])
        pr, yt = clf.predict(X.iloc[te]), y[te]
        lk = lookup_pred(tr, te)
        old = np.array([old_rule(r) for r in X.iloc[te].itertuples()])
        maj = pd.Series(y[tr]).value_counts().index[0]
        seen_keys = set(key[tr])
        row = dict(split=name, n_test=len(te),
                   pct_test_key_seen=100 * np.mean([k in seen_keys for k in key[te]]),
                   model_acc=accuracy_score(yt, pr), model_macro_f1=f1_score(yt, pr, average="macro"),
                   lookup_acc=accuracy_score(yt, lk), lookup_macro_f1=f1_score(yt, lk, average="macro"),
                   old_rule_acc=accuracy_score(yt, old), majority_acc=float(np.mean(yt == maj)))
        rows.append(row)
        print({k: (round(v, 4) if isinstance(v, float) else v) for k, v in row.items()},
              f"[{time.time() - t0:.0f}s]", flush=True)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(args.out, index=False)
    print("wrote", args.out)


if __name__ == "__main__":
    main()
