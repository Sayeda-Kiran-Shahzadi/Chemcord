"""
Chemcord -- Tkinter desktop app for metal-complex geometry and stability
=============================================================================

WHAT THIS APP DOES
  Pick a transition metal, an oxidation state, and ONE OR MORE ligands (each
  with a count). The app then
    1. computes crystal-field quantities (Delta for four geometries, pairing
       energy, spin, CFSE/CFAE, kinetic lability) from the site-weighted
       average field of the chosen ligands (ligandfieldmodel.py);
    2. uses machine-learning models trained on the real tmQM_y DFT dataset
       (~99k complexes) to predict the HOMO-LUMO gap (eV), a Low/Moderate/High
       electronic-stability class and the DFT metal partial charge;
    3. predicts the coordination geometry (octahedral, square planar, trigonal
       bipyramidal, linear, piano stool, ...) with a confidence value, using a
       classifier trained on geometries measured from the tmQM 3D structures
       (label_geometry.py + train_geometry.py);
    4. looks up real tmQM complexes that match your selection (same metal,
       oxidation state and ligands) and shows their actual DFT gaps and the
       geometries those complexes really have;
    5. (Tools menu / "Binding strength (log K1)" button) predicts the aqueous
       binding constant log K1 of a metal ion with ANY ligand you enter as a
       SMILES string, shows the free energy of binding, how far to trust the
       number, any measured values, and ranks all metal ions (selectivity).
       This uses a separate model trained on experimental stability constants
       (logk.py, train_logk.py, logk_ui.py); it needs RDKit (pip install rdkit).

SETUP (once, from this folder)
    pip install pandas numpy scikit-learn joblib rdkit
    python prepare_dataset.py data/tmQM_y.csv   # only needed to rebuild data
    python train_model.py                       # only needed to rebuild models
    python label_geometry.py data/tmQM_X*_xyz.gz  # only to re-measure geometries
    python train_geometry.py                    # only needed to rebuild the geometry model
    python train_logk.py                        # only needed to rebuild the log K1 model (needs data/logk_raw/)
    python chemcord.py
  The prepared data/ and models/ folders are included, so you can start the
  app directly.
"""
import json
import subprocess
import sys
import threading
import tkinter as tk
from pathlib import Path
from tkinter import ttk, messagebox

import numpy as np
import pandas as pd

try:
    import joblib
except ImportError:
    import pickle

    class _J:
        load = staticmethod(lambda p: pickle.load(open(p, "rb")))
    joblib = _J()

from ligandfieldmodel import (
    ATOMIC_NUMBERS, METAL_ROW, GEOMETRIES, d_electron_count, all_deltas,
    base_values, calculate_complex, geometry_from_coordination,
    mixed_ligand, to_cm1, delta_confidence, K_CM1_PER_UNIT,
)
from featurize import feature_row
from geometry_classes import LIGAND_FIELD_KEY, DESCRIPTIONS, to_class

FROZEN = getattr(sys, "frozen", False)
ROOT = Path(getattr(sys, "_MEIPASS", Path(__file__).parent))
DATA_DIR, MODELS_DIR = ROOT / "data", ROOT / "models"
MAX_PER_LIGAND = 8

TMQM_LICENCE_SHORT = ("Data derived from tmQM / tmQM_y (MIT License, Copyright (c) 2024 Bastian Bjerkem Skjelstad). "
                      "See THIRD_PARTY_LICENSES.md for the full licence text and citations.")

INSTRUCTIONS_TEXT = """How to use this app

1. Choose a Metal. The oxidation-state list then shows only states that occur
   in the tmQM_y dataset for that metal (oxidation state = formal metal charge
   in the dataset's SMILES).

2. Select ligands. Tick one or more ligand classes and set how many of each
   you want (the "x" box). Use the search box to filter the list. Examples:
       Fe, ox 2:  Cyanide x6
       Pt, ox 2:  Ammonia x2 + Chloride x2        (mixed ligands)
       Ru, ox 2:  N,N-donor bidentate x3          (chelate, counts as 6 sites)
   Each class occupies a number of coordination "sites" (a bidentate ligand
   = 2, a Cp-type ligand = 3). The total is shown under the list.

3. Click "Predict Complex".

4. Results
   - Crystal field: Delta for four geometries and pairing energy, from the
     site-weighted average ligand field strength. Calculated, not predicted.
   - AI prediction: HOMO-LUMO gap (eV), stability class and metal partial
     charge, learned from real DFT data in tmQM_y. The typical test error of
     each model is shown next to the prediction.
   - Geometry: a classifier trained on the 3D structures in tmQM (geometry
     measured from the metal-donor bond angles) predicts the most likely
     geometry, with a confidence value and the runner-up shapes. It can say
     linear, trigonal planar, T-shaped, tetrahedral, square planar, seesaw,
     piano stool, square pyramidal, trigonal bipyramidal, octahedral,
     trigonal prismatic, pentagonal bipyramidal, ... (a pi-ring such as Cp
     counts as ONE vertex of the polyhedron here).
     For the four geometries the ligand-field model covers (octahedral,
     tetrahedral, square planar, square pyramidal) spin state, CFSE, CFAE and
     kinetic lability are calculated; for any other predicted geometry they
     are shown as n/a.
   - tmQM evidence: real complexes matching your selection, with their DFT
     HOMO-LUMO gaps, CSD codes and the geometries they actually have.

5. Binding strength (log K1). Use the "Binding strength (log K1)..." button or
   Tools menu. Pick a metal ion, then a common ligand or type any ligand
   SMILES (organic ligands neutral, e.g. CC(=O)O; simple anions as ions,
   e.g. [Cl-]). You get the predicted aqueous log K1, the free energy of
   binding, whether the pair was measured, and a typical error that depends
   on how familiar the ligand is. "Compare all metal ions" ranks every ion.
   This is the thermodynamic (equilibrium) stability in water; the gap-based
   "gap class" above is a different quantity.

Limitations: the geometry model sees only the ligand set, metal and oxidation
state (not a 3D structure), so isomers / distortions that the same ligand set
can adopt show up as lower confidence; ligand-field strengths are
textbook-inspired assumptions, not fitted values; ligands are grouped into classes (e.g. all monodentate
N-donors are one class); the main-window gap class (Low/Moderate/High tercile of the HOMO-LUMO gap)
is an electronic proxy, not a binding constant or thermodynamic stability (use the log K1 tool for that). Treat results as trends, not research-grade values.
"""


# Oxidation states offered in the dropdown even though tmQM_y has fewer than MIN_OX_SUPPORT (25)
# complexes for them. tmQM_y contains very few Fe(III) complexes (7), so predictions for Fe(III) rest on
# little data; the app prints a low-support warning whenever such a state is chosen.
# To offer another state, add it here, e.g. {"Fe": [3], "Cu": [2], "Cr": [3], "Mn": [2]}.
EXTRA_OXIDATION_STATES = {"Fe": [3]}
LOW_SUPPORT = 25


class App:
    def __init__(self, root):
        self.root = root
        root.title("Chemcord")
        root.geometry("1260x820")

        self.meta = json.loads((DATA_DIR / "dataset_meta.json").read_text(encoding="utf-8"))
        self._logk = None                                  # log K1 predictor, loaded on first use
        self.classes = self.meta["classes"]
        self.vocab = self.meta["vocab"] + ["Other ligand"]
        self.pickable = list(self.meta["vocab"])          # "Other ligand" is not selectable
        self.metal_ox = {m: list(v) for m, v in self.meta["metal_oxidation_states"].items()}
        for m, extra in EXTRA_OXIDATION_STATES.items():
            if m in self.metal_ox:
                self.metal_ox[m] = sorted(set(self.metal_ox[m]) | set(extra))
        self.gap_lo, self.gap_hi = self.meta["gap_thresholds_eV"]
        self.models = None
        self.geo_model = None
        self.metrics = {}
        self.geo_metrics = {}
        self._load_models(silent=True)
        self._load_reference()

        self.lig_vars = {}     # class name -> (checked BooleanVar, count IntVar)
        self._build_menu()
        self._build_layout()

    # ------------------------------------------------------------------ data
    def _load_reference(self):
        try:
            self.ref = pd.read_csv(DATA_DIR / "tmqm_processed.csv.gz")
            self.ref["sig_wrapped"] = "|" + self.ref.signature.str.replace(r"::\d+", "", regex=True) + "|"
        except Exception:
            self.ref = None
            return
        try:   # measured geometry of every complex (label_geometry.py)
            gl = pd.read_csv(DATA_DIR / "geometry_labels.csv.gz", usecols=["csd_code", "geometry"])
            gl["geometry_class"] = gl.geometry.map(to_class)
            self.ref = self.ref.merge(gl[["csd_code", "geometry_class"]], on="csd_code", how="left")
        except Exception:
            pass

    def _load_models(self, silent=False):
        try:
            self.models = {
                "gap": joblib.load(MODELS_DIR / "gap_model.joblib"),
                "stab": joblib.load(MODELS_DIR / "stability_model.joblib"),
                "charge": joblib.load(MODELS_DIR / "charge_model.joblib"),
                "cols": joblib.load(MODELS_DIR / "feature_columns.joblib"),
            }
            mp = MODELS_DIR / "metrics.json"
            self.metrics = json.loads(mp.read_text()) if mp.exists() else {}
            try:   # the geometry classifier is optional: without it the app falls back to the old rule
                self.geo_model = joblib.load(MODELS_DIR / "geometry_model.joblib")
                gp = MODELS_DIR / "geometry_metrics.json"
                self.geo_metrics = json.loads(gp.read_text()) if gp.exists() else {}
            except Exception:
                self.geo_model, self.geo_metrics = None, {}
            return True
        except Exception:
            self.models = None
            if not silent:
                messagebox.showwarning("Models not found", "No trained models in models/. Use File -> Rebuild "
                                       "dataset & retrain, or run prepare_dataset.py then train_model.py.")
            return False

    # ------------------------------------------------------------------ menu
    def _build_menu(self):
        bar = tk.Menu(self.root)
        fm = tk.Menu(bar, tearoff=0)
        fm.add_command(label="Rebuild dataset && retrain models", command=self.retrain_models)
        fm.add_separator()
        fm.add_command(label="Exit", command=self.root.quit)
        bar.add_cascade(label="File", menu=fm)
        tm = tk.Menu(bar, tearoff=0)
        tm.add_command(label="Binding strength (log K\u2081)...", command=self.open_logk)
        bar.add_cascade(label="Tools", menu=tm)
        hm = tk.Menu(bar, tearoff=0)
        hm.add_command(label="How to use this app", command=self.show_instructions)
        hm.add_command(label="Data licence & citations (tmQM)", command=self.show_licence)
        bar.add_cascade(label="Help", menu=hm)
        self.root.config(menu=bar)

    def open_logk(self):
        """Open the log K1 window (needs RDKit and models/logk_bundle.joblib)."""
        try:
            import logk
            import logk_ui
        except ImportError as exc:
            messagebox.showerror("log K1 tool unavailable", f"A module could not be imported: {exc}")
            return
        if not logk.HAVE_RDKIT:
            messagebox.showinfo("RDKit needed", "The log K1 tool needs RDKit.\n\nInstall it with:  pip install rdkit")
            return
        if self._logk is None:
            path = MODELS_DIR / "logk_bundle.joblib"
            if not path.exists():
                messagebox.showinfo("log K1 model not found",
                                    "models/logk_bundle.joblib is missing.\n\nPut the six logK csv files into data/logk_raw/ "
                                    "and run:  python train_logk.py\n(see the README).")
                return
            try:
                self._logk = logk.LogKPredictor(path)
            except Exception as exc:
                messagebox.showerror("log K1 model could not be loaded", str(exc))
                return
        # pre-fill from the main window where that is unambiguous
        metal, ox = self.metal_var.get().split(" ")[0], self.ox_var.get()
        key = f"{metal}{int(ox):+d}" if ox.lstrip("-").isdigit() else None
        chosen = [n for n, (on, _cnt) in self.lig_vars.items() if on.get()]
        smiles = logk.CLASS_TO_SMILES.get(chosen[0]) if len(chosen) == 1 else None
        return logk_ui.LogKWindow(self.root, self._logk, ion_key=key, smiles=smiles)

    def show_licence(self):
        win = tk.Toplevel(self.root)
        win.title("Data licence & citations")
        win.geometry("700x640")
        t = tk.Text(win, wrap="word", padx=12, pady=12)
        f = ROOT / "THIRD_PARTY_LICENSES.md"
        t.insert("1.0", f.read_text(encoding="utf-8") if f.exists() else TMQM_LICENCE_SHORT)
        t.config(state="disabled")
        t.pack(fill=tk.BOTH, expand=True)
        ttk.Button(win, text="Close", command=win.destroy).pack(pady=6)

    def show_instructions(self):
        win = tk.Toplevel(self.root)
        win.title("How to use this app")
        win.geometry("680x640")
        t = tk.Text(win, wrap="word", padx=12, pady=12)
        t.insert("1.0", INSTRUCTIONS_TEXT)
        t.config(state="disabled")
        t.pack(fill=tk.BOTH, expand=True)
        ttk.Button(win, text="Close", command=win.destroy).pack(pady=6)

    # ---------------------------------------------------------------- layout
    def _build_layout(self):
        body = ttk.Frame(self.root, padding=10)
        body.pack(fill=tk.BOTH, expand=True)

        left = ttk.Frame(body, padding=(0, 0, 12, 0))
        left.pack(side=tk.LEFT, fill=tk.Y)

        ttk.Label(left, text="Metal", font=("", 10, "bold")).pack(anchor="w")
        metals = [f"{m} ({ {1: '3d', 2: '4d', 3: '5d'}[METAL_ROW[m]] })" for m in ATOMIC_NUMBERS if m in self.metal_ox]
        self.metal_var = tk.StringVar(value=next(x for x in metals if x.startswith("Fe")))
        mc = ttk.Combobox(left, textvariable=self.metal_var, values=metals, state="readonly", width=24)
        mc.pack(anchor="w", pady=(0, 8))
        mc.bind("<<ComboboxSelected>>", lambda e: self._refresh_ox())

        ttk.Label(left, text="Oxidation state (in tmQM_y)", font=("", 10, "bold")).pack(anchor="w")
        self.ox_var = tk.StringVar()
        self.ox_cb = ttk.Combobox(left, textvariable=self.ox_var, state="readonly", width=24)
        self.ox_cb.pack(anchor="w", pady=(0, 8))
        self._refresh_ox(prefer="2")

        ttk.Label(left, text="Ligands (tick one or more, set count)", font=("", 10, "bold")).pack(anchor="w")
        sf = ttk.Frame(left)
        sf.pack(anchor="w", fill=tk.X)
        ttk.Label(sf, text="Search:").pack(side=tk.LEFT)
        self.search_var = tk.StringVar()
        self.search_var.trace_add("write", lambda *_: self._filter_ligands())
        ttk.Entry(sf, textvariable=self.search_var, width=24).pack(side=tk.LEFT, padx=4)

        box = ttk.Frame(left, relief="sunken", borderwidth=1)
        box.pack(fill=tk.BOTH, expand=True, pady=(4, 4))
        self.canvas = tk.Canvas(box, width=330, height=330, highlightthickness=0)
        sb = ttk.Scrollbar(box, orient="vertical", command=self.canvas.yview)
        self.inner = ttk.Frame(self.canvas)
        self.inner.bind("<Configure>", lambda e: self.canvas.configure(scrollregion=self.canvas.bbox("all")))
        self.canvas.create_window((0, 0), window=self.inner, anchor="nw")
        self.canvas.configure(yscrollcommand=sb.set)
        self.canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        sb.pack(side=tk.RIGHT, fill=tk.Y)
        self.canvas.bind("<Enter>", lambda e: self._bind_wheel(True))
        self.canvas.bind("<Leave>", lambda e: self._bind_wheel(False))

        self.rows = {}
        for name in self.pickable:
            chk, cnt = tk.BooleanVar(value=False), tk.IntVar(value=1)
            self.lig_vars[name] = (chk, cnt)
            fr = ttk.Frame(self.inner)
            ttk.Checkbutton(fr, text=name, variable=chk, command=self._update_summary).pack(side=tk.LEFT)
            sp = ttk.Spinbox(fr, from_=1, to=MAX_PER_LIGAND, width=3, textvariable=cnt, command=self._update_summary)
            sp.pack(side=tk.RIGHT, padx=(6, 4))
            sp.bind("<KeyRelease>", lambda e: self._update_summary())
            ttk.Label(fr, text="x").pack(side=tk.RIGHT)
            self.rows[name] = fr
        self._filter_ligands()

        self.sel_var = tk.StringVar()
        ttk.Label(left, textvariable=self.sel_var, wraplength=330, foreground="#1a4d8f").pack(anchor="w", pady=(2, 6))
        bf = ttk.Frame(left)
        bf.pack(fill=tk.X)
        ttk.Button(bf, text="Predict Complex", command=self.predict).pack(side=tk.LEFT, fill=tk.X, expand=True)
        ttk.Button(bf, text="Clear ligands", command=self.clear_ligands).pack(side=tk.LEFT, padx=(6, 0))
        ttk.Button(left, text="Binding strength (log K\u2081)...", command=self.open_logk).pack(fill=tk.X, pady=(6, 0))
        self.status_var = tk.StringVar(value=self._status_text())
        ttk.Label(left, textvariable=self.status_var, wraplength=330, foreground="gray").pack(anchor="w", pady=(8, 0))
        self._update_summary()

        right = ttk.Frame(body)
        right.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

        def panel(title):
            f = ttk.LabelFrame(right, text=title, padding=8)
            f.pack(fill=tk.X, pady=(0, 8))
            return f

        def field(parent, label, r, c, big=False):
            ttk.Label(parent, text=label).grid(row=r, column=c * 2, sticky="w", padx=(0, 4), pady=2)
            v = tk.StringVar(value="--")
            ttk.Label(parent, textvariable=v, font=("", 11 if big else 10, "bold")).grid(row=r, column=c * 2 + 1, sticky="w", padx=(0, 22))
            return v

        p = panel("Crystal field splitting (Δ) & pairing energy  [calculated]")
        self.delta = {g: field(p, g.replace("_", " ").title() + ":", i // 2, i % 2) for i, g in enumerate(GEOMETRIES)}
        self.pair_v = field(p, "Pairing energy (P):", 2, 0)
        self.d_v = field(p, "d-electron count:", 2, 1)
        self.sites_v = field(p, "Coordination sites:", 3, 0)
        self.avg_v = field(p, "Mean field strength:", 3, 1)

        p = panel("AI prediction  [trained on tmQM_y DFT data]")
        self.gap_v = field(p, "HOMO–LUMO gap:", 0, 0, True)
        self.stab_v = field(p, "HOMO–LUMO gap class:", 1, 0, True)
        self.q_v = field(p, "Metal partial charge:", 2, 0, True)

        p = panel("Geometry  [classifier trained on tmQM 3D structures]")
        self.geo_v = field(p, "Predicted geometry:", 0, 0, True)
        self.conf_v = field(p, "Confidence:", 0, 1, True)
        self.alt_v = field(p, "Next most likely:", 1, 0)

        p = panel("Ligand-field cross-check  [calculated: octahedral, tetrahedral, square planar, square pyramidal]")
        self.spin_v = field(p, "Spin state:", 0, 0)
        self.kin_v = field(p, "Kinetic stability:", 0, 1)
        self.cfse_v = field(p, "CFSE:", 1, 0)
        self.cfae_v = field(p, "CFAE (SN1 / SN2):", 1, 1)
        self.net_v = field(p, "Net stabilisation:", 2, 0)

        p = ttk.LabelFrame(right, text="tmQM evidence & notes", padding=8)
        p.pack(fill=tk.BOTH, expand=True)
        self.notes = tk.Text(p, wrap="word", height=12, state="disabled")
        sc = ttk.Scrollbar(p, command=self.notes.yview)
        self.notes.configure(yscrollcommand=sc.set)
        sc.pack(side=tk.RIGHT, fill=tk.Y)
        self.notes.pack(fill=tk.BOTH, expand=True)

    # ------------------------------------------------------- ligand picker UI
    def _bind_wheel(self, on):
        if on:
            self.canvas.bind_all("<MouseWheel>", lambda e: self.canvas.yview_scroll(int(-e.delta / 120) or (-1 if e.delta > 0 else 1), "units"))
            self.canvas.bind_all("<Button-4>", lambda e: self.canvas.yview_scroll(-1, "units"))
            self.canvas.bind_all("<Button-5>", lambda e: self.canvas.yview_scroll(1, "units"))
        else:
            for s in ("<MouseWheel>", "<Button-4>", "<Button-5>"):
                self.canvas.unbind_all(s)

    def _filter_ligands(self):
        q = self.search_var.get().strip().lower()
        for name, fr in self.rows.items():
            fr.pack_forget()
            if not q or q in name.lower():
                fr.pack(fill=tk.X, anchor="w")
        self.canvas.yview_moveto(0)

    def _refresh_ox(self, prefer=None):
        metal = self.metal_var.get().split(" ")[0]
        vals = [str(o) for o in self.metal_ox[metal]]
        self.ox_cb["values"] = vals
        cur = prefer or self.ox_var.get()
        self.ox_var.set(cur if cur in vals else vals[min(len(vals) - 1, 1)] if len(vals) > 1 else vals[0])

    def selection(self):
        sel = {}
        for name, (chk, cnt) in self.lig_vars.items():
            if chk.get():
                try:
                    n = max(1, min(MAX_PER_LIGAND, int(cnt.get())))
                except (tk.TclError, ValueError):
                    n = 1
                sel[name] = n
        return sel

    def _update_summary(self):
        sel = self.selection()
        if not sel:
            self.sel_var.set("No ligand selected yet.")
            return
        sites = sum(n * self.classes[c]["sites"] for c, n in sel.items())
        self.sel_var.set(f"{len(sel)} ligand class(es), {sum(sel.values())} ligands, {sites} coordination sites.")

    def clear_ligands(self):
        for chk, cnt in self.lig_vars.values():
            chk.set(False); cnt.set(1)
        self._update_summary()

    # --------------------------------------------------------------- retrain
    def _status_text(self):
        if not self.models:
            return "No trained models yet -- see File menu."
        g, s = self.metrics.get("gap", {}), self.metrics.get("stability", {})
        txt = (f"Models loaded ✓ (test MAE gap {g.get('MAE_eV', float('nan')):.2f} eV, "
               f"stability accuracy {s.get('accuracy', float('nan')):.0%}")
        if self.geo_model is not None:
            txt += f", geometry accuracy {self.geo_metrics.get('accuracy', float('nan')):.0%}"
        else:
            txt += "; no geometry model -- old rule used"
        return txt + ")"

    def retrain_models(self):
        if FROZEN:
            messagebox.showinfo("Not available", "Rebuilding is only available when running the .py version "
                                "(python train_model.py).")
            return
        src = DATA_DIR / "tmQM_y.csv"
        if not src.exists():
            messagebox.showerror("Dataset missing", f"Put the original tmQM_y.csv in {DATA_DIR} first.")
            return
        self.status_var.set("Rebuilding dataset and training... this can take a few minutes")
        self.root.update_idletasks()

        def worker():
            try:
                subprocess.run([sys.executable, "prepare_dataset.py"], cwd=ROOT, check=True)
                subprocess.run([sys.executable, "train_model.py"], cwd=ROOT, check=True)
                if list(DATA_DIR.glob("tmQM_X*_xyz*")):          # re-measure geometries from the xyz files
                    subprocess.run([sys.executable, "label_geometry.py"], cwd=ROOT, check=True)
                if (DATA_DIR / "geometry_labels.csv.gz").exists():
                    subprocess.run([sys.executable, "train_geometry.py"], cwd=ROOT, check=True)
                self.root.after(0, self._after_retrain)
            except Exception as e:
                self.root.after(0, lambda: (self.status_var.set(self._status_text()), messagebox.showerror("Training failed", str(e))))
        threading.Thread(target=worker, daemon=True).start()

    def _after_retrain(self):
        self.meta = json.loads((DATA_DIR / "dataset_meta.json").read_text(encoding="utf-8"))
        self.classes, self.gap_lo, self.gap_hi = self.meta["classes"], *self.meta["gap_thresholds_eV"]
        self._load_models(silent=True)
        self._load_reference()
        self.status_var.set(self._status_text())
        messagebox.showinfo("Done", "Dataset rebuilt and models retrained. Restart the app to refresh the ligand list.")

    # ------------------------------------------------------------ prediction
    def predict(self):
        sel = self.selection()
        if not sel:
            messagebox.showinfo("Choose ligands", "Tick at least one ligand first.")
            return
        metal = self.metal_var.get().split(" ")[0]
        ox = int(self.ox_var.get())
        lig = mixed_ligand(sel, self.classes)
        d = d_electron_count(metal, ox)

        deltas = all_deltas(metal, ox, lig)
        _, _, pairing = base_values(metal, ox, lig)
        for g, v in self.delta.items():
            v.set(f"{deltas[g]:.2f} Δ")
        self.pair_v.set(f"{pairing:.2f} Δ")
        self.d_v.set(f"d{d}")
        self.sites_v.set(str(lig["sites"]))
        self.avg_v.set(f"{lig['strength']:.2f}")

        X = None
        if self.models:
            X = pd.DataFrame([feature_row(metal, ox, sel, self.classes, self.vocab)]).reindex(columns=self.models["cols"], fill_value=0)

        # ---- geometry: classifier trained on tmQM 3D structures ----
        notes = [f"{metal}(ox {ox}), d{d}; ligands: {lig['formula']}.",
                 f"Calibration: 1 Δ unit is about {K_CM1_PER_UNIT:,.0f} cm-1, so the octahedral Δ shown is about "
                 f"{to_cm1(deltas['octahedral']):,.0f} cm-1 (the unit scale and the constants were fitted to experimental values). "
                 + delta_confidence(list(sel), ox)]
        geom = None
        if self.geo_model is not None and X is not None:
            proba = self.geo_model.predict_proba(X)[0]
            order = np.argsort(proba)[::-1]
            label, conf = str(self.geo_model.classes_[order[0]]), float(proba[order[0]])
            self.geo_v.set(label[0].upper() + label[1:])
            self.conf_v.set(f"{conf:.0%}")
            alts = [f"{self.geo_model.classes_[i]} {proba[i]:.0%}" for i in order[1:3] if proba[i] >= 0.05]
            self.alt_v.set("; ".join(alts) if alts else "(none above 5%)")
            notes.append(f"Geometry: {label} -- {DESCRIPTIONS.get(label, '')}. {self._confidence_hint(conf)}")
            if conf < 0.5:
                notes.append("The geometry model is not sure here: this ligand set is found in several different shapes in tmQM "
                             "(see the observed geometries in the evidence below).")
            geom = LIGAND_FIELD_KEY.get(label)
            if geom is None:
                notes.append(f"The ligand-field model only covers octahedral, tetrahedral, square planar and square pyramidal, "
                             f"so spin state, CFSE, CFAE and lability are not calculated for {label}.")
        else:   # no geometry model on disk: fall back to the old coordination-number rule
            geom, geom_note = geometry_from_coordination(d, lig["strength"], lig["sites"], METAL_ROW[metal])
            self.geo_v.set(geom.replace("_", " ").title() if geom else "n/a")
            self.conf_v.set("n/a (rule)")
            self.alt_v.set("--")
            notes.append("No geometry model found -- geometry assigned by the old coordination-number rule. " + geom_note)

        if geom:
            res = calculate_complex(metal, ox, geom, lig)
            self.spin_v.set(res["spin"].title())
            self.cfse_v.set(f"{round(res['cfse'], 2) + 0.0:.2f} Δ")
            self.net_v.set(f"{round(res['net_stabilization'], 2) + 0.0:.2f} Δ")
            if res["cfae"] is None:
                self.cfae_v.set("n/a (octahedral only)")
            else:
                self.cfae_v.set(f"{round(res['cfae_sn1'], 2) + 0.0:.2f} / {round(res['cfae_sn2'], 2) + 0.0:.2f} Δ")
            self.kin_v.set(res["kinetic"])
            notes.append(res["description"])
            if res["cfae"] is not None:
                notes.append("CFAE = crystal-field activation energy: the CFSE lost on reaching the dissociative (SN1, square "
                             "pyramidal) or associative (SN2, pentagonal bipyramidal) transition state; larger relative to "
                             "Delta means slower ligand exchange. Net stabilisation = -(CFSE + pairing cost).")
            if geom == "octahedral" and d == 8:
                notes.append("Octahedral d8: CFAE theory over-predicts inertness, so the kinetic label is capped at Moderate.")
        else:
            for v in (self.spin_v, self.cfse_v, self.cfae_v, self.kin_v, self.net_v):
                v.set("n/a")

        # ---- AI prediction ----
        if self.models:
            gap = float(self.models["gap"].predict(X)[0])
            proba = dict(zip(self.models["stab"].classes_, self.models["stab"].predict_proba(X)[0]))
            label = max(proba, key=proba.get)
            q = float(self.models["charge"].predict(X)[0])
            mae_g = self.metrics.get("gap", {}).get("MAE_eV")
            mae_q = self.metrics.get("metal_charge", {}).get("MAE")
            self.gap_v.set(f"{gap:.2f} eV" + (f"  (typical error ±{mae_g:.2f})" if mae_g else ""))
            self.stab_v.set(f"{label}  ({proba[label] * 100:.0f}% confidence)")
            self.q_v.set(f"{q:+.2f} e" + (f"  (typical error ±{mae_q:.2f})" if mae_q else ""))
            notes.append(f"Stability classes are gap terciles of tmQM_y: Low < {self.gap_lo:.2f} eV ≤ Moderate < {self.gap_hi:.2f} eV ≤ High.")
        else:
            for v in (self.gap_v, self.stab_v, self.q_v):
                v.set("(no model)")
            notes.append("No trained models available -- rebuild via the File menu.")

        if self.ref is not None:
            n_ox = int(((self.ref.metal == metal) & (self.ref.oxidation_state == ox)).sum())
            if n_ox < LOW_SUPPORT:
                notes.append(f"Low-support warning: only {n_ox} tmQM complexes have {metal}(ox {ox}) (the app normally "
                             f"offers states with at least {LOW_SUPPORT}). The AI gap, charge and geometry values for this "
                             "state are weakly supported -- treat them as indicative only.")
        evidence = self._evidence(metal, ox, sel)
        if "exact ligand set & counts" not in evidence:
            notes.append("Caution: no tmQM complex has exactly this metal / oxidation state / ligand set, so the AI "
                         "values are extrapolations -- trust them less than the typical errors shown.")
        notes.append(evidence)
        if lig["sites"] > 12 or lig["sites"] < 2:
            notes.append("Warning: this coordination number is unusual for a mononuclear complex.")
        notes.append("Reminder: ligand-field strengths are textbook-inspired assumptions; ML values are statistical "
                     "estimates from class-level descriptors, not first-principles results. The geometry label counts a pi-ring "
                     "(Cp, arene) as one vertex of the coordination polyhedron.")
        self._set_notes("\n\n".join(notes))

    def _confidence_hint(self, conf):
        """How often predictions with about this confidence were right on held-out tmQM complexes."""
        for row in self.geo_metrics.get("confidence_vs_accuracy", []):
            lo, hi = (float(x.strip("%")) / 100 for x in row["confidence"].split("-"))
            if lo <= conf < hi or (hi >= 1 and conf >= lo):
                return (f"On held-out tmQM complexes, predictions with {row['confidence']} confidence were correct "
                        f"{row['accuracy']:.0%} of the time (n={row['n']:,}).")
        return ""

    def _evidence(self, metal, ox, sel):
        if self.ref is None:
            return "tmQM evidence unavailable (data/tmqm_processed.csv.gz not found)."
        sub = self.ref[(self.ref.metal == metal) & (self.ref.oxidation_state == ox)]
        if sub.empty:
            return f"tmQM evidence: no {metal}(ox {ox}) complexes in the dataset."
        exact_sig = "|".join(f"{k}::{v}" for k, v in sorted(sel.items()))
        exact = sub[sub.signature == exact_sig]
        has_all = sub
        for c in sel:
            has_all = has_all[has_all.sig_wrapped.str.contains(f"|{c}|", regex=False)]

        def describe(df, title):
            g = df.hl_gap_eV
            codes = ", ".join(df.sort_values("hl_gap_eV").csd_code.iloc[[0, len(df) // 2, -1]].unique()) if len(df) > 2 else ", ".join(df.csd_code.head(3))
            geo = ""
            if "geometry_class" in df and df.geometry_class.notna().any():
                vc = df.geometry_class.value_counts(normalize=True).head(3)
                geo = " Measured geometries: " + ", ".join(f"{k} {v:.0%}" for k, v in vc.items()) + "."
            return (f"{title}: {len(df)} complexes. HOMO–LUMO gap median {g.median():.2f} eV "
                    f"(10–90%: {g.quantile(.1):.2f}–{g.quantile(.9):.2f}); metal charge median {df.metal_q.median():+.2f}."
                    f"{geo} Example CSD codes (low/mid/high gap): {codes}.")

        if len(exact):
            out = describe(exact, "tmQM evidence (exact ligand set & counts)")
        elif len(has_all):
            out = ("tmQM evidence: no complex with exactly this ligand set.\n" +
                   describe(has_all, "Closest match (contains all chosen ligand classes, any counts/extra ligands)"))
        else:
            out = ("tmQM evidence: no complex contains all chosen ligand classes together.\n" +
                   describe(sub, f"For reference, all {metal}(ox {ox}) complexes"))
        common = sub.cn_sites.value_counts().head(3)
        out += "\nMost common coordination-site totals for this metal/oxidation state: " + \
               ", ".join(f"{int(k)} ({v / len(sub):.0%})" for k, v in common.items()) + "."
        return out

    def _set_notes(self, text):
        self.notes.config(state="normal")
        self.notes.delete("1.0", tk.END)
        self.notes.insert("1.0", text)
        self.notes.config(state="disabled")


def main():
    root = tk.Tk()
    App(root)
    root.mainloop()


if __name__ == "__main__":
    main()
