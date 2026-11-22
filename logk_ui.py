"""Tkinter windows for the log K1 tool (opened from the main Chemcord window).

HOW TO USE
  1. Choose the metal ion (charge included, e.g. Fe3+).
  2. Choose a common ligand or type a ligand SMILES. Write organic ligands in their NEUTRAL form
     (acetic acid CC(=O)O, glycine NCC(=O)O, EDTA as the neutral acid) and simple inorganic anions
     as ions ([Cl-], [F-], [C-]#N) -- this is how the training data is written.
  3. Set temperature, electrolyte concentration and electrolyte, or tick "Ignore conditions".
  4. "Predict log K1" gives the 1:1 binding constant in water, the binding free energy and how far
     to trust it. "Compare all metal ions" ranks every metal ion for the same ligand (selectivity).
"""
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

import logk

NOTES = ("log K1 is the equilibrium constant of M + L <-> ML in water: larger means stronger binding. "
         "The model does not know which ligand atoms bind the metal, pH, or solvents other than the listed "
         "electrolytes, so treat results as screening estimates. Where the exact metal/ligand pair was "
         "measured, the measured values are usually more trustworthy than the prediction.")
CITATION = ("Model trained on data from Karunaratne, Zahariev & Perez Garcia, J. Chem. Inf. Model. 2025, 65, "
            "11532 (doi:10.1021/acs.jcim.5c01546), extracted from the IUPAC Stability Constants Database.")


class LogKWindow(tk.Toplevel):
    def __init__(self, master, predictor, ion_key=None, smiles=None):
        super().__init__(master)
        self.P = predictor
        self.title("Binding strength (log K\u2081)")
        self.geometry("1020x720")
        self.minsize(900, 640)

        self.ions = predictor.ion_choices()
        self.ion_labels = [f"{i['label']}   ({i['symbol']}, {i['charge']:+d})" for i in self.ions]
        self.key_of_label = dict(zip(self.ion_labels, (i["key"] for i in self.ions)))
        self.label_of_key = {v: k for k, v in self.key_of_label.items()}

        top = ttk.Frame(self, padding=(12, 10))
        top.pack(fill=tk.X)
        ttk.Label(top, text=NOTES, wraplength=980, foreground="#333333").pack(anchor="w")

        body = ttk.Frame(self, padding=(12, 0))
        body.pack(fill=tk.BOTH, expand=True)
        left = ttk.Frame(body)
        left.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 14))
        right = ttk.Frame(body)
        right.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        # ---- inputs -------------------------------------------------------------------------------
        ttk.Label(left, text="Metal ion", font=("", 10, "bold")).pack(anchor="w")
        default_ion = ion_key if ion_key in self.label_of_key else "Cu+2"
        self.ion_var = tk.StringVar(value=self.label_of_key.get(default_ion, self.ion_labels[0]))
        ttk.Combobox(left, textvariable=self.ion_var, values=self.ion_labels, state="readonly", width=30,
                     height=18).pack(anchor="w", pady=(0, 8))

        ttk.Label(left, text="Common ligand", font=("", 10, "bold")).pack(anchor="w")
        self.common_var = tk.StringVar()
        cb = ttk.Combobox(left, textvariable=self.common_var, values=list(logk.COMMON_LIGANDS), state="readonly",
                          width=30, height=18)
        cb.pack(anchor="w", pady=(0, 8))
        cb.bind("<<ComboboxSelected>>", self._on_common)

        ttk.Label(left, text="Ligand SMILES", font=("", 10, "bold")).pack(anchor="w")
        self.smiles_var = tk.StringVar(value=smiles or logk.COMMON_LIGANDS["Ethylenediamine (en)"])
        ttk.Entry(left, textvariable=self.smiles_var, width=33).pack(anchor="w")
        self.smiles_var.trace_add("write", self._on_smiles_edit)   # typing a custom SMILES clears the preset name
        ttk.Label(left, wraplength=250, foreground="gray",
                  text="Organic ligands in neutral form (CC(=O)O); simple inorganic anions as ions ([Cl-]).").pack(
            anchor="w", pady=(2, 8))

        cond = ttk.LabelFrame(left, text="Conditions", padding=8)
        cond.pack(fill=tk.X, pady=(0, 8))
        self.temp_var = tk.StringVar(value="25")
        self.conc_var = tk.StringVar(value="0.1")
        self.elec_var = tk.StringVar(value=logk.DEFAULT_CONDITIONS["electrolyte"])
        self.ignore_var = tk.BooleanVar(value=False)
        ttk.Label(cond, text="Temperature (\u00b0C):").grid(row=0, column=0, sticky="w", pady=2)
        ttk.Entry(cond, textvariable=self.temp_var, width=8).grid(row=0, column=1, sticky="w", padx=6)
        ttk.Label(cond, text="Electrolyte conc. (M):").grid(row=1, column=0, sticky="w", pady=2)
        ttk.Entry(cond, textvariable=self.conc_var, width=8).grid(row=1, column=1, sticky="w", padx=6)
        ttk.Label(cond, text="Electrolyte:").grid(row=2, column=0, sticky="w", pady=2)
        ttk.Combobox(cond, textvariable=self.elec_var, values=list(self.P.electrolytes), state="readonly",
                     width=8).grid(row=2, column=1, sticky="w", padx=6)
        ttk.Checkbutton(cond, text="Ignore conditions (SMILES-only)", variable=self.ignore_var).grid(
            row=3, column=0, columnspan=2, sticky="w", pady=(6, 0))

        ttk.Button(left, text="Predict log K\u2081", command=self.predict).pack(fill=tk.X, pady=(4, 6))
        ttk.Button(left, text="Compare all metal ions...", command=self.compare).pack(fill=tk.X, pady=(0, 6))
        ttk.Button(left, text="Close", command=self.destroy).pack(fill=tk.X)

        # ---- results ------------------------------------------------------------------------------
        res = ttk.LabelFrame(right, text="Result", padding=10)
        res.pack(fill=tk.X)
        self.v = {k: tk.StringVar(value="--") for k in ("logk", "dg", "strength", "rel", "meas", "near")}
        rows = [("logk", "Predicted log K\u2081:"), ("dg", "Binding free energy \u0394G\u00b0:"),
                ("strength", "Binding strength:"), ("rel", "How far to trust it:"),
                ("meas", "Measured in training data:"), ("near", "Closest training ligand:")]
        for r, (k, label) in enumerate(rows):
            ttk.Label(res, text=label).grid(row=r, column=0, sticky="nw", pady=3)
            font = ("", 14, "bold") if k == "logk" else ("", 10, "bold")
            ttk.Label(res, textvariable=self.v[k], font=font, wraplength=380, justify="left").grid(
                row=r, column=1, sticky="w", padx=10, pady=3)
        notes = ttk.LabelFrame(right, text="Notes", padding=8)
        notes.pack(fill=tk.BOTH, expand=True, pady=(10, 0))
        self.notes = tk.Text(notes, height=9, wrap="word", state="disabled")
        self.notes.pack(fill=tk.BOTH, expand=True)
        ttk.Label(self, text=CITATION, wraplength=980, foreground="gray").pack(anchor="w", padx=12, pady=(6, 10))

    # -------------------------------------------------------------------------------- helpers
    def _on_common(self, _event=None):
        self.smiles_var.set(logk.COMMON_LIGANDS[self.common_var.get()])

    def _on_smiles_edit(self, *_):
        name = self.common_var.get()
        if name and logk.COMMON_LIGANDS.get(name) != self.smiles_var.get():
            self.common_var.set("")

    def _conditions(self):
        use = not self.ignore_var.get()
        if not use:
            return 25.0, 0.1, self.elec_var.get(), False
        try:
            temp, conc = float(self.temp_var.get()), float(self.conc_var.get())
        except ValueError:
            raise ValueError("Temperature and concentration must be numbers (e.g. 25 and 0.1).")
        if conc < 0:
            raise ValueError("Concentration cannot be negative.")
        return temp, conc, self.elec_var.get(), True

    def _set_notes(self, text):
        self.notes.config(state="normal")
        self.notes.delete("1.0", tk.END)
        self.notes.insert("1.0", text)
        self.notes.config(state="disabled")

    def _key(self):
        return self.key_of_label[self.ion_var.get()]

    # -------------------------------------------------------------------------------- actions
    def predict(self):
        try:
            temp, conc, elec, use = self._conditions()
            r = self.P.predict(self._key(), self.smiles_var.get(), temp, conc, elec, use)
        except (ValueError, RuntimeError) as exc:
            messagebox.showwarning("Cannot predict", str(exc), parent=self)
            return
        ion = next(i for i in self.ions if i["key"] == self._key())
        err = r["typical_error"]
        self.v["logk"].set(f"{r['logk']:.2f}")
        self.v["dg"].set(f"{r['dg']:.1f} kJ/mol  (at {r['temp_used']:g} \u00b0C)")
        self.v["strength"].set(r["strength"])
        text = {"Measured pair": "this exact metal/ligand pair is in the training data",
                "Known ligand": "ligand is in the training data, but not with this metal ion",
                "New ligand": "ligand is NOT in the training data -- rough estimate"}[r["reliability"]]
        self.v["rel"].set(f"{r['reliability']}: {text}. Typical error about \u00b1{err:.1f} log units.")
        m = r["measured"]
        if m:
            n, med, lo, hi = m
            self.v["meas"].set(f"n = {n}; median {med:.2f} (range {lo:.2f} to {hi:.2f}); "
                               f"prediction differs by {r['logk'] - med:+.2f}")
        else:
            self.v["meas"].set("none for this pair")
        self.v["near"].set(f"{r['nearest']}   (Tanimoto similarity {r['similarity']:.2f})")

        notes = []
        if m:
            notes.append("Experimental values exist for this pair; prefer them over the prediction. Their spread "
                         "reflects different temperatures, media and laboratories.")
        if r["reliability"] == "New ligand":
            notes.append("This ligand is not in the training data, where the typical error is largest.")
        if r["similarity"] < 0.4:
            notes.append("The ligand is structurally very different from anything in the training data "
                         f"(best similarity {r['similarity']:.2f}); the prediction is an extrapolation.")
        if ion["charge"] >= 4:
            notes.append(f"{ion['label']} has charge +4 or higher; such ions are rare in the training data "
                         "(about 4%) and predictions for them are less reliable.")
        if ion["symbol"] in ("Fe",) and ion["charge"] == 3:
            notes.append("Fe3+ hydrolyses easily in water; real binding depends strongly on pH.")
        notes.extend(r["warnings"])
        notes.append("Binding-strength classes are simple orientation cut-offs (log K1 < 3 weak, 3-8 moderate, "
                     "8-15 strong, 15 or more very strong), not literature standards.")
        notes.append("Typical errors are mean absolute errors measured on held-out data of each kind "
                     f"(see models/logk_bundle.joblib metrics); the paper's best deep-learning model reports "
                     f"MAE {self.P.metrics['paper_best_model_external']['mae']} on its external set, this lighter "
                     f"model {self.P.metrics['external']['mae']:.2f}.")
        self._set_notes("\n\n".join(notes))

    def compare(self):
        try:
            temp, conc, elec, use = self._conditions()
            df = self.P.compare(self.smiles_var.get(), temp, conc, elec, use)
        except (ValueError, RuntimeError) as exc:
            messagebox.showwarning("Cannot compare", str(exc), parent=self)
            return
        win = tk.Toplevel(self)
        win.title("Metal ion selectivity")
        win.geometry("640x680")
        ttk.Label(win, padding=8, wraplength=600,
                  text=f"Predicted log K\u2081 of {self.smiles_var.get()} with each metal ion, strongest first. "
                       "Measured medians are shown where the pair is in the training data. The last column is how many training "
                       "rows exist for that metal ion in total: ions with few rows tend to be predicted less reliably "
                       "(individual predictions can still be off by several log units, and different "
                       "rare ions can get near-identical values).").pack(anchor="w")
        def save():
            path = filedialog.asksaveasfilename(parent=win, defaultextension=".csv", filetypes=[("CSV", "*.csv")])
            if path:
                df.drop(columns=["key"]).to_csv(path, index=False)

        ttk.Button(win, text="Save as CSV...", command=save).pack(side=tk.BOTTOM, pady=6)   # packed first so it stays visible
        cols = ("ion", "pred", "meas", "n", "rows")
        tree = ttk.Treeview(win, columns=cols, show="headings")
        for c, h, w in zip(cols, ("Metal ion", "Predicted log K\u2081", "Measured median", "n", "Ion training rows"), (90, 120, 120, 40, 110)):
            tree.heading(c, text=h)
            tree.column(c, width=w, anchor="center")
        for r in df.itertuples():
            tree.insert("", tk.END, values=(r.ion, f"{r.predicted:.2f}",
                                            "" if r.n_measured == 0 else f"{r.measured_median:.2f}",
                                            "" if r.n_measured == 0 else r.n_measured, r.ion_training_rows))
        sb = ttk.Scrollbar(win, orient="vertical", command=tree.yview)
        tree.configure(yscrollcommand=sb.set)
        sb.pack(side=tk.RIGHT, fill=tk.Y, pady=(0, 8))
        tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(8, 0), pady=(0, 8))
