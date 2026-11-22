"""log K1 prediction: how strongly does a metal ion bind a ligand in water?
==========================================================================

This module is shared by the trainer (train_logk.py) and the app (logk_ui.py),
so the features used for training and for prediction are produced by the SAME
code and cannot drift apart.

Model: gradient boosting on
  * metal-ion descriptors (atomic number, density, polarisability, radii,
    electronegativity, ionisation energy, valence electrons, charge),
  * experimental conditions (temperature, electrolyte concentration, electrolyte),
  * ligand descriptors computed with RDKit from the ligand SMILES
    (Morgan count fingerprint, 16 physico-chemical descriptors, element counts).

Data: the training/validation/test tables published with
  Karunaratne, Zahariev, Perez Garcia, J. Chem. Inf. Model. 2025, 65, 11532-11542
  (doi:10.1021/acs.jcim.5c01546), which are extracted from the IUPAC Stability
  Constants Database. Please cite that paper if you use this tool.

What the model can and cannot do (see README): it predicts the FIRST stability
constant (log K1) of one metal ion with one ligand in aqueous solution. It does
not know which ligand atoms bind the metal, pH, or solvent other than the listed
electrolytes, and it is much less accurate for ligands that are not in the
training data.
"""
import math
from pathlib import Path

import numpy as np
import pandas as pd

try:
    from rdkit import Chem, DataStructs, RDLogger
    from rdkit.Chem import Descriptors, rdFingerprintGenerator

    RDLogger.DisableLog("rdApp.*")
    HAVE_RDKIT = True
except ImportError:  # the app still starts without RDKit; the log K tool then explains what to install
    HAVE_RDKIT = False

try:
    import joblib
except ImportError:  # pragma: no cover
    import pickle

    class joblib:  # minimal stand-in
        @staticmethod
        def load(path):
            with open(path, "rb") as f:
                return pickle.load(f)

        @staticmethod
        def dump(obj, path, **_):
            with open(path, "wb") as f:
                pickle.dump(obj, f)

# ---------------------------------------------------------------------------
# Feature definitions (order matters: the trained model depends on it)
# ---------------------------------------------------------------------------
METAL_FEATURES = [
    "Atomic_number", "metal_density", "metal_dipole_polarizability", "metal_vdw_radius",
    "metal_atomic_weight", "metal_covalent_radius_by_pyykko", "metal_Sanderson Electronegativity",
    "IE", "metal_valence_e",
]
CONDITION_COLUMNS = ["Temp", "Concentration of Medium", "Exact MW of solvent (g/mol)", "Density of solvent g/cm-3"]
FP_BITS = 512
DESCRIPTOR_NAMES = [
    "MolWt", "HeavyAtomCount", "NumHDonors", "NumHAcceptors", "NumRotatableBonds", "RingCount",
    "NumAromaticRings", "TPSA", "MolLogP", "MolMR", "FractionCSP3", "NumHeteroatoms", "NHOHCount",
    "NOCount", "LabuteASA", "BertzCT",
]
COUNTED_ELEMENTS = ("N", "O", "S", "P", "F", "Cl")
LIGAND_DIM = FP_BITS + len(DESCRIPTOR_NAMES) + len(COUNTED_ELEMENTS)
N_METAL = len(METAL_FEATURES) + 1          # + metal charge
N_FEATURES = N_METAL + len(CONDITION_COLUMNS) + LIGAND_DIM

# Electrolyte (medium) names for the (exact mass, density) pairs found in the data.
ELECTROLYTE_MASSES = {"KNO3": 100.9515, "NaClO4": 121.9383, "KCl": 73.9326, "NaNO3": 84.9776,
                      "Me4NCl": 109.0658, "NaCl": 57.9586}
DEFAULT_CONDITIONS = {"temp": 25.0, "conc": 0.1, "electrolyte": "KNO3"}

# Ligands in the form the training data uses: organic ligands neutral, simple inorganic anions as anions.
COMMON_LIGANDS = {
    "Ammonia": "N", "Ethylenediamine (en)": "NCCN", "Diethylenetriamine (dien)": "NCCNCCN",
    "Triethylenetetramine (trien)": "NCCNCCNCCN", "Pyridine": "c1ccncc1", "Imidazole": "c1c[nH]cn1",
    "2,2'-Bipyridine": "c1ccc(-c2ccccn2)nc1", "1,10-Phenanthroline": "c1cnc2c(c1)ccc1cccnc12",
    "Acetic acid (acetate)": "CC(=O)O", "Oxalic acid (oxalate)": "OC(=O)C(=O)O",
    "Citric acid (citrate)": "OC(=O)CC(O)(CC(O)=O)C(O)=O", "Salicylic acid": "OC(=O)c1ccccc1O",
    "Glycine": "NCC(=O)O", "Iminodiacetic acid (IDA)": "OC(=O)CNCC(O)=O",
    "Nitrilotriacetic acid (NTA)": "OC(=O)CN(CC(O)=O)CC(O)=O",
    "EDTA": "OC(=O)CN(CCN(CC(O)=O)CC(O)=O)CC(O)=O", "Acetylacetone": "CC(=O)CC(C)=O",
    "8-Hydroxyquinoline": "Oc1cccc2cccnc12", "Chloride": "[Cl-]", "Fluoride": "[F-]",
    "Hydroxide": "[OH-]", "Cyanide": "[C-]#N", "Thiocyanate": "[S-]C#N",
}
# Ligand classes of the geometry app that have an unambiguous aqueous-ligand SMILES.
CLASS_TO_SMILES = {"Chloride (Cl-)": "[Cl-]", "Cyanide (CN-)": "[C-]#N", "Ammonia (NH3)": "N",
                   "Fluoride (F-)": "[F-]", "Hydroxide (OH-)": "[OH-]"}

STRENGTH_CLASSES = [(3.0, "Weak"), (8.0, "Moderate"), (15.0, "Strong"), (math.inf, "Very strong")]
R_KJ = 8.314462618e-3

_FP_GEN = None


def _need_rdkit():
    if not HAVE_RDKIT:
        raise RuntimeError("RDKit is not installed. Install it with:  pip install rdkit")


def strength_class(logk):
    """Simple orientation cut-offs (not literature standards): <3 weak, 3-8 moderate, 8-15 strong, >=15 very strong."""
    for limit, name in STRENGTH_CLASSES:
        if logk < limit:
            return name


def delta_g_kj(logk, temp_c=25.0):
    """Standard Gibbs energy of binding, dG = -RT ln(10) logK, in kJ/mol."""
    return -math.log(10) * R_KJ * (temp_c + 273.15) * logk


# ---------------------------------------------------------------------------
# Ligand handling
# ---------------------------------------------------------------------------
def canonical(smiles):
    """RDKit canonical SMILES, or None if the SMILES cannot be read."""
    mol = Chem.MolFromSmiles(smiles) if smiles else None
    return Chem.MolToSmiles(mol) if mol is not None else None


def parse_ligand(smiles, allowed_elements=None):
    """Validate a user SMILES. Returns (mol, canonical_smiles); raises ValueError with a readable message."""
    _need_rdkit()
    s = (smiles or "").strip()
    if not s:
        raise ValueError("Enter a ligand SMILES (or choose a common ligand).")
    mol = Chem.MolFromSmiles(s)
    if mol is None:
        raise ValueError("RDKit could not read that SMILES. Check brackets, ring numbers and charges.")
    mol = Chem.MolFromSmiles(Chem.MolToSmiles(mol))
    elements = {a.GetSymbol() for a in mol.GetAtoms()}
    if allowed_elements is not None:
        unknown = sorted(elements - set(allowed_elements))
        if unknown:
            raise ValueError("The ligand contains element(s) never seen in the training data: "
                             f"{', '.join(unknown)}. Enter only the ligand here; the metal ion is chosen separately.")
    return mol, Chem.MolToSmiles(mol)


def ligand_vector(mol):
    """534 numbers describing a ligand (used identically in training and prediction)."""
    global _FP_GEN
    if _FP_GEN is None:
        _FP_GEN = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=FP_BITS)
    fp = _FP_GEN.GetCountFingerprintAsNumPy(mol).astype(np.float32)
    desc = np.array([getattr(Descriptors, n)(mol) for n in DESCRIPTOR_NAMES], dtype=np.float32)
    symbols = [a.GetSymbol() for a in mol.GetAtoms()]
    counts = np.array([symbols.count(e) for e in COUNTED_ELEMENTS], dtype=np.float32)
    return np.nan_to_num(np.clip(np.concatenate([fp, desc, counts]), -1e6, 1e6))


def ion_key(symbol, charge):
    return f"{symbol}{charge:+d}"


def ion_label(symbol, charge):
    return f"{symbol}{'' if charge == 1 else charge}+" if charge > 0 else f"{symbol}{abs(charge) if charge < -1 else ''}-"


def condition_vector(temp, conc, electrolyte, electrolytes, use_conditions=True):
    if not use_conditions:
        return np.full(4, np.nan, dtype=np.float32)
    mw, dens = electrolytes[electrolyte]
    return np.array([temp, conc, mw, dens], dtype=np.float32)


# ---------------------------------------------------------------------------
# Predictor
# ---------------------------------------------------------------------------
class LogKPredictor:
    """Loads models/logk_bundle.joblib (made by train_logk.py)."""

    def __init__(self, bundle_path):
        _need_rdkit()
        b = joblib.load(bundle_path)
        self.model = b["model"]
        self.metal_table = b["metal_table"]          # ion_key -> [10 metal numbers]
        self.ions = b["ions"]                        # list of dicts (key, symbol, charge, label, n_rows)
        self.electrolytes = b["electrolytes"]        # name -> (exact mass, density)
        self.metrics = b["metrics"]
        self.ligand_elements = set(b["ligand_elements"])
        self.cond_range = b["condition_range"]
        # Canonicalise the stored training ligands with the LOCAL RDKit so matching works across RDKit versions.
        raw = b["ligand_raw"]
        canon = [canonical(s) for s in raw]
        self._canon_of_raw = dict(zip(raw, canon))
        self.known = {c for c in canon if c}
        lk = b["lookup"].copy()
        lk["canon"] = lk.ligand_raw.map(self._canon_of_raw)
        self.lookup = {(r.canon, r.ion): (int(r.n), float(r.median), float(r.min), float(r.max)) for r in lk.itertuples()}
        self._fps, self._fp_smiles = [], []
        for c in sorted(self.known):
            self._fps.append(self._bitfp(Chem.MolFromSmiles(c)))
            self._fp_smiles.append(c)

    @staticmethod
    def _bitfp(mol):
        return rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048).GetFingerprint(mol)

    def ion_choices(self):
        return [i for i in self.ions if i["symbol"] != "H"]

    def _matrix(self, keys, lig_vec, temp, conc, electrolyte, use_conditions):
        cond = condition_vector(temp, conc, electrolyte, self.electrolytes, use_conditions)
        return np.vstack([np.concatenate([self.metal_table[k], cond, lig_vec]) for k in keys]).astype(np.float32)

    def nearest(self, mol):
        sims = DataStructs.BulkTanimotoSimilarity(self._bitfp(mol), self._fps)
        i = int(np.argmax(sims))
        return self._fp_smiles[i], float(sims[i])

    def predict(self, key, smiles, temp=25.0, conc=0.1, electrolyte="KNO3", use_conditions=True):
        mol, canon = parse_ligand(smiles, self.ligand_elements)
        lig = ligand_vector(mol)
        logk = float(self.model.predict(self._matrix([key], lig, temp, conc, electrolyte, use_conditions))[0])
        measured = self.lookup.get((canon, key))
        if measured:
            reliability, err = "Measured pair", self.metrics["error"]["exact_complex"]
        elif canon in self.known:
            reliability, err = "Known ligand", self.metrics["error"]["known_ligand"]
        else:
            reliability, err = "New ligand", self.metrics["error"]["new_ligand"]
        near_smiles, sim = self.nearest(mol)
        warnings = []
        if use_conditions:
            lo, hi = self.cond_range["Temp"]
            if not lo <= temp <= hi:
                warnings.append(f"Temperature {temp:g} C is outside the training range ({lo:g}-{hi:g} C).")
            lo, hi = self.cond_range["Concentration of Medium"]
            if not lo <= conc <= hi:
                warnings.append(f"Electrolyte concentration {conc:g} M is outside the training range ({lo:g}-{hi:g} M).")
        t = temp if use_conditions else 25.0
        return {"logk": logk, "dg": delta_g_kj(logk, t), "strength": strength_class(logk), "canon": canon,
                "reliability": reliability, "typical_error": err, "measured": measured,
                "nearest": near_smiles, "similarity": sim, "warnings": warnings, "temp_used": t}

    def compare(self, smiles, temp=25.0, conc=0.1, electrolyte="KNO3", use_conditions=True):
        """Predicted log K1 for every metal ion, plus measured medians where the pair is in the training data."""
        mol, canon = parse_ligand(smiles, self.ligand_elements)
        ions = self.ion_choices()
        X = self._matrix([i["key"] for i in ions], ligand_vector(mol), temp, conc, electrolyte, use_conditions)
        pred = self.model.predict(X)
        rows = []
        for i, p in zip(ions, pred):
            m = self.lookup.get((canon, i["key"]))
            rows.append({"ion": i["label"], "key": i["key"], "predicted": float(p),
                         "measured_median": m[1] if m else np.nan, "n_measured": m[0] if m else 0,
                         "ion_training_rows": i["n_rows"]})
        return pd.DataFrame(rows).sort_values("predicted", ascending=False).reset_index(drop=True)
