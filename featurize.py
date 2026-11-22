"""One place that turns (metal, oxidation state, {ligand class: count}) into
the feature vector used by the ML models -- shared by train_model.py (real
tmQM complexes) and the app (user selection), so both sides always agree."""
from ligandfieldmodel import (METAL_GROUP, ATOMIC_NUMBERS, METAL_ROW, d_electron_count,
                              base_values, mixed_ligand)

DONOR_GROUPS = ["C", "N", "O", "P", "S", "Hal", "Other"]
OTHER = "Other ligand"


def donor_group(el):
    if el in ("F", "Cl", "Br", "I"):
        return "Hal"
    return el if el in ("C", "N", "O", "P", "S") else "Other"


def feature_row(metal, ox, counts, classes, vocab):
    """counts: {class_name: n}. classes: metadata dict (sites, strength,
    type, donors={element: n}). vocab: ordered list of class names."""
    lig = mixed_ligand(counts, classes)
    _, delta_oct, pairing = base_values(metal, ox, lig)
    row = {
        "row": METAL_ROW[metal], "group": METAL_GROUP[metal], "atomic_number": ATOMIC_NUMBERS[metal],
        "oxidation_state": ox, "d_electron_count": d_electron_count(metal, ox),
        "cn_sites": lig["sites"], "n_ligands": sum(counts.values()),
        "mean_field_strength": lig["strength"], "delta_octahedral": delta_oct,
        "pairing_energy": pairing,
    }
    dg = {g: 0.0 for g in DONOR_GROUPS}
    for c, n in counts.items():
        for el, k in classes[c]["donors"].items():
            dg[donor_group(el)] += n * k
    for g in DONOR_GROUPS:
        row[f"donor_{g}"] = dg[g]
    for c in vocab:
        row[f"cls::{c}"] = counts.get(c, 0)
    return row
