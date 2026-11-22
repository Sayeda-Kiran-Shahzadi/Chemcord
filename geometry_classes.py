"""Shared geometry vocabulary: how raw structural labels from label_geometry.py are
grouped into the classes the classifier learns, and which of them the ligand-field
model can calculate."""

# raw label (label_geometry.py)  ->  class the classifier is trained on
# labels not listed here are used as they are; labels mapped to None are dropped
# (structure-file artefacts: no donor atoms found, or no single metal atom).
RAW_TO_CLASS = {
    "irregular (CN=3)": "irregular / distorted",
    "irregular (CN=4)": "irregular / distorted",
    "irregular (CN=5)": "irregular / distorted",
    "irregular (CN=6)": "irregular / distorted",
    "trigonal pyramidal": "other (rare)",
    "capped octahedral": "other (rare)",
    "cubic": "other (rare)",
    "high coordination (CN>=9)": "other (rare)",
    "monocoordinate": None,
    "no donors": None,
    "no single metal": None,
}

# classifier class -> key understood by ligandfieldmodel.calculate_complex()
LIGAND_FIELD_KEY = {
    "octahedral": "octahedral",
    "tetrahedral": "tetrahedral",
    "square planar": "square_planar",
    "square pyramidal": "square_pyramidal",
}

# one-line descriptions shown in the app
DESCRIPTIONS = {
    "linear": "2-coordinate, L-M-L close to 180 deg",
    "bent": "2-coordinate, L-M-L well below 180 deg",
    "trigonal planar": "3-coordinate, ligands in one plane ~120 deg apart",
    "T-shaped": "3-coordinate, one ~180 deg and two ~90 deg angles",
    "tetrahedral": "4-coordinate, all angles ~109 deg",
    "square planar": "4-coordinate, ~90 and ~180 deg angles",
    "seesaw": "4-coordinate, one ~180, one ~120 and four ~90 deg angles",
    "piano stool": "pi-ring on top of three legs (e.g. CpM(CO)3, arene-M(CO)3)",
    "square pyramidal": "5-coordinate, one apical + four basal ligands",
    "trigonal bipyramidal": "5-coordinate, two axial (~180 deg) + three equatorial (~120 deg)",
    "octahedral": "6-coordinate, 90 and 180 deg angles",
    "trigonal prismatic": "6-coordinate, two eclipsed triangles",
    "pentagonal bipyramidal": "7-coordinate, five equatorial + two axial",
    "square antiprismatic": "8-coordinate, two staggered squares",
    "irregular / distorted": "does not match any ideal polyhedron within tolerance",
    "other (rare)": "rare shapes: trigonal pyramidal, capped octahedral, cubic, CN>=9",
}


def to_class(raw):
    return RAW_TO_CLASS.get(raw, raw)
