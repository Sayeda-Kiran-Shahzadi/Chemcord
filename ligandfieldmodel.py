"""Ligand-field model (v3) for tmQM-derived ligand classes.

Educational crystal-field-theory (CFT) layer that sits next to the machine-
learning models: it turns a metal + oxidation state + a MIXTURE of ligands
into Delta values for four geometries, pairing energy, spin state, CFSE,
CFAE, and a kinetic-lability label.

Changes vs. v2
  * metals: all 30 metals present in tmQM_y (3d, 4d, 5d + La); d-electron
    count now comes from the periodic-table group (d = group - oxidation).
  * ligands: no more hand-written 9-ligand table. Ligand classes are the ones
    discovered in tmQM_y (see tmqm_features.py / data/dataset_meta.json) and
    each class gets a field-strength from `class_strength()` below.
  * mixed ligands: the metal's field is the site-weighted average of the
    chosen ligands (the "average environmental" rule).
  * FIXED: high-spin octahedral and tetrahedral electron counting in v2 put
    too many electrons in the lower set (e.g. d5 high-spin gave CFSE=-2.0 D
    instead of 0). See `calculate_complex`.

Changes vs. v3 (physics fixes; ML features are NOT affected because base_values()
is unchanged)
  * FIXED: the pairing energy is a COST. Net stabilisation is now
    -(CFSE + P * extra_pairs); before, pairing was subtracted from CFSE, which made
    low-spin complexes look MORE stabilised than their CFSE.
  * FIXED: "CFAE" is now the textbook crystal field ACTIVATION energy: the CFSE lost
    on going from the octahedral ground state to the dissociative (square pyramidal,
    SN1) or associative (pentagonal bipyramidal, SN2) transition state, keeping the
    spin multiplicity (Basolo & Pearson). Defined for octahedral complexes only.
    The kinetic label is now derived from it (CFAE / Delta_oct) instead of a d-count rule.
  * FIXED: orbital-energy patterns are the standard point-charge crystal-field values,
    in units of Delta_oct, and every pattern sums to zero (barycentre rule), so a
    filled d10 shell has CFSE = 0 in every geometry. The geometry "Delta" is the total
    d-orbital splitting span of the pattern (square planar 1.742, square pyramidal
    1.371, tetrahedral 4/9, octahedral 1) instead of the earlier guesses 1.30 / 1.10.
  * Pairs in the low-spin square planar / pyramidal fills are now counted from the
    actual occupancy (not d // 2).

REFIT (v5): the constants below marked "fitted" were fitted to 50 experimental octahedral Delta values
(fit/reference_delta.csv; sources in fit/reference_delta_sources.csv, incl. Figgis & Hitchman 2000), see
fit/fit_ligandfield.py and fit/fit_report.txt. Cross-validated median error: about 9% (old constants 13%).
Everything not marked "fitted" is still a textbook-inspired assumption. The "Delta" unit is arbitrary;
K_CM1_PER_UNIT converts it to cm-1 for simple M(L)6 complexes.
"""
import math


# ---------------------------------------------------------------------------
# Metals (every metal that occurs in tmQM_y)
# ---------------------------------------------------------------------------
METAL_GROUP = {
    "Sc": 3, "Ti": 4, "V": 5, "Cr": 6, "Mn": 7, "Fe": 8, "Co": 9, "Ni": 10, "Cu": 11, "Zn": 12,
    "Y": 3, "Zr": 4, "Nb": 5, "Mo": 6, "Tc": 7, "Ru": 8, "Rh": 9, "Pd": 10, "Ag": 11, "Cd": 12,
    "La": 3, "Hf": 4, "Ta": 5, "W": 6, "Re": 7, "Os": 8, "Ir": 9, "Pt": 10, "Au": 11, "Hg": 12,
}
ATOMIC_NUMBERS = {
    "Sc": 21, "Ti": 22, "V": 23, "Cr": 24, "Mn": 25, "Fe": 26, "Co": 27, "Ni": 28, "Cu": 29, "Zn": 30,
    "Y": 39, "Zr": 40, "Nb": 41, "Mo": 42, "Tc": 43, "Ru": 44, "Rh": 45, "Pd": 46, "Ag": 47, "Cd": 48,
    "La": 57, "Hf": 72, "Ta": 73, "W": 74, "Re": 75, "Os": 76, "Ir": 77, "Pt": 78, "Au": 79, "Hg": 80,
}
# Highest oxidation state treated as chemically sensible per group. The SMILES
# in tmQM_y sometimes carry larger formal metal charges (e.g. Zn+4..+8) that
# are bookkeeping artefacts of the dative-bond notation, not real oxidation states.
MAX_OXIDATION_BY_GROUP = {g: g for g in range(3, 11)}
MAX_OXIDATION_BY_GROUP.update({3: 3, 4: 4, 5: 5, 6: 6, 7: 7, 8: 8, 9: 6, 10: 6, 11: 3, 12: 2})


def max_oxidation(metal):
    return MAX_OXIDATION_BY_GROUP[METAL_GROUP[metal]]


METAL_ROW = {m: (1 if z < 31 else 2 if z < 49 else 3) for m, z in ATOMIC_NUMBERS.items()}

# --- constants refitted to experimental Delta (v5); old values: row 1.45 / 1.75, oxidation 1 + 0.35*ox, strengths below
K_CM1_PER_UNIT = 5480.6                      # fitted: cm-1 per model unit (valid for simple M(L)6 complexes)
ROW_FACTOR = {1: 1.00, 2: 1.623, 3: 1.89}      # fitted (90% bootstrap intervals: 4d 1.51-1.80, 5d 1.72-2.05)
# Oxidation-state factor = 1.7 * g(ox). g(2) = 1 by definition; g(3), g(4) fitted; ox 5+ extrapolated with a power law;
# ox 0 and 1: NO DATA except Cr(CO)6, so Delta is simply held at its +2 value (assumption).
OX_G = {2: 1.0, 3: 1.789, 4: 2.244}
_OX_POWER = math.log(OX_G[4] / OX_G[3]) / math.log(4 / 3)
# Pairing energy depends on the d-electron count (spin-crossover data, Figgis & Hitchman Section 6.9 and Table 8.5).
# d4, d5, d6 from the spin states of 19 complexes; d7 from the book's crossover (no low-spin d7 in the data).
PAIRING_BY_D = {4: 4.254, 5: 4.254, 6: 2.913, 7: 3.23}
PAIRING_OTHER_D = 3.663      # d-counts without a spin choice (only used for energy bookkeeping)
PAIRING_ROW_SCALE = {1: 1.0, 2: 2.00 / 2.45, 3: 1.60 / 2.45}   # 4d/5d scaling NOT constrained by data (old assumption kept)
# Point-charge crystal-field orbital energies in units of Delta_oct, lowest -> highest.
# Each list sums to zero (barycentre rule).
_DT = 4 / 9
GEOMETRY_ORBITALS = {
    "octahedral":       [-0.4] * 3 + [0.6] * 2,                  # t2g, eg
    "square_pyramidal": [-0.457, -0.457, -0.086, 0.086, 0.914],  # dxz,dyz | dxy | dz2 | dx2-y2
    "square_planar":    [-0.514, -0.514, -0.428, 0.228, 1.228],  # dxz,dyz | dz2 | dxy | dx2-y2
    "tetrahedral":      [-0.6 * _DT] * 2 + [0.4 * _DT] * 3,      # e, t2
}
PENTAGONAL_BIPYRAMID = [-0.528, -0.528, 0.282, 0.282, 0.492]    # associative (SN2) transition state
# "Delta" of a geometry = total d-orbital splitting span relative to Delta_oct
GEOMETRY_DELTA_FACTOR = {g: max(e) - min(e) for g, e in GEOMETRY_ORBITALS.items()}
KINETIC_CUTS = (0.08, 0.15)   # CFAE/Delta_oct: below 0.08 labile, 0.08-0.15 moderate, 0.15 or more inert
GEOMETRIES = list(GEOMETRY_DELTA_FACTOR)

# ---------------------------------------------------------------------------
# Ligand field strength per ligand class (spectrochemical-series inspired)
# ---------------------------------------------------------------------------
NAMED_STRENGTH = {
    "Carbonyl (CO)": 2.55, "Nitrosyl (NO)": 2.60, "Cyanide (CN-)": 1.863,          # CN fitted (was 2.45)
    "Carbide/carbyne (C)": 2.30, "Hydride (H-)": 2.15, "Methyl (CH3)": 2.05,
    "Phosphine/phosphite (P)": 2.10, "Phosphine/phosphide (PH3)": 2.10,
    "Isocyanide-type (N-bound CN)": 1.50, "Oxygen-bound CO": 1.00,
    "Ammonia (NH3)": 1.258, "Water (H2O)": 1.05, "Hydroxide (OH-)": 0.95,   # NH3 fitted (was 1.35); water is the anchor (fixed)
    "Oxo/oxide (O)": 0.90, "Amido/imido (N)": 1.10, "Sulfide/thiol (S)": 1.00,
    "Fluoride (F-)": 0.881, "Chloride (Cl-)": 0.766, "Bromide (Br-)": 0.676, "Iodide (I-)": 0.65,   # F, Cl, Br fitted (were 0.85, 0.78, 0.72)
    "Cyclopentadienyl-type (η5)": 2.00, "Arene-type (η6)": 1.90, "Allyl-type (η3)": 1.80,
    "Diene-type (η4)": 1.90, "Alkene/alkyne (η2)": 1.70,
    "N-donor, monodentate (amine/pyridine/imine)": 1.40,
    "O-donor, monodentate (alkoxide/ether/carbonyl)": 1.00,
    "S-donor, monodentate (thiolate/thioether)": 1.10,
    "C-donor, monodentate (alkyl/aryl/carbene)": 2.05,
    "Other ligand": 1.50,
}
DONOR_STRENGTH = {"C": 2.05, "N": 1.40, "O": 1.00, "P": 2.10, "S": 1.10, "Se": 1.00,
                  "As": 1.90, "Si": 1.90, "B": 2.00, "H": 2.15, "F": 0.881,
                  "Cl": 0.766, "Br": 0.676, "I": 0.65}
CHELATE_BONUS = 0.10  # small extra field for chelating ligands (chelate effect)


def class_strength(name, donor_elements=()):
    """Field strength of a ligand class on the same scale as v2's table."""
    if name in NAMED_STRENGTH:
        return NAMED_STRENGTH[name]
    if donor_elements:
        base = sum(DONOR_STRENGTH.get(e, 1.4) for e in donor_elements) / len(donor_elements)
        return round(base + (CHELATE_BONUS if len(donor_elements) > 1 else 0.0), 3)
    return 1.5


def class_type(name, donor_elements=()):
    if "η" in name:
        return "pi ligand"
    if name in ("Carbonyl (CO)", "Nitrosyl (NO)", "Cyanide (CN-)") or name.startswith("Phosphine"):
        return "pi acceptor"
    if any(h in name for h in ("Fluoride", "Chloride", "Bromide", "Iodide", "Hydroxide", "Oxo")):
        return "pi donor"
    return "sigma donor"


def mixed_ligand(selection, classes):
    """Combine several ligand classes into one effective ligand.

    selection: {class_name: count}; classes: {class_name: {"sites", "strength",
    "type"...}}. Returns the dict shape used by calculate_complex()."""
    tot_sites = sum(n * classes[c]["sites"] for c, n in selection.items())
    if tot_sites == 0:
        raise ValueError("No ligands selected.")
    strength = sum(n * classes[c]["sites"] * classes[c]["strength"] for c, n in selection.items()) / tot_sites
    parts = [f"{n}x {c.split(' (')[0]}" if n > 1 else c.split(" (")[0] for c, n in selection.items()]
    types = {classes[c]["type"] for c in selection}
    return {"strength": strength, "sites": tot_sites,
            "formula": " + ".join(parts), "type": types.pop() if len(types) == 1 else "mixed"}


# ---------------------------------------------------------------------------
# Core ligand-field calculations
# ---------------------------------------------------------------------------
def oxidation_factor(oxidation):
    """1.7 * g(ox): fitted for +2..+4, held flat below +2 (no data), power-law extrapolation above +4."""
    ox = max(oxidation, 0)
    if ox <= 2:
        return 1.7
    if ox in OX_G:
        return 1.7 * OX_G[ox]
    return 1.7 * OX_G[4] * (ox / 4) ** _OX_POWER


def d_electron_count(metal, oxidation):
    return max(0, min(10, METAL_GROUP[metal] - oxidation))


def free_ion_pairs(d):
    return max(0, d - 5)


def pairs(electrons, orbitals):
    return max(0, electrons - orbitals)


def base_values(metal, oxidation, ligand):
    row = METAL_ROW[metal]
    bond_strength = ligand["strength"] * ROW_FACTOR[row]
    delta_oct = bond_strength * oxidation_factor(max(oxidation, 0))
    d = d_electron_count(metal, oxidation)
    pairing = PAIRING_BY_D.get(d, PAIRING_OTHER_D) * PAIRING_ROW_SCALE[row]
    return bond_strength, delta_oct, pairing


FITTED_CLASSES = ("Water (H2O)", "Ammonia (NH3)", "Cyanide (CN-)", "Fluoride (F-)", "Chloride (Cl-)", "Bromide (Br-)")
FITTED_OXIDATION_STATES = (2, 3, 4)


def delta_confidence(class_names, oxidation):
    """One sentence on how far Delta can be trusted for this selection, based on what the refit covered."""
    inside = all(n in FITTED_CLASSES for n in class_names) and oxidation in FITTED_OXIDATION_STATES
    if inside:
        text = ("This is inside the experimental data the constants were fitted to (typical error about 9% on held-out "
                "octahedral complexes)")
        if len(class_names) > 1:
            text += "; mixing ligands uses their average strength, which was not tested against data"
        return text + "."
    return ("This is OUTSIDE the data the constants were fitted to (other ligand classes and/or oxidation state 0, 1 or 5+): "
            "an extrapolation; comparable checks outside the fitted range (Cr(CO)6, Jahn-Teller complexes) were off by about 30%.")


def to_cm1(delta_units):
    """Convert model units to cm-1 (calibrated on simple M(L)6 complexes; much less certain for mixed or low-valent ones)."""
    return delta_units * K_CM1_PER_UNIT


def all_deltas(metal, oxidation, ligand):
    _, delta_oct, _ = base_values(metal, oxidation, ligand)
    return {g: delta_oct * f for g, f in GEOMETRY_DELTA_FACTOR.items()}


def _groups(energies):
    groups = []
    for i, e in enumerate(energies):
        if groups and abs(energies[groups[-1][0]] - e) < 1e-6:
            groups[-1].append(i)
        else:
            groups.append([i])
    return groups


def _hund(n, size):
    occ = [0] * size
    for k in range(n):
        occ[k % size] += 1
    return occ


def fill_low_spin(energies, d):
    """Fill whole degenerate sets from the bottom up (Hund's rule inside a set)."""
    occ, remaining = [0] * len(energies), d
    for group in _groups(energies):
        n = min(remaining, 2 * len(group))
        for idx, o in zip(group, _hund(n, len(group))):
            occ[idx] = o
        remaining -= n
        if remaining == 0:
            break
    return occ


def fill_high_spin(energies, d):
    """One electron in every orbital first (lowest first), then pair up."""
    occ = [0] * len(energies)
    for k in range(d):
        occ[k % len(energies)] += 1
    return occ


def fill_fixed_unpaired(energies, d, unpaired):
    """Lowest-energy occupancy with a fixed number of unpaired electrons
    (a transition state keeps the spin multiplicity of the ground state)."""
    pairs_n = (d - unpaired) // 2
    occ = [0] * len(energies)
    for i in range(pairs_n):
        occ[i] = 2
    for i in range(pairs_n, pairs_n + unpaired):
        occ[i] = 1
    return occ


def _energy(occ, energies):
    return sum(o * e for o, e in zip(occ, energies))


def calculate_complex(metal, oxidation, geometry, ligand, spin="auto"):
    """Ligand-field quantities for one geometry (all energies in the model's Delta units).

    Returns cfse (negative = stabilised), net_stabilization (= -(CFSE + P * extra pairs),
    positive = stabilised), cfae_sn1 / cfae_sn2 / cfae (activation energies, octahedral only,
    else None) and a kinetic label derived from cfae."""
    d = d_electron_count(metal, oxidation)
    row = METAL_ROW[metal]
    bond_strength, delta_oct, pairing_energy = base_values(metal, oxidation, ligand)
    delta_geom = delta_oct * GEOMETRY_DELTA_FACTOR[geometry]
    energies = GEOMETRY_ORBITALS[geometry]

    if geometry == "octahedral":
        if spin == "auto":
            spin = "low" if delta_oct > pairing_energy else "high"
        occ = fill_low_spin(energies, d) if spin == "low" else fill_high_spin(energies, d)
        desc = f"Octahedral {spin}-spin occupancy: t2g={sum(occ[:3])}, eg={sum(occ[3:])}."
    elif geometry == "tetrahedral":
        spin = "high"
        occ = fill_high_spin(energies, d)
        desc = f"Tetrahedral high-spin occupancy: e={sum(occ[:2])}, t2={sum(occ[2:])}."
    else:  # square pyramidal / square planar: strong-field (low-spin) filling
        spin = "low"
        occ = fill_low_spin(energies, d)
        desc = f"{geometry.replace('_', ' ').capitalize()}: orbital occupancy (low to high) = {occ}."

    cfse = _energy(occ, energies) * delta_oct          # patterns are in units of Delta_oct
    extra_pairs = max(0, sum(1 for o in occ if o == 2) - free_ion_pairs(d))
    net_stabilization = -(cfse + pairing_energy * extra_pairs)

    cfae_sn1 = cfae_sn2 = cfae = None
    if geometry == "octahedral":
        unpaired = sum(1 for o in occ if o == 1)
        ground = _energy(occ, energies)
        sp, pb = GEOMETRY_ORBITALS["square_pyramidal"], PENTAGONAL_BIPYRAMID
        cfae_sn1 = (_energy(fill_fixed_unpaired(sp, d, unpaired), sp) - ground) * delta_oct
        cfae_sn2 = (_energy(fill_fixed_unpaired(pb, d, unpaired), pb) - ground) * delta_oct
        cfae = min(cfae_sn1, cfae_sn2)

    return {"d_count": d, "spin": spin, "cfse": cfse, "net_stabilization": net_stabilization,
            "extra_pairs": extra_pairs, "cfae": cfae, "cfae_sn1": cfae_sn1, "cfae_sn2": cfae_sn2,
            "kinetic": classify_kinetic_lability(d, spin, geometry, cfae=cfae, delta_oct=delta_oct, row=row),
            "delta_oct": delta_oct, "delta_geom": delta_geom,
            "bond_strength": bond_strength, "pairing_energy": pairing_energy,
            "ligand_formula": ligand["formula"], "description": desc}


def geometry_from_coordination(d_count, avg_strength, sites, row):
    """LEGACY rule: geometry from the coordination number (sum of coordination
    sites of the chosen ligands) plus the textbook d-electron rules.
    The app now uses the geometry classifier (train_geometry.py) instead; this
    rule is kept as a fallback when no geometry model is installed and as the
    baseline that train_geometry.py compares the classifier against.
    Returns (geometry or None, note)."""
    if sites <= 3:
        return None, f"{sites}-coordinate (linear/trigonal): outside the four geometries modelled here."
    if sites == 4:
        if d_count == 8 and (avg_strength >= 1.8 or row >= 2):
            return "square_planar", "4-coordinate d8 with a strong field / 4d-5d metal -> square planar."
        return "tetrahedral", "4-coordinate -> tetrahedral (square planar only for strong-field d8)."
    if sites == 5:
        return "square_pyramidal", "5-coordinate -> square pyramidal (trigonal bipyramidal not modelled)."
    if sites == 6:
        return "octahedral", "6-coordinate -> octahedral."
    return None, f"{sites}-coordinate (>6): outside the four geometries modelled here."


def classify_kinetic_lability(d_count, spin, geometry, cfae=None, delta_oct=None, row=None):
    """Inert / Moderate / Labile.

    Octahedral with a CFAE: ratio = CFAE / Delta_oct (>= 0.15 inert, 0.08-0.15 moderate, below labile).
    Octahedral d8 is capped at Moderate: CFAE theory over-predicts its inertness (e.g. Ni2+ aqua
    exchanges water at about 1e4 per second). Square-planar d8 follows the row (3d labile, 4d moderate,
    5d inert: Ni(II) substitutes fast, Pt(II) slowly). Tetrahedral / 5-coordinate: labile.
    Without cfae/delta_oct the old d-electron-count rule is used (kept for backward compatibility)."""
    if geometry == "octahedral":
        if cfae is not None and delta_oct:
            ratio = cfae / delta_oct
            label = "Inert" if ratio >= KINETIC_CUTS[1] else "Moderate" if ratio >= KINETIC_CUTS[0] else "Labile"
            return "Moderate" if (d_count == 8 and label == "Inert") else label
        if d_count == 3 or (spin == "low" and d_count in (4, 5, 6)):
            return "Inert"
        return "Labile"
    if geometry == "square_planar" and d_count == 8:
        return {1: "Labile", 2: "Moderate", 3: "Inert"}.get(row, "Moderate")
    return "Labile"
