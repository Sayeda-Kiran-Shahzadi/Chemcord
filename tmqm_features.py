"""Turn a tmQM_y dative-bond SMILES into: metal, formal metal charge and a
list of ligands, each classified into a human-readable ligand class."""
from collections import Counter, defaultdict

from smiles_graph import parse, add_implicit_h, METALS

DONOR_ORDER = ["C", "N", "O", "P", "S", "Se", "F", "Cl", "Br", "I", "H", "Si", "B", "As"]
_ORD = {e: i for i, e in enumerate(DONOR_ORDER)}


def _sig(elems):
    return ",".join(sorted(elems, key=lambda e: (_ORD.get(e, 99), e)))


def _components(atoms, bonds, metal):
    """Connected components of the non-metal atoms (ligands), plus the
    atoms bonded to the metal in each."""
    adj = defaultdict(set)
    metal_nb = set()
    for i, j, _ in bonds:
        mi, mj = atoms[i]["sym"] in METALS, atoms[j]["sym"] in METALS
        if mi and mj:
            continue
        if mi:
            metal_nb.add(j)
        elif mj:
            metal_nb.add(i)
        else:
            adj[i].add(j); adj[j].add(i)
    seen, comps = set(), []
    for k, a in enumerate(atoms):
        if k in seen or a["sym"] in METALS:
            continue
        stack, comp = [k], set()
        while stack:
            x = stack.pop()
            if x in comp:
                continue
            comp.add(x); stack.extend(adj[x] - comp)
        seen |= comp
        donors = sorted(comp & metal_nb)
        if donors:
            comps.append((comp, donors))
    return comps, adj


def _eta_groups(donors, adj):
    """Group donor atoms that are bonded to each other (pi / haptic binding)."""
    left, groups = set(donors), []
    while left:
        x = left.pop(); grp, stack = {x}, [x]
        while stack:
            y = stack.pop()
            for z in adj[y] & left:
                left.discard(z); grp.add(z); stack.append(z)
        groups.append(sorted(grp))
    return groups


ETA_NAMES = {2: "Alkene/alkyne (η2)", 3: "Allyl-type (η3)", 4: "Diene-type (η4)",
             5: "Cyclopentadienyl-type (η5)", 6: "Arene-type (η6)"}


def classify_ligand(atoms, adj, comp, donors):
    """Return (class_name, n_donor_atoms, coordination_sites)."""
    n = len(comp)
    syms = Counter(atoms[i]["sym"] for i in comp)
    d_el = [atoms[i]["sym"] for i in donors]
    groups = _eta_groups(donors, adj)
    # --- very small ligands -------------------------------------------
    if n == 1:
        a = atoms[donors[0]]; e = a["sym"]
        if e == "H": return "Hydride (H-)", 1, 1
        if e in ("F", "Cl", "Br", "I"): return {"F": "Fluoride (F-)", "Cl": "Chloride (Cl-)", "Br": "Bromide (Br-)", "I": "Iodide (I-)"}[e], 1, 1
        if e == "O":
            return ("Water (H2O)" if a["h"] >= 2 else "Hydroxide (OH-)" if a["h"] == 1 else "Oxo/oxide (O)"), 1, 1
        if e == "N":
            return ("Ammonia (NH3)" if a["h"] >= 3 else "Amido/imido (N)"), 1, 1
        if e == "S": return "Sulfide/thiol (S)", 1, 1
        if e == "P": return "Phosphine/phosphide (PH3)", 1, 1
        if e == "C": return ("Methyl (CH3)" if a["h"] >= 3 else "Carbide/carbyne (C)"), 1, 1
    if n == 2 and len(donors) == 1:
        if syms == Counter({"C": 1, "O": 1}):
            return ("Carbonyl (CO)" if atoms[donors[0]]["sym"] == "C" else "Oxygen-bound CO"), 1, 1
        if syms == Counter({"C": 1, "N": 1}):
            return ("Cyanide (CN-)" if atoms[donors[0]]["sym"] == "C" else "Isocyanide-type (N-bound CN)"), 1, 1
        if syms == Counter({"N": 1, "O": 1}):
            return "Nitrosyl (NO)", 1, 1
    # --- haptic (pi) ligands: every donor belongs to one bonded donor group
    if len(groups) == 1 and len(groups[0]) >= 2 and set(d_el) == {"C"}:
        k = len(groups[0])
        name = ETA_NAMES.get(k, f"Large π ligand (η{k})")
        return name, k, (k + 1) // 2
    # --- everything else: sigma donors by donor-atom signature ----------
    # donors in a pi-group + extra sigma donors are treated as sigma donors
    sig = _sig(d_el)
    dent = len(donors)
    if dent == 1:
        e = d_el[0]
        special = {"P": "Phosphine/phosphite (P)", "N": "N-donor, monodentate (amine/pyridine/imine)",
                   "O": "O-donor, monodentate (alkoxide/ether/carbonyl)", "S": "S-donor, monodentate (thiolate/thioether)",
                   "C": "C-donor, monodentate (alkyl/aryl/carbene)", "Se": "Se-donor, monodentate"}
        return special.get(e, f"{e}-donor, monodentate"), 1, 1
    word = {2: "bidentate", 3: "tridentate", 4: "tetradentate"}.get(dent, "polydentate")
    sites = min(dent, 6)
    return f"{sig}-donor, {word}", dent, sites


def extract(smiles):
    """Return dict(metal, charge, ligands=[(name, n_donors, sites, donor_elements)])
    for single-metal complexes, else None."""
    atoms, bonds = parse(smiles)
    add_implicit_h(atoms, bonds)
    ms = [i for i, a in enumerate(atoms) if a["sym"] in METALS]
    if len(ms) != 1:
        return None
    m = ms[0]
    comps, adj = _components(atoms, bonds, m)
    ligs = [(*classify_ligand(atoms, adj, c, d), tuple(atoms[i]["sym"] for i in d))
            for c, d in comps]
    return dict(metal=atoms[m]["sym"], charge=atoms[m]["charge"], ligands=ligs)
