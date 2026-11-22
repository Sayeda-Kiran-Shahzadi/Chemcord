"""Tiny dependency-free SMILES graph reader for tmQM_y dative-bond SMILES."""
import re

TOKEN = re.compile(r"(\[[^\]]+\]|Br|Cl|[BCNOSPFIbcnosp]|->|<-|[-=#$:/\\]|\(|\)|\.|%\d\d|\d)")
BRACKET = re.compile(r"\[(\d*)([A-Z][a-z]?|[a-z][a-z]?)(@{0,2})(?:H(\d*))?([+-]+\d*)?(?::\d+)?\]")
ORGANIC_VALENCE = {"B": (3,), "C": (4,), "N": (3, 5), "O": (2,), "P": (3, 5),
                   "S": (2, 4, 6), "F": (1,), "Cl": (1,), "Br": (1,), "I": (1,)}
METALS = set("Sc Ti V Cr Mn Fe Co Ni Cu Zn Y Zr Nb Mo Tc Ru Rh Pd Ag Cd La Hf Ta W Re Os Ir Pt Au Hg".split())
BOND_ORDER = {"-": 1, "=": 2, "#": 3, "$": 4, ":": 1.5, "->": 0, "<-": 0, "/": 1, "\\": 1}


def _charge(s):
    if not s:
        return 0
    sign = 1 if s[0] == "+" else -1
    rest = s[1:]
    if rest.isdigit():
        return sign * int(rest)
    return sign * len(s)


def parse(smiles):
    """Return (atoms, bonds). atoms: dict(sym, charge, arom, h_explicit);
    bonds: list of (i, j, order_token)."""
    atoms, bonds = [], []
    stack, ring, prev, pending = [], {}, None, None
    for tok in TOKEN.findall(smiles):
        if tok == "(":
            stack.append(prev)
        elif tok == ")":
            prev = stack.pop()
        elif tok == ".":
            prev, pending = None, None
        elif tok in BOND_ORDER:
            pending = tok
        elif tok[0].isdigit() or tok[0] == "%":
            key = tok
            if key in ring:
                j, b0 = ring.pop(key)
                bonds.append((prev, j, pending or b0 or "-"))
            else:
                ring[key] = (prev, pending)
            pending = None
        else:
            if tok[0] == "[":
                m = BRACKET.fullmatch(tok)
                if not m:
                    raise ValueError(tok)
                _, sym, _, h, chg = m.groups()
                arom = sym.islower()
                sym = sym.capitalize()
                hexp = 0 if h is None else (int(h) if h else 1)
                atom = dict(sym=sym, charge=_charge(chg), arom=arom, h=hexp)
            else:
                atom = dict(sym=tok.capitalize(), charge=0, arom=tok.islower(), h=None)
            atoms.append(atom)
            idx = len(atoms) - 1
            if prev is not None:
                bonds.append((prev, idx, pending or "-"))
            prev, pending = idx, None
    return atoms, bonds


def add_implicit_h(atoms, bonds):
    """Fill atom['h'] for organic-subset atoms; dative/metal bonds ignored."""
    tot = [0.0] * len(atoms)
    narom = [0] * len(atoms)
    for i, j, b in bonds:
        if atoms[i]["sym"] in METALS or atoms[j]["sym"] in METALS:
            continue
        if b == ":" or (atoms[i]["arom"] and atoms[j]["arom"] and b == "-"):
            narom[i] += 1; narom[j] += 1; tot[i] += 1; tot[j] += 1
        else:
            o = BOND_ORDER[b]
            tot[i] += o; tot[j] += o
    for k, a in enumerate(atoms):
        if a["h"] is None:
            if a["sym"] in ORGANIC_VALENCE:
                used = tot[k] + (1 if a["arom"] and narom[k] else 0)
                allowed = [v for v in ORGANIC_VALENCE[a["sym"]] if v >= used]
                a["h"] = int(round(allowed[0] - used)) if allowed else 0
            else:
                a["h"] = 0
    return atoms
