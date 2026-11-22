import csv, statistics as st
import figgis_tables as FG

SRC = {
 "S": ("Shriver, Atkins & Langford, Inorganic Chemistry 2nd ed. 1994 (table reproduced on LibreTexts, Centre College CHE 332 s6.3)",
       "https://chem.libretexts.org/Courses/Centre_College/CHE_332%3A_Inorganic_Chemistry/06%3A_Coordination_Chemistry-_Bonding/6.03%3A_Factors_That_Affect_Crystal_Field_Splitting"),
 "C": ("LibreTexts Centre College CHE 332 s6.3 (worked example in the text)",
       "https://chem.libretexts.org/Courses/Centre_College/CHE_332%3A_Inorganic_Chemistry/06%3A_Coordination_Chemistry-_Bonding/6.03%3A_Factors_That_Affect_Crystal_Field_Splitting"),
 "M": ("MIT 5.03 lecture 27 notes (D. Nocera)", "https://web.mit.edu/5.03/www/notes/spec_series.pdf"),
 "I": ("Imperial College ChemWiki, Transition Metal Coordination", "https://wiki.ch.ic.ac.uk/wiki/TransitionMetalCoordination"),
 "D": ("IIT Delhi CYL120 lecture notes", "https://web.iitd.ac.in/~ashok/cyl%20120/lectures_II.pdf"),
 "U": ("Univ. Muenster ligand-field lecture notes", "https://www.uni-muenster.de/imperia/md/content/anorganische_und_analytische_chemie/lehre/lehramt/b2fchemie/lftheorie.pdf"),
 "L": ("LibreTexts Saint Mary's College CHEM 431 s11.4", "https://chem.libretexts.org (Saint Mary's College CHEM 431, section 11.4: The Effect of the Metal Ion on d-Orbital Splitting)"),
 "F": ("Figgis & Hitchman, Ligand Field Theory and Its Applications, Wiley-VCH 2000, Table 8.3 p. 215 (book values are rounded to 1000 cm-1)", "book (no URL)"),
 "A": ("LibreTexts Tennessee State CHEM 4210 s3.4.5 (derived from AOM parameters, Cr(III) series)", "https://chem.libretexts.org (Tennessee State CHEM 4210, section 3.4.5)"),
}
# complex, metal, oxidation, ligand, geometry, [(source, value_cm1), ...]
T = [
 ("[Ti(H2O)6]3+","Ti",3,"H2O","octahedral",[("S",20300),("U",20000)]),
 ("[V(H2O)6]2+","V",2,"H2O","octahedral",[("S",12600),("L",11800),("M",12300)]),
 ("[V(H2O)6]3+","V",3,"H2O","octahedral",[("S",18900),("L",17850),("M",18600)]),
 ("[Cr(H2O)6]2+","Cr",2,"H2O","octahedral",[("S",13900),("M",14100)]),
 ("[Cr(H2O)6]3+","Cr",3,"H2O","octahedral",[("S",17400),("M",17000),("D",17400)]),
 ("[Mn(H2O)6]2+","Mn",2,"H2O","octahedral",[("S",8500)]),
 ("[Mn(H2O)6]3+","Mn",3,"H2O","octahedral",[("S",21000)]),
 ("[Fe(H2O)6]2+","Fe",2,"H2O","octahedral",[("S",10400),("D",9400)]),
 ("[Fe(H2O)6]3+","Fe",3,"H2O","octahedral",[("S",14300),("D",14000)]),
 ("[Co(H2O)6]2+","Co",2,"H2O","octahedral",[("S",9300),("M",9300)]),
 ("[Co(H2O)6]3+","Co",3,"H2O","octahedral",[("S",27000),("M",18200),("D",20760)]),
 ("[Ni(H2O)6]2+","Ni",2,"H2O","octahedral",[("S",8500)]),
 ("[Ru(H2O)6]2+","Ru",2,"H2O","octahedral",[("I",19800),("U",19800)]),
 ("[Ru(H2O)6]3+","Ru",3,"H2O","octahedral",[("I",28600),("U",28600)]),
 ("[Rh(H2O)6]3+","Rh",3,"H2O","octahedral",[("S",27000),("M",27000)]),
 ("[Cr(NH3)6]3+","Cr",3,"NH3","octahedral",[("S",21500)]),
 ("[Cr(en)3]3+","Cr",3,"en","octahedral",[("D",22300),("A",21780)]),
 ("[Co(NH3)6]2+","Co",2,"NH3","octahedral",[("D",10200)]),
 ("[Co(NH3)6]3+","Co",3,"NH3","octahedral",[("S",22900),("M",22870),("I",22870),("D",22870),("L",22900)]),
 ("[Ni(NH3)6]2+","Ni",2,"NH3","octahedral",[("S",10800)]),
 ("[Rh(NH3)6]3+","Rh",3,"NH3","octahedral",[("S",34000),("I",34100),("D",34000),("L",34100)]),
 ("[Ir(NH3)6]3+","Ir",3,"NH3","octahedral",[("S",41000),("M",41200),("I",41200),("D",41200),("U",41200),("L",40000)]),
 ("[Fe(CN)6]4-","Fe",2,"CN","octahedral",[("S",32800),("I",32200)]),
 ("[Fe(CN)6]3-","Fe",3,"CN","octahedral",[("S",35000),("I",35000),("D",35000)]),
 ("[Cr(CN)6]3-","Cr",3,"CN","octahedral",[("S",26600),("D",26600),("A",26310)]),
 ("[Co(CN)6]3-","Co",3,"CN","octahedral",[("S",34800)]),
 ("[Rh(CN)6]3-","Rh",3,"CN","octahedral",[("S",45500),("M",50000)]),
 ("[V(CN)6]3-","V",3,"CN","octahedral",[("M",23850)]),
 ("Cr(CO)6","Cr",0,"CO","octahedral",[("S",34150)]),
 ("[CrCl6]3-","Cr",3,"Cl","octahedral",[("S",13000)]),
 ("[MnCl6]4-","Mn",2,"Cl","octahedral",[("S",7500)]),
 ("[MnCl6]3-","Mn",3,"Cl","octahedral",[("S",20000)]),
 ("[RhCl6]3-","Rh",3,"Cl","octahedral",[("S",20400),("M",20300)]),
 ("[IrCl6]3-","Ir",3,"Cl","octahedral",[("S",25000)]),
 ("[FeCl6]3-","Fe",3,"Cl","octahedral",[("C",11600)]),
 ("[VCl6]3-","V",3,"Cl","octahedral",[("M",12200)]),
 ("[VCl6]2-","V",4,"Cl","octahedral",[("I",15400)]),
 ("[VF6]3-","V",3,"F","octahedral",[("M",16100)]),
 ("[CoF6]3-","Co",3,"F","octahedral",[("S",13000)]),
 ("[CrF6]3-","Cr",3,"F","octahedral",[("I",15060),("D",15060),("U",15600)]),
 ("[CrF6]2-","Cr",4,"F","octahedral",[("I",22000),("D",22000),("U",22000)]),
 ("[MnF6]2-","Mn",4,"F","octahedral",[("I",21860)]),
 ("[TcF6]2-","Tc",4,"F","octahedral",[("I",28400)]),
 ("[ReF6]2-","Re",4,"F","octahedral",[("I",32800),("U",32500)]),
 ("[VCl4]","V",4,"Cl","tetrahedral",[("S",9010),("I",7900)]),
 ("[CoCl4]2-","Co",2,"Cl","tetrahedral",[("S",3300)]),
 ("[CoBr4]2-","Co",2,"Br","tetrahedral",[("S",2900)]),
 ("[CoI4]2-","Co",2,"I","tetrahedral",[("S",2700)]),
 ("[FeCl4]-","Fe",3,"Cl","tetrahedral",[("C",5200)]),
 ("[Co(NH3)4]2+","Co",2,"NH3","tetrahedral",[("D",5900)]),
]
assert len(T) == 50 and len({r[0] for r in T}) == 50, len(T)

# ---- merge the Figgis & Hitchman Table 8.3 values --------------------------------------------------------
entries = {r[0]: {"metal": r[1], "ox": r[2], "lig": r[3], "geom": r[4], "vals": list(r[5]), "flags": set()} for r in T}
n_added_values = n_new = 0
for metal, ox, lig, geom, name, v, fl in FG.T83:
    if name in entries:
        e = entries[name]; assert (e["metal"], e["ox"], e["geom"]) == (metal, ox, geom), name
        n_added_values += 1
    else:
        e = entries[name] = {"metal": metal, "ox": ox, "lig": lig, "geom": geom, "vals": [], "flags": set()}
        n_new += 1
    e["vals"].append(("F", round(v * 1000))); e["flags"] |= set(fl)

rows = []
for name, e in entries.items():
    v = [x for _, x in e["vals"]]; med = st.median(v); spread = (max(v) - min(v)) / med
    notes = "; ".join(FG.FLAG_TEXT[c] for c in sorted(e["flags"]))
    if "q" in e["flags"]:
        flag = "UNCERTAIN: printed with ? in the source"
    elif spread > 0.10:
        flag = "CHECK: sources disagree >10%"
    else:
        flag = "ok"
    rows.append({"complex": name, "metal": e["metal"], "oxidation": e["ox"], "ligand": e["lig"], "geometry": e["geom"],
                 "delta_cm1": round(med), "n_sources": len(v), "min_cm1": min(v), "max_cm1": max(v),
                 "spread_pct": round(100 * spread, 1), "flag": flag, "note": notes,
                 "all_values": "; ".join(f"{x} ({k})" for k, x in e["vals"])})
with open("reference_delta.csv", "w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
with open("reference_delta_sources.csv", "w", newline="", encoding="utf-8") as f:
    w = csv.writer(f); w.writerow(["key", "source", "url"]); [w.writerow([k, *SRC[k]]) for k in SRC]
FG.write_csvs()
import collections
print(f"{len(rows)} distinct complexes ({n_new} new from Figgis Table 8.3, {n_added_values} extra values for complexes already listed)")
print(collections.Counter(r["geometry"] for r in rows), "| flags:", collections.Counter(r["flag"].split(":")[0] for r in rows))
print("multi-source rows:", sum(r["n_sources"] > 1 for r in rows), "| 4d/5d:", sum(r["metal"] in ("Ru","Rh","Ir","Tc","Re","Pt") for r in rows))
print("check rows:", [(r["complex"], r["spread_pct"], r["all_values"]) for r in rows if r["flag"].startswith("CHECK")])
