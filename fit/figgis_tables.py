"""Numbers transcribed (by reading the page images) from

    B. N. Figgis and M. A. Hitchman, "Ligand Field Theory and Its Applications",
    Wiley-VCH, New York, 2000.

    Table 8.3  p. 215  ligand-field splitting Delta of octahedral/tetrahedral complexes (units of 10^3 cm^-1)
    Table 8.4  p. 219  empirical parameters g, k (metal ions) and f, h (ligands):  Delta = f*g, B = B_free*(1 - h*k)
    Table 8.5  p. 220  Racah B and nephelauxetic ratio beta for typical complexes
    Section 6.9 p. 143  Delta/B at which the ground spin state changes (Tanabe-Sugano discontinuities, Figs 6.12-6.15)

Only numerical facts are recorded here, with page numbers, so that every value can be checked against the book.
Table 8.3 values are rounded to 1000 cm^-1 in the book (3.2 and 3.5 to 100 cm^-1).
Footnotes of Table 8.3:  a = in-plane ligands of a tetragonally elongated octahedral complex;
b = low-spin complex (other values for that metal ion are high-spin); c = high-spin complex (other values for that
metal ion are low-spin). A '?' printed in the table is kept as flag 'q'.
"""
import csv

# metal, oxidation, ligand code, geometry, complex name, value (10^3 cm^-1), flags
T83 = [
 ("Ti", 3, "Cl", "octahedral", "[TiCl6]3-", 13, ""), ("Ti", 3, "H2O", "octahedral", "[Ti(H2O)6]3+", 20, ""), ("Ti", 3, "NH3", "octahedral", "[Ti(NH3)6]3+", 17, "q"),
 ("V", 3, "F", "octahedral", "[VF6]3-", 16, ""), ("V", 3, "Cl", "octahedral", "[VCl6]3-", 13, ""), ("V", 3, "H2O", "octahedral", "[V(H2O)6]3+", 19, ""),
 ("V", 3, "ox", "octahedral", "[V(ox)3]3-", 18, ""), ("V", 3, "NH3", "octahedral", "[V(NH3)6]3+", 18, "q"),
 ("Cr", 3, "Cl", "octahedral", "[CrCl6]3-", 13, ""), ("Cr", 3, "H2O", "octahedral", "[Cr(H2O)6]3+", 17, ""), ("Cr", 3, "ox", "octahedral", "[Cr(ox)3]3-", 17, ""),
 ("Cr", 3, "NH3", "octahedral", "[Cr(NH3)6]3+", 22, ""), ("Cr", 3, "en", "octahedral", "[Cr(en)3]3+", 22, ""), ("Cr", 3, "CN", "octahedral", "[Cr(CN)6]3-", 26, ""),
 ("Re", 4, "F", "octahedral", "[ReF6]2-", 32, ""), ("Re", 4, "Cl", "octahedral", "[ReCl6]2-", 33, ""),
 ("Cr", 2, "Cl", "octahedral", "[CrCl6]4-", 13, "a"), ("Cr", 2, "H2O", "octahedral", "[Cr(H2O)6]2+", 14, "a"), ("Cr", 2, "en", "octahedral", "[Cr(en)3]2+", 18, "a"),
 ("Mn", 3, "F", "octahedral", "[MnF6]3-", 22, "a"), ("Mn", 3, "Cl", "octahedral", "[MnCl6]3-", 20, "a"), ("Mn", 3, "H2O", "octahedral", "[Mn(H2O)6]3+", 21, "a"),
 ("Mn", 3, "CN", "octahedral", "[Mn(CN)6]3-", 30, "b"),
 ("Mn", 2, "F", "octahedral", "[MnF6]4-", 8, ""), ("Mn", 2, "Cl", "octahedral", "[MnCl6]4-", 8, ""), ("Mn", 2, "H2O", "octahedral", "[Mn(H2O)6]2+", 8, ""),
 ("Mn", 2, "en", "octahedral", "[Mn(en)3]2+", 10, ""), ("Mn", 2, "Cl", "tetrahedral", "[MnCl4]2-", 2, ""),
 ("Fe", 3, "F", "octahedral", "[FeF6]3-", 14, ""), ("Fe", 3, "H2O", "octahedral", "[Fe(H2O)6]3+", 14, ""), ("Fe", 3, "ox", "octahedral", "[Fe(ox)3]3-", 14, ""),
 ("Fe", 3, "CN", "octahedral", "[Fe(CN)6]3-", 35, "b"), ("Fe", 3, "Cl", "tetrahedral", "[FeCl4]-", 5, ""),
 ("Fe", 2, "H2O", "octahedral", "[Fe(H2O)6]2+", 10, ""), ("Fe", 2, "CN", "octahedral", "[Fe(CN)6]4-", 33, "b"), ("Fe", 2, "Cl", "tetrahedral", "[FeCl4]2-", 4, ""),
 ("Co", 3, "F", "octahedral", "[CoF6]3-", 13, "c"), ("Co", 3, "H2O", "octahedral", "[Co(H2O)6]3+", 19, ""), ("Co", 3, "ox", "octahedral", "[Co(ox)3]3-", 18, ""),
 ("Co", 3, "NH3", "octahedral", "[Co(NH3)6]3+", 24, ""), ("Co", 3, "en", "octahedral", "[Co(en)3]3+", 24, ""), ("Co", 3, "CN", "octahedral", "[Co(CN)6]3-", 34, ""),
 ("Rh", 3, "Cl", "octahedral", "[RhCl6]3-", 20, ""), ("Rh", 3, "Br", "octahedral", "[RhBr6]3-", 19, ""), ("Rh", 3, "H2O", "octahedral", "[Rh(H2O)6]3+", 27, ""),
 ("Rh", 3, "ox", "octahedral", "[Rh(ox)3]3-", 26, ""), ("Rh", 3, "NH3", "octahedral", "[Rh(NH3)6]3+", 34, ""), ("Rh", 3, "en", "octahedral", "[Rh(en)3]3+", 35, ""),
 ("Ir", 3, "Cl", "octahedral", "[IrCl6]3-", 25, ""), ("Ir", 3, "Br", "octahedral", "[IrBr6]3-", 23, ""), ("Ir", 3, "en", "octahedral", "[Ir(en)3]3+", 41, ""),
 ("Pt", 4, "F", "octahedral", "[PtF6]2-", 33, ""), ("Pt", 4, "Cl", "octahedral", "[PtCl6]2-", 29, ""), ("Pt", 4, "Br", "octahedral", "[PtBr6]2-", 25, ""),
 ("Co", 2, "H2O", "octahedral", "[Co(H2O)6]2+", 10, ""), ("Co", 2, "ox", "octahedral", "[Co(ox)3]4-", 11, ""), ("Co", 2, "NH3", "octahedral", "[Co(NH3)6]2+", 11, ""),
 ("Co", 2, "en", "octahedral", "[Co(en)3]2+", 11, ""), ("Co", 2, "Cl", "tetrahedral", "[CoCl4]2-", 3.2, ""),
 ("Ni", 2, "Cl", "octahedral", "[NiCl6]4-", 7, ""), ("Ni", 2, "Br", "octahedral", "[NiBr6]4-", 6, ""), ("Ni", 2, "H2O", "octahedral", "[Ni(H2O)6]2+", 9, ""),
 ("Ni", 2, "NH3", "octahedral", "[Ni(NH3)6]2+", 11, ""), ("Ni", 2, "en", "octahedral", "[Ni(en)3]2+", 12, ""), ("Ni", 2, "Cl", "tetrahedral", "[NiCl4]2-", 3.5, ""),
 ("Cu", 2, "H2O", "octahedral", "[Cu(H2O)6]2+", 13, "a"), ("Cu", 2, "NH3", "octahedral", "[Cu(NH3)6]2+", 15, "a"), ("Cu", 2, "en", "octahedral", "[Cu(en)3]2+", 16, "a"),
]
FLAG_TEXT = {"a": "in-plane value of a tetragonally elongated octahedron (Figgis footnote a)",
             "b": "low-spin complex (Figgis footnote b)", "c": "high-spin complex (Figgis footnote c)",
             "q": "printed with '?' in the book (uncertain)"}

# Table 8.4 (p. 219): metal ion -> (g in 10^3 cm^-1, k or None); ligand -> (f, h or None)
G_K = {"Co2+": (9.3, 0.24), "Co3+": (19.0, 0.35), "Cr2+": (14.1, None), "Cr3+": (17.0, 0.21), "Cu2+": (12.0, None),
       "Fe2+": (10.0, None), "Fe3+": (14.0, 0.24), "Ir3+": (32, 0.3), "Mn2+": (8.5, 0.07), "Mn3+": (21, None),
       "Mn4+": (23, 0.5), "Mo3+": (24, 0.15), "Ni2+": (8.9, 0.12), "Pt4+": (36, 0.5), "Re4+": (35, 0.2),
       "Rh3+": (27.0, 0.3), "Ti3+": (20.3, None), "V2+": (12.3, 0.08), "V3+": (18.6, None)}
F_H = {"3acac": (1.2, None), "6Br": (0.76, 2.3), "6CH3CO2": (0.96, None), "4Cl": (-0.36, None), "6Cl": (0.80, 2.0),
       "6CN": (1.7, 2.0), "6NCS": (1.03, None), "2den": (1.28, None), "edta": (1.20, None), "ddtp": (0.86, 2.8),
       "3bipy": (1.43, None), "3en": (1.28, 1.5), "6F": (0.9, 0.8), "3glycine": (1.21, None), "6H2O": (1.00, 1.0),
       "6NH3": (1.25, 1.4), "6NO2": (1.5, None), "6OH": (0.94, None), "3Ox": (0.98, 1.5), "3o-phen": (1.43, None),
       "6py": (1.25, None), "6urea": (0.91, 1.25)}
# map our ligand codes to Table 8.4 ligand names
LIG_F = {"F": "6F", "Cl": "6Cl", "Br": "6Br", "H2O": "6H2O", "NH3": "6NH3", "CN": "6CN", "en": "3en", "ox": "3Ox"}

# Table 8.5 (p. 220): complex, d configuration (as printed), symmetry, B (cm^-1), beta
T85 = [("[VF6]3-", "d2", "Oh", 627, 0.71), ("[VCl6]3-", "d2", "Oh", 536, 0.61), ("[VCl4]-", "d2", "Td", 505, 0.57),
       ("VCl2", "d3", "Oh", 615, 0.81), ("VBr2", "d3", "Oh", 530, 0.70), ("VI2", "d3", "Oh", 510, 0.67),
       ("[V(NH3)6]2+", "d3", "Oh", 660, 0.87), ("[CrF6]3-", "d3", "Oh", 896, 0.96), ("[CrCl6]3-", "d3", "Oh", 512, 0.55),
       ("[Cr(NH3)6]3+", "d3", "Oh", 657, 0.70), ("[MnF6]2-", "d4 (as printed; Mn(IV) is d3)", "Oh", 650, 0.60),
       ("[MnCl4]2-", "d5", "Td", 650, 0.76), ("[MnBr4]2-", "d5", "Td", 630, 0.73), ("[FeF6]3-", "d5", "Oh", 835, 0.84),
       ("[FeBr4]-", "d5", "Td", 470, 0.46), ("[Fe(CN)6]4-", "d6", "Oh", 490, 0.55), ("[CoF6]3-", "d6", "Oh", 787, 0.73),
       ("[Co(NH3)6]3+", "d6", "Oh", 615, 0.57), ("[Co(CN)6]3-", "d6", "Oh", 400, 0.37), ("[Co(NH3)6]2+", "d7", "Oh", 885, 0.89),
       ("[Co(NH3)4]2+", "d7", "Td", 710, 0.72), ("[CoCl4]2-", "d7", "Td", 710, 0.72), ("[CoBr4]2-", "d7", "Td", 695, 0.70),
       ("[CoI4]2-", "d7", "Td", 665, 0.67), ("KNiF3", "d8", "Oh", 843, 0.81), ("CsNiCl3", "d8", "Oh", 838, 0.80),
       ("CsNiBr3", "d8", "Oh", 777, 0.75), ("[Ni(NH3)6]2+", "d8", "Oh", 881, 0.85)]
# Section 6.9 (p. 143): Delta/B at the high-spin -> low-spin change of ground term (octahedral)
CROSSOVER = {4: 26, 5: 27, 6: 18, 7: 20}


def write_csvs(out="."):
    with open(f"{out}/figgis_table83_delta.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f); w.writerow(["complex", "metal", "oxidation", "ligand", "geometry", "delta_cm1", "flags", "note", "book_ref"])
        for m, ox, lg, g, name, v, fl in T83:
            w.writerow([name, m, ox, lg, g, round(v * 1000), fl, "; ".join(FLAG_TEXT[c] for c in fl), "Figgis & Hitchman 2000, Table 8.3, p. 215 (rounded to 1000 cm-1)"])
    with open(f"{out}/figgis_table84_fg.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f); w.writerow(["kind", "name", "main_parameter", "second_parameter", "meaning", "book_ref"])
        for k, (g, kk) in G_K.items():
            w.writerow(["metal ion", k, g, "" if kk is None else kk, "g (10^3 cm-1), k", "Table 8.4, p. 219"])
        for k, (ff, h) in F_H.items():
            w.writerow(["ligand", k, ff, "" if h is None else h, "f, h", "Table 8.4, p. 219"])
    with open(f"{out}/figgis_table85_racahB.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f); w.writerow(["complex", "d_configuration", "symmetry", "B_cm1", "beta", "book_ref"])
        for r in T85:
            w.writerow([*r, "Table 8.5, p. 220"])


if __name__ == "__main__":
    write_csvs()
    print(len(T83), "Table 8.3 entries;", len(G_K), "metal ions and", len(F_H), "ligands in Table 8.4;", len(T85), "rows in Table 8.5")
