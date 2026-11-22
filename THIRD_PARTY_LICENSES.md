# Third-party licences and data attribution

## tmQM dataset

This project uses data from the tmQM dataset (the tmQM xyz, bond-order and charge
files, and the tmQM_y property table). The tmQM repository is distributed under the
MIT License, reproduced below.

```
MIT License

Copyright (c) 2024 Bastian Bjerkem Skjelstad

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

### Files in this project derived from tmQM

- `data/tmqm_processed.csv.gz` - per-complex table built from tmQM_y (metal, oxidation
  state, ligand classes, HOMO-LUMO gap, metal charge).
- `data/geometry_labels.csv.gz` - coordination geometry measured from the tmQM xyz files.
- `models/*.joblib` - machine-learning models trained on the data above.

### Please cite

- Balcells, D.; Skjelstad, B. B. "tmQM Dataset - Quantum Geometries and Properties of
  86k Transition Metal Complexes." J. Chem. Inf. Model. 2020, 60 (12), 6135-6146.
- Kneiding, H.; Lukin, R.; Lang, L.; Reine, S.; Pedersen, T. B.; De Bin, R.; Balcells, D.
  "Deep learning metal complex properties with natural quantum graphs."
  Digital Discovery 2023, 2, 618-633 (tmQM_y).

(Please check these references against the tmQM repository README before publishing.)

The tmQM structures originate from the Cambridge Structural Database (CSD); the xyz
headers carry a "CSD" note. Check the CSD terms before redistributing structure files
themselves.

---

## log K1 tool: data, paper and database

The "Binding strength (log K1)" tool (`logk.py`, `logk_ui.py`, `train_logk.py`) is trained on the
tables published with:

- Karunaratne, E.; Zahariev, F.; Perez Garcia, M. "A Comprehensive Machine Learning Model for
  Metal-Ligand Binding Prediction: Applications in Chemistry and Biology."
  J. Chem. Inf. Model. 2025, 65 (21), 11532-11542. doi:10.1021/acs.jcim.5c01546.
  The article is licensed CC-BY-NC-ND 4.0. Repository with the authors' data and Chemprop models:
  https://github.com/Critical-Materials-Institute/stabilityconstant-ml-models

Those tables were extracted by the authors from the **IUPAC Stability Constants Database**
(Pettit & Powell, SC-Database, 2016).

**Licence status (checked when this project was assembled):** the authors' repository declares no
licence file, and the underlying IUPAC database is a licensed product. Therefore this project does NOT
include the raw tables or the trained log K1 model in anything intended for public release:
`data/logk_raw/`, `models/logk_bundle.joblib` are listed in `.gitignore`. Before publishing them, ask the
authors (contact addresses are in the paper) and check the IUPAC database terms. Using the tables
privately to train and run the tool on your own computer is a separate matter from redistributing them.

Please cite the paper above if you use the log K1 tool in any report or publication.

---

## Ligand-field calibration data (`fit/`)

The experimental Delta values in `fit/reference_delta.csv` are numerical facts collected from the sources below, which are
listed value by value in `fit/reference_delta_sources.csv`. Please cite them if you use the table.

- B. N. Figgis and M. A. Hitchman, *Ligand Field Theory and Its Applications*, Wiley-VCH, New York, 2000
  (Tables 8.3, 8.4, 8.5, Section 6.9). A copyrighted book: only the numbers were transcribed, with page references
  (`fit/figgis_table8*.csv`, `fit/figgis_tables.py`).
- D. F. Shriver, P. W. Atkins and C. H. Langford, *Inorganic Chemistry*, 2nd ed., 1994 (a table reproduced on the
  LibreTexts page of Centre College CHE 332, section 6.3).
- MIT 5.03 lecture notes (D. Nocera); Imperial College ChemWiki "Transition Metal Coordination"; IIT Delhi CYL120
  lecture notes; Univ. Muenster ligand-field notes; LibreTexts pages (Saint Mary's College CHEM 431, Tennessee State
  CHEM 4210).

I did not review each site's reuse terms. Before publishing the repository, decide whether to keep the verbatim
transcriptions of the book's tables (`fit/figgis_table83_delta.csv`, `fit/figgis_table85_racahB.csv`) in it; the
fitted constants and the analysis do not depend on redistributing them.
