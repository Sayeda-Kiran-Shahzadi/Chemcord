# Chemcord

Desktop app (Python, Tkinter) for exploring the **geometry and stability of metal complexes**.
It combines three things in one window:

1. **Ligand-field model** (`ligandfieldmodel.py`): crystal-field splitting for four geometries,
   pairing energy, spin state, CFSE, net stabilisation, the crystal-field *activation* energy (CFAE) for
   octahedral complexes, and a kinetic label derived from it.
2. **Machine learning on real DFT data** (tmQM / tmQM_y, ~99,000 complexes): HOMO-LUMO gap,
   gap-based class (Low/Moderate/High gap tercile), DFT metal partial charge, and coordination geometry
   (classifier trained on geometries *measured* from the 3D structures).
3. **log K1 tool** (Tools menu): predicted aqueous binding constant of a metal ion with *any ligand
   you enter as SMILES*, binding free energy, a reliability label, measured values where they exist,
   and a metal-ion selectivity table. Trained on experimental stability constants.

## Quick start

```
pip install -r requirements.txt
python chemcord.py
```

The prepared `data/` and `models/` folders are included, except the log K1 model: its training data has no
redistribution licence, so the log K1 tool works only after you obtain the data and run `train_logk.py`
(see *Rebuilding*). Everything else runs out of the box. RDKit is only needed for the log K1 tool.

The bundled models were trained with scikit-learn 1.9.1 (pinned in `requirements.txt`); pickled models may
not load in other versions, in which case retrain them (see *Rebuilding*).

## What each number means (and how much to trust it)

| Output | Source | Honest accuracy note |
|---|---|---|
| Delta (also in cm-1), pairing energy, CFSE, net stabilisation, CFAE, kinetic label | formulas in `ligandfieldmodel.py` | calibrated to experimental Delta (see *Calibration*): about 9% typical error for simple M(L)6 complexes, much less certain outside the fitted range; octahedral d8 kinetics capped at Moderate |
| HOMO-LUMO gap | ML on tmQM_y (DFT) | MAE about 0.45 eV (baseline 0.70) |
| HOMO-LUMO gap class (Low / Moderate / High) | tercile class of the predicted gap | an electronic proxy, **not** thermodynamic or kinetic stability |
| Geometry | ML on measured geometries | 82.8% on a random split; 77.2% when whole ligand sets are held out; rare geometries are predicted poorly (see *Geometry classifier: grouped splits*) |
| **log K1** | ML on experimental data | see below |

### log K1 tool: accuracy by situation (mean absolute error, log units)

| Situation | Typical error |
|---|---|
| exact metal/ligand pair is in the training data | about 0.5 |
| ligand known, metal ion new for it | about 1.1 |
| ligand not in the training data | about 1.2 (can be larger for unusual ligands) |

For comparison the authors' deep-learning model reports MAE 0.834 on its external set; this lighter
gradient-boosting model gets 1.20 and needs no PyTorch. It predicts only the **first constant
(log K1) of a 1:1 complex in water**; it does not know which ligand atoms bind, pH, or solvents other
than the listed electrolytes. 43 of the 102 metal ions have fewer than 30 training rows.

## Geometry classifier: grouped splits

A random split is optimistic here: 86% of random-split test complexes share their exact (metal, oxidation
state, ligand set) input with a training complex. `evaluate_geometry_grouped.py` re-measures accuracy when whole
groups are held out. Each model is the same gradient-boosting classifier; the lookup table predicts the majority
geometry for the exact input (falling back to ligand set, then metal + oxidation state, then the overall
majority); the old rule is the hand-set coordination-number rule used before the model.

| Split | Test rows | Model accuracy | Model macro-F1 | Lookup table | Old rule | Majority class |
|---|---|---|---|---|---|---|
| Random (as in the app) | 19,788 | 82.8% | 0.583 | 79.9% | 48.3% | 26.4% |
| Exact metal + oxidation + ligand combinations held out | 20,626 | 82.0% | 0.538 | 64.3% | 45.5% | 23.5% |
| Whole ligand sets held out (any metal) | 23,591 | 77.2% | 0.439 | 58.7% | 46.2% | 22.8% |
| Whole metals held out | 15,725 | 83.4% | 0.397 | 72.3% | 59.2% | 24.0% |

Full table: `results/geometry_grouped_splits.csv`. Reproduce with `python evaluate_geometry_grouped.py`
(about 10 minutes per split on one CPU core; `--sample 6000 --max-iter 30` is a quick smoke test).

What this shows and does not show:
- The model beats the lookup table on every split, by 3 points on the random split and 11-18 points on the
  grouped ones, so it is not just memorising input combinations.
- Macro-F1 falls from 0.58 to 0.40-0.54 on the grouped splits: the common geometries (octahedral, square planar,
  piano stool) stay accurate while rare ones (e.g. seesaw, T-shaped, bent) are predicted poorly. Do not trust a
  rare-geometry prediction without checking its confidence.
- Each split is a single random draw (seed 42) with no repeats or confidence intervals. The held-out-metals
  accuracy depends on which metals landed in the test set, and that was not examined.
- The model sees only metal, oxidation state and ligand classes, not a 3D structure; it estimates which geometry
  complexes with that ligand set usually adopt.

## Oxidation states offered in the app

The dropdown lists oxidation states with at least 25 tmQM_y complexes. Some very common states are rare
there (Fe(III): 7 complexes; Cu(II): 3), so extra states are added by hand through
`EXTRA_OXIDATION_STATES` at the top of `chemcord.py` (currently `{"Fe": [3]}`); a low-support
warning appears whenever such a state is chosen. Note that "oxidation state" in tmQM_y is the formal
charge of the metal in the dataset's SMILES; for some ligand sets (e.g. tetradentate N donors) that is
inflated and is not a true oxidation state.

## Calibration of the ligand-field model (v5)

The ligand-field constants were fitted to **50 experimental octahedral Delta values** (`fit/reference_delta.csv`,
85 complexes in total; every value is listed with its source in `fit/reference_delta_sources.csv`). Model:
`Delta = K * s_ligand * row_factor * 1.7 * g(oxidation)`; water is fixed at 1.05 to anchor the arbitrary unit.

| Constant | Before | After (90% bootstrap interval) | Data points behind it |
|---|---|---|---|
| K, cm-1 per model unit | none (arbitrary unit) | 5481 (5055-6171) | 50 |
| 4d row factor | 1.45 | 1.62 (1.51-1.80) | 8 |
| 5d row factor | 1.75 | 1.89 (1.72-2.05) | 8 |
| g(+3) relative to +2 | 1.21 | 1.79 (1.59-1.94) | 28 |
| g(+4) relative to +2 | 1.41 | 2.24 (2.00-2.56) | 9 |
| NH3 | 1.35 | 1.26 (1.14-1.39) | 6 |
| CN- | 2.45 | 1.86 (1.60-2.34) | 7 |
| F-, Cl-, Br- | 0.85, 0.78, 0.72 | 0.88, 0.77, 0.68 | 10, 11, 4 |
| Pairing energy P | 2.45 (one value per row) | depends on d count: d4 4.25, d5 4.25, d6 2.91, d7 3.23 | spin states of 21 complexes |

**Accuracy** (median absolute error of Delta): 6.1% on the 50 fitted complexes. Honest estimates, where the fit never
sees the complex it predicts: **8.9%** leave-one-complex-out and **10.5%** leave-one-metal-out, against 13.0% and 12.9%
for the old constants (3d metals: 17.0% to 10.3%). Not used in the fit: tetrahedral complexes with the 4/9 rule 11%
(old constants 38%), Jahn-Teller in-plane values 30%. The model now gets the spin state right for all 21 octahedral d4-d7
complexes with a reported spin state (the old one called [Co(H2O)6]3+ high-spin). The ML models were retrained on the
refitted features and their accuracy did not change (differences below 0.005).

**What the data does not constrain** (these are assumptions, not results):
- Oxidation states 0 and 1: the only data point is Cr(CO)6 (observed 34,150 cm-1, model 23,758 cm-1). Delta is simply
  held at its +2 value there. Ligand classes other than H2O, NH3, CN-, F-, Cl-, Br- (CO, hydride, phosphine, Cp, ...)
  keep their old hand-set strengths.
- The pairing energy for d4 and d5 is only bracketed to a factor of 1.8 (17,500-31,000 cm-1) and is set to the middle;
  d7 rests on the book's crossover ratio because the data has no low-spin d7. The 4d/5d scaling of P is the old assumption.
- Mixed-ligand complexes use the average of the ligand strengths; that was not tested against data.
- The experimental values come from one textbook table, lecture notes and a second textbook; many are copies of the same
  compilations, so they are not independent. Table 8.3 of Figgis & Hitchman is rounded to 1000 cm-1.
- Sources disagree (e.g. [Co(H2O)6]3+: 18,200 to 27,000 cm-1); the table keeps every value and uses the median.

Reproduce: `cd fit && python build_table.py && python fit_ligandfield.py && python verify_model.py`
(`fit/fit_report.txt` is the saved output; `fit/legacy_constants.py` holds the pre-refit values).

## Tests

```
python test_label_geometry.py   # geometry labeller on ideal polyhedra
python test_ligandfield.py      # crystal-field physics (orbital sets, occupancies, CFSE, CFAE, kinetic labels)
```

`test_ligandfield.py` checks that every orbital pattern sums to zero, high-spin occupancies for d1-d10,
CFSE = 0 for d0 / high-spin d5 / d10, the classic activation energies (d3: 2.00 and 4.26 Dq, low-spin d6:
4.00 and 8.52 Dq, in units of Dq = Delta_oct / 10), the sign of the pairing term, and inert/labile labels
for well-known complexes.

## Rebuilding

```
python prepare_dataset.py data/tmQM_y.csv      # tmQM data (download separately)
python train_model.py                          # gap / stability / charge models
python label_geometry.py data/tmQM_X*_xyz.gz   # measure geometries
python train_geometry.py                       # geometry model
python train_logk.py                           # log K1 model (needs data/logk_raw/, see below)
```

`data/logk_raw/` must contain the six files `train|test|validation _ input|features .csv` from the
authors' repository (https://github.com/Critical-Materials-Institute/stabilityconstant-ml-models).

## Known issues (not yet fixed)

- CFAE is defined for octahedral complexes only, and CFAE theory over-predicts the inertness of octahedral
  d8 (e.g. Ni(II)), so that case is capped at "Moderate". The 0.08 / 0.15 kinetic cut-offs are hand-calibrated.
- See *Calibration* above for what the Delta fit does and does not constrain.
- The geometry/gap models are trained with a random split, so the headline accuracies are somewhat
  optimistic for ligand sets not seen in training; see *Geometry classifier: grouped splits* for the
  stricter numbers (geometry only; the gap models have not been re-evaluated with grouped splits).
- Rare geometries (macro-F1 0.40-0.54 on grouped splits) are predicted poorly.
- Only closed-shell complexes are common in tmQM, so classic paramagnetic ions (Cu(II), Cr(III), Mn(II))
  are poorly covered by the gap/geometry models; the log K1 tool covers them well.
- Lanthanides: tmQM contains only La.

## Data attribution and licences

See [THIRD_PARTY_LICENSES.md](THIRD_PARTY_LICENSES.md). tmQM / tmQM_y data are MIT-licensed; their
structures originate from the CSD (check its terms before redistributing structure files). The log K1
training data come from the IUPAC Stability Constants Database via Karunaratne et al., J. Chem. Inf.
Model. 2025, 65, 11532 (doi:10.1021/acs.jcim.5c01546) - that repository declares no licence, so the raw
tables and the trained log K1 model are git-ignored until you have permission. Please cite both papers.

This project's own code is released under the MIT licence (see [LICENSE](LICENSE)).
