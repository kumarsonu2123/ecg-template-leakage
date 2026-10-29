# ECG Biometric Templates Leak Diagnoses

Code, notebooks and paper for a controlled measurement of **how much diagnostic information an ECG biometric template exposes**. The template is compared against an untrained copy of the same network, against the raw heartbeat, and against age and sex.

> **Main result (PTB-XL, lead I, 3,372 unseen patients, pre-registered lockbox test):** the identity template predicts conduction disturbance better than each of five untrained copies of the same network. **Δ_rand = +0.028 AUROC [95% CI 0.020, 0.038], p ≤ 0.001 against every copy.** All four diagnostic superclasses leak, the leak grows as the biometric gets better, and it transfers to the Brazilian CODE-15% cohort without retraining.

## Key results

| Question | Result |
|---|---|
| Does the biometric work? | EER 0.059 on 422 unseen identities, against 0.190 for a median-beat matcher |
| Leak beyond an untrained network (Δ_rand), CD vs normal | +0.028 [0.020, 0.038] |
| Leak beyond the raw heartbeat (Δ_raw), CD | +0.044 [0.029, 0.058] |
| Added beyond age + sex (Δ_demo), CD | +0.130 [0.113, 0.146] |
| Other diagnoses (Δ_rand) | MI +0.017, HYP +0.016, STTC +0.013, all significant after BH correction |
| Better biometric, more leak? (H3) | Spearman ρ = −0.75 between identity error and leakage, p = 0.007 |
| Across 3 seeds | Δ_rand = +0.026 ± 0.006 |
| Lead II | +0.008 [0.002, 0.014] |
| CODE-15% external, frozen model | +0.021 [0.007, 0.038] |
| Attackers (MLP, kNN) and Siamese biometric | Leak present in all (+0.012 to +0.046) |

## How it works

1. **Identity encoder.** A CNN + Transformer trained with ArcFace on lead I to recognise people. Patients with 2 or more visits are used.
2. **Templates of unseen patients.** The encoder is frozen and applied to single-visit patients it never saw.
3. **Attacker.** A logistic regression (plus MLP and kNN) is trained on attacker-development patients to predict a diagnosis from templates, then scored once on the sealed lockbox.
4. **Baselines.** The same attacker is run on 5 untrained copies of the network (Δ_rand), the raw median beat (Δ_raw), and age + sex (Δ_demo).
5. **Statistics.** Each gain gets a 95% patient-bootstrap CI and a paired permutation test, with Benjamini–Hochberg correction across diagnoses. Rules were written to dated files before the lockbox was opened.

## Repository layout

```text
config.yaml                  all settings (data paths, preprocessing, model, training, statistics)
src/ecgleak/                 the shared library
  config.py                  load config, seeds, environment record
  models.py                  ECGEncoder (CNN + Transformer), ArcFace head, untrained copies
  train.py                   identity training (ArcFace or Siamese), checkpoint/resume, diagnosis model
  evaluate.py                templates, verification EER, cross-validated attacker AUROC
  leakage.py                 attacker fit/score, paired bootstrap, paired permutation
  analysis.py                MLP/kNN attackers, gains table, BH correction, integrated gradients
notebooks/
  01_setup_and_data.ipynb        Phases 0-1: verify PTB-XL, cohorts, splits, heartbeat cache
  02_baselines_and_pilot.ipynb   Phase 2: baselines, self-tests, resume test, pilot
  03_main_experiment.ipynb       Phase 3: identity model, sealed lockbox, H1/H2
  04_secondary_analyses.ipynb    Phase 4: all diagnoses, H3/H4/H5, attackers, Siamese
  05_robustness_external.ipynb   Phase 5: seeds, lead II, CODE-15% external test
  06_paper_tables_figures.ipynb  Phase 6: corrected main test, tables and figures
  executed_run_with_outputs.ipynb  the actual run, with every printed output and figure
paper/                       IEEE-format LaTeX draft (main.tex, references.bib, figures/)
tests/                       fast checks of the statistics (pytest -q)
```

## Running it

The pipeline was run on **Google Colab** (free T4 GPU) with data on **Google Drive**.

1. **Get PTB-XL 1.0.3** from [PhysioNet](https://physionet.org/content/ptb-xl/1.0.3/). Upload the unzipped folder (with `records500/`, `ptbxl_database.csv`, `scp_statements.csv` and `SHA256SUMS.txt`) to `My Drive/PTB-XL/`. Version 1.0.3 is required; the Kaggle mirror is an older version.
2. **Open the notebooks in Colab** in order, `01` to `06`. Each notebook mounts Drive, writes its outputs to `My Drive/ecg_leak/`, and starts only if the previous phase's gate file (`gates/gate_N.json`) says GO.
3. **Runtime:** CPU is enough for notebooks 01 and 06; use a T4 GPU for 02 to 05. The whole pipeline takes about 4–6 hours of Colab time; training itself takes minutes.
4. **CODE-15%** (notebook 05) is downloaded automatically from [Zenodo](https://zenodo.org/record/4916206). Only part 0 is used, about 2.7 GB, on the Colab disk.

To use the library outside Colab:

```bash
pip install -r requirements.txt
export PYTHONPATH=$PWD/src
python -m pytest -q tests
```

## Reproducibility safeguards

- Every PTB-XL file is checked against PhysioNet's SHA-256 list.
- Patient-disjoint splits are recorded with their SHA-256 hashes.
- Pass/fail rules (`phase3_plan.json`, `phase5_plan.json`) are written before results exist, and changing them afterwards is refused.
- The model used on the lockbox is sealed by its SHA-256 fingerprint; any other model is refused.
- Checkpoints are saved every epoch, with automatic resume after a Colab disconnect.
- Seeds are fixed, and each run stores its frozen config and software environment.

## Limitations

- The leaks are modest in size: +0.008 to +0.046 AUROC.
- The external test is small: 117 CD-equivalent test patients.
- The H3 snapshots come from one training run, so they aren't independent points.
- H5 rests on only 4 diagnoses.

See the paper's Limitations section for details.

## Data licences

PTB-XL and CODE-15% are both released under CC BY 4.0. Cite them as given in `paper/references.bib`. No patient data is included in this repository.

## Citation

> TODO: add the paper citation once it is published.

## Licence

Code is released under the MIT Licence (see `LICENSE`).
