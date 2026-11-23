# Dataset Shortcuts and Apparent Quantum Advantage in IoT Intrusion Detection — Research Artifact

Code, configuration and per-run results for an audited, controlled evaluation of a **variational quantum
classifier (VQC)** against tuned classical models and a parameter-matched neural network on
**CIC-IoT-2023**. 

| | |
|---|---|
| **Models** | VQC (R<sub>Y</sub> angle encoding, EfficientSU2, COBYLA, exact simulation), logistic regression, RBF-SVM, random forest, parameter-matched MLP, random forest on all features |
| **Data** | CIC-IoT-2023, merged CSV release (public, not included) |
| **Scale** | 2–8 qubits · 10–800 training rows per class · 9 detection scenarios · 3 seeds (10 in the replication) |
| **Hardware** | CPU only; no GPU or quantum hardware needed |
| **Runtime** | Analyses: minutes. Full re-training: about 31 h on a 10-core laptop with 8 parallel processes |

> **Shortcut.** All trained-model results are included in `results/`. To check the paper's numbers you
> only need **step 5** (analyses), which take minutes and
> retrain nothing.

---

## 1. Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt          # exact versions used for every result (Python 3.14)
```

Run every command from the repository root.

## 2. Data (only needed to rebuild `prepared/` from scratch)

`prepared/` already contains everything the experiments read: the per-family training pools and fixed
test sets (`prepared/pools/`), the larger evaluation sets (`prepared/eval/`), the selected features and
the cleaning reports. To rebuild it from the raw data:

1. Download the CIC-IoT-2023 **merged CSV** release (63 files) from the Canadian Institute for
   Cybersecurity (<https://www.unb.ca/cic/datasets/iotdataset-2023.html>) into `CIC-IOT-2023/`.
2. In `model_new.py`, uncomment **PART 1** and comment out the **LOAD SAVED RESULTS** block, then run
   ```bash
   python model_new.py      # cleaning rules 1–8, hash split, feature decisions (slow: reads 45M rows)
   python build_pools.py    # per-family training pools and fixed 1,000-row test sets
   ```

## 3. Design checks and pilots (Section "Method" / RQ2)

```bash
python feature_maps.py      # R_Y vs ZZ circuits, depth and CNOT counts
python vqc_model.py         # readout sensitivity per qubit   -> results/readout_sensitivity.csv
python ansatz.py            # expressibility (several minutes) -> results/ansatz_expressibility.csv
python optimizer_pilot.py   # COBYLA vs L-BFGS-B vs SPSA       -> results/optimizer_pilot*.csv
python ansatz_pilot.py      # ansatz family and depth          -> results/ansatz_pilot*.csv
```
`optimizer_pilot.py` and `ansatz_pilot.py` accept `--smoke` for a quick check.

## 4. Experiments (re-training; optional)

All stages are run by `run_experiments.py`. Each stage is split into shards that run in parallel and
can be resumed: finished runs are skipped.

```bash
python run_experiments.py --smoke                                  # 2–3 minute end-to-end check
python run_experiments.py --stage first_look --plan --nshards 8    # count runs, estimate hours
python run_experiments.py --stage first_look --shard i --nshards 8 # i = 0..7, one per terminal
```

| Order | Stage | What it does | VQC trainings |
|---|---|---|---|
| 1 | `first_look` | R<sub>Y</sub> vs ZZ encoding, 4 qubits, 200 rows/class | 54 |
| | `python summarize_results.py --stage first_look` | encoding choice (ZZ dropped) | – |
| 2 | `qubits_features` | 2/4/6/8 qubits × PCA/MI, 200 rows/class | 210 |
| | `python summarize_qubits_features.py` | one setting per scenario (paired one-SE rule) | – |
| 3 | `learning_curves` | 25–800 rows/class, random and *k*-means sampling | 324 |
| | `python evaluate_extra.py --shard i --nshards 8`, then `--summary` | larger test sets, shots, noise, natural traffic mix (no retraining) | – |
| 4 | `small_data` | confirmatory replication, new seeds 3–12 | 120 |

**Follow-ups** (each has its rule fixed in the script header before it was run):

```bash
python analyze_small_data.py                       # replication test (stage small_data)
python analyze_low_far.py                          # 1% false-alarm budget (~12 min) -> results/main/low_far.csv
python run_diagnostic.py --check                   # self-checks of the diagnostic arms
python run_diagnostic.py --shard i --nshards 8     # why the VQC underfits (about 5 h on 8 shards)
python analyze_diagnostic.py                       # decision rule -> results/diagnostic/
```

## 5. Analyses of the stored results (no training, minutes)

```bash
python analyze_claims.py           # trainability, paired tests, cost  -> results/main/claims_*.csv
python analyze_learning_curves.py  # plateaus, catching up with classical models
python compare_best.py             # best VQC vs best classical model per scenario
python compare_vqc_mlp.py          # VQC vs parameter-matched MLP per configuration
python analyze_small_data.py       # replication result      -> results/main/small_data_tests.csv
python analyze_diagnostic.py       # diagnostic verdicts     -> results/diagnostic/diagnostic_*.csv
```

## Repository layout

```
├── model_new.py, build_pools.py           data cleaning, split, feature decisions, pools
├── qml_data.py                             scenarios, sampling, feature pipeline (PCA / MI, angle scaling)
├── feature_maps.py, ansatz.py, vqc_model.py   VQC: encoding, ansatz, readout, loss
├── baselines.py                            classical models and parameter-matched MLP
├── optimizer_pilot.py, ansatz_pilot.py     validation pilots
├── run_experiments.py                      all experiment stages (sharded, resumable)
├── evaluate_extra.py                       re-evaluation of trained models (no retraining)
├── run_diagnostic.py                       diagnostic follow-up (why the VQC underfits)
├── analyze_*.py, compare_*.py, summarize_*.py   analyses of stored results
├── prepared/                               pools, evaluation sets, features, cleaning reports
├── results/                                pilot results; main/ and diagnostic/ per-run results
│   └── main/curves/, diagnostic/curves/    loss after every COBYLA evaluation
└── docs/DECISIONS.md                       every design choice, the rejected alternative and the evidence
```

## Reproducibility notes

- **Determinism.** Data splits use a content hash, not a random seed. Training samples, VQC start
  points and classical models are seeded (seeds 0–2; 3–12 in the replication). Re-running a stage on
  the same machine reproduces the stored runs exactly; the diagnostic's re-run baseline matched all
  18 stored runs evaluation by evaluation. On a different machine, last-digit floating-point
  differences can change the COBYLA path after some evaluations.
- **Validation only.** Every choice uses the validation set (500 rows per class, seed 12345); the
  test set was never used for a decision.
- **Loss units.** qiskit-machine-learning computes cross-entropy with base-2 logarithms, so all stored
  losses are in bits; a uniform guess costs 1 bit (binary) or 3 bits (8-class).
- **Changes after seeing results** are listed and justified in `docs/DECISIONS.md`.

## License

An open-source license will be added on public release. The CIC-IoT-2023 dataset is subject to its own terms of use.
