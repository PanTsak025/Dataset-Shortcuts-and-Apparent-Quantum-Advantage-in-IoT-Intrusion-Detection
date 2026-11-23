# Design decisions

This file lists every choice that shaped the results of the QML NIDS study on CIC-IoT-2023:
what was chosen, what was rejected, and the evidence for it.

- **Validation** always means the fixed validation set: 500 rows per class, seed 12345, drawn from the training pool after removing the task's training rows.
- Four decisions were added or changed after seeing results; they are marked *changed after* or *added after*: D17 (MI redundancy filter), D26 (matched MLP) and D29 (paired rule and 6-qubit cap).
- Three follow-up decisions (D36–D38) were motivated by results. For the two that involve new analyses (D37, D38) the rule was fixed in the script header before the analysis was run or evaluated.
- **No decision used the test set.**
- Result files are in `results/` (pilots) and `results/main/` (experiments).

- Result files of the diagnostic follow-up (D41) are in `results/diagnostic/`.

- **Submitted artifact.** Files named in the project history below but not needed to reproduce the paper were removed from the submitted version: `Tutorials/` (learning Qiskit), `model.py` (first look at one CSV), `repair_results.py` (one-off repair of result files, already applied), `make_figures.py` and `figures/` (superseded by the scripts in `paper/`), Python caches, smoke-test outputs, logs, and the large intermediate splits `prepared/train` and `prepared/test`, which `model_new.py` regenerates from the public dataset. The paper's tables and figures are produced by `paper/`.

---

## 0. Scope and project history

Fixed at the start: a Qiskit variational quantum classifier (VQC) for IoT network intrusion detection on the CIC-IoT-2023 merged CSVs (63 files), written up as a 6-page IEEE conference paper, with every choice explained so the study can be reproduced.

| Phase | Dates | Commits | What was done |
|---|---|---|---|
| 1. Learning Qiskit | 22–27 Oct | `8f3bc34`–`8751f82` | Tutorials (now `Tutorials/`): Iris classifiers, QSVM/QSVC (Pegasos, PCA, clustering), QNN basics, IBM introduction, encodings, quantum kernels. Not part of the pipeline. |
| 2. Dataset exploration | 29–30 Oct | `af1bb14`, `4b2b9ee` | `model.py`: first look at one CSV (labels, NaN/inf, duplicates, constant columns). Superseded by `model_new.py`. |
| 3. Preprocessing | 30 Oct | `c8e60ad` | `model_new.py` (Rules 1–8, PCA exploration), `build_pools.py`, `prepared/` (D1–D12). |
| 4. Model design and pilots | 30 Oct–1 Nov | `13c4b8c`, `72d153d` | Feature maps, ansatz expressibility, readout, optimizer and ansatz pilots (D16–D24). |
| 5. Main stages | 1–4 Nov | `f30dc38`–`88a9aa3` | Baselines and the three stages, committed at checkpoints while running; `repair_results.py` (D25–D35). |
| 6. Extra evaluation and claims | 4 Nov | `6ab246c` | `evaluate_extra.py`, `analyze_claims.py`, `compare_vqc_mlp.py` (D30, D33, D34). |
| 7. Figures, first version | 4 Nov | `ab5ac36` | `make_figures.py`: 8 figures, 3 tables. |
| 8. Follow-ups | 9 Nov | `dd17918`, `f665892`, `088e776` | Stage `small_data`, `analyze_small_data.py`, `analyze_low_far.py`, extraction-window check (D36–D38). `f665892` only adds the last log lines of shards still running at `dd17918`. |
| 9. Final documents | 9–10 Nov | not yet committed | `make_figures.py` with 5 new figures and 2 tables, `low_far.csv`, `DECISIONS.md`, `requirements.txt` (D39, D40). |

Checks on `088e776`: the small_data stage has 120 tasks × 6 models = 720 rows, no duplicates, no errors in any shard log, no restarts; re-running `analyze_small_data.py` reproduces `small_data_tests.csv` exactly.

## 1. Data

| # | Decision | Chosen | Rejected | Evidence and reason | Code |
|---|---|---|---|---|---|
| D1 | Invalid values | Remove rows with NaN or inf | Impute | 991 rows of 45,019,234, so imputing is not worth the assumptions. | `model_new.py` PART 1 |
| D2 | Duplicates | Remove exact duplicates (64-bit hash of the float64 features) | Keep | 24,013,003 rows (53.3%) were duplicates. Kept, they would sit on both sides of the split and inflate F1. | PART 1 |
| D3 | Label conflicts | Remove feature vectors that appear with more than one label | Majority vote | 1,152,528 rows, 99.5% of them DDoS or DoS. A conflicting row cannot be learned and would cap F1 for every model. | PART 1 |
| D4 | Train/test split | Feature hash mod 5 (1 of 5 to test, about 80/20) | Random row split | Identical rows always land on the same side, and the split does not depend on file order or a seed. | PART 1 |
| D5 | Attack grouping | 8 families (Benign, DDoS, DoS, Mirai, Recon, Spoofing, BruteForce, Web) | 34 labels | Families are the operational unit for detection. Seven labels have fewer than 6,000 clean rows (Uploading_Attack only 1,196), too few for per-label pools of 20,000 candidates. | PART 1 |

Result: 19,852,712 clean rows (15,880,857 train and 3,971,855 test). Report: `prepared/cleaning_report_by_family.csv`.

## 2. Features

| # | Decision | Chosen | Rejected | Evidence and reason | Code |
|---|---|---|---|---|---|
| D6 | Constant features | Drop if constant in train (Rule 5) | Keep | They carry no information. None were found. | `model_new.py` PART 2 |
| D7 | Redundant features | Spearman \|ρ\| ≥ 0.999 on train: drop the later column of the pair (Rule 6) | Keep | 7 pairs were found, so 6 columns were dropped: fin_count, rst_count, IPv, LLC, Tot size, Variance. 33 of 39 features remain. | PART 2 |
| D8 | Skew | log1p where it lowers \|skewness\| (Rule 7) | Always / never | It was applied to 31 features. It is only applied where it helps, so it never makes a feature worse. | PART 2 |
| D9 | Negative values | Set to 0 (Rule 8) | Keep | Only IAT has negatives. A time gap cannot be negative, and log1p needs non-negative input. | PART 2 |
| D10 | Fit statistics | All statistics from train only | Fit on all data | Prevents test information leaking into preprocessing. | everywhere |

Output: `prepared/features.json`. Versions: `prepared/config.json`.

## 3. Scenarios, samples and evaluation

| # | Decision | Chosen | Rejected | Evidence and reason | Code |
|---|---|---|---|---|---|
| D11 | Scenarios | 7 binary (Benign vs one family), Benign vs all attacks (equal family shares), 8-class | Only 8-class | Detection of a specific attack is the target use. 8-class tests the readout limit. | `qml_data.py` |
| D12 | Pools | 20,000 training candidates and 1,000 test rows per family, ordered by a stable hash | Sample on the fly | The same rows for every run, and nested training sets (25 ⊂ 50 ⊂ … ⊂ 800). | `build_pools.py` |
| D13 | Balance | Equal rows per class in training, validation and fixed test | Natural mix | Macro-F1 then compares models fairly. The natural mix is tested separately (D33). | `qml_data.py` |
| D14 | Metric | Macro-F1 (also accuracy, detection rate, false-alarm rate, precision) | Accuracy only | Macro-F1 treats every class equally. | `run_experiments.py` |
| D15 | Seeds | 3 run seeds (0, 1, 2): training sample, VQC start point, k-means, PCA/MI, RF and MLP random_state | 1 seed | Gives a standard error per cell. Strong claims rest on paired tests over many cells (D30). | `run_experiments.py` |

## 4. Reduction and encoding

| # | Decision | Chosen | Rejected | Evidence and reason | Code |
|---|---|---|---|---|---|
| D16 | Features per qubit | n features = n qubits (one angle each) | Several features per qubit | Keeps the encoding simple and comparable to the classical models fed the same n numbers. | `qml_data.to_qubits` |
| D17 | Reduction (*changed after a first run*) | PCA and MI (mRMR, skip a feature if \|ρ\| ≥ 0.95 with one already chosen). Both are kept as a factor | Plain top-k MI | Plain top-k MI picked near-duplicate features. Neither PCA nor MI won overall, so both stay. | `to_qubits` |
| D18 | Angle scaling | 1st to 99th percentile of train mapped to [0, angle_max], values outside clipped | Min-max | Traffic features are heavy-tailed, so min-max would squash most rows into a tiny angle range. | `to_qubits` |
| D19 | Feature map | RY angle encoding, angle_max = π | ZZ feature map (2 reps, linear, angle_max = π/2) | ZZ was lower in all 9 scenarios, by 0.045 to 0.148 validation macro-F1 (first_look stage). | `feature_maps.py` |

## 5. Quantum model

| # | Decision | Chosen | Rejected | Evidence and reason | Code |
|---|---|---|---|---|---|
| D20 | Readout | Binary: last qubit. 8-class: last 3 qubits, `x >> (n - 3)` | Qiskit default (`x % C`, qubit 0) | Under linear entanglement, qubit 0 has zero sensitivity to later features (`results/readout_sensitivity.csv`). The check used a RealAmplitudes ansatz with the same linear CNOT chain. A single-qubit (local) cost avoids cost-induced barren plateaus (Cerezo et al., Nat. Commun. 2021). | `vqc_model.py` |
| D21 | Loss | Cross-entropy | Not compared (Qiskit default kept) | Maximum likelihood and a proper scoring rule. The classes are balanced, so no weights are needed. | `vqc_model.py` |
| D22 | Simulation | Exact probabilities (QMLSampler, no shots) | Shots during training | Deterministic and reproducible. Shots and noise were checked after training (D34). | `vqc_model.py` |
| D23 | Optimizer | COBYLA, maxiter = 25 × parameters, start uniform on [0, 2π) from the run seed | L-BFGS-B, SPSA | Pilot results: L-BFGS-B 0.709 ± 0.065 (287 s), COBYLA 0.696 ± 0.069 (72 s), SPSA 0.694 (123 s). All are within noise; COBYLA is 4× faster than L-BFGS-B. | `optimizer_pilot.py` |
| D24 | Ansatz | EfficientSU2, 2 reps, linear entanglement | RealAmplitudes; EfficientSU2 with 1 or 3 reps | 3 reps scored 0.737 and 2 reps 0.725. The gap is 0.013 ± 0.014, within one SE, so the smaller circuit was kept (20 effective parameters at 4 qubits). | `ansatz_pilot.py` |

## 6. Classical baselines

| # | Decision | Chosen | Rejected | Evidence and reason | Code |
|---|---|---|---|---|---|
| D25 | Same-feature models | LogReg, SVM-RBF and RF on the same n scaled features, tuned on validation | Untuned defaults | A fair comparison needs equally tuned models. | `baselines.py` |
| D26 | Parameter-matched MLP (*added after the qubits stage*) | tanh, lbfgs, hidden units set so the parameter count matches the VQC, alpha tuned on validation | No matched model | Separates "quantum vs classical" from "number of trainable parameters". Back-filled into the finished stages without retraining any VQC. | `baselines.py` |
| D27 | Ceiling | RF on all 33 features | None | Shows how much information the n-feature reduction loses. | `baselines.py` |

## 7. Experiment design and analysis

| # | Decision | Chosen | Rejected | Evidence and reason | Code |
|---|---|---|---|---|---|
| D28 | Stages | first_look → qubits_features → learning_curves, each with 200 per class unless varied; later the small_data follow-up (D37) | One full grid | Each stage narrows the next one and saves compute. | `run_experiments.py` |
| D29 | Learning-curve setting per scenario (*changed after the qubits stage*) | Paired one-SE rule on validation, at most 6 qubits, ties broken by fewest qubits, then PCA | Unpaired one-SE rule; best validation score | The unpaired rule broke when an SE was 0. The paired rule compares the same seeds. The cap is 6 qubits because one 8-qubit training costs about 13× a 4-qubit one and showed no gain. | `summarize_qubits_features.py` |
| D30 | Statistics | Wilcoxon signed-rank on paired cell means, Holm correction over 12 tests | t-test per cell | 3 seeds are too few for per-cell tests. The test needs no normality assumption. Cells share training rows (nested samples), so the p-values are descriptive support, not exact error rates. | `analyze_claims.py` |
| D31 | Data requirement | Report both A (VQC within one SE of best classical from n onward) and B (plateau: within one SE of own best) | One definition | A asks "is it competitive", B asks "how much data does it need". | `analyze_learning_curves.py` |
| D32 | Training-sample selection | Random nested samples (reported); k-means prototypes (tested) | Random only | k-means did not help: the VQC's best F1 was lower in 5 of 9 scenarios, equal in 1 and higher by at most 0.011 in 3; RF's was lower in 6 of 9. | `run_experiments.py` |
| D33 | Extra test sets | Extended (up to 10,000 per class), realistic (20,000 rows, natural mix), seed 2026; VQCs rebuilt from saved weights, classical models refitted with saved hyperparameters | Fixed test only | Checks that the 2,000-row test set is large enough and shows precision at real base rates. | `evaluate_extra.py` |
| D34 | Robustness | 1,024 shots; Aer noise 0.1% per 1-qubit gate, 1% per CNOT, 2% readout, applied to trained weights | Noise-aware training | Asks whether the trained model survives a realistic noise level. Change was at most 0.01 F1. | `evaluate_extra.py` |
| D35 | Compute | 8 processes, `OMP_NUM_THREADS=1`, shards balanced longest-first using measured cost factors 0.47 / 1 / 2.24 / 6.40 at 2 / 4 / 6 / 8 qubits | Equal task counts per shard | A linear cost estimate underestimated 8 qubits by about 3×. | `run_experiments.py` |

## 8. Follow-ups and final documents

| # | Decision | Chosen | Rejected | Evidence and reason | Code |
|---|---|---|---|---|---|
| D36 | Flood scenarios (DDoS, DoS, Mirai) | Report them as separable by construction; base model comparisons on the window-matched scenarios (Recon, Spoofing, BruteForce, Web) | Treat flood scores as evidence about the models. Re-extract Benign and Mirai flows from the raw captures with a common 10-packet window: not done, it needs the raw PCAP captures, a working copy of the authors' feature extractor, and first reproducing the original scores on the chosen captures (future work). Re-run the floods without `Number` and `Tot sum`: not done, `Max`, `Min` and `Std` over 100 vs 10 packets still carry the window, so it cannot settle the question | The dataset authors used 10-packet windows for Benign and 100-packet windows for these floods. `Number` is 10 in 100% of Benign training rows and 100 in 99.4–99.9% of flood rows; the rule "attack if `Number` > 55" reaches test macro-F1 0.999 / 0.9995 / 0.9985, but 0.33 on Recon (same window as Benign). MI selects `Number` in every flood scenario. | analysis of `prepared/pools/` |
| D37 | Small-data hypothesis | Confirmatory replication: stage `small_data` (Recon, Spoofing, BruteForce, Web; 10, 25, 50 per class; learning-curve settings; new seeds 3–12; 120 VQC trainings). Primary test fixed before running: two-sided Wilcoxon on 40 (scenario × seed) VQC-minus-MLP gaps averaged over sizes; confirmed only if p < 0.05 and median gap > 0 | Report the pattern from seeds 0–2 as a finding | The pattern (VQC ahead of the matched MLP in 12 of 16 cells at ≤ 50 per class) was noticed after seeing results. **Not confirmed:** median gap +0.006, mean +0.005 (95% CI −0.002 to +0.012), 22 ahead / 18 behind, p = 0.19. Against the best tuned classical model: median −0.021, 4 / 36, p = 2.7e-9; largest gap at 10 per class (−0.023). The one significant secondary test (25 per class, p_Holm = 0.033) is reported as exploratory. | `run_experiments.py`, `analyze_small_data.py` |
| D38 | Operating point | Threshold at ≤ 1% benign false alarms, chosen on 5,000 benign calibration rows (training pool, disjoint from the training sample, seed 12345), frozen and applied to the realistic set; reported next to the 0.5 threshold. BruteForce and Web, 800 per class, random, seeds 0–2; VQC, RF, matched MLP; no retraining | Only the 0.5 threshold; 500 calibration rows | An IDS runs at a false-alarm budget. 5,000 rows put 50 false alarms at the 1% point instead of 5. Recall at ≤ 1%: BruteForce VQC 0.19, RF 0.24, MLP 0.24; Web VQC 0.05, RF 0.11, MLP 0.09 (achieved rates 1.0–1.2%). | `analyze_low_far.py` |
| D39 | Paper figures | `make_figures.py` v2: values printed on bars; `fig_summary` forest plot of every claim (median, 95% bootstrap CI of the median with 10,000 resamples and seed 0, p-values copied from the analysis outputs); `fig_training` split into window-matched, flood and 8-class scenarios; floods marked † in `fig_best` and `fig_window`; fixed seeds for jitter | First versions: dot plot of gaps (`fig_best`), heatmap of qubit gaps (`fig_qubits`), precision-only realistic figure without numbers | The first versions were hard to read or hid the numbers. Medians match the Wilcoxon tests. Pooling all binary scenarios in the training-loss figure hid how much higher the non-flood losses are than the flood losses (reference lines corrected to bits, see Sect. 9). Recommended main-paper set: method figure, `fig_best` + `tab_main`, `fig_summary`, `fig_learning_curves`, `fig_training` (or `fig_qubits`), `fig_window` (or one sentence), `tab_small_data`, `tab_low_far`. | `make_figures.py` |
| D40 | Software environment | `requirements.txt` with exact versions of the 9 imported packages: numpy 2.5.4, pandas 3.1.0, scipy 1.18.1, scikit-learn 1.9.1, matplotlib 3.11.2, qiskit 2.6.1, qiskit-aer 0.18.2, qiskit-machine-learning 0.9.1, pyarrow 26.0.0 (Python 3.14.8) | Unpinned installs; a full `pip freeze` | Classical models (RF tie-breaking) and simulators can change between versions. The Qiskit versions were not recorded automatically, so they were read from the environment afterwards. Only imported packages are listed, so the file stays readable. | `requirements.txt` |
| D41 | Diagnostic of the underfitting | Follow-up `run_diagnostic.py` / `analyze_diagnostic.py`, rule fixed before running. Setting: 4 qubits, PCA, 200 per class, random, seeds 0–2; Recon, Spoofing, BruteForce, Web, all attacks (primary) and 8-class (descriptive). Arms, each one change from the main VQC (A0, re-run): A1 best of 5 starts by training loss (optimiser); A2 ⟨Z⟩ of all qubits + linear layer trained jointly (readout; 24+5 parameters); R2 the same layer on cos(x) without a circuit (control for A2); A3 EfficientSU2 4 reps (40 parameters); A4 RY encoding before each rotation layer, same 24 parameters; A5 SVM on the fidelity kernel of the RY encoding (reference). Rule: over 15 (scenario, seed) units, training loss / log2 C lower and validation macro-F1 higher, both Wilcoxon with Holm over the 4 circuit arms p < 0.05, both in the right direction in ≥ 4 of 5 scenarios; A2 must also beat R2 on validation (p < 0.05). Test set not used. Result: A0 reproduced all 18 stored runs exactly (identical loss traces). A2 explains part of the underfitting (Δ fit −0.047, Holm p 0.00024; Δ val F1 +0.013, Holm p 0.034; 5/5; beats R2 by +0.017, p 0.002), halving the mean test-F1 gap to the best classical model (−0.020 → −0.009) without closing it. A1 (Δ val 0.000, p 0.49), A3 (Δ val +0.008, Holm p 0.098) and A4 (worse: Δ fit +0.019, Δ val −0.016) do not. A5 beats A0 on validation (+0.013, p 0.0001) and is within 0.016 test F1 of the best classical model in every scenario; its kernel equals ∏cos²((x−x′)/2), i.e. classical. 8-class: A2 0.35 → 0.46 test F1, but R2 alone 0.48, so the gain there is the layer's. | Citing Thomos & Andronikos 2026b as the only evidence for the cause; a broad search over circuits | Answers RQ4 with own evidence; one change per arm identifies the cause; the decision rule and the no-circuit control were fixed before running, and only training and validation data decide. | `run_diagnostic.py`, `analyze_diagnostic.py`, `results/diagnostic/` |

## 9. Corrections made during the study

These are recorded so that the history in `results/` makes sense.

- **Mixed column layouts.** The MLP back-fill appended 30-field rows to 29-column files. `repair_results.py` fixed the files (originals kept as `*.csv.bak`), and `run_experiments.py` now enforces a fixed column layout.
- **SPSA callback.** The qiskit-machine-learning SPSA callback was unreliable, so the optimizer pilot reads results from `fit_result` instead.
- **Withdrawn claim.** "VQC beats the matched MLP" was withdrawn after the paired tests. There is no significant difference at 2–4 qubits, and the MLP is significantly better at 6–8 qubits (p_Holm = 2.9e-5).
- **Hypothesis not confirmed.** "With very little data the VQC beats the same-size MLP" was tested on new seeds and not confirmed (D37).
- **Loss units corrected (bits, not nats).** qiskit-machine-learning 0.9.1 `CrossEntropyLoss` computes −Σ y·log₂ p averaged over samples, so every stored loss is in bits. Checked on two runs: Recon_n200_q4_mi_random_s0 stores 0.732 = mean −log₂ p (−ln p would give 0.507); multiclass_n200_q6_pca_random_s1 stores 2.363 = mean −log₂ p. A uniform guess therefore costs log₂ C = 1 bit (binary) and 3 bits (8-class), not ln 2 = 0.693 and ln 8 = 2.079. An earlier entry compared the bit losses with ln C and concluded that non-flood losses stay above a uniform guess and that the probabilities are poorly calibrated; that conclusion was wrong and is withdrawn. In bits, the final training loss relative to a uniform guess (qubits stage, 200 per class, mean of 3 seeds per setting) is 0.01–0.36 on the floods and 0.68–0.95 on every other scenario (binary per-scenario means: all attacks 0.70, Recon 0.74, Web 0.80, Brute force 0.81, Spoofing 0.88 bits; 8-class 2.28–2.45 bits by qubit count). The VQC learns better-than-uniform probabilities everywhere. What still holds: it underfits (training F1 within 0.01 of test F1), uses the full COBYLA budget with < 0.13% gain in the last 10% of evaluations, and classical models score higher on the same features. The conclusion is narrowed to: the limit is the trained circuit (optimisation, single-qubit readout or ansatz), not the information in the inputs; the experiments do not separate the three. The console text in `analyze_claims.py` ("ln 2 = 0.693 and ln 8 = 2.079 are the losses of uniform guessing") was wrong for the same reason and is corrected in the submitted artifact; the correct references are 1 and 3 bits. `make_figures.py` `fig_training` now draws the reference lines at 1 and 3 bits.
- **Analysis-script bug in the diagnostic (fixed before use).** The first version of `analyze_diagnostic.py` assigned Holm-corrected p-values in sorted order rather than arm order (`.values` on a Series sorted by p). No verdict changed, because the corrected values were recomputed by hand and then by the fixed script (`.reindex(r.index)`); the reported numbers come from the fixed script.
- **Flood scores reinterpreted.** The near-perfect DDoS, DoS and Mirai scores were first read as a VQC strength; they are explained by the extraction window (D36).

## 10. Fixed parameters

| Item | Value |
|---|---|
| Split | feature hash mod 5 |
| Validation | 500 per class, seed 12345 |
| Fixed test | 1,000 per family |
| Run seeds | 0, 1, 2 (small_data follow-up: 3–12) |
| Candidate pools (`build_pools.py`) | seed 42 |
| `to_qubits` (PCA, MI) | run seed |
| Extra-evaluation seed | 2026 |
| Training sizes | 25, 50, 100, 200, 400, 800 per class (small_data follow-up: 10, 25, 50) |
| Qubits | 2, 4, 6, 8 (8-class: 4, 6, 8) |
| Ansatz | EfficientSU2, reps 2, linear |
| Optimizer | COBYLA, maxiter = 25 × parameters |
| Angle range | RY [0, π]; ZZ [0, π/2] |
| Clip | 1st and 99th percentile |
| MI redundancy cap | \|ρ\| < 0.95 |
| Redundancy rule (cleaning) | Spearman ≥ 0.999 |
| False-alarm calibration (D38) | 5,000 benign rows per scenario, seed 12345, budget 1% |
