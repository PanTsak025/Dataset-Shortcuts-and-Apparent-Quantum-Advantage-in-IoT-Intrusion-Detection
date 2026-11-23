import glob, json
import numpy as np
import pandas as pd
from qiskit_machine_learning.primitives import QMLSampler

from qml_data import load_scenario, load_validation, to_qubits
from feature_maps import ANGLE_MAX
from run_experiments import tasks_for, task_id
from evaluate_extra import vqc_qnn, refit_classical, realistic_set

OUT = "results/main"
SCENARIOS = ["BruteForce", "Web"]
MODELS = ["VQC-RY", "RF", "MLP-matched"]
N_TRAIN, SAMPLING, SEEDS = 800, "random", [0, 1, 2]
FAR_BUDGET, N_CAL, CAL_SEED = 0.01, 5000, 12345


def threshold_at_budget(benign_scores, budget):
    """Lowest t with share(benign score > t) <= budget."""
    s = np.sort(np.asarray(benign_scores))
    cand = np.unique(s)
    far = 1 - np.searchsorted(s, cand, side="right") / len(s)
    return float(cand[far <= budget][0])


def rates(y, score, t):
    pred = score > t
    tp, fp = int((pred & (y == 1)).sum()), int((pred & (y == 0)).sum())
    return {"false_alarm_rate": fp / int((y == 0).sum()), "recall": tp / int((y == 1).sum()),
            "precision": tp / (tp + fp) if tp + fp else np.nan, "alerts": tp + fp}


def scores_for(t, stored):
    train, fixed, _ = load_scenario(t["scenario"], t["n_train"], t["seed"], strategy=t["sampling"])
    cal, _ = load_validation(t["scenario"], N_CAL, train["row_id"], seed=CAL_SEED)
    sets = {"calibration": cal, "fixed_test": fixed, "realistic": realistic_set(t["scenario"])}
    big = pd.concat([d.assign(_set=k) for k, d in sets.items()], ignore_index=True)
    Xtr, Xbig, ytr, _, _ = to_qubits(train, big, reducer=t["reducer"], n_qubits=t["n_qubits"],
                                     angle_max=ANGLE_MAX["ry"], seed=t["seed"])
    out = {}
    for m in MODELS:
        r = stored[(stored.task == task_id(t)) & (stored.model == m)]
        if not len(r):
            continue
        if m == "VQC-RY":
            w = np.array(json.loads(r["weights"].iloc[0]))
            out[m] = vqc_qnn(t["n_qubits"], 2, QMLSampler()).forward(Xbig, w)[:, 1]
        else:
            model = refit_classical(m, str(r["hyperparameter"].iloc[0]), Xtr, ytr, t["seed"])
            out[m] = model.predict_proba(Xbig)[:, 1]
    return big["_set"].to_numpy(), big["y"].to_numpy(), out


if __name__ == "__main__":
    stored = pd.concat([pd.read_csv(f) for f in glob.glob(f"{OUT}/runs_learning_curves_shard*.csv")],
                       ignore_index=True).drop_duplicates(["task", "model"])
    tasks = [t for t in tasks_for("learning_curves")[0]
             if t["scenario"] in SCENARIOS and t["n_train"] == N_TRAIN
             and t["sampling"] == SAMPLING and t["seed"] in SEEDS]
    rows = []
    for t in tasks:
        which, y, scores = scores_for(t, stored)
        cal_benign = (which == "calibration") & (y == 0)
        for m, s in scores.items():
            rules = {"default 0.5": 0.5, "FAR <= 1% (calibrated)": threshold_at_budget(s[cal_benign], FAR_BUDGET)}
            for rule, thr in rules.items():
                for es in ["calibration", "fixed_test", "realistic"]:
                    k = which == es
                    rows.append({"scenario": t["scenario"], "seed": t["seed"], "model": m, "rule": rule,
                                 "threshold": thr, "eval_set": es, **rates(y[k], s[k], thr)})
        print(f"done {task_id(t)}", flush=True)
    d = pd.DataFrame(rows)
    d.to_csv(f"{OUT}/low_far.csv", index=False)

    pd.set_option("display.width", 220)
    held = d[d.eval_set != "calibration"]
    cols = ["false_alarm_rate", "recall", "precision"]
    g = held.groupby(["scenario", "eval_set", "rule", "model"])[cols]
    table = (g.mean().round(3).astype(str) + " ± " + (g.std() / np.sqrt(g.count())).round(3).astype(str))
    print(f"\nHeld-out results, mean ± SE over seeds {SEEDS} ({N_TRAIN} per class, {SAMPLING}).")
    print("fixed_test: 1,000 per class (its precision is at 50% prevalence, not meaningful);"
          " realistic: natural attack share.\n")
    print(table.to_string())
    cal = d[(d.eval_set == "calibration") & (d.rule != "default 0.5")]
    print("\nCalibration check (benign false-alarm rate on the calibration set, should be <= 0.01):\n",
          cal.groupby(["scenario", "model"])["false_alarm_rate"].max().round(4).to_string())
    print(f"\nwrote {OUT}/low_far.csv")
