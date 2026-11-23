import argparse, glob, json, os
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon
from run_experiments import tasks_for, task_id, MODEL

OUT = "results/main"
CLASSICAL = ["LogReg", "SVM-RBF", "RF"]
pd.set_option("display.width", 250, "display.max_colwidth", 60)


def load(stage):
    runs = pd.concat([pd.read_csv(f) for f in glob.glob(f"{OUT}/runs_*.csv")], ignore_index=True)
    ids = {task_id(t) for t in tasks_for(stage)[0]}
    d = runs[runs.task.isin(ids)].drop_duplicates(["task", "model"]).copy()
    d["setting"] = d["reducer"].str.upper() + "-" + d["n_qubits"].astype(str) + "q"
    return d


def order():
    return list(dict.fromkeys(t["scenario"] for t in tasks_for("qubits_features")[0]))

def trainability():
    v = load("qubits_features")
    v = v[v.model == "VQC-RY"].copy()
    v["budget"] = MODEL["evals_per_parameter"] * v["parameters"]
    v["used_full_budget"] = v["loss_evaluations"] >= v["budget"]

    def late_gain(row):
        path = f"{OUT}/curves/{row.task}_VQC-RY.json"
        if not os.path.exists(path):
            return np.nan
        loss = np.minimum.accumulate(np.array(json.load(open(path))))   
        if len(loss) < 20:
            return np.nan
        k = int(len(loss) * 0.9)
        return 100 * (loss[k] - loss[-1]) / loss[k]                     
    v["late_gain_pct"] = v.apply(late_gain, axis=1)

    for col, title in [("train_macro_f1", "1a) VQC TRAINING macro-F1 vs qubits (mean over seeds)"),
                       ("final_train_loss", "1b) VQC final TRAINING loss (cross-entropy) vs qubits"),
                       ("test_macro_f1", "1c) VQC TEST macro-F1 vs qubits (for comparison)")]:
        t = v.pivot_table(index="scenario", columns="setting", values=col).round(3)
        cols = [f"{r}-{q}q" for r in ["PCA", "MI"] for q in [2, 4, 6, 8]]
        print(f"\n{title}\n", t.reindex(index=order(), columns=[c for c in cols if c in t]).to_string())
    agg = v.assign(type=np.where(v.scenario == "multiclass", "8-class", "binary")) \
           .groupby(["type", "n_qubits"]).agg(runs=("task", "count"),
                                              train_f1=("train_macro_f1", "mean"),
                                              test_f1=("test_macro_f1", "mean"),
                                              train_loss=("final_train_loss", "mean"),
                                              full_budget_pct=("used_full_budget", lambda s: 100 * s.mean()),
                                              late_gain_pct=("late_gain_pct", "median")).round(3)
    print("\n1d) Summary per qubit count: training fit, budget use, late-training improvement\n"
          "    (late_gain_pct = median % loss drop in the last 10% of COBYLA evaluations;\n"
          "     losses are in bits (qiskit-machine-learning uses log2): uniform guessing costs 1 bit\n"
          "     for binary and 3 bits for 8-class)\n", agg.to_string())
    agg.to_csv(f"{OUT}/claims_trainability.csv")


def gaps(stage, other, keys):
    d = load(stage)
    rows = []
    for key, g in d.groupby(keys):
        if other == "best_classical":
            cls = g[g.model.isin(CLASSICAL)].groupby("model")["val_macro_f1"].mean()
            if cls.empty:
                continue
            comp = cls.idxmax()                    
        else:
            comp = other
        p = g.pivot_table(index="seed", columns="model", values="test_macro_f1")
        if "VQC-RY" not in p or comp not in p:
            continue
        rows.append(dict(zip(keys, key), gap=float((p["VQC-RY"] - p[comp]).mean())))
    return pd.DataFrame(rows)


def significance():
    tests = []
    lc_keys = ["scenario", "sampling", "n_train"]
    qf_keys = ["scenario", "setting", "n_qubits"]
    for other in ["MLP-matched", "best_classical"]:
        lc = gaps("learning_curves", other, lc_keys)
        qf = gaps("qubits_features", other, qf_keys)
        tests += [(f"VQC vs {other}: learning curves, all", lc),
                  (f"VQC vs {other}: learning curves, <= 200 per class", lc[lc.n_train <= 200]),
                  (f"VQC vs {other}: learning curves, >= 400 per class", lc[lc.n_train >= 400]),
                  (f"VQC vs {other}: qubits stage, all", qf),
                  (f"VQC vs {other}: qubits stage, 2-4 qubits", qf[qf.n_qubits <= 4]),
                  (f"VQC vs {other}: qubits stage, 6-8 qubits", qf[qf.n_qubits >= 6])]
    rows = []
    for name, g in tests:
        x = g["gap"].dropna().to_numpy()
        nz = x[x != 0]
        p = wilcoxon(nz).pvalue if len(nz) >= 6 else np.nan
        rows.append({"comparison": name, "cells": len(x), "median_gap": np.median(x),
                     "mean_gap": x.mean(), "VQC_ahead": int((x > 0).sum()), "VQC_behind": int((x < 0).sum()),
                     "p_wilcoxon": p})
    r = pd.DataFrame(rows)
    valid = r["p_wilcoxon"].notna()
    m = valid.sum()
    ranked = r.loc[valid, "p_wilcoxon"].sort_values()
    adj, running = {}, 0.0
    for i, (idx, p) in enumerate(ranked.items()):
        running = max(running, min(1.0, (m - i) * p))
        adj[idx] = running
    r["p_holm"] = pd.Series(adj)
    show = r.round({"median_gap": 4, "mean_gap": 4})
    for c in ["p_wilcoxon", "p_holm"]:
        show[c] = show[c].map(lambda p: "" if pd.isna(p) else f"{p:.2g}")
    print("\n2) Significance: VQC minus comparison model, TEST macro-F1, one value per cell "
          "(mean over seeds); two-sided Wilcoxon signed-rank, Holm-corrected\n",
          show.to_string(index=False))
    r.to_csv(f"{OUT}/claims_significance.csv", index=False)

def cost():
    d = load("learning_curves")
    d = d[d.cpu_seconds.notna()]
    d["type"] = np.where(d.scenario == "multiclass", "8-class", "binary")
    t = d.pivot_table(index=["type", "n_qubits", "n_train"], columns="model", values="cpu_seconds",
                      aggfunc="median").round(1)
    t = t.reindex(columns=[c for c in ["VQC-RY", "LogReg", "SVM-RBF", "RF", "MLP-matched", "RF-all33"] if c in t])
    print("\n3a) Median CPU seconds per training (classical: incl. 3-value tuning)\n", t.to_string())
    ratio = (t["VQC-RY"] / t[["LogReg", "SVM-RBF", "RF", "MLP-matched"]].max(axis=1)).round(0)
    print("\n3b) VQC CPU time / slowest same-feature classical model\n", ratio.to_string())
    v = d[d.model == "VQC-RY"].groupby(["type", "n_qubits"]).agg(
        parameters=("parameters", "first"), loss_evaluations=("loss_evaluations", "median")).astype(int)
    print("\n3c) VQC size and optimiser evaluations per training\n", v.to_string())
    t.to_csv(f"{OUT}/claims_cost.csv")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--part", choices=["trainability", "significance", "cost", "all"], default="all")
    a = ap.parse_args()
    for name, fn in [("trainability", trainability), ("significance", significance), ("cost", cost)]:
        if a.part in (name, "all"):
            fn()