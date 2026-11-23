import glob
import numpy as np
import pandas as pd
from run_experiments import tasks_for, task_id

CLASSICAL = ["LogReg", "SVM-RBF", "RF"]


def from_size_onwards(sizes, ok):
    result = None
    for s, good in sorted(zip(sizes, ok), reverse=True):
        if not good:
            break
        result = s
    return result


def plateau_size(curve):
    best = curve["mean"].idxmax()
    threshold = curve.loc[best, "mean"] - curve.loc[best, "se"]
    n_b = from_size_onwards(curve.index, curve["mean"] >= threshold)
    if n_b is None:   
        n_b = f"no stable plateau (best at {best})"
    return n_b, round(curve.loc[best, "mean"], 3)


def analyse(res, vqc_name="VQC-RY"):
    rows_a, rows_b = [], []
    for (sc, samp), d in res.groupby(["scenario", "sampling"]):
        sizes = sorted(d["n_train"].unique())

        ok, details = [], []
        for n in sizes:
            dn = d[d.n_train == n]
            best_cls = dn[dn.model.isin(CLASSICAL)].groupby("model")["val_macro_f1"].mean().idxmax()
            q = dn[dn.model == vqc_name].set_index("seed")["test_macro_f1"]
            c = dn[dn.model == best_cls].set_index("seed")["test_macro_f1"]
            gap = (q - c).dropna()
            se = gap.std() / np.sqrt(len(gap)) if len(gap) > 1 else np.nan
            ok.append(bool(gap.mean() >= -se))
            details.append(f"{n}:{gap.mean():+.3f}±{se:.3f}({best_cls})")
        n_a = from_size_onwards(sizes, ok)
        rows_a.append({"scenario": sc, "sampling": samp,
                       "n_A_matches_classical": n_a if n_a is not None else f"not within {max(sizes)}",
                       "gap_per_size": "  ".join(details)})

        row = {"scenario": sc, "sampling": samp}
        for m in [vqc_name] + CLASSICAL + ["MLP-matched"]:
            g = d[d.model == m].groupby("n_train")["test_macro_f1"]
            curve = pd.DataFrame({"mean": g.mean(), "se": g.std() / np.sqrt(g.count())})
            n_b, best = plateau_size(curve)
            row[f"n_B_{m}"] = n_b
            row[f"best_F1_{m}"] = best
        rows_b.append(row)
    return pd.DataFrame(rows_a), pd.DataFrame(rows_b)


if __name__ == "__main__":
    out_dir = "results/main"
    runs = pd.concat([pd.read_csv(f) for f in glob.glob(f"{out_dir}/runs_*.csv")], ignore_index=True)
    tasks, fmaps = tasks_for("learning_curves")
    res = runs[runs["task"].isin({task_id(t) for t in tasks})].drop_duplicates(["task", "model"])
    print(f"learning_curves rows used: {len(res)}")
    a, b = analyse(res)
    pd.set_option("display.width", 250, "display.max_colwidth", 200)
    print("\nA) smallest training size (per class) from which the VQC matches the best classical "
          "model on the same features (within 1 SE, paired by seed)\n", a.to_string(index=False))
    print("\nB) smallest training size (per class) from which each model stays within 1 SE of its "
          "own best F1 (plateau)\n", b.to_string(index=False))
    a.to_csv(f"{out_dir}/learning_curves_A_match_classical.csv", index=False)
    b.to_csv(f"{out_dir}/learning_curves_B_plateau.csv", index=False)