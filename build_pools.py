import json, os
import numpy as np
import pandas as pd

OUT = "prepared"
CFG = json.load(open(f"{OUT}/config.json"))["config"]
FEAT = json.load(open(f"{OUT}/features.json"))
KEPT = FEAT["kept"]          # the 33 features chosen by the rules in step 2
SEED = CFG["seed"]

POOL_CFG = {
    "train_candidates_per_family": 20000,
    "test_per_family": 1000,
}
FAMILIES = ["Benign", "BruteForce", "DDoS", "DoS", "Mirai", "Recon", "Spoofing", "Web"]
os.makedirs(f"{OUT}/pools", exist_ok=True)


def stable_sample(df, n, seed):
    key = pd.util.hash_pandas_object(df[KEPT + ["Label"]], index=False).to_numpy()
    df = df.assign(row_id=key).sort_values("row_id", kind="stable")
    return df.sample(n=min(n, len(df)), random_state=seed).reset_index(drop=True)


rows = []
for fam in FAMILIES:
    for split, n in [("train", POOL_CFG["train_candidates_per_family"]),
                     ("test", POOL_CFG["test_per_family"])]:
        d = pd.read_parquet(f"{OUT}/{split}", columns=KEPT + ["Label"],
                            filters=[("family", "==", fam)])
        s = stable_sample(d, n, SEED)
        s.to_parquet(f"{OUT}/pools/{split}_{fam}.parquet", index=False)
        rows.append({"family": fam, "split": split, "available": len(d), "sampled": len(s)})
        print(f"{fam:<11} {split:<5} available={len(d):>10,}  sampled={len(s):,}")
        del d  

summary = pd.DataFrame(rows).pivot(index="family", columns="split",
                                   values=["available", "sampled"])
summary.to_csv(f"{OUT}/pools/pool_summary.csv")
with open(f"{OUT}/pools/pool_config.json", "w") as fh:
    json.dump(POOL_CFG, fh, indent=2)
print(summary)