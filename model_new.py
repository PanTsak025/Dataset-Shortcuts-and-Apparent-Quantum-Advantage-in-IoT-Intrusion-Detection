import glob, json, os, platform, shutil, sys
import numpy as np
import pandas as pd
import sklearn


CFG = {
    "seed": 42,                                # one seed for all random operations
    "label_col": "Label",
    "test_hash_modulus": 5,                    # a row goes to the test pool if hash % 5 == 0  -> ~20 %
    "analysis_rows_per_family_per_file": 500,  # size of the sample used for feature analysis only
    "redundancy_spearman_threshold": 0.999,    # |rho| at or above this = same information
}
OUT = "prepared"
os.makedirs(OUT, exist_ok=True)

VERSIONS = {"python": sys.version.split()[0], "numpy": np.__version__,
            "pandas": pd.__version__, "scikit-learn": sklearn.__version__,
            "platform": platform.platform()}
with open(f"{OUT}/config.json", "w") as fh:
    json.dump({"config": CFG, "versions": VERSIONS}, fh, indent=2)

SEED, LABEL = CFG["seed"], CFG["label_col"]


# 34 labels -> 8 families
def family(label):
    s = label.upper()
    if s == "BENIGN":                                     return "Benign"
    if s.startswith("DDOS"):                              return "DDoS"
    if s.startswith("DOS"):                               return "DoS"
    if s.startswith("MIRAI"):                             return "Mirai"
    if s.startswith("RECON") or s == "VULNERABILITYSCAN": return "Recon"
    if "SPOOFING" in s:                                   return "Spoofing"
    if s == "DICTIONARYBRUTEFORCE":                       return "BruteForce"
    return "Web"   # BrowserHijacking, SQLInjection, CommandInjection, XSS, Backdoor, Uploading


files = sorted(glob.glob(os.path.join("CIC-IOT-2023", "*.csv")))   # sorted -> same order on every machine
print(len(files), "files")

# BEGIN LOAD
analysis = pd.read_parquet(f"{OUT}/analysis_sample.parquet")
feature_stats = pd.read_csv(f"{OUT}/feature_stats_train.csv", index_col=0)
FEATURES = list(feature_stats.index)
# END LOAD

print(analysis["family"].value_counts())


NEG_TO_ZERO = feature_stats.index[feature_stats["min"] < 0].tolist()
print("negative values set to 0 in:", NEG_TO_ZERO)
analysis[NEG_TO_ZERO] = analysis[NEG_TO_ZERO].clip(lower=0)
feature_stats.loc[NEG_TO_ZERO, "min"] = 0.0

constant = feature_stats.index[feature_stats["min"] == feature_stats["max"]].tolist()
print("constant in training pool:", constant)

candidates = [c for c in FEATURES if c not in constant]
rho = analysis[candidates].corr(method="spearman").abs()
redundant, pairs = [], []
for a_i, a in enumerate(candidates):
    for b in candidates[a_i + 1:]:
        if rho.loc[a, b] >= CFG["redundancy_spearman_threshold"]:
            pairs.append((a, b, round(float(rho.loc[a, b]), 5)))
            if a not in redundant and b not in redundant:
                redundant.append(b)
print("redundant pairs (kept, dropped, |rho|):", pairs)

upper = rho.where(np.triu(np.ones(rho.shape, dtype=bool), k=1)).stack()
print(upper[(upper >= 0.9) & (upper < CFG["redundancy_spearman_threshold"])]
      .sort_values(ascending=False))

KEPT = [c for c in candidates if c not in redundant]
print(len(KEPT), "features kept")

skew_before = analysis[KEPT].skew()
skew_after = np.log1p(analysis[KEPT].clip(lower=0)).skew()
nonneg = feature_stats.loc[KEPT, "min"] >= 0
LOG_COLS = [c for c in KEPT if nonneg[c] and abs(skew_after[c]) < abs(skew_before[c])]

skew_table = pd.DataFrame({"skew_raw": skew_before, "skew_log1p": skew_after,
                           "log1p_applied": [c in LOG_COLS for c in KEPT]}).round(2)
skew_table.to_csv(f"{OUT}/skewness_train.csv")
print(skew_table)

with open(f"{OUT}/features.json", "w") as fh:
    json.dump({"all_features": FEATURES, "constant": constant, "redundant_pairs": pairs,
               "dropped": constant + redundant, "kept": KEPT, "log1p": LOG_COLS,
               "negative_to_zero": NEG_TO_ZERO},
              fh, indent=2)

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

Z = analysis[KEPT].astype("float64").copy()
Z[LOG_COLS] = np.log1p(Z[LOG_COLS])          # 1. compress heavy tails
Z = StandardScaler().fit_transform(Z)        # 2. mean 0, std 1 -> no feature wins by scale
pca = PCA(random_state=SEED).fit(Z)          # 3. all components, to see the full picture

cum = np.cumsum(pca.explained_variance_ratio_)
print(pd.DataFrame({"k_qubits": np.arange(1, 13),
                    "variance_%": (100 * pca.explained_variance_ratio_[:12]).round(1),
                    "cumulative_%": (100 * cum[:12]).round(1)}).to_string(index=False))

loadings = pd.DataFrame(pca.components_[:4].T, index=KEPT,
                        columns=["PC1", "PC2", "PC3", "PC4"])
for pc in loadings:
    top = loadings[pc].abs().sort_values(ascending=False).head(5).index
    print(pc, ", ".join(f"{c} ({loadings.loc[c, pc]:+.2f})" for c in top))