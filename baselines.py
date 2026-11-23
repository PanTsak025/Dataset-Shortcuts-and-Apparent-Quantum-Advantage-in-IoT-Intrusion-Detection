import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from qml_data import KEPT, LOG_COLS, NEG_TO_ZERO

CANDIDATES = {
    "LogReg":  (lambda p, s: LogisticRegression(C=p, max_iter=2000),          [0.1, 1, 10]),
    "SVM-RBF": (lambda p, s: SVC(C=p, kernel="rbf", gamma="scale"),         [0.1, 1, 10]),
    "RF":      (lambda p, s: RandomForestClassifier(n_estimators=200, max_depth=p,
                                                    random_state=s, n_jobs=1), [5, 10, None]),
}


def all_features(train, *others):
    def prep(df):
        X = df[KEPT].astype("float64").copy()
        X[NEG_TO_ZERO] = X[NEG_TO_ZERO].clip(lower=0)
        X[LOG_COLS] = np.log1p(X[LOG_COLS].clip(lower=0))
        return X
    scaler = StandardScaler().fit(prep(train))
    return [scaler.transform(prep(d)) for d in (train, *others)]


def fit_tuned(name, Xtr, ytr, Xval, yval, seed):
    make, grid = CANDIDATES[name]
    best = None
    for p in grid:
        model = make(p, seed).fit(Xtr, ytr)
        f1 = f1_score(yval, model.predict(Xval), average="macro")
        if best is None or f1 > best[0]:
            best = (f1, p, model)
    return best[2], best[1]

MLP_ALPHAS = [1e-4, 1e-3, 1e-2]


def mlp_hidden_units(n_in, n_classes, target_params):
    out = 1 if n_classes == 2 else n_classes
    return max(1, int(round((target_params - out) / (n_in + 1 + out))))


def mlp_parameters(n_in, n_classes, h):
    out = 1 if n_classes == 2 else n_classes
    return n_in * h + h + h * out + out


def fit_matched_mlp(Xtr, ytr, Xval, yval, seed, n_classes, target_params):
    from sklearn.neural_network import MLPClassifier
    h = mlp_hidden_units(Xtr.shape[1], n_classes, target_params)
    best = None
    for a in MLP_ALPHAS:
        model = MLPClassifier(hidden_layer_sizes=(h,), activation="tanh", solver="lbfgs",
                              alpha=a, max_iter=2000, random_state=seed).fit(Xtr, ytr)
        f1 = f1_score(yval, model.predict(Xval), average="macro")
        if best is None or f1 > best[0]:
            best = (f1, a, model)
    return best[2], f"h={h}, alpha={best[1]}", mlp_parameters(Xtr.shape[1], n_classes, h)