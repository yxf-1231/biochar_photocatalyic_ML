#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
03_mcirp_classification_rf.py
=============================
Classification of the dominant reaction pathway (MCIRP: radical / non-radical /
mixed): screening of ten candidate classifiers under an identical protocol,
then five-fold cross-validated grid search for the selected Random Forest and
final held-out test evaluation. Corresponds to Tables S10/S11, Fig. 2 and
Fig. S7 of the manuscript.
"""

from __future__ import annotations

import argparse
import json
from importlib import import_module
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import (AdaBoostClassifier, ExtraTreesClassifier,
                              GradientBoostingClassifier, RandomForestClassifier)
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                             precision_score, recall_score)
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC

pp = import_module("01_data_preprocessing")

MODEL_FILE = "rf_mcirp_model.pkl"

# ---- candidate classifiers screened under the identical protocol (Fig. S7) ----
def make_candidates(seed: int) -> dict:
    from catboost import CatBoostClassifier
    from lightgbm import LGBMClassifier
    from xgboost import XGBClassifier
    return {
        "RF": RandomForestClassifier(random_state=seed),
        "GBDT": GradientBoostingClassifier(random_state=seed),
        "CatBoost": CatBoostClassifier(verbose=0, random_seed=seed, allow_writing_files=False),
        "MLP": MLPClassifier(max_iter=2000, random_state=seed),
        "XGBoost": XGBClassifier(eval_metric="mlogloss", random_state=seed),
        "LightGBM": LGBMClassifier(random_state=seed, verbose=-1),
        "ExtraTrees": ExtraTreesClassifier(random_state=seed),
        "AdaBoost": AdaBoostClassifier(random_state=seed),
        "KNN": KNeighborsClassifier(),
        "SVM": SVC(probability=True, random_state=seed),
    }

# ---- Random Forest search space (cf. Table S10) ----
PARAM_GRID = {
    "estimator__n_estimators": [100, 200, 300, 500],
    "estimator__max_depth": [4, 8, 12, 16, 20, 25, None],
    "estimator__min_samples_split": [2, 4, 8, 16, 20, 32],
    "estimator__min_samples_leaf": [1, 2, 4, 8],
    "estimator__max_features": ["sqrt", "log2", 0.3, 0.5, 0.7, None],
    "estimator__criterion": ["gini", "entropy"],
}

# Reference values reported for the Random Forest in SI Table S10
# (informational consistency printout only -- never overrides the search).
TABLE_S10_REFERENCE = {"n_estimators": 100, "max_depth": 16,
                       "min_samples_split": 20, "min_samples_leaf": 1}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", default=pp.DATA_PATH, help="input workbook")
    ap.add_argument("--sheet", default=None, help="worksheet (default: Dataset)")
    ap.add_argument("--out", default=".", help="output directory for model and reports")
    args = ap.parse_args(argv)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    # ---- data (single stratified train/test split, random_state = 42) ----
    frame, y = pp.load_task_frame(args.data, task="mc", sheet_name=args.sheet)
    X = frame[pp.FEATURES]
    X_train, X_test, y_train, y_test = pp.split_task(X, y, task="mc")
    print(f"train/test: {len(X_train)}/{len(X_test)} records "
          f"(stratified; classes {list(pp.CLASSES)})")

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=pp.SEED)

    # ---- screening of ten classifiers (5-fold macro-F1 on the training set) ----
    # Every candidate is evaluated through the SAME leakage-safe pipeline, so
    # imputation/scaling are refitted inside each fold for each classifier.
    le = LabelEncoder().fit(y_train)          # integer labels for XGBoost & co.
    y_train_enc = pd.Series(le.transform(y_train), index=y_train.index)
    screening = []
    for name, clf in make_candidates(pp.SEED).items():
        scores = cross_val_score(pp.build_pipeline(clf), X_train, y_train_enc,
                                 cv=cv, scoring="f1_macro", n_jobs=-1)
        screening.append({"Classifier": name, "Macro_F1_mean": scores.mean(),
                          "Macro_F1_std": scores.std()})
        print(f"{name:10s} 5-fold macro-F1 = {scores.mean():.4f} +/- {scores.std():.4f}")
    screening_df = pd.DataFrame(screening).sort_values("Macro_F1_mean", ascending=False)
    screening_df.to_csv(out / "mcirp_classifier_screening.csv", index=False,
                        encoding="utf-8-sig")

    # ---- Random Forest hyperparameter search (5-fold CV on training set) ----
    search = GridSearchCV(
        estimator=pp.build_pipeline(RandomForestClassifier(random_state=pp.SEED)),
        param_grid=PARAM_GRID,
        cv=cv,
        scoring="f1_macro",
        n_jobs=-1,
        refit=True,                # refits the winner on the whole training set
        return_train_score=True,
        verbose=1,
    )
    n_fits = int(np.prod([len(v) for v in PARAM_GRID.values()])) * 5
    print(f"grid search: {int(n_fits/5)} combinations x 5 folds = {n_fits} fits")
    search.fit(X_train, y_train)
    print(f"best 5-fold CV macro-F1 = {search.best_score_:.4f}")
    print("best parameters:", search.best_params_)

    # ---- final model = the tuned model (no hand-copied parameters) ----
    model = search.best_estimator_

    # ---- held-out test evaluation (cf. Table S11) ----
    pred = model.predict(X_test)
    acc = accuracy_score(y_test, pred)
    prec = precision_score(y_test, pred, average="macro", zero_division=0)
    rec = recall_score(y_test, pred, average="macro", zero_division=0)
    f1 = f1_score(y_test, pred, average="macro", zero_division=0)
    cm = confusion_matrix(y_test, pred, labels=pp.CLASSES)
    print(f"Accuracy  = {acc:.4f}\nPrecision = {prec:.4f}\nRecall    = {rec:.4f}"
          f"\nF1 (macro)= {f1:.4f}")
    print("Confusion matrix (rows = observed, cols = predicted,",
          pp.CLASSES, "):\n", cm)

    # ---- persist model + full tuning record ----
    joblib.dump(model, out / MODEL_FILE)
    best_params = {k.replace("estimator__", ""): v for k, v in search.best_params_.items()}
    with open(out / "mcirp_grid_search_best_params.json", "w", encoding="utf-8") as fh:
        json.dump({"best_params": best_params,
                   "best_cv_macro_F1": float(search.best_score_),
                   "test_accuracy": float(acc), "test_precision_macro": float(prec),
                   "test_recall_macro": float(rec), "test_f1_macro": float(f1),
                   "confusion_matrix_labels": pp.CLASSES,
                   "confusion_matrix": cm.tolist(),
                   "n_train": int(len(X_train)), "n_test": int(len(X_test))},
                  fh, indent=2)
    results = pd.DataFrame(search.cv_results_)
    keep = ([c for c in results.columns if c.startswith("param_")]
            + ["mean_test_score", "std_test_score", "mean_train_score",
               "std_train_score", "rank_test_score", "mean_fit_time"])
    results[keep].sort_values("rank_test_score").to_csv(
        out / "mcirp_grid_search_results.csv", index=False, encoding="utf-8-sig")

    # ---- informational consistency check against SI Table S10 ----
    match = {k: (best_params.get(k) == v) for k, v in TABLE_S10_REFERENCE.items()}
    print("Table S10 reference values:", TABLE_S10_REFERENCE)
    print("match with search result:", match,
          "-> the reported values are reproduced" if all(match.values())
          else "-> NOTE: search result differs from Table S10; report the values "
               "produced by this run.")

    print(f"model saved to {out / MODEL_FILE}")


if __name__ == "__main__":
    main()
