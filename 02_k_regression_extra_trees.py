#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
02_k_regression_extra_trees.py
==============================
Regression of the apparent degradation rate constant k (min-1):
five-fold cross-validated grid search over the Extra Trees hyperparameter
space, followed by final-model training and held-out test evaluation.
Corresponds to Table S8 (best hyperparameters), Table S9 (evaluation metrics)
"""

from __future__ import annotations

import argparse
import json
from importlib import import_module
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV, KFold

pp = import_module("01_data_preprocessing")

# ---- hyperparameter search space (Extra Trees; cf. Table S8) ----
PARAM_GRID = {
    "estimator__n_estimators": [100, 200, 300, 500],
    "estimator__max_depth": [4, 6, 8, 10, 12, 15, None],
    "estimator__min_samples_leaf": [1, 2, 4, 8, 16],
    "estimator__min_samples_split": [2, 4, 8, 16, 32, 64],
    "estimator__max_features": [0.3, 0.5, 0.7, "sqrt", None],
    "estimator__ccp_alpha": [0.0, 1e-6, 1e-5, 1e-4, 3e-4, 1e-3],
    "estimator__criterion": ["squared_error", "friedman_mse", "absolute_error"],
}

# Reference values reported for Extra Trees in SI Table S8 (n_estimators is
# not part of Table S8). Used only for the informational consistency printout
# at the end of the run -- never to override the search result.
TABLE_S8_REFERENCE = {"min_samples_split": 8, "min_samples_leaf": 1,
                      "max_features": 0.7, "max_depth": 15,
                      "criterion": "friedman_mse", "ccp_alpha": 1e-6}

MODEL_FILE = "extra_trees_k_model.pkl"


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", default=pp.DATA_PATH, help="input workbook")
    ap.add_argument("--sheet", default=None, help="worksheet (default: Dataset_S2_k)")
    ap.add_argument("--out", default=".", help="output directory for model and reports")
    args = ap.parse_args(argv)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    # ---- data ----
    frame, y = pp.load_task_frame(args.data, task="k", sheet_name=args.sheet)
    X = frame[pp.FEATURES]
    X_train, X_test, y_train, y_test = pp.split_task(X, y, task="k")
    print(f"train/test: {len(X_train)}/{len(X_test)} records")

    # ---- hyperparameter search (5-fold CV on the training set) ----
    # The search operates on the full pipeline, so preprocessing (KNN
    # imputation + standardization of the numeric block only) is refitted
    # inside every CV fold; OA and fingerprint bits are never scaled.
    search = GridSearchCV(
        estimator=pp.build_pipeline(
            ExtraTreesRegressor(random_state=pp.SEED, n_jobs=-1)),
        param_grid=PARAM_GRID,
        cv=KFold(n_splits=5, shuffle=True, random_state=pp.SEED),
        scoring="r2",
        n_jobs=-1,
        refit=True,                # refits the winner on the whole training set
        return_train_score=True,
        verbose=1,
    )
    n_fits = int(np.prod([len(v) for v in PARAM_GRID.values()])) * 5
    print(f"grid search: {int(n_fits/5)} combinations x 5 folds = {n_fits} fits")
    search.fit(X_train, y_train)
    print(f"best 5-fold CV R2 = {search.best_score_:.4f}")
    print("best parameters:", search.best_params_)

    # ---- final model = the tuned model (no hand-copied parameters) ----
    model = search.best_estimator_

    # ---- held-out test evaluation (R2, RMSE, MAE; cf. Table S9) ----
    pred = model.predict(X_test)
    r2 = r2_score(y_test, pred)
    rmse = float(np.sqrt(mean_squared_error(y_test, pred)))
    mae = float(mean_absolute_error(y_test, pred))
    print(f"test set: R2 = {r2:.3f}, RMSE = {rmse:.3f} min-1, MAE = {mae:.3f} min-1")

    # ---- persist model + full tuning record ----
    joblib.dump(model, out / MODEL_FILE)
    best_params = {k.replace("estimator__", ""): (v if not isinstance(v, float) else float(v))
                   for k, v in search.best_params_.items()}
    with open(out / "k_grid_search_best_params.json", "w", encoding="utf-8") as fh:
        json.dump({"best_params": best_params,
                   "best_cv_R2": float(search.best_score_),
                   "test_R2": float(r2), "test_RMSE": rmse, "test_MAE": mae,
                   "n_train": int(len(X_train)), "n_test": int(len(X_test))},
                  fh, indent=2)
    results = pd.DataFrame(search.cv_results_)
    keep = ([c for c in results.columns if c.startswith("param_")]
            + ["mean_test_score", "std_test_score", "mean_train_score",
               "std_train_score", "rank_test_score", "mean_fit_time"])
    results[keep].sort_values("rank_test_score").to_csv(
        out / "k_grid_search_results.csv", index=False, encoding="utf-8-sig")

    # ---- informational consistency check against SI Table S8 ----
    match = {k: (best_params.get(k) == v) for k, v in TABLE_S8_REFERENCE.items()}
    print("Table S8 reference values:", TABLE_S8_REFERENCE)
    print("match with search result:", match,
          "-> the reported values are reproduced" if all(match.values())
          else "-> NOTE: search result differs from Table S8; report the values "
               "produced by this run.")

    print(f"model saved to {out / MODEL_FILE}")


if __name__ == "__main__":
    main()
