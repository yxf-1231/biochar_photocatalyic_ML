#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
01_data_preprocessing.py
========================
Shared data loading and leakage-safe preprocessing for the dual-task framework
(regression of the apparent rate constant *k*; classification of the dominant
reaction pathway, MCIRP: radical / non-radical / mixed).

"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import KNNImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# --------------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------------
SEED = 42
RANDOM_STATE = SEED                       # kept for backward compatibility
RADIUS, N_BITS = 2, 2048

DATA_PATH = "Dataset.xlsx"                # consolidated dataset (711 records)

NAMES = ["C", "N", "O", "SSA", "IdIg", "CD", "LI", "RT", "T", "AC", "PH"]
FP = [f"fp{i:04d}" for i in range(N_BITS)]
CLASSES = ["radical", "non-radical", "mixed"]

# Publication-style axis labels keyed by the internal names above.
DISPLAY = {"C": "C (%)", "N": "N (%)", "O": "O (%)", "SSA": "SSA (m²/g)",
           "IdIg": "Id/Ig", "CD": "CD (g/L)", "LI": "LI (W)", "RT": "RT (min)",
           "T": "T (℃)", "AC": "AC (mg/L)", "PH": "PH", "OA_binary": "OA"}

# Both header conventions encountered in the source workbooks are accepted.
ALIASES = {
    "C %": "C", "C (%)": "C", "N %": "N", "N (%)": "N", "O %": "O", "O (%)": "O",
    "SSA m2/g": "SSA", "SSA (m2/g)": "SSA", "Id/Ig": "IdIg",
    "CD g/L": "CD", "CD (g/L)": "CD", "LI W": "LI", "LI (W)": "LI",
    "RT min": "RT", "RT (min)": "RT", "T ℃": "T", "T (℃)": "T",
    "AC mg/L": "AC", "AC (mg/L)": "AC", "PH": "PH", "OA": "OA",
    "Smiles": "Smiles", "SMILES": "Smiles",
}
K_TARGETS = ["k min-1", "k (min-1)", "k (min−1)", "k min−1", "k(min-1)", "k"]
MC_TARGETS = ["MCIRP", "Mcirp", "mcirp", "Mechanism"]

EXPECTED_N = {"k": 382, "mc": 711}        # consolidated dataset accompanying the revision


# --------------------------------------------------------------------------
# Data loading
# --------------------------------------------------------------------------
def _resolve_sheet(path: Path, preferred: str | None) -> str | int:
    sheets = pd.ExcelFile(path).sheet_names
    if preferred and preferred in sheets:
        return preferred
    return sheets[0]


def morgan_matrix(smiles: pd.Series) -> np.ndarray:
    """Morgan fingerprints (radius = 2, 2048 bits); invalid SMILES -> all-zero row."""
    from rdkit import Chem, RDLogger
    from rdkit.Chem import rdFingerprintGenerator
    RDLogger.DisableLog("rdApp.*")
    gen = rdFingerprintGenerator.GetMorganGenerator(radius=RADIUS, fpSize=N_BITS)
    mols = [Chem.MolFromSmiles(str(s)) for s in smiles]
    out = np.zeros((len(mols), N_BITS), dtype=np.float32)
    valid = [i for i, m in enumerate(mols) if m is not None]
    for row, i in enumerate(valid):
        out[row] = gen.GetFingerprintAsNumPy(mols[i])
    n_bad = len(mols) - len(valid)
    if n_bad:
        print(f"  warning: {n_bad} SMILES could not be parsed; "
              f"their fingerprint rows are all zeros.")
    return out


def load_task_frame(file_path: str | Path = DATA_PATH, task: str = "k",
                    sheet_name: str | None = None):
    """Read one task worksheet and return ``(frame, y)``.

    ``frame`` contains the 11 numeric descriptors (internal names), OA_binary,
    Excel_row (traceability back to the source file) and the 2048 fingerprint
    columns. Rows without a valid target are dropped; targets are never
    imputed. All other missing descriptor values are left as NaN and are
    handled exclusively by the fold-internal KNN imputer.
    """
    file_path = Path(file_path)
    if not file_path.exists():
        raise FileNotFoundError(f"input workbook not found: {file_path}")
    if task not in ("k", "mc", "mcirp"):
        raise ValueError("task must be 'k' or 'mc'")
    task = "mc" if task == "mcirp" else task
    default_sheet = K_SHEET if task == "k" else MC_SHEET
    d = pd.read_excel(file_path, sheet_name=_resolve_sheet(file_path, sheet_name or default_sheet))
    d.columns = d.columns.astype(str).str.strip().str.replace(r"\s+", " ", regex=True)

    target_col = next((c for c in (K_TARGETS if task == "k" else MC_TARGETS)
                       if c in d.columns), None)
    if target_col is None:
        raise KeyError(f"{task}: no target column found in {file_path.name} "
                       f"(tried {K_TARGETS if task == 'k' else MC_TARGETS})")
    missing = [c for c in ("Smiles", "OA") if c not in d.columns]
    if missing:
        raise KeyError(f"{file_path.name}: required columns missing: {missing}")

    out = pd.DataFrame({new: d[old] for old, new in ALIASES.items()
                        if old in d.columns and new not in ("Smiles", "OA")})
    out["Smiles"] = d["Smiles"].astype(str).str.strip()
    out["OA"] = d["OA"].astype(str).str.strip().str.upper()
    bad = sorted(set(out["OA"]) - {"Y", "N"})
    if bad:
        raise ValueError(f"unexpected OA labels {bad}; expected Y/N")
    out["OA_binary"] = out["OA"].map({"Y": 1, "N": 0}).astype(int)
    out["Excel_row"] = np.arange(2, len(out) + 2)

    for col in NAMES:
        out[col] = pd.to_numeric(out[col], errors="coerce")

    if task == "k":
        y = pd.to_numeric(d[target_col], errors="coerce")
        keep = y > 0                       # targets are never imputed
    else:
        y = d[target_col].astype(str).str.strip().str.lower().replace({"both": "mixed"})
        keep = y.isin(CLASSES)
    out = out.loc[keep].reset_index(drop=True)
    y = y.loc[keep].reset_index(drop=True)

    fp = morgan_matrix(out["Smiles"])
    out = pd.concat([out, pd.DataFrame(fp, columns=FP)], axis=1)

    n_expected = EXPECTED_N[task]
    status = "OK" if len(out) == n_expected else "WARNING"
    print(f"  [{task}] {len(out)} records loaded from {file_path.name} "
          f"(expected {n_expected} for the consolidated dataset: {status})")
    return out, y


# --------------------------------------------------------------------------
# Leakage-safe preprocessing (Manuscript Section 2.3)
# --------------------------------------------------------------------------
def build_preprocessor() -> ColumnTransformer:
    """ColumnTransformer: numeric KNN-impute + scale; OA and fingerprints pass through.

    The transformer is intentionally UNFITTED: it is refitted inside every
    train/test split and every cross-validation fold via ``build_pipeline``.
    """
    numeric = Pipeline([
        ("KNN5_training_only", KNNImputer(n_neighbors=5, weights="distance",
                                          keep_empty_features=True)),
        ("StandardScaler_training_only", StandardScaler()),
    ])
    return ColumnTransformer(
        [("numeric", numeric, NAMES),
         ("OA", "passthrough", ["OA_binary"]),
         ("Morgan2048", "passthrough", FP)],
        sparse_threshold=0.0,
    )


FEATURES = NAMES + ["OA_binary"] + FP


def build_pipeline(estimator) -> Pipeline:
    """Leakage-safe model pipeline: preprocessing (refit per fold) + estimator."""
    return Pipeline([("preprocess", build_preprocessor()),
                     ("estimator", estimator)])


def split_task(X, y, task: str = "k", test_size: float = 0.2):
    """The manuscript's train/test split.

    Regression (k): random 80/20 split. Classification (MCIRP): stratified
    80/20 split ("stratified splitting applied to the MCIRP classification
    task", Section 2.3). Hyperparameters are optimized separately by five-fold
    cross-validation on the training set (Section 2.4), so no additional
    validation split is carved out here.
    """
    stratify = y if task in ("mc", "mcirp") else None
    return train_test_split(X, y, test_size=test_size,
                            random_state=SEED, stratify=stratify)
