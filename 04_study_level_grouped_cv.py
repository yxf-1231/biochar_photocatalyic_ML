#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
04_study_level_grouped_cv.py
============================
Full-dataset 5-fold out-of-fold (OOF) validation of the two published models
under both a conventional random record-wise split and the
experimental-setting-grouped split described in the response letter
"""

from __future__ import annotations

import argparse
import hashlib
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import ExtraTreesRegressor, RandomForestClassifier
from sklearn.impute import KNNImputer
from sklearn.metrics import (accuracy_score, balanced_accuracy_score,
                             confusion_matrix, f1_score, mean_absolute_error,
                             mean_squared_error, r2_score, recall_score)
from sklearn.model_selection import GroupKFold, KFold, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=FutureWarning)

SEED = 42
NAMES = ['C', 'N', 'O', 'SSA', 'IdIg', 'CD', 'LI', 'RT', 'T', 'AC', 'PH']
FP = [f'fp{i:04d}' for i in range(2048)]
CLASSES = ['radical', 'non-radical', 'mixed']
# Columns defining one experimental setting ("complete experimental-setting
# groups ... were withheld"). This is a convenient proxy, NOT a verified
# publication ID.
SETTING = ['Smiles', 'CD', 'LI', 'RT', 'T', 'AC', 'PH', 'OA']
EXPECTED_N = {'k': 382, 'mc': 711}   # consolidated dataset accompanying the revision

# Header conventions accepted on input (renamed to the internal names above).
KIN = {'C %': 'C', 'N %': 'N', 'O %': 'O', 'SSA m2/g': 'SSA', 'Id/Ig': 'IdIg',
       'CD g/L': 'CD', 'LI W': 'LI', 'RT min': 'RT', 'T ℃': 'T', 'AC mg/L': 'AC',
       'PH': 'PH', 'OA': 'OA', 'Smiles': 'Smiles', 'k (min-1)': 'target'}
MIN = {'C (%)': 'C', 'N (%)': 'N', 'O (%)': 'O', 'SSA (m2/g)': 'SSA', 'Id/Ig': 'IdIg',
       'CD (g/L)': 'CD', 'LI (W)': 'LI', 'RT (min)': 'RT', 'T (℃)': 'T',
       'AC (mg/L)': 'AC', 'PH': 'PH', 'OA': 'OA', 'Smiles': 'Smiles', 'MCIRP': 'target'}

MODEL_CONFIG = {
    'k': {'n_estimators': 100, 'max_depth': 15, 'min_samples_split': 8,
          'min_samples_leaf': 1, 'max_features': 0.7, 'ccp_alpha': 1e-6},
    'mc': {'n_estimators': 100, 'max_depth': 16, 'min_samples_split': 20,
           'min_samples_leaf': 1},
}


# --------------------------------------------------------------------------
# Shared model core (inlined from original_model_core.py)
# --------------------------------------------------------------------------

def read_dataset(file_path: str | Path, task: str, sheet_name=None):
    """Read one task workbook and return (frame, y).

    The frame carries the 11 numeric descriptors, OA_binary, Proxy_group,
    Excel_row (1-based data row number matching the original Excel file) and
    the 2048 Morgan-fingerprint columns; rows without a valid target are
    dropped.
    """
    file_path = Path(file_path)
    if not file_path.exists():
        raise FileNotFoundError(f'{task} input file not found: {file_path}')
    d = pd.read_excel(file_path, sheet_name=sheet_name)
    if isinstance(d, dict):                      # sheet_name=None -> all sheets
        raise ValueError(f'Could not resolve a worksheet in {file_path}; '
                         f'available: {list(d)}. Pass --k-sheet/--mc-sheet.')
    rename = {}
    for canon, cand in (('k (min-1)', ['k (min-1)', 'k min-1', 'k (min−1)', 'k min−1',
                                       'k(min-1)', 'k (min^-1)', 'k']),
                        ('MCIRP', ['MCIRP', 'Mcirp', 'mcirp', 'Mechanism'])):
        for c in cand:
            if c in d.columns:
                rename[c] = canon
                break
    d = d.rename(columns=rename)
    # accept both header conventions for either task
    mapping = dict(MIN)
    mapping.update(KIN)
    src = {}
    for old, new in mapping.items():
        if old in d.columns:
            src[new] = d[old]
    target_col = 'k (min-1)' if task == 'k' else 'MCIRP'
    if target_col not in d.columns:
        raise KeyError(f'{task}: target column {target_col!r} not found in {file_path.name}')
    src['target'] = d[target_col]
    out = pd.DataFrame(src)
    out['Excel_row'] = np.arange(2, len(out) + 2)   # Excel row incl. header
    for col in NAMES:
        out[col] = pd.to_numeric(out[col], errors='coerce')
    out['Smiles'] = out['Smiles'].astype(str).str.strip()
    out['OA'] = out['OA'].astype(str).str.strip().str.upper()
    bad_oa = sorted(set(out['OA']) - {'Y', 'N'})
    if bad_oa:
        raise ValueError(f'Unexpected OA labels {bad_oa}; expected Y/N')
    out['OA_binary'] = out['OA'].map({'Y': 1, 'N': 0}).astype(int)
    if task == 'k':
        out['target'] = pd.to_numeric(out['target'], errors='coerce')
        out = out[out['target'] > 0].copy()
    else:
        out['target'] = (out['target'].astype(str).str.strip().str.lower()
                         .replace({'both': 'mixed'}))
        out = out[out['target'].isin(CLASSES)].copy()
    out['Proxy_group'] = pd.util.hash_pandas_object(
        out[SETTING].fillna('<NA>').astype(str), index=False).astype(str)
    out = out.reset_index(drop=True)
    y = out['target'].copy()

    # Morgan fingerprints, radius 2, 2048 bits (invalid SMILES -> all-zero row)
    from rdkit import Chem, RDLogger
    from rdkit.Chem import rdFingerprintGenerator
    RDLogger.DisableLog('rdApp.*')
    gen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
    smis = out['Smiles'].tolist()
    mols = [Chem.MolFromSmiles(s) for s in smis]
    valid = [i for i, m in enumerate(mols) if m is not None]
    bits = np.zeros((len(valid), 2048), dtype=np.uint8)
    for row, i in enumerate(valid):
        bits[row] = gen.GetFingerprintAsNumPy(mols[i])
    fp_mat = np.zeros((len(out), 2048), dtype=np.float32)
    if len(valid):
        fp_mat[np.array(valid)] = bits.astype(np.float32)
    if len(valid) < len(out):
        print(f'  [{task}] warning: {len(out) - len(valid)} SMILES could not be '
              f'parsed; their fingerprint rows are all zeros.')
    out = pd.concat([out.reset_index(drop=True),
                     pd.DataFrame(fp_mat, columns=FP)], axis=1)
    missing = [c for c in NAMES + ['OA_binary'] + FP if c not in out.columns]
    if missing:
        raise KeyError(f'Missing feature columns: {missing[:5]}...')
    return out, y


def pipeline(task: str) -> Pipeline:
    """Fixed model + fold-internal preprocessing.

    The KNN imputer and the scaler are refit on the training rows of every
    fold, so imputation/scaling statistics never see held-out data.
    """
    pre = ColumnTransformer(
        [('numeric', Pipeline([
            ('KNN5_training_only', KNNImputer(n_neighbors=5, weights='distance',
                                              keep_empty_features=True)),
            ('StandardScaler_training_only', StandardScaler())]), NAMES),
         ('OA', 'passthrough', ['OA_binary']),
         ('Morgan2048', 'passthrough', FP)],
        sparse_threshold=0.0)
    if task == 'k':
        est = ExtraTreesRegressor(random_state=SEED, n_jobs=-1, **MODEL_CONFIG['k'])
    else:
        est = RandomForestClassifier(random_state=SEED, n_jobs=-1, **MODEL_CONFIG['mc'])
    return Pipeline([('preprocess_refits_per_fold', pre), ('fixed_estimator', est)])


def score(task: str, y_true, y_pred) -> dict:
    """Metric set reported in the manuscript / SI (Table S15 convention)."""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    if task == 'k':
        rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
        mae = float(mean_absolute_error(y_true, y_pred))
        r2 = float(r2_score(y_true, y_pred))
        yt = np.log10(np.clip(y_true, 1e-12, None))
        yp = np.log10(np.clip(y_pred, 1e-9, None))
        mask = y_true <= 0.1
        return {'R2': r2, 'RMSE': rmse, 'MAE': mae,
                'Log10_R2': float(r2_score(yt, yp)),
                'Log10_MAE': float(mean_absolute_error(yt, yp)),
                'N_nonpositive_k_pred': int((y_pred <= 0).sum()),
                'N_k_le_0.1': int(mask.sum()),
                'R2_k_le_0.1': (float(r2_score(y_true[mask], y_pred[mask]))
                                if mask.sum() >= 2 else np.nan)}
    return {'Accuracy': float(accuracy_score(y_true, y_pred)),
            'Macro_F1': float(f1_score(y_true, y_pred, labels=CLASSES,
                                       average='macro', zero_division=0)),
            'Balanced_accuracy': float(balanced_accuracy_score(y_true, y_pred)),
            'Recall_radical': float(recall_score(y_true, y_pred, labels=CLASSES,
                                                 average=None, zero_division=0)[0]),
            'Recall_non_radical': float(recall_score(y_true, y_pred, labels=CLASSES,
                                                     average=None, zero_division=0)[1]),
            'Recall_mixed': float(recall_score(y_true, y_pred, labels=CLASSES,
                                               average=None, zero_division=0)[2])}


# --------------------------------------------------------------------------
# 5-fold full-data OOF validation (random + experimental-setting-grouped)
# --------------------------------------------------------------------------

def get_folds(task, y, groups, mode, group_col='Proxy_group'):
    """Yield (train_idx, test_idx) for the requested split protocol."""
    X = np.zeros((len(y), 1))
    if mode == 'random':
        cv = (KFold(n_splits=5, shuffle=True, random_state=SEED) if task == 'k'
              else StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED))
        return list(cv.split(X, y))
    if mode == 'grouped':
        return list(GroupKFold(n_splits=5, shuffle=True, random_state=SEED)
                    .split(X, y, groups))
    raise ValueError(f'unknown mode: {mode}')


def full_cv(task, d, y, mode, group_col='Proxy_group'):
    """Run one 5-fold OOF validation and return (summary, folds, predictions)."""
    groups = d[group_col].astype(str)
    n_groups = int(groups.nunique())
    feats = NAMES + ['OA_binary'] + FP
    fold_rows, pred_rows = [], []
    for fold, (tr, te) in enumerate(get_folds(task, y, groups, mode, group_col), 1):
        overlap = set(groups.iloc[tr]) & set(groups.iloc[te])
        if mode == 'grouped':
            assert not overlap, f'group leakage in fold {fold}: {sorted(overlap)[:5]}'
        model = pipeline(task)
        model.fit(d.iloc[tr][feats], y.iloc[tr])
        pred = model.predict(d.iloc[te][feats])
        s = score(task, y.iloc[te], pred)
        fold_rows.append({'Task': task, 'Validation': mode, 'Fold': fold,
                          'N_train': int(len(tr)), 'N_test': int(len(te)),
                          'Train_groups': int(groups.iloc[tr].nunique()),
                          'Test_groups': int(groups.iloc[te].nunique()),
                          'Proxy_group_overlap': int(len(overlap)), **s})
        pred_rows.append(pd.DataFrame({
            'Task': task, 'Validation': mode, 'Fold': fold,
            'Excel_row': d.iloc[te]['Excel_row'].to_numpy(),
            'Proxy_group': groups.iloc[te].to_numpy(),
            'Observed': y.iloc[te].to_numpy(),
            'Predicted': np.asarray(pred)}))
    preds = pd.concat(pred_rows, ignore_index=True)
    assert len(preds) == len(y), 'every record must be predicted exactly once'
    assert preds['Excel_row'].is_unique, 'duplicate OOF predictions detected'
    pooled = score(task, y.to_numpy(),
                   preds.sort_values('Excel_row')['Predicted'].to_numpy())
    summary = {'Task': task, 'Validation': mode, 'Split': '5-fold full-data OOF',
               'Group_column': group_col if mode == 'grouped' else '(none)',
               'N_total': int(len(y)), 'N_distinct_proxy_groups': n_groups,
               'N_folds': 5, 'Coverage': 'each record held out exactly once',
               **pooled}
    return summary, pd.DataFrame(fold_rows), preds


# --------------------------------------------------------------------------
# Output helpers
# --------------------------------------------------------------------------

def formatted_excel(tables: dict, save_path: Path):
    """Write all result tables to one formatted workbook."""
    from openpyxl.styles import Alignment, Border, Font, PatternFill, Side
    from openpyxl.utils import get_column_letter
    with pd.ExcelWriter(save_path, engine='openpyxl') as writer:
        for name, df in tables.items():
            df.to_excel(writer, sheet_name=name[:31], index=False)
    from openpyxl import load_workbook
    wb = load_workbook(save_path)
    fill = PatternFill('solid', start_color='163754')
    band = PatternFill('solid', start_color='D9E2F3')
    thin = Border(*[Side(style='thin', color='BFBFBF')] * 4)
    for ws in wb.worksheets:
        for cell in ws[1]:
            cell.font = Font(bold=True, color='FFFFFF')
            cell.fill = fill
            cell.alignment = Alignment(horizontal='center', vertical='center',
                                       wrap_text=True)
        for row in ws.iter_rows(min_row=2):
            for cell in row:
                cell.border = thin
                if isinstance(cell.value, float):
                    cell.number_format = '0.0000'
            if row[0].row % 2 == 0:
                for cell in row:
                    cell.fill = band
        for idx, col in enumerate(ws.columns, 1):
            width = max((len(str(c.value)) for c in col if c.value is not None),
                        default=8)
            ws.column_dimensions[get_column_letter(idx)].width = min(max(width + 2, 10), 42)
        ws.freeze_panes = 'A2'
    wb.save(save_path)


def choose_font():
    """Prefer Arial/Times-style fonts; fall back to whatever is installed."""
    from matplotlib import font_manager
    installed = {f.name for f in font_manager.fontManager.ttflist}
    for cand in ('Arial', 'Times New Roman', 'Liberation Sans', 'DejaVu Sans'):
        if cand in installed:
            return cand
    return 'DejaVu Sans'


MODE_LABEL = {'random': 'Random 5-fold split',
              'grouped': 'Experimental-setting-grouped 5-fold split'}
FOLD_COLORS = plt.get_cmap('Set2').colors


def k_parity(preds_k: dict, outdir: Path, font: str):
    """Log-log parity plot of OOF predicted vs observed k, coloured by fold."""
    fig, axes = plt.subplots(1, 2, figsize=(11, 5.2), sharex=True, sharey=True)
    for ax, mode in zip(axes, ['random', 'grouped']):
        p = preds_k[mode]
        x = np.clip(p['Observed'].astype(float).to_numpy(), 1e-12, None)
        yv = np.clip(p['Predicted'].astype(float).to_numpy(), 1e-9, None)
        for fold in sorted(p['Fold'].unique()):
            m = p['Fold'].to_numpy() == fold
            ax.scatter(x[m], yv[m], s=14, alpha=0.75,
                       color=FOLD_COLORS[(fold - 1) % len(FOLD_COLORS)],
                       edgecolors='none', label=f'Fold {fold}')
        lims = [1e-5, 1e1]
        ax.plot(lims, lims, 'k-', lw=1)
        ax.plot(lims, [l * 10 ** 0.5 for l in lims], 'k:', lw=0.8)
        ax.plot(lims, [l * 10 ** -0.5 for l in lims], 'k:', lw=0.8)
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.set_xlim(lims)
        ax.set_ylim(lims)
        ax.set_xlabel('Observed $\\it{k}$ (min$^{-1}$)', fontname=font, fontsize=11)
        ax.set_title(MODE_LABEL[mode], fontname=font, fontsize=12)
        ax.grid(alpha=0.25, which='both')
    axes[0].set_ylabel('Out-of-fold predicted $\\it{k}$ (min$^{-1}$)',
                       fontname=font, fontsize=11)
    axes[1].legend(frameon=False, fontsize=9, loc='lower right')
    fig.tight_layout()
    for ext in ('png', 'pdf'):
        fig.savefig(outdir / f'Figure_04_K_setting_5fold_parity.{ext}', dpi=300)
    plt.close(fig)


def mc_confusion(preds_mc: dict, outdir: Path, font: str):
    """Confusion matrices of the pooled OOF predictions (random vs grouped)."""
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.8))
    vmax = max(confusion_matrix(preds_mc[m]['Observed'], preds_mc[m]['Predicted'],
                                labels=CLASSES).max() for m in ('random', 'grouped'))
    for ax, mode in zip(axes, ['random', 'grouped']):
        cm = confusion_matrix(preds_mc[mode]['Observed'],
                              preds_mc[mode]['Predicted'], labels=CLASSES)
        im = ax.imshow(cm, cmap='Blues', vmin=0, vmax=vmax)
        for i in range(len(CLASSES)):
            for j in range(len(CLASSES)):
                ax.text(j, i, str(cm[i, j]), ha='center', va='center',
                        fontname=font, fontsize=12,
                        color='white' if cm[i, j] > vmax / 2 else 'black')
        ax.set_xticks(range(len(CLASSES)), CLASSES, fontname=font)
        ax.set_yticks(range(len(CLASSES)), CLASSES, fontname=font)
        ax.set_xlabel('Predicted pathway', fontname=font, fontsize=11)
        ax.set_title(MODE_LABEL[mode], fontname=font, fontsize=12)
    axes[0].set_ylabel('Observed pathway', fontname=font, fontsize=11)
    fig.colorbar(im, ax=axes, shrink=0.85, label='Number of records')
    fig.tight_layout()
    for ext in ('png', 'pdf'):
        fig.savefig(outdir / f'Figure_05_MC_setting_5fold_confusion.{ext}', dpi=300)
    plt.close(fig)


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------

def _autodetect(patterns):
    here = Path.cwd()
    for pat in patterns:
        hits = sorted(here.glob(pat))
        if hits:
            return hits[0]
    return None


def _resolve_sheet(path: Path, preferred: str | None, fallbacks) -> str:
    if preferred:
        return preferred
    sheets = pd.ExcelFile(path).sheet_names
    for cand in fallbacks:
        if cand in sheets:
            return cand
    return sheets[0]


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--k-file', default=None,
                    help='workbook with the k records (default: auto-detect)')
    ap.add_argument('--mc-file', default=None,
                    help='workbook with the MCIRP records (default: auto-detect)')
    ap.add_argument('--k-sheet', default=None,
                    help="sheet of --k-file (default: 'Dataset_S2_k' if present)")
    ap.add_argument('--mc-sheet', default=None,
                    help="sheet of --mc-file (default: 'Dataset' if present)")
    ap.add_argument('--group-col', default='Proxy_group',
                    help="grouping column for the grouped protocol "
                         "(default: Proxy_group; use e.g. 'Study_ID' for a "
                         "genuinely study-disjoint split)")
    ap.add_argument('--out', default='grouped_cv_output', help='output directory')
    args = ap.parse_args()

    k_file = Path(args.k_file) if args.k_file else _autodetect(
        ['Dataset.xlsx', '*dedup.xlsx', '*3.xlsx', '*.xlsx'])
    mc_file = Path(args.mc_file) if args.mc_file else k_file
    if k_file is None or not Path(k_file).exists():
        sys.exit('No input workbook found. Put Dataset.xlsx next to this script '
                 'or pass --k-file/--mc-file.')
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    k_sheet = _resolve_sheet(Path(k_file), args.k_sheet,
                             ['Dataset_S2_k', 'k records', 'Sheet1'])
    mc_sheet = _resolve_sheet(Path(mc_file), args.mc_sheet,
                              ['Dataset', 'MCIRP records', 'Sheet1'])
    print(f'k input     : {k_file} [{k_sheet}]')
    print(f'MCIRP input : {mc_file} [{mc_sheet}]')

    dk, yk = read_dataset(k_file, 'k', sheet_name=k_sheet)
    dm, ym = read_dataset(mc_file, 'mc', sheet_name=mc_sheet)
    for task, n in (('k', len(dk)), ('mc', len(dm))):
        status = 'OK' if n == EXPECTED_N[task] else 'WARNING'
        print(f'  [{task}] N = {n} (expected {EXPECTED_N[task]} for the '
              f'consolidated dataset: {status})')

    summaries, folds_all = [], []
    preds_k, preds_mc = {}, {}
    for mode in ('random', 'grouped'):
        print(f'\n=== k regression, {mode} 5-fold OOF ===')
        s, f, p = full_cv('k', dk, yk, mode, args.group_col)
        summaries.append(s)
        folds_all.append(f)
        preds_k[mode] = p
        print('  pooled:', {k2: round(v, 4) for k2, v in s.items()
                             if isinstance(v, float)})
        print(f'=== MCIRP classification, {mode} 5-fold OOF ===')
        s, f, p = full_cv('mc', dm, ym, mode, args.group_col)
        summaries.append(s)
        folds_all.append(f)
        preds_mc[mode] = p
        print('  pooled:', {k2: round(v, 4) for k2, v in s.items()
                             if isinstance(v, float)})

    cv = pd.DataFrame(summaries)
    fl = pd.concat(folds_all, ignore_index=True)
    keep = ['Task', 'Validation', 'Split', 'N_total', 'N_distinct_proxy_groups',
            'R2', 'RMSE', 'MAE', 'Log10_R2', 'Log10_MAE', 'Accuracy', 'Macro_F1',
            'Balanced_accuracy', 'Recall_radical', 'Recall_non_radical',
            'Recall_mixed', 'N_nonpositive_k_pred', 'N_k_le_0.1', 'R2_k_le_0.1']
    cv = cv[[c for c in keep if c in cv.columns]]
    pr = pd.concat([preds_k['random'], preds_k['grouped'],
                    preds_mc['random'], preds_mc['grouped']], ignore_index=True)
    scope = pd.DataFrame([
        {'Scope': 'Models',
         'Note': 'Extra Trees (k, SI Table S8) and Random Forest (MCIRP, SI '
                 'Table S10); fixed hyperparameters; n_estimators=100 assumed.'},
        {'Scope': 'Features',
         'Note': 'C, N, O, SSA, Id/Ig, CD, LI, RT, T, AC, PH + OA_binary + '
                 'Morgan fingerprints (radius 2, 2048 bits).'},
        {'Scope': 'Preprocessing',
         'Note': 'KNN imputation (n_neighbors=5, distance weights) and '
                 'standard scaling refit within every training fold only.'},
        {'Scope': 'Evaluation',
         'Note': 'Full 5-fold out-of-fold validation on all records; every '
                 'record is held out exactly once; metrics are pooled over '
                 'all OOF predictions (Table S15 convention).'},
        {'Scope': 'Random protocol',
         'Note': 'Record-wise KFold (k) / StratifiedKFold (MCIRP), '
                 'n_splits=5, shuffle=True, random_state=42.'},
        {'Scope': 'Grouped protocol',
         'Note': 'GroupKFold (n_splits=5, shuffle=True, random_state=42) over '
                 'experimental-setting proxy groups (Smiles + CD, LI, RT, T, '
                 'AC, PH, OA); zero train/test group overlap asserted per fold. '
                 'Proxy only: pass --group-col Study_ID for a study-disjoint '
                 'split once a curated study identifier is available.'},
        {'Scope': 'Reproducibility',
         'Note': f'Global seed = {SEED}. Expected N: k={EXPECTED_N["k"]}, '
                 f'MCIRP={EXPECTED_N["mc"]} for the consolidated dataset.'},
    ])

    tables = {'01_Pooled_OOF_metrics': cv,
              '02_Per_fold_metrics': fl,
              '03_OOF_predictions': pr,
              '04_Protocol_notes': scope}
    cv.to_csv(out / '04_grouped_cv_comparison.csv', index=False, encoding='utf-8-sig')
    fl.to_csv(out / '04_grouped_cv_fold_metrics.csv', index=False, encoding='utf-8-sig')
    preds_k_all = pd.concat([preds_k['random'], preds_k['grouped']], ignore_index=True)
    preds_mc_all = pd.concat([preds_mc['random'], preds_mc['grouped']], ignore_index=True)
    preds_k_all.to_csv(out / '04_grouped_cv_k_oof_predictions.csv', index=False,
                       encoding='utf-8-sig')
    preds_mc_all.to_csv(out / '04_grouped_cv_mc_oof_predictions.csv', index=False,
                        encoding='utf-8-sig')
    formatted_excel(tables, out / '04_grouped_cv_tables.xlsx')

    font = choose_font()
    k_parity(preds_k, out, font)
    mc_confusion(preds_mc, out, font)

    print('\n==== Pooled out-of-fold metrics (Table S15 convention) ====')
    with pd.option_context('display.width', 200, 'display.max_columns', 50):
        print(cv.to_string(index=False))
    print(f'\nOutputs written to: {out.resolve()}')


if __name__ == '__main__':
    main()
