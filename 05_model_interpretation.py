#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
05_model_interpretation.py
==========================
Interpretation of the trained models: SHAP analysis (global importance,
dependence plots, feature-group contributions), partial dependence plots
(1D/2D), and accumulated local effects (ALE) of carbon content on pathway
classification. Corresponds to Figs. 3-4, Figs. S8-S11 and Fig. S17 of the
manuscript.
"""

from __future__ import annotations

import argparse
from importlib import import_module
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

pp = import_module("01_data_preprocessing")

FEATURE_GROUPS = {
    "Physicochemical": ["C", "N", "O", "SSA", "IdIg"],
    "Experimental": ["CD", "LI", "RT", "T", "AC", "PH", "OA_binary"],
    "Fingerprint": "Smiles",          
}
SHAP_DEPENDENCE_FEATURES = ["C", "IdIg", "SSA"]
PDP_PAIRS = {"k": [("RT", "CD"), ("SSA", "C")],
             "mc": [("C", "O"), ("N", "LI")]}


def assign_group(feature: str) -> str:
    for group, members in FEATURE_GROUPS.items():
        if isinstance(members, str):
            if feature.startswith(members):
                return group
        elif feature in members:
            return group
    return "Other"


def _pretty(df: pd.DataFrame) -> pd.DataFrame:
    """Rename internal column names to publication-style labels for plotting."""
    return df.rename(columns={c: pp.DISPLAY.get(c, c) for c in df.columns})


def _as_class_list(shap_values, n_classes: int):
    """Normalize TreeExplainer output to a list of (n_samples, n_features)."""
    if isinstance(shap_values, list):
        return shap_values
    arr = np.asarray(shap_values)
    if arr.ndim == 3:                       # (n, p, n_classes) in recent shap
        return [arr[:, :, c] for c in range(arr.shape[2])]
    return [arr]                            # regression


def shap_analysis(model, X_raw: pd.DataFrame, task: str, outdir: Path):
    """Global importance, dependence plots and feature-group contributions."""
    import shap

    # Exactly the preprocessing the estimator was trained on.
    pre = model.named_steps["preprocess"]
    est = model.named_steps["estimator"]
    Xs = pd.DataFrame(pre.transform(X_raw), columns=pp.FEATURES, index=X_raw.index)
    Xs_plot = _pretty(Xs)

    explainer = shap.TreeExplainer(est)
    sv_list = _as_class_list(explainer.shap_values(Xs),
                             getattr(est, "n_classes_", 1))

    # ---- global importance ----
    if task == "k":
        shap.summary_plot(sv_list[0], Xs_plot, show=False)
        plt.savefig(outdir / f"SHAP_global_{task}.png", dpi=600, bbox_inches="tight")
        plt.close()
        dep_targets = [(sv_list[0], "")]
    else:
        shap.summary_plot(sv_list, Xs_plot, plot_type="bar",
                          class_names=list(getattr(est, "classes_", pp.CLASSES)),
                          show=False)
        plt.savefig(outdir / f"SHAP_global_{task}.png", dpi=600, bbox_inches="tight")
        plt.close()
        classes = [str(c) for c in getattr(est, "classes_", pp.CLASSES)]
        dep_targets = [(sv, f"_{classes[i].replace('-', '')}")
                       for i, sv in enumerate(sv_list)]

    # ---- dependence plots of leading descriptors (Figs. S9/S11) ----
    for sv, suffix in dep_targets:
        for feat in SHAP_DEPENDENCE_FEATURES:
            shap.dependence_plot(pp.DISPLAY[feat], sv, Xs_plot, show=False)
            plt.savefig(outdir / f"SHAP_dependence_{task}_{feat}{suffix}.png",
                        dpi=600, bbox_inches="tight")
            plt.close()

    # ---- feature-group contributions (Fig. 3) ----
    abs_sv = np.mean([np.abs(sv) for sv in sv_list], axis=0)
    imp = pd.DataFrame({"Feature": pp.FEATURES,
                        "MeanAbsSHAP": abs_sv.mean(axis=0)})
    imp["Group"] = imp["Feature"].map(assign_group)
    group_imp = (imp.groupby("Group")["MeanAbsSHAP"].sum()
                 .sort_values(ascending=False))
    print(f"[{task}] feature-group contributions (sum of mean |SHAP|):")
    print(group_imp)
    imp.sort_values("MeanAbsSHAP", ascending=False).to_csv(
        outdir / f"SHAP_feature_importance_{task}.csv", index=False,
        encoding="utf-8-sig")
    group_imp.to_csv(outdir / f"SHAP_group_contributions_{task}.csv",
                     encoding="utf-8-sig")


def _pd2d_grid(est, Xs: pd.DataFrame, f1: str, f2: str,
               n_grid: int = 50, proba_class: int | None = None,
               batch: int = 25):
    """2D partial dependence of the estimator in its own feature space.

    Averaged over the dataset on an n_grid x n_grid mesh (5th-95th percentile
    range), evaluated in batches to bound memory. For classifiers the
    predicted probability of ``proba_class`` is used.
    """
    x1, x2 = Xs[f1].to_numpy(), Xs[f2].to_numpy()
    g1 = np.linspace(*np.percentile(x1, [5, 95]), n_grid)
    g2 = np.linspace(*np.percentile(x2, [5, 95]), n_grid)
    pts = np.array([(a, b) for a in g1 for b in g2])
    i1, i2 = Xs.columns.get_loc(f1), Xs.columns.get_loc(f2)
    X = Xs.to_numpy()
    n = len(Xs)
    preds = np.empty(len(pts))
    for s in range(0, len(pts), batch):
        chunk = pts[s:s + batch]
        Xe = np.repeat(X, len(chunk), axis=0)
        Xe[:, i1] = np.tile(chunk[:, 0], n)
        Xe[:, i2] = np.tile(chunk[:, 1], n)
        p = (est.predict(Xe) if proba_class is None
             else est.predict_proba(Xe)[:, proba_class])
        preds[s:s + len(chunk)] = p.reshape(len(chunk), n).mean(axis=1)
    return g1, g2, preds.reshape(n_grid, n_grid)


def pdp_plots(model, X_raw: pd.DataFrame, task: str, outdir: Path,
              n_grid: int = 50):
    """2D partial dependence of leading descriptor pairs (Figs. 4, S10, S11).

    Computed for the estimator in its own (preprocessed) feature space -- the
    monotonic per-feature standardization only rescales the axes, which are
    relabelled back to raw physical units -- so the result is identical to
    evaluating the full pipeline but avoids re-running the preprocessing for
    millions of grid points. For the MCIRP model one panel per pathway class
    (predicted probability) is drawn, cf. Fig. S11.
    """
    pre = model.named_steps["preprocess"]
    est = model.named_steps["estimator"]
    scaler = pre.named_transformers_["numeric"].named_steps[
        "StandardScaler_training_only"]

    def to_raw(g, feature):
        if feature in pp.NAMES:
            idx = pp.NAMES.index(feature)
            return g * scaler.scale_[idx] + scaler.mean_[idx]
        return g

    Xs = pd.DataFrame(pre.transform(X_raw), columns=pp.FEATURES, index=X_raw.index)
    for f1, f2 in PDP_PAIRS[task]:
        if task == "k":
            panels = [("Predicted $\\it{k}$ (min$^{-1}$)",
                       _pd2d_grid(est, Xs, f1, f2, n_grid))]
            fig, axes = plt.subplots(1, 1, figsize=(6, 5))
            axes = [axes]
        else:
            classes = [str(c) for c in est.classes_]
            panels = [(f"P({cls})", _pd2d_grid(est, Xs, f1, f2, n_grid, proba_class=c))
                      for c, cls in enumerate(classes)]
            fig, axes = plt.subplots(1, len(panels), figsize=(5 * len(panels), 4.6),
                                     sharey=True)
        g1r = to_raw(np.linspace(*np.percentile(Xs[f1], [5, 95]), n_grid), f1)
        g2r = to_raw(np.linspace(*np.percentile(Xs[f2], [5, 95]), n_grid), f2)
        for ax, (label, (_, _, Z)) in zip(np.atleast_1d(axes), panels):
            im = ax.imshow(Z, origin="lower", aspect="auto",
                           extent=[g1r[0], g1r[-1], g2r[0], g2r[-1]], cmap="viridis")
            ax.set_xlabel(pp.DISPLAY.get(f1, f1))
            ax.set_ylabel(pp.DISPLAY.get(f2, f2))
            fig.colorbar(im, ax=ax, label=label)
        fig.tight_layout()
        fig.savefig(outdir / f"PDP2D_{task}_{f1}_{f2}.png", dpi=600,
                    bbox_inches="tight")
        plt.close(fig)


def ale_carbon_content(model, X_raw: pd.DataFrame, outdir: Path,
                       feature: str = "C", n_bins: int = 20):
    """Accumulated local effects of carbon content on pathway predictions (Fig. S17c).

    Computed in the estimator's preprocessed feature space (measured/missing C
    values are exactly the KNN-imputed, standardized inputs the Random Forest
    was trained on); the bin edges are converted back to raw C (%) for the
    figure axes.
    """
    pre = model.named_steps["preprocess"]
    est = model.named_steps["estimator"]
    scaler = pre.named_transformers_["numeric"].named_steps[
        "StandardScaler_training_only"]
    Xs = pd.DataFrame(pre.transform(X_raw), columns=pp.FEATURES, index=X_raw.index)

    x = Xs[feature].to_numpy(dtype=float)
    edges = np.unique(np.quantile(x, np.linspace(0, 1, n_bins + 1)))
    if len(edges) < 3:
        raise ValueError("not enough distinct C values for ALE")
    idx = np.clip(np.digitize(x, edges[1:-1]), 0, len(edges) - 2)
    X_lo, X_hi = Xs.copy(), Xs.copy()
    X_lo.loc[:, feature] = edges[idx]
    X_hi.loc[:, feature] = edges[idx + 1]
    delta = (est.predict_proba(X_hi.to_numpy())
             - est.predict_proba(X_lo.to_numpy()))             # (n, n_classes)

    # standardized -> raw C (%) axis
    fidx = pp.NAMES.index(feature)
    centers_std = (edges[:-1] + edges[1:]) / 2
    centers = centers_std * scaler.scale_[fidx] + scaler.mean_[fidx]
    for c, cls in enumerate(est.classes_):
        bin_mean = np.array([delta[idx == b, c].mean() if (idx == b).any() else 0.0
                             for b in range(len(edges) - 1)])
        ale = np.cumsum(bin_mean)
        ale -= ale.mean()                        # center the ALE curve
        plt.plot(centers, ale, marker="o", label=str(cls))
    plt.xlabel("C (%)")
    plt.ylabel("Accumulated local effect")
    plt.legend()
    plt.tight_layout()
    plt.savefig(outdir / "ALE_Cpercent_mcirp.png", dpi=600, bbox_inches="tight")
    plt.close()


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", default=pp.DATA_PATH, help="input workbook")
    ap.add_argument("--k-sheet", default=None, help="worksheet for the k records")
    ap.add_argument("--mc-sheet", default=None, help="worksheet for the MCIRP records")
    ap.add_argument("--modeldir", default=".", help="directory with the saved models")
    ap.add_argument("--out", default=".", help="output directory for figures/tables")
    args = ap.parse_args(argv)
    modeldir, outdir = Path(args.modeldir), Path(args.out)
    outdir.mkdir(parents=True, exist_ok=True)

    frame_k, _ = pp.load_task_frame(args.data, task="k", sheet_name=args.k_sheet)
    frame_mc, _ = pp.load_task_frame(args.data, task="mc", sheet_name=args.mc_sheet)
    X_k, X_mc = frame_k[pp.FEATURES], frame_mc[pp.FEATURES]

    model_k = joblib.load(modeldir / "extra_trees_k_model.pkl")
    model_mc = joblib.load(modeldir / "rf_mcirp_model.pkl")

    shap_analysis(model_k, X_k, task="k", outdir=outdir)
    pdp_plots(model_k, X_k, task="k", outdir=outdir)
    shap_analysis(model_mc, X_mc, task="mc", outdir=outdir)
    pdp_plots(model_mc, X_mc, task="mc", outdir=outdir)
    ale_carbon_content(model_mc, X_mc, outdir=outdir)


if __name__ == "__main__":
    main()
