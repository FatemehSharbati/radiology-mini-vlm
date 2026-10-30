"""Subgroup fairness audit for binary classifier scores (educational demo).

Given a CSV with columns ``y_true`` (0/1), ``y_score`` (model score) and
``subgroup`` (e.g. sex, age band, scanner site), this module reports per-subgroup
AUROC with 95% bootstrap confidence intervals and the absolute AUROC gap to the
best subgroup.

Pure scikit-learn / numpy / pandas / matplotlib — no model weights needed.
"""

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

import matplotlib

matplotlib.use("Agg")  # headless-safe
import matplotlib.pyplot as plt

REQUIRED_COLUMNS = {"y_true", "y_score", "subgroup"}


def make_demo_csv(path, n=2000, seed=0):
    """Write a synthetic audit CSV with a deliberate small performance gap.

    Subgroup A gets cleaner scores (higher AUROC), subgroup B noisier scores
    (lower AUROC). This is *synthetic* data for demonstrating the audit —
    it represents no real patients or model.
    """
    rng = np.random.default_rng(seed)
    frames = []
    for subgroup, noise, frac in (("A", 0.9, 0.6), ("B", 1.6, 0.4)):
        m = int(n * frac)
        y_true = rng.integers(0, 2, size=m)
        latent = np.where(y_true == 1, 1.2, -1.2)
        logits = latent + rng.normal(0.0, noise, size=m)
        y_score = 1.0 / (1.0 + np.exp(-logits))
        frames.append(
            pd.DataFrame({"y_true": y_true, "y_score": y_score, "subgroup": subgroup})
        )
    df = pd.concat(frames, ignore_index=True)
    df.to_csv(path, index=False)
    return path


def _bootstrap_auroc(y_true, y_score, n_boot=1000, seed=0):
    """Mean AUROC and 95% percentile CI from nonparametric bootstrap."""
    rng = np.random.default_rng(seed)
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score)
    n = len(y_true)
    aucs = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        if len(np.unique(y_true[idx])) < 2:
            continue  # resample without both classes is uninformative
        aucs.append(roc_auc_score(y_true[idx], y_score[idx]))
    if not aucs:
        raise ValueError("bootstrap produced no valid resamples")
    lo, hi = np.percentile(aucs, [2.5, 97.5])
    return float(np.mean(aucs)), float(lo), float(hi)


def audit_csv(path, n_boot=1000, plot_path=None):
    """Audit a CSV file; returns a DataFrame with per-subgroup AUROC + gaps.

    Columns: subgroup, n, auroc, ci_lo, ci_hi, gap_vs_best.
    """
    df = pd.read_csv(path)
    missing = REQUIRED_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(f"CSV is missing required columns: {sorted(missing)}")
    rows = []
    for subgroup, group in df.groupby("subgroup"):
        if len(np.unique(group["y_true"].to_numpy())) < 2:
            raise ValueError(
                f"subgroup '{subgroup}' must contain both classes to compute AUROC"
            )
        mean, lo, hi = _bootstrap_auroc(
            group["y_true"].to_numpy(), group["y_score"].to_numpy(), n_boot=n_boot
        )
        rows.append(
            {"subgroup": subgroup, "n": len(group), "auroc": mean, "ci_lo": lo, "ci_hi": hi}
        )
    out = pd.DataFrame(rows).sort_values("subgroup").reset_index(drop=True)
    out["gap_vs_best"] = out["auroc"].max() - out["auroc"]
    if plot_path is not None:
        plot_gaps(out, plot_path)
    return out


def print_gap_table(results: pd.DataFrame):
    """Print the audit table to stdout."""
    print(f"{'subgroup':<10}{'n':>8}{'AUROC':>10}{'95% CI':>22}{'gap':>10}")
    for _, r in results.iterrows():
        ci = f"[{r['ci_lo']:.3f}, {r['ci_hi']:.3f}]"
        print(
            f"{str(r['subgroup']):<10}{int(r['n']):>8}"
            f"{r['auroc']:>10.3f}{ci:>22}{r['gap_vs_best']:>10.3f}"
        )


def plot_gaps(results: pd.DataFrame, path):
    """Save a bar plot of per-subgroup AUROC with 95% CI whiskers."""
    fig, ax = plt.subplots(figsize=(6, 4))
    x = np.arange(len(results))
    y = results["auroc"].to_numpy()
    lower = y - results["ci_lo"].to_numpy()
    upper = results["ci_hi"].to_numpy() - y
    ax.bar(x, y, yerr=[lower, upper], capsize=5, color="steelblue")
    ax.set_xticks(x, results["subgroup"].tolist())
    ax.set_ylabel("AUROC (95% bootstrap CI)")
    ax.set_title("Subgroup performance audit (demo data)")
    ax.set_ylim(0.4, 1.0)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
