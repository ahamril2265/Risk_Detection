"""
Shared publication-quality utilities for the Credit Risk Scoring System notebooks.

Used across Notebooks 05-15 for consistent figure styling, reusable metric
implementations (KS statistic, lift/gain, McNemar, DeLong), the
append-or-create CSV logging pattern used throughout the project, and
progress-bar helpers for long-running searches and loops.
"""

import contextlib
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from scipy import stats
from tqdm.auto import tqdm

FIGURES_DIR = Path("../figures")
RESULTS_DIR = Path("../results")


def set_pub_style():
    """Apply a consistent, publication-quality style (300 DPI) to all subsequent matplotlib figures."""
    plt.rcParams.update({
        "figure.dpi": 300,
        "savefig.dpi": 300,
        "font.size": 11,
        "axes.titlesize": 13,
        "axes.titleweight": "bold",
        "axes.labelsize": 11,
        "legend.fontsize": 10,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.alpha": 0.3,
    })
    sns.set_palette("colorblind")


def save_fig(fig, name, directory=FIGURES_DIR):
    """Save a matplotlib figure to the figures directory at 300 DPI."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{name}.png"
    fig.savefig(path, dpi=300, bbox_inches="tight")
    print(f"Figure saved to: {path}")
    return path


def append_csv(path, row_df):
    """Append a results row to a CSV, creating the file with headers if it does not exist yet."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        existing = pd.read_csv(path)
        combined = pd.concat([existing, row_df], ignore_index=True)
        combined.to_csv(path, index=False)
    else:
        row_df.to_csv(path, index=False)
    return path


def ks_statistic(y_true, y_prob):
    """Kolmogorov-Smirnov statistic: maximum separation between cumulative good/bad distributions."""
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)
    order = np.argsort(y_prob)
    y_true_sorted = y_true[order]
    n_pos = y_true_sorted.sum()
    n_neg = len(y_true_sorted) - n_pos
    cum_pos = np.cumsum(y_true_sorted) / n_pos
    cum_neg = np.cumsum(1 - y_true_sorted) / n_neg
    return float(np.max(np.abs(cum_pos - cum_neg)))


def lift_gain_table(y_true, y_prob, n_bins=10):
    """Build a decile-based lift/gain table, ranking applicants from highest to lowest predicted risk."""
    df = pd.DataFrame({"y_true": np.asarray(y_true), "y_prob": np.asarray(y_prob)})
    df = df.sort_values("y_prob", ascending=False).reset_index(drop=True)
    df["decile"] = pd.qcut(df.index, n_bins, labels=False) + 1

    total_positives = df["y_true"].sum()
    table = (
        df.groupby("decile")
        .agg(applicants=("y_true", "size"), actual_defaults=("y_true", "sum"))
        .reset_index()
    )
    table["cumulative_defaults"] = table["actual_defaults"].cumsum()
    table["cumulative_applicants"] = table["applicants"].cumsum()
    table["gain_pct"] = 100 * table["cumulative_defaults"] / total_positives
    table["pct_population"] = 100 * table["cumulative_applicants"] / len(df)
    table["lift"] = table["gain_pct"] / table["pct_population"]
    return table


def mcnemar_test(y_true, pred_a, pred_b):
    """
    Exact McNemar test for two paired classifiers' correctness on the same test set.
    Uses scipy.stats only (exact binomial for small discordant counts, continuity-
    corrected chi-square otherwise) so no statsmodels dependency is required.
    """
    y_true = np.asarray(y_true)
    pred_a = np.asarray(pred_a)
    pred_b = np.asarray(pred_b)

    correct_a = (pred_a == y_true)
    correct_b = (pred_b == y_true)

    b = int(np.sum(correct_a & ~correct_b))
    c = int(np.sum(~correct_a & correct_b))
    n = b + c

    if n == 0:
        return {"b": b, "c": c, "statistic": 0.0, "p_value": 1.0, "method": "no discordant pairs"}

    if n < 25:
        result = stats.binomtest(min(b, c), n, p=0.5, alternative="two-sided")
        return {"b": b, "c": c, "statistic": float(min(b, c)), "p_value": result.pvalue, "method": "exact binomial"}

    statistic = (abs(b - c) - 1) ** 2 / n
    p_value = 1 - stats.chi2.cdf(statistic, df=1)
    return {"b": b, "c": c, "statistic": statistic, "p_value": p_value, "method": "chi-square (continuity corrected)"}


def _compute_midrank(x):
    """Midrank helper for the fast DeLong algorithm (handles tied values)."""
    j = np.argsort(x)
    z = x[j]
    n = len(x)
    t = np.zeros(n, dtype=float)
    i = 0
    while i < n:
        k = i
        while k < n and z[k] == z[i]:
            k += 1
        t[i:k] = 0.5 * (i + k - 1) + 1
        i = k
    t2 = np.empty(n, dtype=float)
    t2[j] = t
    return t2


def _fast_delong(predictions_sorted_transposed, label_1_count):
    """Core DeLong covariance computation (Sun & Xu, 2014 fast implementation)."""
    m = label_1_count
    n = predictions_sorted_transposed.shape[1] - m
    positive_examples = predictions_sorted_transposed[:, :m]
    negative_examples = predictions_sorted_transposed[:, m:]
    k = predictions_sorted_transposed.shape[0]

    tx = np.empty([k, m], dtype=float)
    ty = np.empty([k, n], dtype=float)
    tz = np.empty([k, m + n], dtype=float)
    for r in range(k):
        tx[r, :] = _compute_midrank(positive_examples[r, :])
        ty[r, :] = _compute_midrank(negative_examples[r, :])
        tz[r, :] = _compute_midrank(predictions_sorted_transposed[r, :])

    aucs = tz[:, :m].sum(axis=1) / m / n - float(m + 1.0) / (2.0 * n)
    v01 = (tz[:, :m] - tx[:, :]) / n
    v10 = 1.0 - (tz[:, m:] - ty[:, :]) / m
    sx = np.cov(v01)
    sy = np.cov(v10)
    delongcov = sx / m + sy / n
    return aucs, delongcov


def delong_roc_test(y_true, prob_a, prob_b):
    """
    DeLong's test for two correlated ROC-AUCs estimated on the same test set
    (Sun & Xu, 2014 fast DeLong algorithm). Returns both AUCs, their difference,
    the z-statistic, and a two-sided p-value for whether the difference is
    statistically significant.
    """
    y_true = np.asarray(y_true)
    order = np.argsort(-y_true, kind="stable")
    y_sorted = y_true[order]
    label_1_count = int(y_sorted.sum())

    preds = np.vstack([np.asarray(prob_a)[order], np.asarray(prob_b)[order]])
    aucs, delongcov = _fast_delong(preds, label_1_count)

    auc_a, auc_b = aucs
    var = delongcov[0, 0] + delongcov[1, 1] - 2 * delongcov[0, 1]
    var = max(var, 1e-12)
    z = (auc_a - auc_b) / np.sqrt(var)
    p_value = 2 * (1 - stats.norm.cdf(abs(z)))

    return {
        "auc_a": float(auc_a),
        "auc_b": float(auc_b),
        "auc_diff": float(auc_a - auc_b),
        "z_statistic": float(z),
        "p_value": float(p_value),
    }


@contextlib.contextmanager
def tqdm_joblib(desc, total):
    """
    Show a live tqdm progress bar for any joblib.Parallel-backed call (GridSearchCV,
    RandomizedSearchCV, etc.) by patching joblib's batch-completion callback for the
    duration of the `with` block. `total` should be the number of individual fits
    (candidates x cv folds), since joblib reports progress per completed batch.

    Usage:
        with pu.tqdm_joblib("Tuning Decision Tree", total=n_candidates * n_splits):
            grid_search.fit(X_train, y_train)
    """
    progress_bar = tqdm(desc=desc, total=total)

    class TqdmBatchCompletionCallback(joblib.parallel.BatchCompletionCallBack):
        def __call__(self, *args, **kwargs):
            progress_bar.update(n=self.batch_size)
            return super().__call__(*args, **kwargs)

    original_callback = joblib.parallel.BatchCompletionCallBack
    joblib.parallel.BatchCompletionCallBack = TqdmBatchCompletionCallback
    try:
        yield progress_bar
    finally:
        joblib.parallel.BatchCompletionCallBack = original_callback
        progress_bar.close()


def n_candidates(param_space):
    """Number of hyperparameter combinations in a GridSearchCV param_grid dict."""
    from sklearn.model_selection import ParameterGrid
    return len(ParameterGrid(param_space))
