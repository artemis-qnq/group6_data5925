"""
Shared preprocessing module for 5-Fold Stratified Cross-Validation.

Provides prepare_data_cv() which produces leakage-free preprocessed data for
one CV fold, including both sequence (Track B) and flat (Track A) features.

Two bug fixes relative to the original LSTM notebook prepare_data_cv:
  1. Flat z-score now uses continuous_mask (binary cols not scaled).
  2. Flat features concatenate S_raw_enc (not S_norm) to avoid double z-score
     on Age/Height/Weight.
"""

import warnings
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.utils import resample as sk_resample
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    roc_auc_score, average_precision_score, f1_score,
    roc_curve, confusion_matrix, brier_score_loss,
)

from paths import RAW_DATA_DIR, PROCESSED_DATA_DIR

warnings.filterwarnings("ignore", category=pd.errors.PerformanceWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning, message=".*empty slice.*")
warnings.filterwarnings("ignore", category=RuntimeWarning, message=".*All-NaN slice.*")
warnings.filterwarnings("ignore", category=RuntimeWarning,
                        message=".*Degrees of freedom.*")

# ══════════════════════════════════════════════════════════════════════════════
# Constants
# ══════════════════════════════════════════════════════════════════════════════
N_HOURS = 48

STATIC_COLS = ["Age", "Gender", "Height", "ICUType"]
ADMISSION_EXTRAS = ["Weight"]
STATIC_FINAL_COLS = [
    "Age", "Gender", "Height", "Weight",
    "Height_missing", "Weight_missing",
    "ICUType_1", "ICUType_2", "ICUType_3", "ICUType_4",
]
TS_BINARY_COLS = ["MechVent"]

LOG_TRANSFORM_COLS = {
    "ALP", "ALT", "AST", "Bilirubin", "BUN", "Creatinine",
    "Glucose", "Lactate", "TroponinI", "TroponinT", "Urine", "WBC",
}

PHYS_BOUNDS = {
    "ALP": (1, 5_000), "ALT": (1, 10_000), "AST": (1, 10_000),
    "Albumin": (0.5, 6.0), "Bilirubin": (0.1, 80.0), "BUN": (1, 300),
    "Cholesterol": (50, 600), "Creatinine": (0.1, 30.0),
    "DiasABP": (1, 200), "MAP": (10, 200), "SysABP": (30, 300),
    "NIDiasABP": (1, 200), "NIMAP": (10, 200), "NISysABP": (30, 300),
    "FiO2": (0.21, 1.0), "PaCO2": (5, 200), "PaO2": (20, 700),
    "RespRate": (1, 80), "SaO2": (0, 100), "HR": (10, 300),
    "Glucose": (20, 1_000), "HCO3": (5, 60), "HCT": (5, 70),
    "K": (1, 12), "Lactate": (0.1, 30), "Mg": (0.5, 6.0),
    "Na": (100, 180), "Platelets": (5, 2_000), "WBC": (0.1, 200),
    "pH": (6.5, 8.0), "TroponinI": (0, 1_000), "TroponinT": (0, 100),
    "GCS": (3, 15), "MechVent": (0, 1), "Temp": (25, 45),
    "Urine": (0, 6_000), "Weight": (30, 250),
}

PATIENT_DIR = RAW_DATA_DIR / "Patients"
OUTCOMES_FILE = RAW_DATA_DIR / "Outcomes-6.txt"


# ══════════════════════════════════════════════════════════════════════════════
# Raw data loading
# ══════════════════════════════════════════════════════════════════════════════

def read_patient_long(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, header=0)
    df["Value"] = pd.to_numeric(df["Value"], errors="coerce")
    t = pd.to_timedelta(df["Time"] + ":00")
    df["minute"] = (t.dt.total_seconds() // 60).astype(int)
    df = df[df["minute"] < N_HOURS * 60].copy()
    df["hour"] = (df["minute"] // 60).astype(int)
    df["RecordID"] = int(path.stem)
    return df


def extract_static(df_long: pd.DataFrame) -> dict:
    POSITIVE_SENTINEL = {"Height", "Weight"}
    VALID_BINARY = {"Gender": {0.0, 1.0}}
    VALID_CATEGORICAL = {"ICUType": {1.0, 2.0, 3.0, 4.0}}
    result = {"RecordID": df_long["RecordID"].iloc[0]}
    for col in STATIC_COLS:
        vals = df_long.loc[df_long["Parameter"] == col, "Value"].dropna()
        if col in POSITIVE_SENTINEL:
            vals = vals[vals > 0]
        if col in VALID_BINARY:
            vals = vals[vals.isin(VALID_BINARY[col])]
        if col in VALID_CATEGORICAL:
            vals = vals[vals.isin(VALID_CATEGORICAL[col])]
        result[col] = float(vals.iloc[0]) if len(vals) > 0 else np.nan
    for col in ADMISSION_EXTRAS:
        admit = df_long.loc[
            (df_long["Parameter"] == col) & (df_long["hour"] == 0), "Value"
        ].dropna()
        if col in POSITIVE_SENTINEL:
            admit = admit[admit > 0]
        result[col] = float(admit.iloc[0]) if len(admit) > 0 else np.nan
    return result


def build_hourly_grid(df_long: pd.DataFrame) -> pd.DataFrame:
    record_id = df_long["RecordID"].iloc[0]
    EXCLUDE = set(STATIC_COLS) | {"RecordID"}
    ts_df = df_long[~df_long["Parameter"].isin(EXCLUDE)].copy()
    agg = (ts_df
           .sort_values(["hour", "minute"])
           .groupby(["hour", "Parameter"], as_index=False)["Value"]
           .last())
    pivot = agg.pivot(index="hour", columns="Parameter", values="Value")
    pivot.columns.name = None
    pivot = pivot.reindex(range(N_HOURS))
    pivot.index.name = "hour"
    pivot.insert(0, "RecordID", record_id)
    return pivot.reset_index()


def load_raw_data():
    """Parse all patient files → (df_static, df_ts, ts_cols)."""
    print("Loading patient files...")
    static_list, ts_list = [], []
    for fpath in sorted(PATIENT_DIR.glob("*.txt")):
        df_long = read_patient_long(fpath)
        static_list.append(extract_static(df_long))
        ts_list.append(build_hourly_grid(df_long))
    df_static = pd.DataFrame(static_list)
    df_ts = pd.concat(ts_list, ignore_index=True)
    ts_cols = sorted(
        [c for c in df_ts.columns if c not in ["RecordID", "hour"]]
    )
    print(f"  Loaded {len(df_static)} patients, {len(ts_cols)} ts variables")
    return df_static, df_ts, ts_cols


def load_outcomes(df_static):
    """Load labels → (all_record_ids, y_all)."""
    df_out = pd.read_csv(OUTCOMES_FILE)
    labels = df_out.set_index("RecordID")["In-hospital_death"]
    all_record_ids = df_static["RecordID"].values
    y_all = np.array([labels[rid] for rid in all_record_ids], dtype=np.int64)
    return all_record_ids, y_all


# ══════════════════════════════════════════════════════════════════════════════
# Preprocessing helpers
# ══════════════════════════════════════════════════════════════════════════════

def df_to_3d(df, record_ids, cols):
    full_idx = pd.MultiIndex.from_product(
        [record_ids, range(N_HOURS)], names=["RecordID", "hour"]
    )
    return (df.set_index(["RecordID", "hour"])[cols]
              .reindex(full_idx)
              .values
              .reshape(len(record_ids), N_HOURS, len(cols))
              .astype(np.float32))


def apply_phys_bounds(X, cols):
    X = X.copy()
    for j, col in enumerate(cols):
        if col == "FiO2":
            pct = X[:, :, j] > 1.0
            X[:, :, j] = np.where(pct, X[:, :, j] / 100.0, X[:, :, j])
        if col in PHYS_BOUNDS:
            lo, hi = PHYS_BOUNDS[col]
            bad = (X[:, :, j] < lo) | (X[:, :, j] > hi)
            X[:, :, j] = np.where(bad, np.nan, X[:, :, j])
    return X


def compute_delta(mask_3d):
    N, T, V = mask_3d.shape
    delta = np.zeros((N, T, V), dtype=np.float32)
    last_obs = np.full((N, V), -1.0, dtype=np.float32)
    for t in range(T):
        observed = mask_3d[:, t, :] == 1
        delta[:, t, :] = np.where(last_obs >= 0, t - last_obs, float(t))
        last_obs = np.where(observed, float(t), last_obs)
    return delta


def locf_3d(X):
    X = X.copy()
    for t in range(1, N_HOURS):
        missing = np.isnan(X[:, t, :])
        X[:, t, :] = np.where(missing, X[:, t - 1, :], X[:, t, :])
    return X


def fill_with_means(X, means):
    X = X.copy()
    for j in range(X.shape[2]):
        nan_mask = np.isnan(X[:, :, j])
        X[:, :, j][nan_mask] = means[j]
    return X


def apply_log1p_transform(X, col_indices):
    X = X.copy()
    for j in col_indices:
        X[:, :, j] = np.log1p(np.maximum(X[:, :, j], 0.0))
    return X


def zscore_3d(X, mean, std):
    return ((X - mean) / std).astype(np.float32)


def get_static_df(df_static, record_ids):
    return df_static.set_index("RecordID").reindex(record_ids).reset_index()


def _process_static_norm(df, stats):
    """Z-scored continuous + raw binary/one-hot → 10 cols (sequence models)."""
    h_miss = df["Height"].isna().astype(np.float32).values
    w_miss = df["Weight"].isna().astype(np.float32).values
    df = df.copy()
    df["Age"] = df["Age"].fillna(stats["age_m"])
    df["Height"] = df["Height"].fillna(stats["h_m"])
    df["Weight"] = df["Weight"].fillna(stats["w_m"])
    df["Gender"] = df["Gender"].fillna(stats["g_mode"])
    df["ICUType"] = df["ICUType"].fillna(stats["icu_mode"]).astype(int)
    age_z = ((df["Age"].values - stats["age_m"]) / stats["age_s"]).astype(np.float32)
    h_z = ((df["Height"].values - stats["h_m"]) / stats["h_s"]).astype(np.float32)
    w_z = ((df["Weight"].values - stats["w_m"]) / stats["w_s"]).astype(np.float32)
    icu_oh = (pd.get_dummies(df["ICUType"], prefix="ICUType")
                .reindex(columns=["ICUType_1", "ICUType_2",
                                  "ICUType_3", "ICUType_4"], fill_value=0)
                .values.astype(np.float32))
    return np.column_stack([
        age_z, df["Gender"].values, h_z, w_z, h_miss, w_miss, icu_oh,
    ])


def _process_static_raw(df, stats):
    """Original-unit + binary/one-hot → 10 cols (flat features)."""
    h_miss = df["Height"].isna().astype(np.float32).values
    w_miss = df["Weight"].isna().astype(np.float32).values
    df = df.copy()
    df["Age"] = df["Age"].fillna(stats["age_m"])
    df["Height"] = df["Height"].fillna(stats["h_m"])
    df["Weight"] = df["Weight"].fillna(stats["w_m"])
    df["Gender"] = df["Gender"].fillna(stats["g_mode"])
    df["ICUType"] = df["ICUType"].fillna(stats["icu_mode"]).astype(int)
    icu_oh = (pd.get_dummies(df["ICUType"], prefix="ICUType")
                .reindex(columns=["ICUType_1", "ICUType_2",
                                  "ICUType_3", "ICUType_4"], fill_value=0)
                .values.astype(np.float32))
    return np.column_stack([
        df["Age"].values.astype(np.float32),
        df["Gender"].values.astype(np.float32),
        df["Height"].values.astype(np.float32),
        df["Weight"].values.astype(np.float32),
        h_miss, w_miss, icu_oh,
    ])


def extract_ts_stats(X_pre, mask, fallback_means, log_indices):
    """8 summary stats per ts variable from observed positions → (N, 8*V)."""
    N, T, V = X_pre.shape
    feats = []
    for j in range(V):
        x = X_pre[:, :, j]
        m = mask[:, :, j]
        x_obs = np.where(m == 1, x, np.nan).astype(np.float64)
        if j in log_indices:
            x_obs = np.where(m == 1, np.log1p(np.maximum(x_obs, 0.0)), np.nan)

        f_mean = np.nanmean(x_obs, axis=1)
        f_std = np.nanstd(x_obs, axis=1)
        f_min = np.nanmin(x_obs, axis=1)
        f_max = np.nanmax(x_obs, axis=1)

        has_obs = m.any(axis=1)
        first_idx = np.argmax(m, axis=1)
        last_idx = T - 1 - np.argmax(m[:, ::-1], axis=1)
        row_idx = np.arange(N)
        first = np.where(has_obs, x_obs[row_idx, first_idx], np.nan)
        last = np.where(has_obs, x_obs[row_idx, last_idx], np.nan)

        f_count = m.sum(axis=1).astype(np.float64)
        f_missing = 1.0 - m.mean(axis=1)

        never_obs = f_count == 0
        fb = float(fallback_means[j])
        for arr in [f_mean, f_min, f_max, first, last]:
            arr[never_obs] = fb
        f_std[never_obs] = 0.0

        feats.extend([f_mean, f_std, f_min, f_max, first, last,
                      f_count, f_missing])

    return np.column_stack(feats).astype(np.float32)


def _build_continuous_mask(n_ts_vars, n_flat):
    """True for continuous columns that should be z-scored in flat features."""
    n_ts_feats = 8 * n_ts_vars
    cm = np.zeros(n_flat, dtype=bool)
    for v in range(n_ts_vars):
        cm[v * 8: v * 8 + 6] = True   # mean, std, min, max, first, last
    cm[n_ts_feats + 0] = True   # Age
    cm[n_ts_feats + 2] = True   # Height
    cm[n_ts_feats + 3] = True   # Weight
    return cm


# ══════════════════════════════════════════════════════════════════════════════
# Evaluation helpers
# ══════════════════════════════════════════════════════════════════════════════

def compute_metrics(y_true, y_prob, threshold):
    y_pred = (y_prob >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return dict(
        AUROC=roc_auc_score(y_true, y_prob),
        AUPRC=average_precision_score(y_true, y_prob),
        F1=f1_score(y_true, y_pred, zero_division=0),
        Sensitivity=tp / (tp + fn + 1e-8),
        Specificity=tn / (tn + fp + 1e-8),
        PPV=tp / (tp + fp + 1e-8),
        NPV=tn / (tn + fn + 1e-8),
        Brier=brier_score_loss(y_true, y_prob),
        Threshold=threshold,
    )


def youden_threshold(y_true, y_prob):
    fpr, tpr, thresholds = roc_curve(y_true, y_prob)
    return float(thresholds[np.argmax(tpr - fpr)])


def calibrate_platt(val_y, val_prob, target_prob):
    cal = LogisticRegression(C=1e10, max_iter=1000)
    cal.fit(val_prob.reshape(-1, 1), val_y)
    return cal.predict_proba(target_prob.reshape(-1, 1))[:, 1]


def random_oversample_indices(y, seed):
    """Return balanced indices (minority upsampled to match majority)."""
    idx = np.arange(len(y))
    y_int = y.astype(int)
    idx_maj = idx[y_int == 0]
    idx_min = idx[y_int == 1]
    idx_min_up = sk_resample(idx_min, replace=True,
                             n_samples=len(idx_maj), random_state=seed)
    idx_bal = np.concatenate([idx_maj, idx_min_up])
    return np.random.RandomState(seed).permutation(idx_bal)


def print_cv_summary(all_fold_metrics, metric_keys=None):
    if metric_keys is None:
        metric_keys = ["AUROC", "AUPRC", "F1", "Sensitivity", "Specificity",
                       "PPV", "NPV", "Brier", "Threshold"]
    print(f"\n{'='*60}")
    print("5-Fold CV Summary (mean ± std)")
    print(f"{'='*60}")
    summary = {}
    for k in metric_keys:
        vals = [m[k] for m in all_fold_metrics]
        mean, std = float(np.mean(vals)), float(np.std(vals))
        print(f"  {k:<14s}: {mean:.4f} ± {std:.4f}")
        summary[k] = {"mean": mean, "std": std}
    return summary


# ══════════════════════════════════════════════════════════════════════════════
# Main CV preprocessing
# ══════════════════════════════════════════════════════════════════════════════

def prepare_data_cv(train_ids, val_ids, test_ids,
                    df_ts, df_static, y_all, all_record_ids, ts_cols,
                    held_out_ids=None):
    """
    Leakage-free preprocessing for one CV fold.

    Returns dict with:
      - Sequence branch: X/S/mask/delta for train/val/test
      - Flat branch: flat_raw and flat_scaled for train/val/test
      - Labels and IDs
      - Held-out arrays (if held_out_ids provided)
    """
    id_to_y = dict(zip(all_record_ids, y_all))

    # ── Reshape + physiological bounds ────────────────────────────────────
    X_tr_raw = apply_phys_bounds(df_to_3d(df_ts, train_ids, ts_cols), ts_cols)
    X_va_raw = apply_phys_bounds(df_to_3d(df_ts, val_ids, ts_cols), ts_cols)
    X_te_raw = apply_phys_bounds(df_to_3d(df_ts, test_ids, ts_cols), ts_cols)

    # ── Mask + delta ──────────────────────────────────────────────────────
    m_tr = (~np.isnan(X_tr_raw)).astype(np.float32)
    m_va = (~np.isnan(X_va_raw)).astype(np.float32)
    m_te = (~np.isnan(X_te_raw)).astype(np.float32)
    delta_tr = compute_delta(m_tr)
    delta_va = compute_delta(m_va)
    delta_te = compute_delta(m_te)

    X_tr_pre = X_tr_raw.copy()
    X_va_pre = X_va_raw.copy()
    X_te_pre = X_te_raw.copy()

    log_indices = [j for j, col in enumerate(ts_cols)
                   if col in LOG_TRANSFORM_COLS]

    # ── Fallback means (train only, shared by both branches) ──────────────
    fallback = np.zeros(len(ts_cols), dtype=np.float32)
    for j in range(len(ts_cols)):
        obs = X_tr_pre[:, :, j][m_tr[:, :, j] == 1]
        if len(obs) > 0:
            if j in log_indices:
                fallback[j] = float(
                    np.nanmean(np.log1p(np.maximum(obs, 0.0)))
                )
            else:
                fallback[j] = float(np.nanmean(obs))
    fallback = np.nan_to_num(fallback, nan=0.0)
    if "MechVent" in ts_cols:
        fallback[ts_cols.index("MechVent")] = 0.0

    # ── Sequence branch: LOCF → log1p → fill → z-score ───────────────────
    X_tr = fill_with_means(
        apply_log1p_transform(locf_3d(X_tr_raw), log_indices), fallback)
    X_va = fill_with_means(
        apply_log1p_transform(locf_3d(X_va_raw), log_indices), fallback)
    X_te = fill_with_means(
        apply_log1p_transform(locf_3d(X_te_raw), log_indices), fallback)

    norm_m = np.zeros(len(ts_cols), dtype=np.float32)
    norm_s = np.ones(len(ts_cols), dtype=np.float32)
    for j, col in enumerate(ts_cols):
        if col in TS_BINARY_COLS:
            continue
        obs = X_tr[:, :, j][m_tr[:, :, j] == 1]
        if len(obs) > 1:
            norm_m[j] = float(obs.mean())
            s = float(obs.std())
            norm_s[j] = s if s > 0 else 1.0

    X_tr_norm = zscore_3d(X_tr, norm_m, norm_s)
    X_va_norm = zscore_3d(X_va, norm_m, norm_s)
    X_te_norm = zscore_3d(X_te, norm_m, norm_s)

    # ── Static features ──────────────────────────────────────────────────
    S_tr_df = get_static_df(df_static, train_ids)
    S_va_df = get_static_df(df_static, val_ids)
    S_te_df = get_static_df(df_static, test_ids)

    s_stats = {
        "age_m": float(S_tr_df["Age"].mean()),
        "age_s": float(max(S_tr_df["Age"].std(), 1e-8)),
        "h_m": float(S_tr_df["Height"].mean()),
        "h_s": float(max(S_tr_df["Height"].std(), 1e-8)),
        "w_m": float(S_tr_df["Weight"].mean()),
        "w_s": float(max(S_tr_df["Weight"].std(), 1e-8)),
        "g_mode": (float(S_tr_df["Gender"].mode().iloc[0])
                   if not S_tr_df["Gender"].dropna().empty else 0.0),
        "icu_mode": (int(S_tr_df["ICUType"].mode().iloc[0])
                     if not S_tr_df["ICUType"].dropna().empty else 1),
    }

    S_tr_norm = _process_static_norm(S_tr_df, s_stats)
    S_va_norm = _process_static_norm(S_va_df, s_stats)
    S_te_norm = _process_static_norm(S_te_df, s_stats)
    S_tr_raw = _process_static_raw(S_tr_df, s_stats)
    S_va_raw = _process_static_raw(S_va_df, s_stats)
    S_te_raw = _process_static_raw(S_te_df, s_stats)

    # ── Flat features ─────────────────────────────────────────────────────
    ts_stats_tr = extract_ts_stats(X_tr_pre, m_tr, fallback, log_indices)
    ts_stats_va = extract_ts_stats(X_va_pre, m_va, fallback, log_indices)
    ts_stats_te = extract_ts_stats(X_te_pre, m_te, fallback, log_indices)

    # BUG FIX #2: use S_raw (original units) not S_norm (z-scored)
    flat_raw_tr = np.concatenate(
        [ts_stats_tr, S_tr_raw], axis=1).astype(np.float32)
    flat_raw_va = np.concatenate(
        [ts_stats_va, S_va_raw], axis=1).astype(np.float32)
    flat_raw_te = np.concatenate(
        [ts_stats_te, S_te_raw], axis=1).astype(np.float32)

    # BUG FIX #1: only z-score continuous columns
    n_flat = flat_raw_tr.shape[1]
    cont_mask = _build_continuous_mask(len(ts_cols), n_flat)

    flat_mean = np.zeros(n_flat, dtype=np.float32)
    flat_std = np.ones(n_flat, dtype=np.float32)
    flat_mean[cont_mask] = flat_raw_tr[:, cont_mask].mean(axis=0)
    flat_std[cont_mask] = flat_raw_tr[:, cont_mask].std(axis=0)
    flat_std = np.where(flat_std == 0, 1.0, flat_std).astype(np.float32)

    flat_sc_tr = ((flat_raw_tr - flat_mean) / flat_std).astype(np.float32)
    flat_sc_va = ((flat_raw_va - flat_mean) / flat_std).astype(np.float32)
    flat_sc_te = ((flat_raw_te - flat_mean) / flat_std).astype(np.float32)

    result = {
        "X_train": X_tr_norm, "X_val": X_va_norm, "X_test": X_te_norm,
        "S_train": S_tr_norm, "S_val": S_va_norm, "S_test": S_te_norm,
        "mask_train": m_tr, "mask_val": m_va, "mask_test": m_te,
        "delta_train": delta_tr, "delta_val": delta_va, "delta_test": delta_te,
        "flat_raw_train": flat_raw_tr, "flat_raw_val": flat_raw_va,
        "flat_raw_test": flat_raw_te,
        "flat_scaled_train": flat_sc_tr, "flat_scaled_val": flat_sc_va,
        "flat_scaled_test": flat_sc_te,
        "y_train": np.array([id_to_y[i] for i in train_ids]),
        "y_val": np.array([id_to_y[i] for i in val_ids]),
        "y_test": np.array([id_to_y[i] for i in test_ids]),
        "train_ids": train_ids, "val_ids": val_ids, "test_ids": test_ids,
    }

    # ── Held-out test (same normalisation params) ─────────────────────────
    if held_out_ids is not None and len(held_out_ids) > 0:
        X_ho_raw = apply_phys_bounds(
            df_to_3d(df_ts, held_out_ids, ts_cols), ts_cols)
        m_ho = (~np.isnan(X_ho_raw)).astype(np.float32)
        delta_ho = compute_delta(m_ho)
        X_ho_pre = X_ho_raw.copy()
        X_ho = fill_with_means(
            apply_log1p_transform(locf_3d(X_ho_raw), log_indices), fallback)
        X_ho_norm = zscore_3d(X_ho, norm_m, norm_s)

        S_ho_df = get_static_df(df_static, held_out_ids)
        S_ho_norm = _process_static_norm(S_ho_df, s_stats)
        S_ho_raw = _process_static_raw(S_ho_df, s_stats)

        ts_stats_ho = extract_ts_stats(X_ho_pre, m_ho, fallback, log_indices)
        flat_raw_ho = np.concatenate(
            [ts_stats_ho, S_ho_raw], axis=1).astype(np.float32)
        flat_sc_ho = ((flat_raw_ho - flat_mean) / flat_std).astype(np.float32)

        result.update({
            "X_held_out": X_ho_norm,
            "S_held_out": S_ho_norm,
            "mask_held_out": m_ho,
            "delta_held_out": delta_ho,
            "flat_raw_held_out": flat_raw_ho,
            "flat_scaled_held_out": flat_sc_ho,
            "y_held_out": np.array([id_to_y[i] for i in held_out_ids]),
            "held_out_ids": held_out_ids,
        })

    return result


# ══════════════════════════════════════════════════════════════════════════════
# CV fold setup
# ══════════════════════════════════════════════════════════════════════════════

def setup_cv_splits(all_record_ids, y_all, n_splits=5, random_state=42,
                    held_out_test_ids=None, val_ratio=0.15):
    """
    Build CV fold splits with optional held-out test set.

    Parameters
    ----------
    held_out_test_ids : 'auto' | ndarray | None
        'auto' loads from data/processed/ids_test.npy.

    Returns
    -------
    folds : list of (fold_idx, train_ids, val_ids, fold_test_ids)
    held_out_test_ids : ndarray or None
    """
    if held_out_test_ids is not None:
        if isinstance(held_out_test_ids, str) and held_out_test_ids == "auto":
            p = PROCESSED_DATA_DIR / "ids_test.npy"
            if p.exists():
                held_out_test_ids = np.load(p)
                print(f"Held-out test: {len(held_out_test_ids)} patients "
                      f"loaded from {p}")
            else:
                print(f"Warning: {p} not found — CV without held-out test")
                held_out_test_ids = None

    if held_out_test_ids is not None:
        ho_set = set(int(x) for x in held_out_test_ids)
        cv_mask = np.array([int(rid) not in ho_set
                            for rid in all_record_ids])
        cv_ids = all_record_ids[cv_mask]
        cv_y = y_all[cv_mask]
    else:
        cv_ids = all_record_ids
        cv_y = y_all

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True,
                          random_state=random_state)
    folds = []
    for fold_idx, (tv_idx, te_idx) in enumerate(skf.split(cv_ids, cv_y)):
        fold_te_ids = cv_ids[te_idx]
        tv_ids = cv_ids[tv_idx]
        tv_y = cv_y[tv_idx]
        tr_ids, vl_ids = train_test_split(
            tv_ids, test_size=val_ratio, stratify=tv_y,
            random_state=fold_idx,
        )
        folds.append((fold_idx, tr_ids, vl_ids, fold_te_ids))

    return folds, held_out_test_ids
