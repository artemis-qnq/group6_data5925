# ─────────────────────────────────────────────────────────────────────────────
# ML Model Training & Evaluation — 5-Run Cross Validation
#
# For each seed, the full preprocessing pipeline is re-executed:
#   - Stratified 80/10/10 split
#   - Static feature imputation (fit on train only)
#   - Flat feature extraction and z-scoring (fit on train only)
# ─────────────────────────────────────────────────────────────────────────────

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import (roc_curve, auc, f1_score,
                             average_precision_score, confusion_matrix,
                             precision_score, recall_score)
from imblearn.over_sampling import SMOTE
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

# ── Seeds for 5 independent runs ──
ML_SEEDS = [42, 123, 456, 789, 1024]


def find_best_threshold_youden(model, X_val, y_val):
    """
    Select optimal threshold using Youden Index on validation set.
    使用Youden Index在验证集上选择最优决策阈值。
    Youden Index = Sensitivity + Specificity - 1 = TPR + (1 - FPR) - 1
    """
    y_prob               = model.predict_proba(X_val)[:, 1]
    fpr, tpr, thresholds = roc_curve(y_val, y_prob)
    youden_index         = tpr + (1 - fpr) - 1
    return float(thresholds[np.argmax(youden_index)])


def evaluate_model(model, X_test, y_test, threshold):
    """
    Compute all evaluation metrics on test set.
    在测试集上计算所有评估指标。
    """
    y_prob              = model.predict_proba(X_test)[:, 1]
    y_pred              = (y_prob >= threshold).astype(int)
    fpr, tpr, _         = roc_curve(y_test, y_prob)
    tn, fp, fn, tp      = confusion_matrix(y_test, y_pred, labels=[0, 1]).ravel()
    prec                = precision_score(y_test, y_pred, zero_division=0)
    rec                 = recall_score(y_test, y_pred, zero_division=0)
    return {
        'AUROC':       auc(fpr, tpr),
        'AUPRC':       average_precision_score(y_test, y_prob),
        'F1':          f1_score(y_test, y_pred, zero_division=0),
        'Sensitivity': tp / (tp + fn) if (tp + fn) > 0 else 0.0,
        'Specificity': tn / (tn + fp) if (tn + fp) > 0 else 0.0,
        'Threshold':   threshold,
        'Score1':      min(prec, rec),
    }


def undersample_data(X, y, seed):
    """
    Undersample majority class to match minority class size.
    """
    df      = pd.DataFrame(X)
    df['y'] = y
    dead    = df[df['y'] == 1]
    alive   = df[df['y'] == 0].sample(n=len(dead), random_state=seed)
    bal     = pd.concat([dead, alive]).sample(frac=1, random_state=seed)
    return bal.drop(columns=['y']).values, bal['y'].values


def run_preprocessing_for_seed(seed):
    """
    Re-execute stratified split + flat feature preprocessing for a given seed.

    Reuses parsed data already in memory (df_static, df_ts, all_record_ids, y_all).
    """
    # ── Stratified split / 分层划分 ──────────────────────
    train_ids_s, temp_ids_s, y_train_s, y_temp_s = train_test_split(
        all_record_ids, y_all,
        test_size    = VAL_RATIO + TEST_RATIO,
        stratify     = y_all,
        random_state = seed,
    )
    val_ids_s, test_ids_s, y_val_s, y_test_s = train_test_split(
        temp_ids_s, y_temp_s,
        test_size    = TEST_RATIO / (VAL_RATIO + TEST_RATIO),
        stratify     = y_temp_s,
        random_state = seed,
    )

    # ── Reshape to 3D ─────────────────────
    X_train_raw_s = df_to_3d(df_ts, train_ids_s, ts_cols)
    X_val_raw_s   = df_to_3d(df_ts, val_ids_s,   ts_cols)
    X_test_raw_s  = df_to_3d(df_ts, test_ids_s,  ts_cols)

    # ── Apply physiological bounds ───────
    X_train_s = apply_phys_bounds(X_train_raw_s, ts_cols, verbose=False)
    X_val_s   = apply_phys_bounds(X_val_raw_s,   ts_cols, verbose=False)
    X_test_s  = apply_phys_bounds(X_test_raw_s,  ts_cols, verbose=False)

    # ── Missingness mask ────────────────────
    mask_train_s = (~np.isnan(X_train_s)).astype(np.float32)
    mask_val_s   = (~np.isnan(X_val_s  )).astype(np.float32)
    mask_test_s  = (~np.isnan(X_test_s )).astype(np.float32)

    # Pre-imputation copy for flat features 
    X_train_pre_s = X_train_s.copy()
    X_val_pre_s   = X_val_s.copy()
    X_test_pre_s  = X_test_s.copy()

    # ── Static features (fit on train only) ──
    S_train_df_s = get_static_df(df_static, train_ids_s)
    S_val_df_s   = get_static_df(df_static, val_ids_s)
    S_test_df_s  = get_static_df(df_static, test_ids_s)

    S_train_raw_s = process_static_raw_encoded(S_train_df_s)
    S_val_raw_s   = process_static_raw_encoded(S_val_df_s)
    S_test_raw_s  = process_static_raw_encoded(S_test_df_s)

    # ── Flat feature extraction  ───────────
    # Fallback means fit on training observed values only
    fb_means_s = np.zeros(len(ts_cols), dtype=np.float32)
    for j in range(len(ts_cols)):
        obs = X_train_pre_s[:, :, j][mask_train_s[:, :, j] == 1]
        if len(obs) > 0:
            if j in log_col_indices:
                fb_means_s[j] = float(np.nanmean(np.log1p(np.maximum(obs, 0.0))))
            else:
                fb_means_s[j] = float(np.nanmean(obs))
    fb_means_s = np.nan_to_num(fb_means_s, nan=0.0)

    ts_stats_train_s = extract_ts_stats(X_train_pre_s, mask_train_s, fb_means_s, log_col_indices)
    ts_stats_val_s   = extract_ts_stats(X_val_pre_s,   mask_val_s,   fb_means_s, log_col_indices)
    ts_stats_test_s  = extract_ts_stats(X_test_pre_s,  mask_test_s,  fb_means_s, log_col_indices)

    flat_raw_train_s = np.concatenate([ts_stats_train_s, S_train_raw_s], axis=1).astype(np.float32)
    flat_raw_val_s   = np.concatenate([ts_stats_val_s,   S_val_raw_s  ], axis=1).astype(np.float32)
    flat_raw_test_s  = np.concatenate([ts_stats_test_s,  S_test_raw_s ], axis=1).astype(np.float32)

    # ── Z-score scaling (fit on train only) ──
    flat_mean_s = np.zeros(n_flat, dtype=np.float32)
    flat_std_s  = np.ones( n_flat, dtype=np.float32)
    flat_mean_s[continuous_mask] = flat_raw_train_s[:, continuous_mask].mean(axis=0)
    flat_std_s [continuous_mask] = flat_raw_train_s[:, continuous_mask].std(axis=0)
    flat_std_s  = np.where(flat_std_s == 0, 1.0, flat_std_s)

    flat_scaled_train_s = ((flat_raw_train_s - flat_mean_s) / flat_std_s).astype(np.float32)
    flat_scaled_val_s   = ((flat_raw_val_s   - flat_mean_s) / flat_std_s).astype(np.float32)
    flat_scaled_test_s  = ((flat_raw_test_s  - flat_mean_s) / flat_std_s).astype(np.float32)

    return (flat_scaled_train_s, flat_scaled_val_s, flat_scaled_test_s,
            y_train_s, y_val_s, y_test_s)


# ── Storage for results  ────────────────────────
metric_keys = ['AUROC', 'AUPRC', 'F1', 'Sensitivity', 'Specificity',
               'Threshold', 'Score1']
model_names = ['log_undersample', 'log_smote', 'svm_undersample', 'svm_smote']
results     = {m: {k: [] for k in metric_keys} for m in model_names}

# ── 5-run loop  ──────────────────────────────────
for run, seed in enumerate(ML_SEEDS):
    print(f"\n{'='*60}")
    print(f"Run {run+1}/5  (seed={seed})")
    print(f"{'='*60}")

    # Re-run preprocessing for this seed 
    (X_tr, X_va, X_te,
     y_tr, y_va, y_te) = run_preprocessing_for_seed(seed)
    print(f"  Train: {len(y_tr)}  Val: {len(y_va)}  Test: {len(y_te)}")

    # ── Logistic Regression — Undersample ─────────────────
    X_bal, y_bal = undersample_data(X_tr, y_tr, seed)
    m = LogisticRegression(penalty='l2', C=1.0, max_iter=1000, random_state=42)
    m.fit(X_bal, y_bal)
    thresh  = find_best_threshold_youden(m, X_va, y_va)
    metrics = evaluate_model(m, X_te, y_te, thresh)
    for k in metric_keys: results['log_undersample'][k].append(metrics[k])
    print(f"  Log  Undersample — AUROC: {metrics['AUROC']:.4f}  F1: {metrics['F1']:.4f}")

    # ── Logistic Regression — SMOTE ───────────────────────
    smote      = SMOTE(random_state=seed)
    X_sm, y_sm = smote.fit_resample(X_tr, y_tr)
    m = LogisticRegression(penalty='l2', C=1.0, max_iter=1000, random_state=42)
    m.fit(X_sm, y_sm)
    thresh  = find_best_threshold_youden(m, X_va, y_va)
    metrics = evaluate_model(m, X_te, y_te, thresh)
    for k in metric_keys: results['log_smote'][k].append(metrics[k])
    print(f"  Log  SMOTE        — AUROC: {metrics['AUROC']:.4f}  F1: {metrics['F1']:.4f}")

    # ── SVM — Undersample ─────────────────────────────────
    X_bal, y_bal = undersample_data(X_tr, y_tr, seed)
    m = SVC(kernel='linear', probability=True, random_state=42)
    m.fit(X_bal, y_bal)
    thresh  = find_best_threshold_youden(m, X_va, y_va)
    metrics = evaluate_model(m, X_te, y_te, thresh)
    for k in metric_keys: results['svm_undersample'][k].append(metrics[k])
    print(f"  SVM  Undersample  — AUROC: {metrics['AUROC']:.4f}  F1: {metrics['F1']:.4f}")

    # ── SVM — SMOTE ───────────────────────────────────────
    smote      = SMOTE(random_state=seed)
    X_sm, y_sm = smote.fit_resample(X_tr, y_tr)
    m = SVC(kernel='linear', probability=True, random_state=42)
    m.fit(X_sm, y_sm)
    thresh  = find_best_threshold_youden(m, X_va, y_va)
    metrics = evaluate_model(m, X_te, y_te, thresh)
    for k in metric_keys: results['svm_smote'][k].append(metrics[k])
    print(f"  SVM  SMOTE        — AUROC: {metrics['AUROC']:.4f}  F1: {metrics['F1']:.4f}")

# ── Summary  ────────────────────────────────────────
print(f"\n{'='*60}")
print(f"Summary across 5 runs (mean ± std)")
print(f"{'='*60}")

summary_rows = []
for model_name in model_names:
    print(f"\n{model_name}:")
    row = {'Model': model_name}
    for k in metric_keys:
        mean = np.mean(results[model_name][k])
        std  = np.std(results[model_name][k])
        print(f"  {k:<12s}: {mean:.4f} (±{std:.4f})")
        row[f'{k}_mean'] = round(mean, 4)
        row[f'{k}_std']  = round(std, 4)
    summary_rows.append(row)

summary_df = pd.DataFrame(summary_rows)
summary_df.to_csv(OUTPUT_DIR / 'ml_model_comparison_5runs.csv', index=False)
print(f"\nSaved: {OUTPUT_DIR / 'ml_model_comparison_5runs.csv'}")