import os
import glob
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer

# ── Path configuration ──
FEATURES_DIR  = 'feature'
OUTCOMES_FILE = 'Outcomes-6.txt'

# ── Static descriptor fields──
META_PARAMS = ['RecordID', 'Age', 'Gender', 'Height', 'ICUType', 'Weight']

# ── Reference values for outlier filtering (μ, σ) ──
REFERENCE = {
    'Albumin':     (4.25,  0.75),
    'ALP':         (64,    28),
    'ALT':         (20,    15),
    'AST':         (23.5,  16.5),
    'Bilirubin':   (0.75,  0.55),
    'BUN':         (34,    12),
    'Cholesterol': (100,   100),
    'Creatinine':  (0.9,   0.3),
    'DiasABP':     (70,    30),
    'FiO2':        (0.5,   0.5),
    'GCS':         (9,     6),
    'Glucose':     (90,    20),
    'HCO3':        (26,    3),
    'HCT':         (45.5,  8.5),
    'HR':          (73,    25),
    'K':           (4.3,   0.8),
    'Lactate':     (0.8,   0.5),
    'MAP':         (100,   45),
    'Mg':          (1.7,   0.4),
    'Na':          (140,   5),
    'NIDiasABP':   (70,    30),
    'NIMAP':       (100,   45),
    'NISysABP':    (130,   60),
    'PaCO2':       (40,    5),
    'PaO2':        (90,    10),
    'pH':          (7.4,   0.05),
    'Platelets':   (275,   125),
    'RespRate':    (18,    2),
    'SaO2':        (96.5,  2.5),
    'SysABP':      (130,   60),
    'Temp':        (36.65, 0.55),
    'TroponinI':   (0.02,  0.02),
    'TroponinT':   (0.05,  0.05),
    'Urine':       (246.79, 17.93),
    'WBC':         (7.75,  3.25),
}

# MechVent is binary (0/1), no outlier filtering needed / MechVent 
BINARY_PARAMS = ['MechVent']

MISSING_THRESHOLD = 0.70  # Drop parameter if missing in more than 70% of patients


# ─────────────────────────────────────────────────────────
# Data Loading 
# ─────────────────────────────────────────────────────────

def load_patient_file(filepath):
    """
    Read a single patient file and return static info and time-series data.
    读取单个患者文件，返回静态信息和时序数据。

    Returns / 返回:
      - meta : dict, static descriptors (RecordID, Age, Gender, etc.)
               字典，静态信息（RecordID、Age、Gender 等）
      - ts   : DataFrame, columns = [minutes, Parameter, Value]
               时序数据，列为 [minutes, Parameter, Value]
    """
    df = pd.read_csv(filepath)

    # Separate static info and time-series rows 
    meta_df = df[df['Parameter'].isin(META_PARAMS)]
    ts_df   = df[~df['Parameter'].isin(META_PARAMS)].copy()

    # Convert static info to dict, replace -1 with NaN 
    meta = {}
    for _, row in meta_df.iterrows():
        val = float(row['Value'])
        meta[row['Parameter']] = np.nan if val == -1 else val

    # Convert Value to numeric, replace -1 with NaN
    ts_df['Value'] = pd.to_numeric(ts_df['Value'], errors='coerce')
    ts_df.loc[ts_df['Value'] == -1, 'Value'] = np.nan

    # Convert time string (HH:MM) to minutes since ICU admission 
    ts_df['minutes'] = ts_df['Time'].apply(
        lambda t: int(t.split(':')[0]) * 60 + int(t.split(':')[1])
    )

    ts_df = ts_df[['minutes', 'Parameter', 'Value']].reset_index(drop=True)

    return meta, ts_df


def load_all_patients(features_dir):
    """
    Load all patient files from the features directory.

    Returns :
      - all_meta : DataFrame, one row per patient with static info
      - all_ts   : DataFrame, all patients' time-series concatenated, with RecordID column
    """
    files = sorted(glob.glob(os.path.join(features_dir, '*.txt')))
    print(f"Found {len(files)} patient files.")

    meta_list = []
    ts_list   = []

    for i, filepath in enumerate(files):
        if (i + 1) % 500 == 0:
            print(f"  Processed {i + 1} / {len(files)} files...")
        try:
            meta, ts = load_patient_file(filepath)
            record_id = int(meta['RecordID'])
            meta_list.append(meta)
            ts['RecordID'] = record_id
            ts_list.append(ts)
        except Exception as e:
            print(f"  WARNING: Skipping {os.path.basename(filepath)}: {e}")

    all_meta = pd.DataFrame(meta_list)
    all_ts   = pd.concat(ts_list, ignore_index=True)

    print(f"Loading complete.")
    print(f"  Static info table : {all_meta.shape[0]} patients, {all_meta.shape[1]} columns")
    print(f"  Time-series table : {all_ts.shape[0]} records, {all_ts['Parameter'].nunique()} unique parameters")

    return all_meta, all_ts


def load_outcomes(outcomes_file):
    """
    Load the Outcomes file, keep only relevant columns.
    """
    outcomes = pd.read_csv(outcomes_file)
    outcomes = outcomes[['RecordID', 'In-hospital_death']]
    print(f"Outcomes loaded.")
    print(f"  Total patients   : {len(outcomes)}")
    print(f"  In-hospital death: {outcomes['In-hospital_death'].sum()} "
          f"({outcomes['In-hospital_death'].mean():.1%})")
    return outcomes


# ─────────────────────────────────────────────────────────
# Data Cleaning 
# ─────────────────────────────────────────────────────────

def filter_outliers(all_ts):
    """
    Remove observations outside μ ± 10σ for each parameter.
    """
    original_len = len(all_ts)
    cleaned_parts = []

    for param, group in all_ts.groupby('Parameter'):
        if param in BINARY_PARAMS or param not in REFERENCE:
            # Keep as-is if binary or not in reference table
            cleaned_parts.append(group)
            continue

        mu, sigma = REFERENCE[param]
        lower = mu - 10 * sigma
        upper = mu + 10 * sigma

        mask = group['Value'].isna() | ((group['Value'] >= lower) & (group['Value'] <= upper))
        cleaned_parts.append(group[mask])

    all_ts_clean = pd.concat(cleaned_parts, ignore_index=True)
    removed = original_len - len(all_ts_clean)
    print(f"Outlier filtering complete.")
    print(f"  Records removed  : {removed} ({removed / original_len:.2%})")
    print(f"  Records remaining: {len(all_ts_clean)}")

    return all_ts_clean


def handle_duplicates(all_ts):
    """
    For duplicate records (same patient, same time, same parameter), take the mean.
    """
    before = len(all_ts)
    all_ts_deduped = (
        all_ts.groupby(['RecordID', 'minutes', 'Parameter'], as_index=False)['Value']
        .mean()
    )
    removed = before - len(all_ts_deduped)
    print(f"Duplicate handling complete.")
    print(f"  Duplicate records merged: {removed}")
    print(f"  Records remaining       : {len(all_ts_deduped)}")

    return all_ts_deduped


def filter_missing_parameters(all_ts, n_patients, threshold=MISSING_THRESHOLD):
    """
    Drop parameters where more than `threshold` of patients have no record.
    """
    # Count how many unique patients have at least one record per parameter
    # 计算每个指标覆盖的患者数
    patient_coverage = (
        all_ts.groupby('Parameter')['RecordID']
        .nunique()
        .reset_index()
        .rename(columns={'RecordID': 'n_patients_with_data'})
    )
    patient_coverage['missing_rate'] = (
        1 - patient_coverage['n_patients_with_data'] / n_patients
    )
    patient_coverage = patient_coverage.sort_values('missing_rate', ascending=False)

    print(f"Missing rate per parameter (out of {n_patients} patients):")
    print(patient_coverage.to_string(index=False))

    # Parameters to keep 
    keep_params = patient_coverage[
        patient_coverage['missing_rate'] <= threshold
    ]['Parameter'].tolist()

    dropped = patient_coverage[
        patient_coverage['missing_rate'] > threshold
    ]['Parameter'].tolist()

    print(f"Parameters dropped (missing rate > {threshold:.0%}): {dropped}")
    print(f"Parameters kept: {len(keep_params)}")

    all_ts_filtered = all_ts[all_ts['Parameter'].isin(keep_params)]

    return all_ts_filtered, keep_params


# ─────────────────────────────────────────────────────────
# Feature Engineering
# ─────────────────────────────────────────────────────────

def compute_mean_features(all_ts, kept_params, hours=24):
    """
    Compute the mean value of each parameter within the first `hours` hours
    for each patient.

    Returns :
      - features : DataFrame, shape = (n_patients, n_params)
    """
    max_minutes = hours * 60

    # Filter to first `hours` hours only 
    ts_window = all_ts[all_ts['minutes'] <= max_minutes].copy()

    # Compute mean per patient per parameter
    features = (
        ts_window.groupby(['RecordID', 'Parameter'])['Value']
        .mean()
        .unstack(level='Parameter')
        .reset_index()
    )

    # Ensure all kept parameters are present as columns
    for param in kept_params:
        if param not in features.columns:
            features[param] = np.nan

    # Keep only kept_params columns plus RecordID 
    features = features[['RecordID'] + kept_params]

    print(f"Feature matrix computed (0-{hours}h mean).")
    print(f"  Shape: {features.shape[0]} patients x {features.shape[1] - 1} parameters")

    return features


def build_final_dataset(features, all_meta, outcomes):
    """
    Merge time-series features with static info and outcome label.

    Returns:
      - dataset : DataFrame, ready for modelling 
    """
    # Merge static info 
    static_cols = ['RecordID', 'Age', 'Gender', 'Height', 'ICUType', 'Weight']
    dataset = features.merge(all_meta[static_cols], on='RecordID', how='left')

    # Merge outcome label 
    dataset = dataset.merge(outcomes, on='RecordID', how='left')

    print(f"Final dataset assembled.")
    print(f"  Shape: {dataset.shape[0]} patients x {dataset.shape[1] - 2} features")

    return dataset


# ─────────────────────────────────────────────────────────
# PCA 
# ─────────────────────────────────────────────────────────

def apply_pca(dataset, kept_params, variance_threshold=0.90):
    """
    Apply PCA on time-series features (static info excluded).

    Steps / 步骤:
      1. Impute missing values with median 
      2. Standardise features 
      3. Fit PCA, keep components explaining `variance_threshold` of variance

    Returns :
      - dataset_pca : DataFrame with PCA components + static info + label
      - pca         : fitted PCA object 
      - scaler      : fitted StandardScaler 
      - imputer     : fitted SimpleImputer 
    """
    static_cols = ['Age', 'Gender', 'Height', 'ICUType', 'Weight']
    ts_cols     = [c for c in kept_params if c in dataset.columns]

    X_ts = dataset[ts_cols].values

    # Step 1: Impute missing values with median 
    imputer   = SimpleImputer(strategy='median')
    X_imputed = imputer.fit_transform(X_ts)

    # Step 2: Standardise 
    scaler   = StandardScaler()
    X_scaled = scaler.fit_transform(X_imputed)

    # Step 3: Fit PCA 
    pca   = PCA(n_components=variance_threshold, random_state=42)
    X_pca = pca.fit_transform(X_scaled)

    n_components = pca.n_components_
    explained    = pca.explained_variance_ratio_.cumsum()[-1]

    print(f"PCA complete.")
    print(f"  Components selected : {n_components} (explaining {explained:.1%} of variance)")
    print(f"  Variance per component:")
    for i, v in enumerate(pca.explained_variance_ratio_):
        print(f"    PC{i+1}: {v:.3%}")

    # Assemble output dataset 
    pc_cols     = [f'PC{i+1}' for i in range(n_components)]
    dataset_pca = pd.DataFrame(X_pca, columns=pc_cols)

    # Impute static info with median 
    static_imputer = SimpleImputer(strategy='median')
    X_static       = dataset[static_cols].values
    X_static_imp   = static_imputer.fit_transform(X_static)

    # Attach RecordID, static info and label 
    dataset_pca['RecordID']          = dataset['RecordID'].values
    for i, col in enumerate(static_cols):
        dataset_pca[col]             = X_static_imp[:, i]
    dataset_pca['In-hospital_death'] = dataset['In-hospital_death'].values

    return dataset_pca, pca, scaler, imputer, static_imputer


# ─────────────────────────────────────────────────────────
# Main 
# ─────────────────────────────────────────────────────────

if __name__ == '__main__':

    # ── Step 1: Load data 
    all_meta, all_ts = load_all_patients(FEATURES_DIR)
    outcomes         = load_outcomes(OUTCOMES_FILE)
    n_patients       = len(all_meta)

    # ── Step 2a: Filter outliers 
    all_ts = filter_outliers(all_ts)

    # ── Step 2b: Handle duplicates 
    all_ts = handle_duplicates(all_ts)

    # ── Step 2c: Filter high-missing parameters 
    all_ts, kept_params = filter_missing_parameters(all_ts, n_patients)

    # ── Step 3: Feature engineering 
    features = compute_mean_features(all_ts, kept_params, hours=24)
    dataset  = build_final_dataset(features, all_meta, outcomes)

    # ── Step 4: PCA 
    dataset_pca, pca, scaler, imputer, static_imputer = apply_pca(dataset, kept_params)

    # ── Save output 
    dataset.to_csv('features_raw.csv', index=False)
    dataset_pca.to_csv('features_pca.csv', index=False)
    print("Saved: features_raw.csv and features_pca.csv")