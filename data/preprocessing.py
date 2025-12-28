import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from imblearn.over_sampling import SMOTE, ADASYN

def preprocess_features(train_df, df, exclude_cols=None, clip_quantile=0.999):
    """
    Apply leakage-free preprocessing to train/test datasets:
      - Handle infinity values FIRST
      - Apply log transformations to skewed features
      - Encode categorical variables
      - Handle missing values
      - Clip extreme values
      - Scale numerical features
    
    Args:
        train_df: training dataframe
        df: full dataframe to preprocess
        exclude_cols: list of columns to exclude 
                      (e.g. ['Label', 'IPV4_SRC_ADDR', 'IPV4_DST_ADDR'])
        clip_quantile: upper quantile for clipping numeric features
        
    Returns:
        df (preprocessed), scaler, label_encoders
    """
    print(" CRITICAL: Handling infinity values BEFORE any processing...")
    
    # Get numeric columns first
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    
    # Replace inf/-inf with NaN in BOTH dataframes
    for col in numeric_cols:
        if col in df.columns:
            inf_count = np.isinf(df[col]).sum()
            if inf_count > 0:
                print(f"  Fixing {inf_count} infinity values in {col}")
            df[col] = df[col].replace([np.inf, -np.inf], np.nan)
        
        if col in train_df.columns:
            train_df[col] = train_df[col].replace([np.inf, -np.inf], np.nan)
    
    # Fill NaN with medians from training data
    for col in numeric_cols:
        if col in df.columns:
            train_median = train_df[col].median()
            if np.isnan(train_median):
                train_median = 0.0  # Fallback if all training values were inf
            df[col] = df[col].fillna(train_median)
    
    print(" Infinity values handled")
    
    # *** ADD LOG TRANSFORMATIONS FOR NETWORK DATA ***
    LOG_TRANSFORM_COLS = [
        'IN_BYTES', 'OUT_BYTES', 'IN_PKTS', 'OUT_PKTS', 
        'FLOW_DURATION_MILLISECONDS', 'SRC_TO_DST_SECOND_BYTES',
        'DST_TO_SRC_SECOND_BYTES', 'RETRANSMITTED_IN_BYTES',
        'RETRANSMITTED_OUT_BYTES', 'SRC_TO_DST_AVG_THROUGHPUT',
        'DST_TO_SRC_AVG_THROUGHPUT', 'NUM_PKTS_UP_TO_128_BYTES'
    ]
    
    print(" Applying log transformations...")
    for col in LOG_TRANSFORM_COLS:
        if col in df.columns and col in numeric_cols:
            # Handle zeros and negatives (add small constant)
            df[col] = df[col].clip(lower=1e-10)  # Avoid log(0)
            train_df[col] = train_df[col].clip(lower=1e-10)
            
            # Apply log1p (log(1+x)) - more stable than log(x)
            df[col] = np.log1p(df[col])
            train_df[col] = np.log1p(train_df[col])
            
            print(f"   Log-transformed {col}")
    
    if exclude_cols is None:
        exclude_cols = ['Label', 'IPV4_SRC_ADDR', 'IPV4_DST_ADDR', 'Attack']
    
    # --- Separate columns ---
    feature_cols = [c for c in train_df.columns if c not in exclude_cols]
    cat_cols = [c for c in feature_cols if train_df[c].dtype == 'object']
    num_cols = [c for c in feature_cols if train_df[c].dtype != 'object']
    
    df = df.copy()
    
    print(f" Preprocessing {len(feature_cols)} features ({len(cat_cols)} categorical, {len(num_cols)} numerical)")
    
    # --- Encode categorical variables ---
    label_encoders = {}
    for col in cat_cols:
        if col in df.columns:
            le = LabelEncoder()
            train_vals = train_df[col].astype(str).fillna("NA")
            full_vals = df[col].astype(str).fillna("NA")
            
            # Fit on combined unique values to handle unseen categories
            all_unique = list(set(train_vals.unique()).union(set(full_vals.unique())))
            le.fit(all_unique)
            
            df[col] = le.transform(full_vals)
            label_encoders[col] = le
    
    # --- Handle missing values for numeric columns ---
    print(" Handling missing values...")
    missing_counts = {}
    
    for col in num_cols:
        if col in df.columns:
            missing_count = df[col].isna().sum()
            if missing_count > 0:
                missing_counts[col] = missing_count
                
            # Use training median for imputation (leakage-free)
            train_median = train_df[col].median()
            if np.isnan(train_median):  # All training values were NaN
                train_median = 0.0
            df[col] = df[col].fillna(train_median)
    
    if missing_counts:
        print(f"  Handled missing values in {len(missing_counts)} columns")
    
    # --- Clip outliers based on training data ---
    if num_cols:
        clip_vals = train_df[num_cols].quantile(clip_quantile)
        df[num_cols] = df[num_cols].clip(upper=clip_vals, axis=1)
        print(f"   Clipped outliers at {clip_quantile} quantile")
    
    # --- Scaling ---
    scaler = StandardScaler()
    if num_cols:
        scaler.fit(train_df[num_cols])
        df[num_cols] = scaler.transform(df[num_cols])
        print("   Applied StandardScaler")
    
    return df, scaler, label_encoders


def apply_iot_imbalance_handling(df, dataset_name, feature_cols):
    """
    Apply oversampling techniques for IoT datasets with extreme class imbalance
    
    Args:
        df: Input dataframe
        dataset_name: Name of dataset for conditional processing
        feature_cols: List of feature columns for SMOTE
        
    Returns:
        Processed dataframe with balanced classes
    """
    print(f" Applying imbalance handling for {dataset_name}...")
    
    # Check current class distribution
    class_counts = df['Label'].value_counts()
    total_samples = len(df)
    benign_ratio = class_counts[0] / total_samples if 0 in class_counts else 0
    
    print(f"   Original class distribution:")
    print(f"   Benign: {class_counts.get(0, 0):,} ({benign_ratio:.3f})")
    print(f"   Attack: {class_counts.get(1, 0):,} ({1-benign_ratio:.3f})")
    
    # Apply oversampling for extremely imbalanced IoT datasets
    if ("ToN" in dataset_name or "BoT" in dataset_name) and benign_ratio < 0.05:
        print(f"    Extreme imbalance detected - applying oversampling...")
        
        try:
            # Prepare numerical features only for SMOTE
            X = df[feature_cols].select_dtypes(include=[np.number])
            y = df['Label']
            X = X.fillna(X.median())
            
            # Select appropriate sampling strategy based on imbalance severity
            if benign_ratio < 0.001:  # BoT-IoT case (99.9% attacks)
                print(f"   Using ADASYN for extreme imbalance...")
                sampler = ADASYN(sampling_strategy=0.1, random_state=42)
            else:
                print(f"   Using SMOTE for moderate imbalance...")
                sampler = SMOTE(sampling_strategy=0.2, random_state=42)
            
            X_resampled, y_resampled = sampler.fit_resample(X, y)
            
            # Reconstruct dataframe with oversampled data
            df_resampled = pd.DataFrame(X_resampled, columns=X.columns)
            df_resampled['Label'] = y_resampled
            
            # Add back non-numeric columns (IP addresses)
            _add_ip_addresses_to_resampled_data(df, df_resampled, len(X_resampled) - len(df))
            
            print(f"    Oversampling completed: {len(df_resampled):,} rows")
            return df_resampled
            
        except Exception as e:
            print(f"   ⚠️ Oversampling failed: {e}")
            print(f"   Continuing with original data...")
    
    return df


def _add_ip_addresses_to_resampled_data(original_df, resampled_df, n_synthetic):
    """Helper function to add IP addresses to synthetic samples"""
    non_numeric_cols = ['IPV4_SRC_ADDR', 'IPV4_DST_ADDR']
    
    for col in non_numeric_cols:
        if col in original_df.columns:
            # Extend with original IPs and random benign IPs for synthetic samples
            extended_col = original_df[col].tolist()
            
            benign_ips = original_df[original_df['Label'] == 0][col].unique()
            synthetic_ips = np.random.choice(benign_ips, n_synthetic, replace=True)
            extended_col.extend(synthetic_ips)
            
            resampled_df[col] = extended_col[:len(resampled_df)]


def filter_dataset_for_efficiency(df, dataset_name, max_rows=500000):
    """Apply dataset-specific filtering after global preprocessing"""
    print(f" Filtering {dataset_name} dataset for efficiency...")
    print(f"   Original size: {len(df):,} rows")
    
    if "ToN" in dataset_name or "BoT" in dataset_name:
        return _filter_iot_dataset(df, max_rows)
    elif "CIC" in dataset_name:
        return _filter_cic_dataset(df, max_rows)
    else:
        return _filter_standard_dataset(df, max_rows)


def _filter_iot_dataset(df, max_rows):
    benign_df = df[df['Label'] == 0]
    attack_df = df[df['Label'] == 1]
    if len(df) > max_rows:
        available_space = max_rows - len(benign_df)
        if available_space > 0 and len(attack_df) > available_space:
            attack_sample = attack_df.sample(n=available_space, random_state=42)
            return pd.concat([benign_df, attack_sample], ignore_index=True)
    return df


def _filter_cic_dataset(df, max_rows):
    ip_counts = pd.concat([df['IPV4_SRC_ADDR'], df['IPV4_DST_ADDR']]).value_counts()
    active_ips = ip_counts[ip_counts >= 10]
    top_ips = set(active_ips.head(2000).index.tolist())
    df_filtered = df[(df['IPV4_SRC_ADDR'].isin(top_ips)) & (df['IPV4_DST_ADDR'].isin(top_ips))]
    if len(df_filtered) > max_rows:
        df_filtered = _apply_stratified_sampling(df_filtered, max_rows)
    return df_filtered


def _filter_standard_dataset(df, max_rows):
    if len(df) > max_rows:
        return _apply_stratified_sampling(df, max_rows)
    return df


def _apply_stratified_sampling(df, max_rows):
    benign_df = df[df['Label'] == 0]
    attack_df = df[df['Label'] == 1]
    benign_ratio = len(benign_df) / len(df)
    benign_sample_size = int(max_rows * benign_ratio)
    attack_sample_size = max_rows - benign_sample_size
    
    benign_sample = benign_df.sample(n=min(benign_sample_size, len(benign_df)), random_state=42)
    attack_sample = attack_df.sample(n=min(attack_sample_size, len(attack_df)), random_state=42)
    return pd.concat([benign_sample, attack_sample], ignore_index=True)


def create_leakage_free_split(df, test_size=0.15):
    benign_ips = set(df[df['Label'] == 0]['IPV4_SRC_ADDR'].unique()) | \
                 set(df[df['Label'] == 0]['IPV4_DST_ADDR'].unique())
    malicious_ips = set(df[df['Label'] == 1]['IPV4_SRC_ADDR'].unique()) | \
                    set(df[df['Label'] == 1]['IPV4_DST_ADDR'].unique())
    
    contaminated_ips = benign_ips & malicious_ips
    pure_benign_ips = benign_ips - contaminated_ips
    all_malicious_ips = malicious_ips | contaminated_ips
    
    if len(pure_benign_ips) >= 10:
        train_benign_ips, test_benign_ips = train_test_split(
            list(pure_benign_ips), test_size=test_size, random_state=123
        )
    else:
        split_point = max(1, int(len(pure_benign_ips) * (1 - test_size)))
        train_benign_ips = list(pure_benign_ips)[:split_point]
        test_benign_ips = list(pure_benign_ips)[split_point:]
    
    test_ips = set(test_benign_ips) | all_malicious_ips
    return set(train_benign_ips), test_ips, all_malicious_ips