"""
Helper utility functions for data loading and general processing
"""
import pandas as pd

def load_dataset_chunks(file_path, max_rows=1000000, chunk_size=20000, required_features=None):
    """
    Load dataset in chunks for memory efficiency
    
    Args:
        file_path: Path to CSV file
        max_rows: Maximum rows to load
        chunk_size: Size of each chunk
        required_features: List of required feature columns
    
    Returns:
        Combined DataFrame
    """
    print(f"Loading dataset from: {file_path}")
    
    chunks = []
    total_read = 0
    
    for chunk in pd.read_csv(file_path, chunksize=chunk_size, engine='python'):
        remaining = max_rows - total_read
        if remaining <= 0:
            break
            
        chunk = chunk.head(remaining)
        
        # Filter to required columns
        if required_features:
            available_cols = [col for col in required_features + 
                            ['IPV4_SRC_ADDR', 'IPV4_DST_ADDR', 'Label'] 
                            if col in chunk.columns]
            chunk = chunk[available_cols]
        
        # Drop critical missing values
        chunk = chunk.dropna(subset=['IPV4_SRC_ADDR', 'IPV4_DST_ADDR', 'Label'])
        
        if len(chunk) > 0:
            chunks.append(chunk)
            total_read += len(chunk)
            print(f"   Loaded chunk: {len(chunk):,} rows (total: {total_read:,})")
    
    if not chunks:
        raise ValueError("No valid data loaded from file")
    
    df = pd.concat(chunks, ignore_index=True)
    print(f"Total loaded: {len(df):,} rows")
    
    return df


def dynamic_feature_selection(df, all_features, top_k=7):
    """
    Robust dynamic feature selection using variance on benign flows.
    
    Converts columns to numeric, handles missing data, and selects most 
    variable features from benign traffic to ensure model learns normal behavior.
    
    Args:
        df: Input dataframe
        all_features: List of candidate feature column names
        top_k: Total number of features to select (default 7)
        node_features_count: How many to assign to nodes (default 4)
        random_state: For reproducibility (not used here, but kept for consistency)
        
    Returns:
        dict: {'node_features': [...], 'edge_features': [...]}
    """
    print(f"🔍 Starting dynamic feature selection... (top {top_k} from {len(all_features)} candidates)")
    
    # === Step 1: Filter to available features only ===
    available_features = [f for f in all_features if f in df.columns]
    if not available_features:
        print("❌ No candidate features found in data.")
        return {'node_features': [], 'edge_features': []}
    
    # === Step 2: Extract benign samples (unsupervised: train on normal traffic) ===
    if 'Label' not in df.columns:
        print("⚠️ 'Label' column missing! Using full dataset.")
        benign = df.copy()
    else:
        benign = df[df['Label'] == 0]
        if len(benign) == 0:
            print("⚠️ No benign (Label=0) samples! Using full dataset.")
            benign = df.copy()
    
    # === Step 3: Convert to numeric safely ===
    X = benign[available_features].copy()
    for col in X.columns:
        # Convert strings, commas, etc. to float; invalid → NaN
        X[col] = pd.to_numeric(X[col], errors='coerce')
    
    # Warn about failed conversions
    conversion_failures = X.isna().mean()
    problematic = conversion_failures[conversion_failures > 0.5].index.tolist()
    if problematic:
        print(f"⚠️  High NaN after conversion (possible non-numeric): {problematic}")
    
    # Keep only columns that have some numeric data
    X_numeric = X.select_dtypes(include=[np.number])
    valid_features = X_numeric.columns.tolist()
    
    if not valid_features:
        print("❌ No valid numeric features after conversion.")
        return {'node_features': [], 'edge_features': []}
    
    # Fill remaining NaNs with median
    X_numeric = X_numeric.fillna(X_numeric.median(numeric_only=True))
    
    # === Step 4: Select top-k by variance ===
    variances = X_numeric.var(numeric_only=True)
    top_k = min(top_k, len(variances))
    if top_k == 0:
        print("❌ No features to select.")
        return {'node_features': [], 'edge_features': []}
    
    top_features = variances.sort_values(ascending=False).head(top_k).index.tolist()
    

    
    return top_features