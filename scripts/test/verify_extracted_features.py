# verify_extracted_features.py
import pandas as pd
import numpy as np

# Load the extracted features
extracted_df = pd.read_csv('ior_all_ranks_features.csv')
print("Extracted features shape:", extracted_df.shape)
print("\nFirst few columns:")
print(extracted_df.columns[:10].tolist())

# Load your training data for comparison
train_df = pd.read_csv('/work/hdd/bdau/mbanisharifdehkordi/GNN_4_IO_5/data/100/aiio_sample_100.csv')
print("\nTraining data shape:", train_df.shape)

# Compare columns (excluding jobid which is metadata)
extracted_cols = set(extracted_df.columns) - {'jobid'}
train_cols = set(train_df.columns)

# Check if we have the right columns
if 'tag' in train_cols and 'performance' in extracted_cols:
    print("\n✓ Performance metric present (called 'performance' in extracted, 'tag' in training)")
    train_cols = train_cols - {'tag'}
    extracted_cols = extracted_cols - {'performance'}

missing = train_cols - extracted_cols
extra = extracted_cols - train_cols

if not missing and not extra:
    print("✓ Perfect match! All feature columns align.")
else:
    if missing:
        print(f"⚠ Missing columns: {missing}")
    if extra:
        print(f"⚠ Extra columns: {extra}")

# Show some statistics
print("\nExtracted data statistics:")
print(extracted_df[['POSIX_READS', 'POSIX_WRITES', 'POSIX_BYTES_READ', 
                    'POSIX_BYTES_WRITTEN', 'performance']].describe())

# Check if normalization looks correct (should be log10 transformed)
print("\nNormalization check (should be log10 values):")
sample_features = ['POSIX_BYTES_READ', 'POSIX_BYTES_WRITTEN', 'LUSTRE_STRIPE_SIZE']
for feat in sample_features:
    if feat in extracted_df.columns:
        val = extracted_df[feat].iloc[0]
        print(f"  {feat}: {val:.4f} (raw would be ~{10**val:.0f})")