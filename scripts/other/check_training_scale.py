# Quick check script - save as check_training_scale.py
import pandas as pd
import numpy as np

# Load training data
train_df = pd.read_csv('data/1M/aiio_sample_1000000_normalized.csv', nrows=100)

print("Training data 'tag' column statistics:")
print(f"  Min: {train_df['tag'].min():.4f}")
print(f"  Max: {train_df['tag'].max():.4f}")
print(f"  Mean: {train_df['tag'].mean():.4f}")

# Check if values look like log10 (should be 0-4 range) or linear (could be 0-10000+)
if train_df['tag'].max() < 10:
    print("  -> Looks like log10 scale")
else:
    print("  -> Looks like linear scale")

# Show some examples
print("\nFirst 5 tag values:")
for i in range(5):
    tag_val = train_df.iloc[i]['tag']
    print(f"  {tag_val:.4f} (linear: {10**tag_val:.2f} MB/s)")