import pandas as pd
import numpy as np

# Load your dataset
df = pd.read_csv('/work/hdd/bdau/mbanisharifdehkordi/GNN_4_IO_5/data/1M/aiio_sample_1000000.csv')

# Check current state
print(f"Before normalization:")
print(f"POSIX_FILENOS max: {df['POSIX_FILENOS'].max():,.0f}")
print(f"POSIX_FILENOS min: {df['POSIX_FILENOS'].min():,.0f}")

# Apply log normalization ONLY to POSIX_FILENOS
df['POSIX_FILENOS'] = np.log10(df['POSIX_FILENOS'] + 1)

print(f"\nAfter normalization:")
print(f"POSIX_FILENOS max: {df['POSIX_FILENOS'].max():.2f}")
print(f"POSIX_FILENOS min: {df['POSIX_FILENOS'].min():.2f}")

# Verify all features are now in similar ranges
print(f"\nOverall data range check:")
feature_cols = [col for col in df.columns if col != 'tag']
print(f"Max across all features: {df[feature_cols].max().max():.2f}")
print(f"Min across all features: {df[feature_cols].min().min():.2f}")

# Save the properly normalized dataset
output_path = '/work/hdd/bdau/mbanisharifdehkordi/GNN_4_IO_5/data/1M/aiio_sample_1000000_normalized.csv'
df.to_csv(output_path, index=False)
print(f"\nSaved normalized data to: {output_path}")