import pandas as pd

# Load the normalized dataset
file_path = '/work/hdd/bdau/mbanisharifdehkordi/GNN_4_IO_5/data/total/sample_train_total_normalized.csv'

print("Checking POSIX_FILENOS in normalized dataset...")
print("="*60)

# Load just the first row to check columns
df_header = pd.read_csv(file_path, nrows=1)
print(f"Total columns in dataset: {len(df_header.columns)}")
print(f"Columns list: {list(df_header.columns)[:5]}... (showing first 5)")

# Check if POSIX_FILENOS exists
if 'POSIX_FILENOS' in df_header.columns:
    print(f"\n✓ POSIX_FILENOS column EXISTS in the dataset")
    
    # Load just that column to check min/max
    df_filenos = pd.read_csv(file_path, usecols=['POSIX_FILENOS'])
    
    min_val = df_filenos['POSIX_FILENOS'].min()
    max_val = df_filenos['POSIX_FILENOS'].max()
    mean_val = df_filenos['POSIX_FILENOS'].mean()
    
    print(f"\nPOSIX_FILENOS statistics:")
    print(f"  Min: {min_val:.6f}")
    print(f"  Max: {max_val:.6f}")
    print(f"  Mean: {mean_val:.6f}")
    
    # Check if normalized
    if max_val < 20:
        print(f"\n✅ POSIX_FILENOS appears NORMALIZED (max < 20)")
    else:
        print(f"\n❌ POSIX_FILENOS appears NOT NORMALIZED (max = {max_val:.0f})")
        
else:
    print(f"\n❌ POSIX_FILENOS column is MISSING from the dataset!")
    print(f"\nActual columns in dataset:")
    for i, col in enumerate(df_header.columns):
        print(f"  {i+1}. {col}")
