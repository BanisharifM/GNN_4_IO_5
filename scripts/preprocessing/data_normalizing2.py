import pandas as pd
import numpy as np
import gc

def normalize_large_dataset(input_path, output_path, chunk_size=100000):
    """
    Normalize POSIX_FILENOS column in batches for large datasets
    """
    
    # First pass: Get statistics about POSIX_FILENOS
    print("=" * 60)
    print("PASS 1: Analyzing POSIX_FILENOS column")
    print("=" * 60)
    
    max_val = 0
    min_val = float('inf')
    total_rows = 0
    
    for i, chunk in enumerate(pd.read_csv(input_path, chunksize=chunk_size)):
        chunk_max = chunk['POSIX_FILENOS'].max()
        chunk_min = chunk['POSIX_FILENOS'].min()
        max_val = max(max_val, chunk_max)
        min_val = min(min_val, chunk_min)
        total_rows += len(chunk)
        
        if i % 10 == 0:
            print(f"  Analyzed {total_rows:,} rows...")
    
    print(f"\nOriginal POSIX_FILENOS stats:")
    print(f"  Total rows: {total_rows:,}")
    print(f"  Max value: {max_val:,.0f}")
    print(f"  Min value: {min_val:,.0f}")
    
    # Second pass: Apply normalization and write
    print("\n" + "=" * 60)
    print("PASS 2: Applying normalization and saving")
    print("=" * 60)
    
    # Process first chunk to get header
    first_chunk = True
    processed_rows = 0
    
    # Track new min/max after normalization
    new_max = 0
    new_min = float('inf')
    
    for i, chunk in enumerate(pd.read_csv(input_path, chunksize=chunk_size)):
        # Apply log normalization to POSIX_FILENOS
        original_values = chunk['POSIX_FILENOS'].copy()
        chunk['POSIX_FILENOS'] = np.log10(chunk['POSIX_FILENOS'] + 1)
        
        # Verify the transformation
        chunk_new_max = chunk['POSIX_FILENOS'].max()
        chunk_new_min = chunk['POSIX_FILENOS'].min()
        new_max = max(new_max, chunk_new_max)
        new_min = min(new_min, chunk_new_min)
        
        # Save chunk
        if first_chunk:
            chunk.to_csv(output_path, index=False, mode='w')
            first_chunk = False
        else:
            chunk.to_csv(output_path, index=False, mode='a', header=False)
        
        processed_rows += len(chunk)
        
        # Progress update with verification
        if i % 10 == 0:
            print(f"  Processed {processed_rows:,} rows...")
            print(f"    Sample original value: {original_values.iloc[0]:,.0f}")
            print(f"    Sample normalized value: {chunk['POSIX_FILENOS'].iloc[0]:.4f}")
        
        # Clear memory
        del chunk
        gc.collect()
    
    print(f"\nNormalized POSIX_FILENOS stats:")
    print(f"  Max value: {new_max:.4f}")
    print(f"  Min value: {new_min:.4f}")
    
    # Third pass: Verify the result
    print("\n" + "=" * 60)
    print("PASS 3: Verifying normalized dataset")
    print("=" * 60)
    
    # Check a few random chunks to ensure normalization was applied
    verification_chunks = [0, total_rows // (2 * chunk_size), total_rows // chunk_size - 1]
    
    for chunk_idx in verification_chunks:
        skip_rows = chunk_idx * chunk_size
        verify_chunk = pd.read_csv(output_path, 
                                  skiprows=skip_rows + 1,  # +1 for header
                                  nrows=min(1000, chunk_size),
                                  header=None,
                                  names=pd.read_csv(output_path, nrows=0).columns)
        
        posix_max = verify_chunk['POSIX_FILENOS'].max()
        posix_min = verify_chunk['POSIX_FILENOS'].min()
        
        print(f"  Chunk starting at row {skip_rows:,}:")
        print(f"    POSIX_FILENOS range: [{posix_min:.4f}, {posix_max:.4f}]")
        
        # Check if values are in expected log range
        if posix_max > 15:  # log10(10^15) = 15, which would be huge
            print(f"    ⚠️ WARNING: Found suspiciously large values!")
        else:
            print(f"    ✓ Values are in expected range")
    
    # Final check: Overall statistics
    print("\n" + "=" * 60)
    print("FINAL VERIFICATION")
    print("=" * 60)
    
    # Read first and last 10000 rows to check overall range
    df_head = pd.read_csv(output_path, nrows=10000)
    
    # For last rows, we need to know total rows first
    df_tail = pd.read_csv(output_path, skiprows=range(1, total_rows - 10000))
    
    feature_cols = [col for col in df_head.columns if col != 'tag']
    
    print(f"First 10,000 rows - Max across all features: {df_head[feature_cols].max().max():.4f}")
    print(f"First 10,000 rows - Min across all features: {df_head[feature_cols].min().min():.4f}")
    print(f"Last 10,000 rows - Max across all features: {df_tail[feature_cols].max().max():.4f}")
    print(f"Last 10,000 rows - Min across all features: {df_tail[feature_cols].min().min():.4f}")
    
    # Specific check for POSIX_FILENOS
    print(f"\nPOSIX_FILENOS specific check:")
    print(f"  First 10K - range: [{df_head['POSIX_FILENOS'].min():.4f}, {df_head['POSIX_FILENOS'].max():.4f}]")
    print(f"  Last 10K - range: [{df_tail['POSIX_FILENOS'].min():.4f}, {df_tail['POSIX_FILENOS'].max():.4f}]")
    
    print(f"\n✓ Normalization complete! Saved to: {output_path}")
    print(f"  Total rows processed: {processed_rows:,}")
    
    return True

# Run the normalization
if __name__ == "__main__":
    input_file = '/work/hdd/bdau/mbanisharifdehkordi/GNN_4_IO_5/data/total/sample_train_total.csv'
    output_file = '/work/hdd/bdau/mbanisharifdehkordi/GNN_4_IO_5/data/total/sample_train_total_normalized.csv'
    
    # You can adjust chunk_size based on your available memory
    # Smaller chunks = less memory but slower
    # Larger chunks = more memory but faster
    success = normalize_large_dataset(input_file, output_file, chunk_size=100000)
    
    if success:
        print("\n" + "🎉" * 30)
        print("SUCCESS: Dataset normalized successfully!")
        print("🎉" * 30)