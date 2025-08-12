import os
import logging
import json
import pandas as pd
import numpy as np
from datetime import datetime
from tqdm import tqdm

class AIIOStratifiedSampler:
    """
    Stratified sampler optimized for AIIO-style I/O performance data
    """
    
    def __init__(self, input_file, output_dir='./sampled_data'):
        self.input_file = input_file
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
    def get_dataset_info(self):
        """Get basic info about the dataset without loading it all"""
        logging.info("Getting dataset information...")
        
        # Count rows more safely
        total_rows = 0
        with open(self.input_file, 'r') as f:
            for _ in f:
                total_rows += 1
        total_rows -= 1  # Subtract header
        
        # Get columns from first row
        df_sample = pd.read_csv(self.input_file, nrows=5)
        
        logging.info(f"Total rows: {total_rows:,}")
        logging.info(f"Total columns: {len(df_sample.columns)}")
        
        return total_rows, df_sample.columns.tolist()
    
    def stratified_sample_for_gnn(self, k=100000, n_strata=20):
        """
        Simplified stratified sampling by performance
        """
        logging.info(f"Starting stratified sampling for {k:,} samples...")
        
        # Step 1: Get performance values
        logging.info("Reading performance values...")
        perf_values = []
        chunksize = 50000
        
        for chunk in tqdm(pd.read_csv(self.input_file, chunksize=chunksize, usecols=['tag']), 
                         desc="Reading performance"):
            perf_values.extend(chunk['tag'].values)
        
        perf_values = np.array(perf_values)
        
        # Step 2: Create strata
        percentiles = np.percentile(perf_values, np.linspace(0, 100, n_strata + 1))
        logging.info(f"Performance strata boundaries: {percentiles[:5]}... (showing first 5)")
        
        # Step 3: Assign samples to strata
        logging.info("Assigning samples to strata...")
        strata_indices = {i: [] for i in range(n_strata)}
        
        for idx, perf in enumerate(perf_values):
            stratum = np.searchsorted(percentiles[1:-1], perf)
            stratum = min(stratum, n_strata - 1)  # Ensure within bounds
            strata_indices[stratum].append(idx)
        
        # Step 4: Sample from each stratum
        samples_per_stratum = k // n_strata
        extra_samples = k % n_strata
        
        selected_indices = []
        np.random.seed(42)  # For reproducibility
        
        for i in range(n_strata):
            n_samples = samples_per_stratum + (1 if i < extra_samples else 0)
            stratum_indices = np.array(strata_indices[i])
            
            if len(stratum_indices) == 0:
                logging.warning(f"Stratum {i} is empty!")
                continue
            elif len(stratum_indices) <= n_samples:
                selected = stratum_indices
            else:
                selected = np.random.choice(stratum_indices, n_samples, replace=False)
            
            selected_indices.extend(selected)
            logging.info(f"Stratum {i}: selected {len(selected)} from {len(stratum_indices)}")
        
        return np.sort(selected_indices)
    
    def extract_samples(self, indices, output_file):
        """
        Extract selected samples from the large file
        """
        logging.info(f"Extracting {len(indices):,} samples...")
        
        indices_set = set(indices)
        selected_data = []
        current_idx = 0
        chunksize = 10000
        
        with tqdm(total=len(indices), desc="Extracting samples") as pbar:
            for chunk in pd.read_csv(self.input_file, chunksize=chunksize):
                # Check which rows to keep
                chunk_indices = range(current_idx, current_idx + len(chunk))
                mask = [idx in indices_set for idx in chunk_indices]
                
                if any(mask):
                    selected_data.append(chunk[mask])
                    pbar.update(sum(mask))
                
                current_idx += len(chunk)
                
                # Stop if we've found all samples
                if len(selected_data) > 0 and sum(len(df) for df in selected_data) >= len(indices):
                    break
        
        # Combine selected data
        df_final = pd.concat(selected_data, ignore_index=True)
        
        # Shuffle
        df_final = df_final.sample(frac=1, random_state=42).reset_index(drop=True)
        
        # Save
        output_path = os.path.join(self.output_dir, output_file)
        df_final.to_csv(output_path, index=False)
        logging.info(f"Saved {len(df_final):,} samples to {output_path}")
        
        return df_final


def main():
    """
    Simple main function to run the sampling
    """
    # Configuration - UPDATE THIS PATH
    INPUT_FILE = '/work/hdd/bdau/mbanisharifdehkordi/GNN_4_IO_5/data/total/sample_train_total.csv'
    OUTPUT_DIR = '/work/hdd/bdau/mbanisharifdehkordi/GNN_4_IO_5/data/100K'
    K = 100000
    
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(message)s',
        handlers=[
            logging.StreamHandler()
        ]
    )
    
    print("="*60)
    print(f"AIIO Dataset Stratified Sampling")
    print(f"Sampling {K:,} rows from dataset")
    print("="*60)
    
    # Check if input file exists
    if not os.path.exists(INPUT_FILE):
        print(f"ERROR: Input file not found: {INPUT_FILE}")
        print("Please update the INPUT_FILE path in the script")
        return
    
    # Create output directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    try:
        # Initialize sampler
        sampler = AIIOStratifiedSampler(INPUT_FILE, OUTPUT_DIR)
        
        # Get dataset info
        print("\nStep 1: Getting dataset info...")
        total_rows, columns = sampler.get_dataset_info()
        print(f"Dataset: {total_rows:,} rows x {len(columns)} columns")
        
        # Check for required column
        if 'tag' not in columns:
            print("ERROR: 'tag' column not found in dataset!")
            return
        
        # Perform sampling
        print(f"\nStep 2: Stratified sampling of {K:,} rows...")
        selected_indices = sampler.stratified_sample_for_gnn(k=K, n_strata=20)
        
        # Extract samples
        print("\nStep 3: Extracting selected samples...")
        output_filename = f'aiio_sample_{K}.csv'
        df_sampled = sampler.extract_samples(selected_indices, output_filename)
        
        # Verify
        print("\n" + "="*60)
        print("SAMPLING COMPLETE!")
        print(f"✓ Sampled {len(df_sampled):,} rows")
        print(f"✓ Saved to: {os.path.join(OUTPUT_DIR, output_filename)}")
        print(f"✓ Performance range: {df_sampled['tag'].min():.2f} - {df_sampled['tag'].max():.2f}")
        print(f"✓ Performance mean: {df_sampled['tag'].mean():.2f}")
        print("="*60)
        
        # Save simple metadata
        metadata = {
            'original_rows': total_rows,
            'sampled_rows': len(df_sampled),
            'output_file': output_filename,
            'performance_mean': float(df_sampled['tag'].mean()),
            'performance_std': float(df_sampled['tag'].std()),
            'performance_min': float(df_sampled['tag'].min()),
            'performance_max': float(df_sampled['tag'].max())
        }
        
        metadata_file = os.path.join(OUTPUT_DIR, 'metadata.json')
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)
        print(f"✓ Metadata saved to: {metadata_file}")
        
    except Exception as e:
        print(f"\nERROR: {str(e)}")
        import traceback
        traceback.print_exc()
        

if __name__ == "__main__":
    # Test mode - uncomment to test with smaller sample
    # K = 1000  # Test with 1000 samples first
    
    main()