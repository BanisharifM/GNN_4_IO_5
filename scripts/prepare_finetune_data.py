#!/usr/bin/env python
"""
Prepare fine-tuning data from existing IOR results
Save as: scripts/prepare_finetune_data.py
"""

import numpy as np
import pandas as pd
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class FinetuneDataCollector:
    """Create fine-tuning dataset from existing IOR data"""
    
    def __init__(self, base_output_dir: str = "data/finetune"):
        self.output_dir = Path(base_output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def augment_existing_data(self):
        """Create variations of existing IOR data"""
        
        # Load your current IOR features
        ior_df = pd.read_csv('ior_all_ranks_features.csv')
        
        augmented_data = []
        
        for idx, row in ior_df.iterrows():
            # Original data
            augmented_data.append(row.to_dict())
            
            # Create 30 variations of each rank (4 ranks * 30 = 120 variations)
            for i in range(30):
                noisy_row = row.copy()
                
                # Add small noise to features
                feature_cols = [col for col in ior_df.columns 
                              if col not in ['performance', 'jobid']]
                
                for col in feature_cols:
                    # Add gaussian noise scaled to feature magnitude
                    current_val = noisy_row[col]
                    if current_val != 0 and current_val != -10:  # Skip special values
                        noise = np.random.normal(0, abs(current_val) * 0.05)  # 5% noise
                        noisy_row[col] = current_val + noise
                
                # Slightly vary performance
                # Your actual performance is ~2.4, so vary around that
                base_perf = noisy_row['performance']
                perf_noise = np.random.normal(0, 0.1)  # Small variation
                noisy_row['performance'] = base_perf + perf_noise
                
                augmented_data.append(noisy_row.to_dict())
        
        return pd.DataFrame(augmented_data)
    
    def generate_synthetic_ior_patterns(self, n_samples: int = 100):
        """Generate synthetic data similar to IOR patterns"""
        
        # Load IOR data as template
        ior_df = pd.read_csv('ior_all_ranks_features.csv')
        template = ior_df.iloc[0]
        feature_cols = [col for col in ior_df.columns 
                       if col not in ['performance', 'jobid']]
        
        synthetic_data = []
        
        for i in range(n_samples):
            sample = {}
            
            # Create variations based on IOR patterns
            for col in feature_cols:
                base_val = template[col]
                
                if base_val == -10:  # Special "no data" value
                    # Sometimes keep it, sometimes change
                    if np.random.random() > 0.7:
                        sample[col] = np.random.uniform(-1, 1)
                    else:
                        sample[col] = -10
                elif 'SIZE_WRITE' in col or 'SIZE_READ' in col:
                    # These features vary more in real workloads
                    if base_val > -5:
                        sample[col] = base_val + np.random.uniform(-1, 2)
                    else:
                        sample[col] = base_val
                elif 'STRIPE' in col:
                    # Stripe settings usually stay similar
                    sample[col] = base_val + np.random.uniform(-0.1, 0.1)
                else:
                    # Moderate variation for others
                    sample[col] = base_val + np.random.normal(0, 0.3)
            
            # Generate performance in similar range to IOR
            # Your IOR is ~2.4, so generate around that
            sample['performance'] = np.random.uniform(2.0, 3.0)
            
            synthetic_data.append(sample)
        
        return pd.DataFrame(synthetic_data)
    
    def create_finetune_dataset(self):
        """Create complete fine-tuning dataset"""
        
        logger.info("Creating fine-tuning dataset...")
        
        # 1. Load original IOR data (4 ranks)
        ior_base = pd.read_csv('ior_all_ranks_features.csv')
        
        # 2. Create augmented variations (4 * 31 = 124 samples)
        augmented = self.augment_existing_data()
        
        # 3. Add synthetic patterns (100 samples)
        synthetic = self.generate_synthetic_ior_patterns(n_samples=100)
        
        # Combine all data
        all_dfs = [augmented, synthetic]  # augmented already includes originals
        finetune_df = pd.concat(all_dfs, ignore_index=True)
        
        # Remove jobid column
        if 'jobid' in finetune_df.columns:
            finetune_df = finetune_df.drop('jobid', axis=1)
        
        # Rename performance to tag for consistency with training format
        if 'performance' in finetune_df.columns:
            finetune_df = finetune_df.rename(columns={'performance': 'tag'})
        
        # Ensure all values are numeric
        for col in finetune_df.columns:
            finetune_df[col] = pd.to_numeric(finetune_df[col], errors='coerce')
        
        # Fill any NaN values
        finetune_df = finetune_df.fillna(0)
        
        # Save dataset
        output_path = self.output_dir / 'ior_finetune_data.csv'
        finetune_df.to_csv(output_path, index=False)
        
        logger.info(f"Saved {len(finetune_df)} samples to {output_path}")
        
        return finetune_df

if __name__ == "__main__":
    collector = FinetuneDataCollector()
    finetune_df = collector.create_finetune_dataset()
    
    print(f"Created fine-tuning dataset with {len(finetune_df)} samples")
    
    # Fix: Use .values to get the actual values before formatting
    tag_values = finetune_df['tag'].values
    print(f"Performance range: {tag_values.min():.2f} - {tag_values.max():.2f}")
    print(f"Mean performance: {tag_values.mean():.2f}")
    
    # Show sample distribution
    print(f"\nDataset composition:")
    print(f"  - Original IOR samples: 4")
    print(f"  - Augmented variations: ~120") 
    print(f"  - Synthetic IOR-like: 100")
    print(f"  - Total: {len(finetune_df)}")