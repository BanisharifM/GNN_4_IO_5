import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import json
import os

class IODataAnalyzer:
    """
    Comprehensive analyzer for I/O performance data
    Checks normalization, distributions, and AIIO paper compliance
    """
    
    def __init__(self, data_paths):
        """
        data_paths: dict with keys like 'sample_100', 'sample_10k', 'sample_100k'
        """
        self.data_paths = data_paths
        self.datasets = {}
        
    def load_all_datasets(self):
        """Load all datasets for comparison"""
        for name, path in self.data_paths.items():
            if os.path.exists(path):
                print(f"\nLoading {name} from {path}")
                df = pd.read_csv(path)
                self.datasets[name] = df
                print(f"  Shape: {df.shape}")
            else:
                print(f"  Warning: {path} not found")
    
    def analyze_normalization(self):
        """
        Check if data follows AIIO normalization: log(x+1) for features, raw for tag
        """
        print("\n" + "="*70)
        print("NORMALIZATION ANALYSIS (AIIO Paper Check)")
        print("="*70)
        
        for name, df in self.datasets.items():
            print(f"\n{name.upper()} Dataset:")
            print("-"*40)
            
            # Separate features and tag
            features = df.drop('tag', axis=1) if 'tag' in df.columns else df
            tag = df['tag'] if 'tag' in df.columns else None
            
            # 1. Check feature ranges
            print("\nFeature Statistics:")
            feature_max = features.max().max()
            feature_min = features.min().min()
            feature_mean = features.mean().mean()
            
            print(f"  Overall Min: {feature_min:.6f}")
            print(f"  Overall Max: {feature_max:.6f}")
            print(f"  Overall Mean: {feature_mean:.6f}")
            
            # 2. Check if log-normalized
            if feature_max > 100:
                print("  ⚠️ Features appear NOT log-normalized (max > 100)")
                
                # Test what log normalization would give
                log_features = np.log(features + 1)
                print(f"  After log(x+1): Min={log_features.min().min():.2f}, Max={log_features.max().max():.2f}")
                
                # Check specific I/O counters mentioned in AIIO
                important_counters = ['POSIX_BYTES_READ', 'POSIX_BYTES_WRITTEN', 
                                    'POSIX_SEQ_READS', 'POSIX_SEQ_WRITES']
                for counter in important_counters:
                    if counter in features.columns:
                        raw_val = features[counter].max()
                        log_val = np.log(raw_val + 1)
                        print(f"    {counter}: raw={raw_val:.0f}, log(x+1)={log_val:.2f}")
            else:
                print("  ✅ Features appear log-normalized")
            
            # 3. Check tag (performance) - should NOT be normalized
            if tag is not None:
                print(f"\nTag (Performance) Statistics:")
                print(f"  Min: {tag.min():.6f}")
                print(f"  Max: {tag.max():.6f}")
                print(f"  Mean: {tag.mean():.6f}")
                print(f"  Std: {tag.std():.6f}")
                
                # According to AIIO, tag is log of performance
                # So actual performance would be exp(tag) - 1
                actual_perf_min = np.exp(tag.min()) - 1
                actual_perf_max = np.exp(tag.max()) - 1
                print(f"  Actual Performance Range: {actual_perf_min:.2f} - {actual_perf_max:.2f} MB/s")
    
    def analyze_feature_distributions(self):
        """
        Analyze distribution of key I/O features
        """
        print("\n" + "="*70)
        print("FEATURE DISTRIBUTION ANALYSIS")
        print("="*70)
        
        # Key features from AIIO paper
        key_features = [
            'POSIX_SEQ_WRITES', 'POSIX_SEQ_READS',
            'POSIX_CONSEC_WRITES', 'POSIX_CONSEC_READS',
            'POSIX_SIZE_READ_0_100', 'POSIX_SIZE_WRITE_0_100',
            'POSIX_SIZE_READ_1K_10K', 'POSIX_SIZE_WRITE_1K_10K',
            'POSIX_MEM_NOT_ALIGNED', 'POSIX_FILE_NOT_ALIGNED',
            'POSIX_BYTES_READ', 'POSIX_BYTES_WRITTEN'
        ]
        
        for name, df in self.datasets.items():
            print(f"\n{name.upper()} - Key Features:")
            print("-"*40)
            
            for feature in key_features:
                if feature in df.columns:
                    values = df[feature].values
                    non_zero = (values > 0).sum()
                    zero_pct = (values == 0).mean() * 100
                    
                    if non_zero > 0:
                        non_zero_values = values[values > 0]
                        print(f"{feature:30} | Zero: {zero_pct:5.1f}% | "
                              f"NonZero Range: [{non_zero_values.min():.2e}, {non_zero_values.max():.2e}]")
                    else:
                        print(f"{feature:30} | All zeros")
    
    def check_data_integrity(self):
        """
        Check for data quality issues
        """
        print("\n" + "="*70)
        print("DATA INTEGRITY CHECK")
        print("="*70)
        
        for name, df in self.datasets.items():
            print(f"\n{name.upper()}:")
            
            # Check for nulls
            null_counts = df.isnull().sum().sum()
            print(f"  Null values: {null_counts}")
            
            # Check for negative values (shouldn't exist in I/O counters)
            neg_counts = (df < 0).sum().sum()
            print(f"  Negative values: {neg_counts}")
            
            # Check for inf values
            inf_counts = np.isinf(df.values).sum()
            print(f"  Inf values: {inf_counts}")
            
            # Check data types
            print(f"  Data types: {df.dtypes.value_counts().to_dict()}")
    
    def compare_similarity_impact(self):
        """
        Analyze why cosine similarity is so high
        """
        print("\n" + "="*70)
        print("SIMILARITY ANALYSIS")
        print("="*70)
        
        for name, df in self.datasets.items():
            if len(df) > 1000:
                # Sample for efficiency
                sample_df = df.sample(min(1000, len(df)), random_state=42)
            else:
                sample_df = df
            
            features = sample_df.drop('tag', axis=1) if 'tag' in sample_df.columns else sample_df
            
            print(f"\n{name.upper()} - Feature Characteristics:")
            
            # 1. Sparsity
            sparsity = (features == 0).mean().mean()
            print(f"  Overall sparsity: {sparsity*100:.1f}% zeros")
            
            # 2. Feature variance
            variances = features.var()
            low_var_features = (variances < 0.01).sum()
            print(f"  Low variance features: {low_var_features}/{len(variances)}")
            
            # 3. Sample a few similarities
            from sklearn.metrics.pairwise import cosine_similarity
            sample_features = features.values[:100]  # First 100 samples
            
            # Apply log if needed
            if sample_features.max() > 100:
                print("  Applying log(x+1) for similarity test...")
                sample_features = np.log(sample_features + 1)
            
            sims = cosine_similarity(sample_features)
            np.fill_diagonal(sims, 0)  # Remove self-similarity
            
            print(f"  Similarity distribution (100 samples):")
            print(f"    Min: {sims.min():.4f}")
            print(f"    25%: {np.percentile(sims, 25):.4f}")
            print(f"    50%: {np.percentile(sims, 50):.4f}")
            print(f"    75%: {np.percentile(sims, 75):.4f}")
            print(f"    95%: {np.percentile(sims, 95):.4f}")
            print(f"    Max: {sims.max():.4f}")
            
            # Count high similarities
            high_sim = (sims > 0.9).sum() / 2  # Divide by 2 for symmetry
            total_pairs = (100 * 99) / 2
            print(f"    Pairs with similarity > 0.9: {high_sim}/{total_pairs} ({high_sim/total_pairs*100:.1f}%)")
    
    def suggest_fixes(self):
        """
        Provide recommendations based on analysis
        """
        print("\n" + "="*70)
        print("RECOMMENDATIONS")
        print("="*70)
        
        # Check if normalization is needed
        for name, df in self.datasets.items():
            features = df.drop('tag', axis=1) if 'tag' in df.columns else df
            
            if features.max().max() > 100:
                print(f"\n{name}: Data needs log(x+1) normalization!")
                print("  Fix: df.iloc[:, :-1] = np.log(df.iloc[:, :-1] + 1)  # All except 'tag'")
                break
        else:
            print("\n✅ Data appears properly normalized")
        
        print("\nFor Graph Construction:")
        print("  1. Use MAX_NEIGHBORS = 50-100")
        print("  2. Use THRESHOLD = 0.85-0.90")
        print("  3. Consider feature selection to reduce similarity")

def main():
    """
    Main analysis pipeline
    """
    # Configure paths
    data_paths = {
        # 'sample_10k': '/work/hdd/bdau/mbanisharifdehkordi/GNN_4_IO_5/data/10K/aiio_sample_10000.csv',
        # 'sample_100k': '/work/hdd/bdau/mbanisharifdehkordi/GNN_4_IO_5/data/1M/aiio_sample_1000000_normalized.csv'
        'sample_total': '/work/hdd/bdau/mbanisharifdehkordi/GNN_4_IO_5/data/total/sample_train_total_normalized.csv'
    }
    
    print("="*70)
    print("I/O DATASET COMPREHENSIVE ANALYSIS")
    print("="*70)
    
    analyzer = IODataAnalyzer(data_paths)
    
    # Run all analyses
    analyzer.load_all_datasets()
    analyzer.analyze_normalization()
    analyzer.analyze_feature_distributions()
    analyzer.check_data_integrity()
    analyzer.compare_similarity_impact()
    analyzer.suggest_fixes()
    
    print("\n" + "="*70)
    print("ANALYSIS COMPLETE")
    print("="*70)

if __name__ == "__main__":
    main()
