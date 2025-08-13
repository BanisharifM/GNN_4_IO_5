import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

class DetailedColumnAnalyzer:
    """
    Analyze each column individually to determine normalization status
    """
    
    def __init__(self, data_path):
        self.data_path = data_path
        self.df = None
        
    def analyze_all_columns(self):
        """
        Check normalization status of each column
        """
        print("Loading data...")
        self.df = pd.read_csv(self.data_path, nrows=100000)  # Sample for speed
        
        print("\n" + "="*80)
        print("COLUMN-BY-COLUMN NORMALIZATION ANALYSIS")
        print("="*80)
        
        results = {
            'normalized': [],
            'not_normalized': [],
            'ambiguous': []
        }
        
        for col in self.df.columns:
            col_data = self.df[col].values
            col_data = col_data[col_data > 0]  # Only non-zero values
            
            if len(col_data) == 0:
                continue
                
            min_val = col_data.min()
            max_val = col_data.max()
            mean_val = col_data.mean()
            
            # Check for log normalization indicators
            is_normalized = self.check_if_normalized(min_val, max_val, mean_val)
            
            status = "✅ NORMALIZED" if is_normalized == 'yes' else \
                     "❌ NOT NORMALIZED" if is_normalized == 'no' else \
                     "⚠️  AMBIGUOUS"
            
            print(f"\n{col:30} | {status}")
            print(f"  Range: [{min_val:.6f}, {max_val:.6f}]")
            print(f"  Mean: {mean_val:.6f}")
            
            # Check for telltale signs
            if min_val > 0.2 and min_val < 0.4:  # Likely log10(2) ≈ 0.301
                print(f"  → Min ≈ log10(2), suggests normalized")
            if max_val > 100:
                print(f"  → Max > 100, suggests NOT normalized")
            if max_val < 20:
                print(f"  → Max < 20, suggests normalized")
                
            # Categorize
            if is_normalized == 'yes':
                results['normalized'].append(col)
            elif is_normalized == 'no':
                results['not_normalized'].append(col)
            else:
                results['ambiguous'].append(col)
        
        # Summary
        print("\n" + "="*80)
        print("SUMMARY")
        print("="*80)
        print(f"\n✅ Normalized columns ({len(results['normalized'])}): ")
        for col in results['normalized']:
            print(f"   - {col}")
            
        print(f"\n❌ NOT normalized columns ({len(results['not_normalized'])}): ")
        for col in results['not_normalized']:
            print(f"   - {col}")
            
        print(f"\n⚠️  Ambiguous columns ({len(results['ambiguous'])}): ")
        for col in results['ambiguous']:
            print(f"   - {col}")
            
        return results
    
    def check_if_normalized(self, min_val, max_val, mean_val):
        """
        Determine if values suggest log normalization
        """
        # Clear indicators of normalization
        if max_val < 20 and min_val > 0.2:
            return 'yes'
        
        # Clear indicators of NO normalization  
        if max_val > 1000:
            return 'no'
            
        # Check if min is close to log10(2) = 0.301
        if abs(min_val - 0.301) < 0.01 and max_val < 50:
            return 'yes'
            
        # Ambiguous
        return 'maybe'
    
    def visualize_column(self, col_name):
        """
        Visualize a specific column's distribution
        """
        if self.df is None:
            self.df = pd.read_csv(self.data_path, nrows=100000)
            
        col_data = self.df[col_name].values
        col_data = col_data[col_data > 0]  # Remove zeros
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
        
        # Raw distribution
        ax1.hist(col_data, bins=50, edgecolor='black')
        ax1.set_title(f'{col_name} - Raw Values')
        ax1.set_xlabel('Value')
        ax1.set_ylabel('Count')
        ax1.set_yscale('log')
        
        # Log-transformed distribution
        log_data = np.log10(col_data + 1)
        ax2.hist(log_data, bins=50, edgecolor='black')
        ax2.set_title(f'{col_name} - After log10(x+1)')
        ax2.set_xlabel('log10(value + 1)')
        ax2.set_ylabel('Count')
        
        plt.tight_layout()
        plt.savefig(f'{col_name}_distribution.png')
        plt.show()
        
        print(f"\nColumn: {col_name}")
        print(f"Raw - Min: {col_data.min():.6f}, Max: {col_data.max():.6f}")
        print(f"Log - Min: {log_data.min():.6f}, Max: {log_data.max():.6f}")

def main():
    # Analyze your dataset
    analyzer = DetailedColumnAnalyzer('/work/hdd/bdau/mbanisharifdehkordi/GNN_4_IO_5/data/total/sample_train_total_normalized.csv')
    
    # Check all columns
    results = analyzer.analyze_all_columns()
    
    # Visualize suspicious columns
    # Example: Check columns that might be problematic
    suspicious_cols = ['POSIX_BYTES_READ', 'POSIX_BYTES_WRITTEN', 'nprocs', 
                      'LUSTRE_STRIPE_SIZE', 'LUSTRE_STRIPE_WIDTH']
    
    print("\n" + "="*80)
    print("VISUALIZING SUSPICIOUS COLUMNS")
    print("="*80)
    
    for col in suspicious_cols:
        if col in analyzer.df.columns:
            analyzer.visualize_column(col)

if __name__ == "__main__":
    main()
