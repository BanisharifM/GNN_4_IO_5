import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import spearmanr, kendalltau
from sklearn.feature_selection import mutual_info_regression
from sklearn.preprocessing import StandardScaler
import warnings
import os
import sys
from datetime import datetime

warnings.filterwarnings('ignore')

# Set matplotlib to use non-interactive backend for HPC
import matplotlib
matplotlib.use('Agg')

class CorrelationAnalyzer:
    """
    Comprehensive correlation analyzer for I/O bottleneck detection
    Supports multiple correlation methods and visualization
    """
    
    def __init__(self, data_path):
        """
        Initialize with data path
        Args:
            data_path: Path to the CSV file
        """
        print(f"[{datetime.now().strftime('%H:%M:%S')}] Loading data from: {data_path}")
        self.df = pd.read_csv(data_path)
        self.feature_cols = [col for col in self.df.columns if col != 'tag']
        self.n_features = len(self.feature_cols)
        print(f"[{datetime.now().strftime('%H:%M:%S')}] Loaded {len(self.df)} samples with {self.n_features} features")
        
        # Store correlation matrices
        self.correlation_matrices = {}
        
    def calculate_pearson(self):
        """Calculate Pearson correlation (linear relationships)"""
        print(f"[{datetime.now().strftime('%H:%M:%S')}] Calculating Pearson correlation...")
        corr_matrix = self.df[self.feature_cols].corr(method='pearson')
        self.correlation_matrices['pearson'] = corr_matrix
        return corr_matrix
    
    def calculate_spearman(self):
        """Calculate Spearman correlation (monotonic relationships)"""
        print(f"[{datetime.now().strftime('%H:%M:%S')}] Calculating Spearman correlation...")
        corr_matrix = self.df[self.feature_cols].corr(method='spearman')
        self.correlation_matrices['spearman'] = corr_matrix
        return corr_matrix
    
    def calculate_kendall(self):
        """Calculate Kendall Tau correlation (ordinal relationships)"""
        print(f"[{datetime.now().strftime('%H:%M:%S')}] Calculating Kendall Tau correlation...")
        # Kendall can be slow for large datasets, consider sampling
        if len(self.df) > 10000:
            print(f"  Dataset large ({len(self.df)} samples), using sample of 10000 for Kendall")
            sample_df = self.df.sample(n=10000, random_state=42)
        else:
            sample_df = self.df
        
        corr_matrix = sample_df[self.feature_cols].corr(method='kendall')
        self.correlation_matrices['kendall'] = corr_matrix
        return corr_matrix
    
    def calculate_mutual_information(self, n_neighbors=3):
        """
        Calculate Mutual Information (any type of relationship)
        Args:
            n_neighbors: Number of neighbors for MI estimation
        """
        print(f"[{datetime.now().strftime('%H:%M:%S')}] Calculating Mutual Information...")
        
        # Initialize MI matrix
        mi_matrix = pd.DataFrame(
            np.zeros((self.n_features, self.n_features)),
            index=self.feature_cols,
            columns=self.feature_cols
        )
        
        # Standardize features for MI calculation
        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(self.df[self.feature_cols])
        scaled_df = pd.DataFrame(scaled_data, columns=self.feature_cols)
        
        # Calculate MI for each pair
        for i, col1 in enumerate(self.feature_cols):
            if i % 10 == 0:
                print(f"  Processing feature {i+1}/{self.n_features}")
            
            for j, col2 in enumerate(self.feature_cols):
                if i == j:
                    mi_matrix.loc[col1, col2] = 1.0
                elif j > i:  # Calculate only upper triangle
                    mi_score = mutual_info_regression(
                        scaled_df[[col1]].values,
                        scaled_df[col2].values.ravel(),
                        n_neighbors=n_neighbors,
                        random_state=42
                    )[0]
                    # Normalize to [0, 1] range
                    mi_normalized = mi_score / (mi_score + 1)
                    mi_matrix.loc[col1, col2] = mi_normalized
                    mi_matrix.loc[col2, col1] = mi_normalized
        
        self.correlation_matrices['mutual_information'] = mi_matrix
        return mi_matrix
    
    def calculate_all(self, methods=None):
        """
        Calculate all specified correlation methods
        """
        if methods is None:
            methods = ['pearson', 'spearman', 'kendall', 'mutual_information']
        
        results = {}
        for method in methods:
            if method == 'pearson':
                results['pearson'] = self.calculate_pearson()
            elif method == 'spearman':
                results['spearman'] = self.calculate_spearman()
            elif method == 'kendall':
                results['kendall'] = self.calculate_kendall()
            elif method == 'mutual_information':
                results['mutual_information'] = self.calculate_mutual_information()
        
        return results
    
    def plot_heatmap(self, method='spearman', figsize=(20, 16), save_path=None):
        """
        Plot correlation heatmap
        """
        if method not in self.correlation_matrices:
            print(f"Method '{method}' not calculated yet. Running calculation...")
            if method == 'pearson':
                self.calculate_pearson()
            elif method == 'spearman':
                self.calculate_spearman()
            elif method == 'kendall':
                self.calculate_kendall()
            elif method == 'mutual_information':
                self.calculate_mutual_information()
        
        corr_matrix = self.correlation_matrices[method]
        
        # Create figure
        plt.figure(figsize=figsize)
        
        # Create heatmap
        sns.heatmap(
            corr_matrix,
            annot=False,  # Don't annotate all cells (too many)
            cmap='RdBu_r',
            center=0,
            vmin=-1 if method != 'mutual_information' else 0,
            vmax=1,
            square=True,
            linewidths=0.1,
            cbar_kws={"shrink": 0.8}
        )
        
        plt.title(f'{method.upper()} Correlation Matrix - I/O Features', fontsize=16, pad=20)
        plt.xlabel('Features', fontsize=12)
        plt.ylabel('Features', fontsize=12)
        plt.xticks(rotation=90, ha='right', fontsize=8)
        plt.yticks(rotation=0, fontsize=8)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"[{datetime.now().strftime('%H:%M:%S')}] Heatmap saved to: {save_path}")
        
        return corr_matrix
    
    def plot_comparison(self, methods=None, figsize=(24, 6), save_path=None):
        """
        Plot comparison of different correlation methods
        """
        if methods is None:
            methods = ['pearson', 'spearman', 'kendall', 'mutual_information']
        
        n_methods = len(methods)
        fig, axes = plt.subplots(1, n_methods, figsize=figsize)
        
        if n_methods == 1:
            axes = [axes]
        
        for idx, method in enumerate(methods):
            if method not in self.correlation_matrices:
                print(f"Calculating {method}...")
                self.calculate_all(methods=[method])
            
            corr_matrix = self.correlation_matrices[method]
            
            im = axes[idx].imshow(
                corr_matrix,
                cmap='RdBu_r',
                aspect='auto',
                vmin=-1 if method != 'mutual_information' else 0,
                vmax=1
            )
            axes[idx].set_title(f'{method.upper()}', fontsize=12)
            axes[idx].set_xticks([])
            axes[idx].set_yticks([])
            
            # Add colorbar for each subplot
            plt.colorbar(im, ax=axes[idx], fraction=0.046, pad=0.04)
        
        plt.suptitle('Correlation Methods Comparison', fontsize=16, y=1.02)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"[{datetime.now().strftime('%H:%M:%S')}] Comparison saved to: {save_path}")
    
    def analyze_feature_independence(self, feature_name, method='spearman', threshold=0.3):
        """
        Analyze independence of a specific feature
        """
        if method not in self.correlation_matrices:
            self.calculate_all(methods=[method])
        
        corr_matrix = self.correlation_matrices[method]
        
        # Get correlations for this feature
        feature_corrs = corr_matrix[feature_name].drop(feature_name)
        
        # Classify correlations
        independent_features = feature_corrs[feature_corrs.abs() < threshold]
        moderate_corr = feature_corrs[(feature_corrs.abs() >= threshold) & (feature_corrs.abs() < 0.7)]
        high_corr = feature_corrs[feature_corrs.abs() >= 0.7]
        
        print(f"\n{'='*60}")
        print(f"Independence Analysis for: {feature_name}")
        print(f"Method: {method.upper()}")
        print(f"{'='*60}")
        
        print(f"\nIndependent Features (|corr| < {threshold}):")
        for feat, corr in independent_features.items():
            print(f"  {feat:30s}: {corr:+.3f}")
        
        print(f"\nModerately Correlated ({threshold} <= |corr| < 0.7):")
        for feat, corr in moderate_corr.items():
            print(f"  {feat:30s}: {corr:+.3f}")
        
        print(f"\nHighly Correlated (|corr| >= 0.7):")
        for feat, corr in high_corr.items():
            print(f"  {feat:30s}: {corr:+.3f}")
        
        # Calculate independence score
        independence_score = 1 - feature_corrs.abs().mean()
        print(f"\nIndependence Score: {independence_score:.3f}")
        print(f"  (1.0 = completely independent, 0.0 = highly dependent)")
        
        return {
            'independence_score': independence_score,
            'independent_features': independent_features.to_dict(),
            'moderate_correlations': moderate_corr.to_dict(),
            'high_correlations': high_corr.to_dict()
        }
    
    def save_correlation_matrices(self, output_dir):
        """Save all correlation matrices to CSV files"""
        os.makedirs(output_dir, exist_ok=True)
        
        for method, matrix in self.correlation_matrices.items():
            output_path = os.path.join(output_dir, f'correlation_{method}.csv')
            matrix.to_csv(output_path)
            print(f"[{datetime.now().strftime('%H:%M:%S')}] Saved {method} correlation matrix to: {output_path}")

def main():
    # Set paths
    data_path = "/work/hdd/bdau/mbanisharifdehkordi/GNN_4_IO_5/data/1M/aiio_sample_1000000_normalized.csv"
    output_dir = "/work/hdd/bdau/mbanisharifdehkordi/GNN_4_IO_5/results/correlation_analysis"
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    print("="*80)
    print("CORRELATION ANALYSIS FOR I/O BOTTLENECK DETECTION")
    print("="*80)
    
    # Initialize analyzer
    analyzer = CorrelationAnalyzer(data_path)
    
    # Calculate all correlation methods
    print(f"\n[{datetime.now().strftime('%H:%M:%S')}] Starting correlation calculations...")
    all_correlations = analyzer.calculate_all(methods=['pearson', 'spearman', 'kendall', 'mutual_information'])
    
    # Save correlation matrices
    print(f"\n[{datetime.now().strftime('%H:%M:%S')}] Saving correlation matrices...")
    analyzer.save_correlation_matrices(output_dir)
    
    # Generate individual heatmaps
    print(f"\n[{datetime.now().strftime('%H:%M:%S')}] Generating heatmaps...")
    for method in ['pearson', 'spearman', 'kendall', 'mutual_information']:
        save_path = os.path.join(output_dir, f'heatmap_{method}.png')
        analyzer.plot_heatmap(method=method, save_path=save_path)
    
    # Generate comparison plot
    print(f"\n[{datetime.now().strftime('%H:%M:%S')}] Generating comparison plot...")
    comparison_path = os.path.join(output_dir, 'correlation_comparison.png')
    analyzer.plot_comparison(save_path=comparison_path)
    
    # Analyze root cause candidates
    print(f"\n[{datetime.now().strftime('%H:%M:%S')}] Analyzing root cause candidates...")
    root_cause_candidates = [
        'LUSTRE_STRIPE_SIZE',
        'LUSTRE_STRIPE_WIDTH', 
        'POSIX_FILE_NOT_ALIGNED',
        'POSIX_MEM_NOT_ALIGNED',
        'POSIX_FILE_ALIGNMENT',
        'POSIX_MEM_ALIGNMENT'
    ]
    
    independence_results = {}
    for feature in root_cause_candidates:
        if feature in analyzer.feature_cols:
            results = analyzer.analyze_feature_independence(feature, method='spearman')
            independence_results[feature] = results['independence_score']
    
    # Print summary
    print("\n" + "="*80)
    print("INDEPENDENCE SCORES SUMMARY (Spearman)")
    print("="*80)
    sorted_features = sorted(independence_results.items(), key=lambda x: x[1], reverse=True)
    for feat, score in sorted_features:
        print(f"{feat:30s}: {score:.3f} {'(HIGH independence - likely root cause)' if score > 0.7 else ''}")
    
    print(f"\n[{datetime.now().strftime('%H:%M:%S')}] Analysis complete!")
    print(f"Results saved to: {output_dir}")

if __name__ == "__main__":
    main()