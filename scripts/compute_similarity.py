import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm
import torch
import logging
import os
import json
from datetime import datetime
import gc

class IOSimilarityComputer:
    """
    Compute cosine similarity for I/O performance data
    Optimized for 100K samples with float64 precision
    """
    
    def __init__(self, data_path, output_dir='./similarity_output'):
        self.data_path = data_path
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        # Setup logging
        log_file = os.path.join(output_dir, 'similarity_computation.log')
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
        
    def load_and_prepare_data(self):
        """
        Load data and prepare for similarity computation
        IMPORTANT: Data is already log-normalized, no additional scaling needed
        """
        self.logger.info(f"Loading data from {self.data_path}")
        
        # Load with float64 precision as discussed
        df = pd.read_csv(self.data_path, dtype=np.float64)
        
        # Separate features and target
        self.target = df['tag'].values
        self.features = df.drop('tag', axis=1).values  # (100000, 45)
        self.feature_names = df.drop('tag', axis=1).columns.tolist()
        
        self.logger.info(f"Loaded {len(df):,} samples with {self.features.shape[1]} features")
        self.logger.info(f"Data type: {self.features.dtype} (preserving precision)")
        self.logger.info(f"Memory usage: {self.features.nbytes / 1e9:.2f} GB")
        
        # Verify log-normalized data (should be in reasonable range)
        self.logger.info(f"Feature value ranges (log-normalized):")
        self.logger.info(f"  Min: {self.features.min():.6f}")
        self.logger.info(f"  Max: {self.features.max():.6f}")
        self.logger.info(f"  Mean: {self.features.mean():.6f}")
        
        return self.features
    
    def compute_similarity_exact(self, threshold=0.75, batch_size=1000):
        """
        Compute exact cosine similarity with batch processing
        As discussed: No scaling after log normalization, preserve float64
        """
        n_samples = self.features.shape[0]
        edge_list = []
        edge_weights = []
        
        self.logger.info(f"Computing cosine similarity with threshold={threshold}")
        self.logger.info(f"Batch size: {batch_size}")
        
        # Track statistics
        degree_counts = np.zeros(n_samples, dtype=np.int32)
        
        # Process in batches to manage memory
        for i in tqdm(range(0, n_samples, batch_size), desc="Computing similarities"):
            end_i = min(i + batch_size, n_samples)
            batch_i = self.features[i:end_i]
            
            # Compute similarity with ALL samples (not just forward)
            # Using float64 as discussed
            similarities = cosine_similarity(
                batch_i.astype(np.float64), 
                self.features.astype(np.float64)
            ).astype(np.float64)
            
            # Process each sample in batch
            for local_idx in range(len(batch_i)):
                global_idx = i + local_idx
                sim_row = similarities[local_idx]
                
                # Set self-similarity to 0
                sim_row[global_idx] = 0
                
                # Find neighbors above threshold
                neighbor_mask = sim_row > threshold
                neighbors = np.where(neighbor_mask)[0]
                neighbor_sims = sim_row[neighbor_mask]
                
                # Add edges
                for neighbor, sim in zip(neighbors, neighbor_sims):
                    edge_list.append([global_idx, neighbor])
                    edge_weights.append(sim)
                
                # Track degree
                degree_counts[global_idx] = len(neighbors)
            
            # Periodic memory cleanup
            if (i + batch_size) % 10000 == 0:
                gc.collect()
        
        # Convert to arrays
        edges = np.array(edge_list, dtype=np.int64)
        weights = np.array(edge_weights, dtype=np.float64)
        
        # Log statistics
        self.logger.info(f"\n=== Graph Statistics ===")
        self.logger.info(f"Total edges: {len(edges):,}")
        self.logger.info(f"Average degree: {degree_counts.mean():.1f}")
        self.logger.info(f"Median degree: {np.median(degree_counts):.1f}")
        self.logger.info(f"Max degree: {degree_counts.max()}")
        self.logger.info(f"Min degree: {degree_counts.min()}")
        self.logger.info(f"Isolated nodes: {(degree_counts == 0).sum()}")
        self.logger.info(f"Weight range: [{weights.min():.6f}, {weights.max():.6f}]")
        
        return edges, weights, degree_counts
    
    def save_results(self, edges, weights, degree_counts, threshold):
        """
        Save results in multiple formats for flexibility
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # 1. Save as PyTorch tensors (for GNN training)
        pt_file = os.path.join(self.output_dir, f'similarity_graph_{timestamp}.pt')
        torch.save({
            'edge_index': torch.tensor(edges.T, dtype=torch.long),  # (2, n_edges) format
            'edge_weight': torch.tensor(weights, dtype=torch.float64),
            'num_nodes': len(self.features),
            'threshold': threshold,
            'degree_counts': torch.tensor(degree_counts, dtype=torch.long),
            'timestamp': timestamp
        }, pt_file)
        self.logger.info(f"Saved PyTorch format to: {pt_file}")
        
        # 2. Save as compressed NumPy (for analysis)
        npz_file = os.path.join(self.output_dir, f'similarity_graph_{timestamp}.npz')
        np.savez_compressed(
            npz_file,
            edges=edges,
            weights=weights,
            degree_counts=degree_counts,
            features=self.features,  # Include features for reference
            target=self.target,
            threshold=threshold
        )
        self.logger.info(f"Saved NumPy format to: {npz_file}")
        
        # 3. Save metadata
        metadata = {
            'timestamp': timestamp,
            'data_file': self.data_path,
            'num_nodes': len(self.features),
            'num_edges': len(edges),
            'threshold': threshold,
            'statistics': {
                'avg_degree': float(degree_counts.mean()),
                'median_degree': float(np.median(degree_counts)),
                'max_degree': int(degree_counts.max()),
                'min_degree': int(degree_counts.min()),
                'isolated_nodes': int((degree_counts == 0).sum()),
                'edge_density': len(edges) / (len(self.features) * (len(self.features) - 1))
            },
            'weight_stats': {
                'min': float(weights.min()),
                'max': float(weights.max()),
                'mean': float(weights.mean()),
                'std': float(weights.std())
            },
            'feature_info': {
                'num_features': self.features.shape[1],
                'feature_names': self.feature_names
            }
        }
        
        metadata_file = os.path.join(self.output_dir, f'similarity_metadata_{timestamp}.json')
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)
        self.logger.info(f"Saved metadata to: {metadata_file}")
        
        return pt_file, npz_file, metadata_file
    
    def analyze_graph_quality(self, degree_counts, threshold):
        """
        Analyze if the graph is suitable for GNN training
        """
        self.logger.info("\n=== Graph Quality Analysis ===")
        
        isolated = (degree_counts == 0).sum()
        low_degree = (degree_counts < 5).sum()
        high_degree = (degree_counts > 500).sum()
        
        # Quality checks
        quality_issues = []
        
        if isolated > len(degree_counts) * 0.01:  # More than 1% isolated
            quality_issues.append(f"⚠️  {isolated} isolated nodes - consider lowering threshold")
        
        if low_degree > len(degree_counts) * 0.1:  # More than 10% with < 5 neighbors
            quality_issues.append(f"⚠️  {low_degree} nodes with < 5 neighbors - might affect GNN learning")
        
        if high_degree > 100:  # More than 100 hub nodes
            quality_issues.append(f"⚠️  {high_degree} potential hub nodes (>500 neighbors) - might dominate")
        
        if degree_counts.mean() < 10:
            quality_issues.append(f"⚠️  Low average degree ({degree_counts.mean():.1f}) - graph too sparse")
        
        if degree_counts.mean() > 1000:
            quality_issues.append(f"⚠️  High average degree ({degree_counts.mean():.1f}) - graph too dense")
        
        if quality_issues:
            self.logger.warning("Quality concerns:")
            for issue in quality_issues:
                self.logger.warning(issue)
            self.logger.info(f"\nConsider adjusting threshold (current: {threshold})")
        else:
            self.logger.info("✅ Graph quality looks good for GNN training!")
        
        # Recommendations
        if degree_counts.mean() < 50:
            suggested_threshold = threshold - 0.05
            self.logger.info(f"Suggestion: Try threshold={suggested_threshold:.2f} for more connections")
        elif degree_counts.mean() > 500:
            suggested_threshold = threshold + 0.05
            self.logger.info(f"Suggestion: Try threshold={suggested_threshold:.2f} for fewer connections")

def main():
    """
    Main function to compute similarity for I/O dataset
    """
    # Configuration
    DATA_PATH = '/work/hdd/bdau/mbanisharifdehkordi/GNN_4_IO_5/data/100K/aiio_sample_100000.csv'  # Your sampled data
    OUTPUT_DIR = '/work/hdd/bdau/mbanisharifdehkordi/GNN_4_IO_5/data/100K/similarity_output'
    THRESHOLD = 0.75  # As discussed
    BATCH_SIZE = 1000  # Adjust based on memory
    
    print("="*70)
    print("I/O Performance Data - Cosine Similarity Computation")
    print("="*70)
    print(f"Data: {DATA_PATH}")
    print(f"Threshold: {THRESHOLD}")
    print(f"Output: {OUTPUT_DIR}")
    print("="*70)
    
    # Initialize computer
    computer = IOSimilarityComputer(DATA_PATH, OUTPUT_DIR)
    
    # Load and prepare data
    print("\nStep 1: Loading and preparing data...")
    features = computer.load_and_prepare_data()
    
    # Compute similarity
    print(f"\nStep 2: Computing cosine similarity (threshold={THRESHOLD})...")
    edges, weights, degree_counts = computer.compute_similarity_exact(
        threshold=THRESHOLD,
        batch_size=BATCH_SIZE
    )
    
    # Save results
    print("\nStep 3: Saving results...")
    pt_file, npz_file, metadata_file = computer.save_results(
        edges, weights, degree_counts, THRESHOLD
    )
    
    # Analyze quality
    print("\nStep 4: Analyzing graph quality...")
    computer.analyze_graph_quality(degree_counts, THRESHOLD)
    
    print("\n" + "="*70)
    print("COMPUTATION COMPLETE!")
    print(f"✅ PyTorch file: {pt_file}")
    print(f"✅ NumPy file: {npz_file}")
    print(f"✅ Metadata: {metadata_file}")
    print("="*70)
    
    print("\nNext steps:")
    print("1. Check the graph statistics in the log")
    print("2. If too sparse/dense, adjust threshold and rerun")
    print("3. Use the .pt file for GNN training")

if __name__ == "__main__":
    # You can test with different thresholds
    # THRESHOLD = 0.70  # More connections
    # THRESHOLD = 0.80  # Fewer connections
    
    main()