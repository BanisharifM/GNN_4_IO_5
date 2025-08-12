import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json

class SimilarityGraphAnalyzer:
    """
    Analyze computed similarity graphs for quality and structure
    """
    
    def __init__(self, pt_path, npz_path, metadata_path=None):
        self.pt_path = pt_path
        self.npz_path = npz_path
        self.metadata_path = metadata_path
        
        # Load data
        self.pt_data = None
        self.npz_data = None
        self.metadata = None
        
    def load_data(self):
        """Load all similarity graph files"""
        print("="*80)
        print("LOADING SIMILARITY GRAPH DATA")
        print("="*80)
        
        # Load PyTorch file
        print(f"\nLoading PyTorch file: {self.pt_path}")
        self.pt_data = torch.load(self.pt_path)
        print(f"✅ PyTorch data loaded")
        
        # Load NumPy file
        print(f"\nLoading NumPy file: {self.npz_path}")
        self.npz_data = np.load(self.npz_path)
        print(f"✅ NumPy data loaded")
        
        # Load metadata if available
        if self.metadata_path and Path(self.metadata_path).exists():
            print(f"\nLoading metadata: {self.metadata_path}")
            with open(self.metadata_path, 'r') as f:
                self.metadata = json.load(f)
            print(f"✅ Metadata loaded")
    
    def analyze_pytorch_structure(self):
        """Analyze PyTorch graph structure"""
        print("\n" + "="*80)
        print("PYTORCH FILE STRUCTURE")
        print("="*80)
        
        print("\nKeys in PyTorch file:")
        for key in self.pt_data.keys():
            if isinstance(self.pt_data[key], torch.Tensor):
                print(f"  - {key}: torch.Tensor, shape={self.pt_data[key].shape}, dtype={self.pt_data[key].dtype}")
            else:
                print(f"  - {key}: {type(self.pt_data[key])}, value={self.pt_data[key]}")
        
        # Analyze edge_index
        if 'edge_index' in self.pt_data:
            edge_index = self.pt_data['edge_index']
            print(f"\nEdge Index Analysis:")
            print(f"  Shape: {edge_index.shape}")
            print(f"  Min node ID: {edge_index.min().item()}")
            print(f"  Max node ID: {edge_index.max().item()}")
            print(f"  Number of unique source nodes: {len(torch.unique(edge_index[0]))}")
            print(f"  Number of unique target nodes: {len(torch.unique(edge_index[1]))}")
            
            # Check if graph is directed or undirected
            edges_set = set(zip(edge_index[0].numpy(), edge_index[1].numpy()))
            reverse_edges = set(zip(edge_index[1].numpy(), edge_index[0].numpy()))
            bidirectional = len(edges_set.intersection(reverse_edges))
            print(f"  Bidirectional edges: {bidirectional}/{len(edges_set)} ({bidirectional/len(edges_set)*100:.1f}%)")
        
        # Analyze edge weights
        if 'edge_weight' in self.pt_data:
            edge_weight = self.pt_data['edge_weight']
            print(f"\nEdge Weight Analysis:")
            print(f"  Shape: {edge_weight.shape}")
            print(f"  Min weight: {edge_weight.min().item():.6f}")
            print(f"  Max weight: {edge_weight.max().item():.6f}")
            print(f"  Mean weight: {edge_weight.mean().item():.6f}")
            print(f"  Std weight: {edge_weight.std().item():.6f}")
            print(f"  Median weight: {edge_weight.median().item():.6f}")
    
    def analyze_numpy_structure(self):
        """Analyze NumPy graph structure"""
        print("\n" + "="*80)
        print("NUMPY FILE STRUCTURE")
        print("="*80)
        
        print("\nArrays in NumPy file:")
        for key in self.npz_data.files:
            arr = self.npz_data[key]
            print(f"  - {key}: shape={arr.shape}, dtype={arr.dtype}")
            
            if key == 'edges':
                print(f"    First 5 edges: \n{arr[:5]}")
            elif key == 'weights':
                print(f"    Weight stats: min={arr.min():.4f}, max={arr.max():.4f}, mean={arr.mean():.4f}")
            elif key == 'degree_counts':
                print(f"    Degree stats: min={arr.min()}, max={arr.max()}, mean={arr.mean():.1f}")
    
    def analyze_degree_distribution(self):
        """Analyze node degree distribution"""
        print("\n" + "="*80)
        print("DEGREE DISTRIBUTION ANALYSIS")
        print("="*80)
        
        if 'degree_counts' in self.npz_data.files:
            degrees = self.npz_data['degree_counts']
            
            print(f"\nDegree Statistics:")
            print(f"  Min degree: {degrees.min()}")
            print(f"  Max degree: {degrees.max()}")
            print(f"  Mean degree: {degrees.mean():.2f}")
            print(f"  Median degree: {np.median(degrees):.2f}")
            print(f"  Std degree: {degrees.std():.2f}")
            
            # Check for isolated nodes
            isolated = (degrees == 0).sum()
            print(f"  Isolated nodes: {isolated} ({isolated/len(degrees)*100:.2f}%)")
            
            # Degree distribution
            unique_degrees, counts = np.unique(degrees, return_counts=True)
            print(f"\nDegree Distribution:")
            for deg, cnt in zip(unique_degrees[:10], counts[:10]):  # Show first 10
                print(f"  Degree {deg}: {cnt} nodes ({cnt/len(degrees)*100:.2f}%)")
            
            # Plot degree distribution
            plt.figure(figsize=(10, 4))
            
            plt.subplot(1, 2, 1)
            plt.hist(degrees, bins=50, edgecolor='black')
            plt.xlabel('Degree')
            plt.ylabel('Count')
            plt.title('Degree Distribution')
            plt.yscale('log')
            
            plt.subplot(1, 2, 2)
            plt.boxplot(degrees)
            plt.ylabel('Degree')
            plt.title('Degree Boxplot')
            
            plt.tight_layout()
            plt.savefig('degree_distribution.png')
            plt.show()
    
    def analyze_weight_distribution(self):
        """Analyze edge weight (similarity) distribution"""
        print("\n" + "="*80)
        print("SIMILARITY WEIGHT DISTRIBUTION")
        print("="*80)
        
        if 'weights' in self.npz_data.files:
            weights = self.npz_data['weights']
            
            print(f"\nWeight Statistics:")
            print(f"  Total edges: {len(weights):,}")
            print(f"  Min similarity: {weights.min():.6f}")
            print(f"  Max similarity: {weights.max():.6f}")
            print(f"  Mean similarity: {weights.mean():.6f}")
            print(f"  Median similarity: {np.median(weights):.6f}")
            print(f"  Std similarity: {weights.std():.6f}")
            
            # Percentiles
            percentiles = [1, 5, 10, 25, 50, 75, 90, 95, 99]
            print(f"\nPercentiles:")
            for p in percentiles:
                val = np.percentile(weights, p)
                print(f"  {p:3d}%: {val:.6f}")
            
            # Plot weight distribution
            plt.figure(figsize=(12, 4))
            
            plt.subplot(1, 3, 1)
            plt.hist(weights, bins=100, edgecolor='black', alpha=0.7)
            plt.xlabel('Similarity')
            plt.ylabel('Count')
            plt.title('Similarity Distribution')
            plt.axvline(weights.mean(), color='red', linestyle='--', label=f'Mean: {weights.mean():.3f}')
            plt.legend()
            
            plt.subplot(1, 3, 2)
            plt.hist(weights, bins=100, cumulative=True, density=True, edgecolor='black', alpha=0.7)
            plt.xlabel('Similarity')
            plt.ylabel('Cumulative Probability')
            plt.title('Cumulative Distribution')
            plt.grid(True, alpha=0.3)
            
            plt.subplot(1, 3, 3)
            plt.boxplot(weights)
            plt.ylabel('Similarity')
            plt.title('Similarity Boxplot')
            
            plt.tight_layout()
            plt.savefig('weight_distribution.png')
            plt.show()
    
    def check_graph_connectivity(self):
        """Check if graph has expected connectivity patterns"""
        print("\n" + "="*80)
        print("CONNECTIVITY ANALYSIS")
        print("="*80)
        
        if 'edge_index' in self.pt_data:
            edge_index = self.pt_data['edge_index'].numpy()
            num_nodes = self.pt_data['num_nodes']
            
            # Create adjacency list
            from collections import defaultdict
            adj_list = defaultdict(set)
            for src, dst in edge_index.T:
                adj_list[src].add(dst)
            
            # Check connectivity patterns
            print(f"\nConnectivity Patterns:")
            print(f"  Total nodes: {num_nodes:,}")
            print(f"  Nodes with edges: {len(adj_list):,}")
            print(f"  Total directed edges: {edge_index.shape[1]:,}")
            
            # Sample some nodes to check their neighborhoods
            sample_nodes = np.random.choice(list(adj_list.keys()), min(5, len(adj_list)), replace=False)
            print(f"\nSample Node Neighborhoods:")
            for node in sample_nodes:
                neighbors = adj_list[node]
                print(f"  Node {node}: {len(neighbors)} neighbors")
                
                if 'edge_weight' in self.pt_data:
                    # Get weights for this node's edges
                    node_edges = edge_index[0] == node
                    node_weights = self.pt_data['edge_weight'][node_edges]
                    if len(node_weights) > 0:
                        print(f"    Weight range: [{node_weights.min():.4f}, {node_weights.max():.4f}]")
    
    def verify_data_consistency(self):
        """Verify consistency between PyTorch and NumPy files"""
        print("\n" + "="*80)
        print("DATA CONSISTENCY CHECK")
        print("="*80)
        
        # Check edge counts
        pt_edges = self.pt_data['edge_index'].shape[1] if 'edge_index' in self.pt_data else 0
        np_edges = len(self.npz_data['edges']) if 'edges' in self.npz_data.files else 0
        
        print(f"\nEdge Count Consistency:")
        print(f"  PyTorch edges: {pt_edges:,}")
        print(f"  NumPy edges: {np_edges:,}")
        print(f"  Match: {'✅ Yes' if pt_edges == np_edges else '❌ No'}")
        
        # Check weight consistency
        if 'edge_weight' in self.pt_data and 'weights' in self.npz_data.files:
            pt_weights = self.pt_data['edge_weight'].numpy()
            np_weights = self.npz_data['weights']
            
            print(f"\nWeight Consistency:")
            print(f"  PyTorch weight range: [{pt_weights.min():.6f}, {pt_weights.max():.6f}]")
            print(f"  NumPy weight range: [{np_weights.min():.6f}, {np_weights.max():.6f}]")
            
            # Check if weights are identical
            if len(pt_weights) == len(np_weights):
                weight_diff = np.abs(pt_weights - np_weights).max()
                print(f"  Max weight difference: {weight_diff:.10f}")
                print(f"  Weights match: {'✅ Yes' if weight_diff < 1e-6 else '❌ No'}")
    
    def summary_report(self):
        """Generate summary report"""
        print("\n" + "="*80)
        print("SUMMARY REPORT")
        print("="*80)
        
        if self.metadata:
            print("\nFrom Metadata:")
            print(f"  Threshold: {self.metadata.get('threshold', 'N/A')}")
            print(f"  Nodes: {self.metadata.get('num_nodes', 'N/A'):,}")
            print(f"  Edges: {self.metadata.get('num_edges', 'N/A'):,}")
            
            if 'statistics' in self.metadata:
                stats = self.metadata['statistics']
                print(f"  Avg Degree: {stats.get('avg_degree', 'N/A')}")
                print(f"  Edge Density: {stats.get('edge_density', 'N/A'):.6f}")
        
        print("\n✅ Graph Structure Ready for GNN:")
        print("  - Edge index in COO format (shape: [2, num_edges])")
        print("  - Edge weights available (similarities)")
        print("  - Degree distribution available")
        print("  - Node features needed from original dataset")

def main():
    # Paths to your files
    base_dir = "/work/hdd/bdau/mbanisharifdehkordi/GNN_4_IO_5/data/1M/similarity_output_0.75"
    pt_path = f"{base_dir}/similarity_graph_20250812_043913.pt"
    npz_path = f"{base_dir}/similarity_graph_20250812_043913.npz"
    metadata_path = f"{base_dir}/similarity_metadata_20250812_043913.json"
    
    # Create analyzer
    analyzer = SimilarityGraphAnalyzer(pt_path, npz_path, metadata_path)
    
    # Run analysis
    analyzer.load_data()
    analyzer.analyze_pytorch_structure()
    analyzer.analyze_numpy_structure()
    analyzer.analyze_degree_distribution()
    analyzer.analyze_weight_distribution()
    analyzer.check_graph_connectivity()
    analyzer.verify_data_consistency()
    analyzer.summary_report()

if __name__ == "__main__":
    main()