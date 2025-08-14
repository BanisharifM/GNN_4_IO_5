#!/usr/bin/env python
"""
Standalone script to test interpretability methods on saved GAT model
Processes nodes one at a time with subgraph extraction to avoid OOM
"""

import os
import sys
import torch
import numpy as np
import pandas as pd
from pathlib import Path
import logging
import json
from typing import Dict, List, Tuple
import warnings
warnings.filterwarnings('ignore')

# Add src to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data.dataset import IOPerformanceGraphDataset
from src.models.gat import create_gat_model
from src.interpretability.attention_analyzer import AttentionAnalyzer
from src.interpretability.gnn_explainer import IOGNNExplainer
from src.interpretability.gradient_methods import GradientAnalyzer, BottleneckIdentifier
from torch_geometric.utils import k_hop_subgraph

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class InterpretabilityTester:
    """Test interpretability methods on trained model"""
    
    def __init__(
        self,
        checkpoint_path: str,
        data_paths: Dict[str, str],
        device: str = 'cpu',
        max_subgraph_size: int = 500
    ):
        """
        Args:
            checkpoint_path: Path to saved model checkpoint
            data_paths: Paths to data files (similarity_pt, features_csv, etc.)
            device: Device to use (recommend 'cpu' to avoid OOM)
            max_subgraph_size: Maximum nodes in subgraph
        """
        self.device = torch.device(device)
        self.max_subgraph_size = max_subgraph_size
        
        # Load model
        logger.info("Loading model...")
        self.model = self._load_model(checkpoint_path)
        
        # Load data
        logger.info("Loading data...")
        self.data = self._load_data(data_paths)
        
        # Get feature names
        self.feature_names = self._get_feature_names()
        
        # Initialize analyzers
        logger.info("Initializing analyzers...")
        self.attention_analyzer = AttentionAnalyzer(
            self.model, self.feature_names, self.device
        )
        self.gnn_explainer = IOGNNExplainer(
            self.model, device=self.device
        )
        self.gradient_analyzer = GradientAnalyzer(
            self.model, self.feature_names, self.device
        )
        self.bottleneck_identifier = BottleneckIdentifier(
            self.model, self.feature_names, self.device
        )
    
    def _load_model(self, checkpoint_path: str):
        """Load saved model"""
        # Load checkpoint
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        # Recreate model architecture
        model = create_gat_model(
            num_features=45,  # Your feature count
            model_type='standard',
            hidden_channels=256,
            num_layers=3,
            heads=[8, 8, 1],
            dropout=0.2,
            residual=True,
            layer_norm=True,
            dtype=torch.float64
        )
        
        # Load weights
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)
        
        model = model.to(self.device)
        model.eval()
        
        logger.info(f"Model loaded from {checkpoint_path}")
        return model
    
    def _load_data(self, data_paths: Dict[str, str]):
        """Load graph data"""
        dataset = IOPerformanceGraphDataset(
            root='./data/processed',
            similarity_pt_path=data_paths['similarity_pt'],
            similarity_npz_path=data_paths['similarity_npz'],
            features_csv_path=data_paths['features_csv'],
            use_edge_weights=True,
            dtype=torch.float64,
            lazy_load=True
        )
        
        data = dataset[0]
        
        # Load splits if available
        if 'splits_path' in data_paths:
            splits = torch.load(data_paths['splits_path'])
            data.train_mask = splits['train_mask']
            data.val_mask = splits['val_mask']
            data.test_mask = splits['test_mask']
        
        # Move to device
        data = data.to(self.device)
        
        logger.info(f"Data loaded: {data.num_nodes} nodes, {data.num_edges} edges")
        return data
    
    def _get_feature_names(self) -> List[str]:
        """Get feature names"""
        return [
            'nprocs', 'POSIX_OPENS', 'LUSTRE_STRIPE_SIZE', 'LUSTRE_STRIPE_WIDTH',
            'POSIX_FILENOS', 'POSIX_MEM_ALIGNMENT', 'POSIX_FILE_ALIGNMENT',
            'POSIX_READS', 'POSIX_WRITES', 'POSIX_SEEKS', 'POSIX_STATS',
            'POSIX_BYTES_READ', 'POSIX_BYTES_WRITTEN', 'POSIX_CONSEC_READS',
            'POSIX_CONSEC_WRITES', 'POSIX_SEQ_READS', 'POSIX_SEQ_WRITES',
            'POSIX_RW_SWITCHES', 'POSIX_MEM_NOT_ALIGNED', 'POSIX_FILE_NOT_ALIGNED',
            'POSIX_SIZE_READ_0_100', 'POSIX_SIZE_READ_100_1K', 'POSIX_SIZE_READ_1K_10K',
            'POSIX_SIZE_READ_100K_1M', 'POSIX_SIZE_WRITE_0_100', 'POSIX_SIZE_WRITE_100_1K',
            'POSIX_SIZE_WRITE_1K_10K', 'POSIX_SIZE_WRITE_10K_100K', 'POSIX_SIZE_WRITE_100K_1M',
            'POSIX_STRIDE1_STRIDE', 'POSIX_STRIDE2_STRIDE', 'POSIX_STRIDE3_STRIDE',
            'POSIX_STRIDE4_STRIDE', 'POSIX_STRIDE1_COUNT', 'POSIX_STRIDE2_COUNT',
            'POSIX_STRIDE3_COUNT', 'POSIX_STRIDE4_COUNT', 'POSIX_ACCESS1_ACCESS',
            'POSIX_ACCESS2_ACCESS', 'POSIX_ACCESS3_ACCESS', 'POSIX_ACCESS4_ACCESS',
            'POSIX_ACCESS1_COUNT', 'POSIX_ACCESS2_COUNT', 'POSIX_ACCESS3_COUNT',
            'POSIX_ACCESS4_COUNT'
        ]
    
    def extract_subgraph(self, node_idx: int, num_hops: int = 2):
        """Extract k-hop subgraph around node"""
        subset, edge_index, mapping, edge_mask = k_hop_subgraph(
            node_idx, 
            num_hops, 
            self.data.edge_index,
            relabel_nodes=True,
            num_nodes=self.data.num_nodes
        )
        
        # Check size
        if len(subset) > self.max_subgraph_size:
            logger.warning(f"Subgraph has {len(subset)} nodes, limiting to {self.max_subgraph_size}")
            subset = subset[:self.max_subgraph_size]
        
        # Create subgraph data object
        from torch_geometric.data import Data
        subgraph = Data(
            x=self.data.x[subset],
            edge_index=edge_index,
            edge_attr=self.data.edge_attr[edge_mask] if self.data.edge_attr is not None else None,
            y=self.data.y[subset] if self.data.y is not None else None
        )
        
        logger.info(f"Extracted subgraph: {len(subset)} nodes, {edge_index.size(1)} edges")
        return subgraph, mapping
    
    def find_test_nodes(self, n_per_category: int = 5) -> Dict[str, List[int]]:
        """Find nodes in different performance categories"""
        # Use test mask if available, otherwise sample
        if hasattr(self.data, 'test_mask'):
            mask = self.data.test_mask
        else:
            # Use last 20% as test
            n_test = int(0.2 * self.data.num_nodes)
            mask = torch.zeros(self.data.num_nodes, dtype=torch.bool)
            mask[-n_test:] = True
        
        test_indices = torch.where(mask)[0]
        test_performance = self.data.y[mask]
        
        # Find percentiles
        perf_25 = torch.quantile(test_performance, 0.25)
        perf_75 = torch.quantile(test_performance, 0.75)
        
        # Categorize
        poor_mask = test_performance < perf_25
        medium_mask = (test_performance >= perf_25) & (test_performance <= perf_75)
        good_mask = test_performance > perf_75
        
        # Sample nodes
        categories = {
            'poor': test_indices[poor_mask][:n_per_category].tolist(),
            'medium': test_indices[medium_mask][:n_per_category].tolist(),
            'good': test_indices[good_mask][:n_per_category].tolist()
        }
        
        logger.info(f"Selected test nodes:")
        logger.info(f"  Poor performers (<{perf_25:.2f}): {len(categories['poor'])} nodes")
        logger.info(f"  Medium performers: {len(categories['medium'])} nodes")
        logger.info(f"  Good performers (>{perf_75:.2f}): {len(categories['good'])} nodes")
        
        return categories
    
    def analyze_node(self, node_idx: int) -> Dict:
        """Analyze single node with all methods"""
        logger.info(f"\n{'='*60}")
        logger.info(f"Analyzing node {node_idx}")
        logger.info(f"Performance: {self.data.y[node_idx]:.4f}")
        
        # Extract subgraph
        try:
            subgraph, mapping = self.extract_subgraph(node_idx)
        except Exception as e:
            logger.error(f"Failed to extract subgraph: {e}")
            return {}
        
        results = {
            'node_idx': node_idx,
            'performance': self.data.y[node_idx].item(),
            'methods': {}
        }
        
        # 1. Attention Analysis
        logger.info("Running attention analysis...")
        try:
            attention_bottlenecks = self.attention_analyzer.attention_based_bottleneck_detection(
                subgraph, mapping
            )
            results['methods']['attention'] = attention_bottlenecks
            
            # Log top features
            if attention_bottlenecks:
                top_features = list(attention_bottlenecks.items())[:5]
                logger.info("  Top attention features:")
                for feat, score in top_features:
                    logger.info(f"    - {feat}: {score:.4f}")
        except Exception as e:
            logger.warning(f"Attention analysis failed: {e}")
            results['methods']['attention'] = {}
        
        # 2. GNNExplainer
        logger.info("Running GNNExplainer...")
        try:
            gnn_bottlenecks = self.gnn_explainer.explain_bottleneck_pattern(
                subgraph, mapping, self.feature_names
            )
            results['methods']['gnn_explainer'] = gnn_bottlenecks
            
            if gnn_bottlenecks:
                top_features = list(gnn_bottlenecks.items())[:5]
                logger.info("  Top GNNExplainer features:")
                for feat, score in top_features:
                    logger.info(f"    - {feat}: {score:.4f}")
        except Exception as e:
            logger.warning(f"GNNExplainer failed: {e}")
            results['methods']['gnn_explainer'] = {}
        
        # 3. Gradient Methods
        logger.info("Running gradient analysis...")
        try:
            gradient_bottlenecks = self.gradient_analyzer.integrated_gradients(
                subgraph, mapping
            )
            results['methods']['gradients'] = gradient_bottlenecks
            
            if gradient_bottlenecks:
                top_features = list(gradient_bottlenecks.items())[:5]
                logger.info("  Top gradient features:")
                for feat, score in top_features:
                    logger.info(f"    - {feat}: {score:.4f}")
        except Exception as e:
            logger.warning(f"Gradient analysis failed: {e}")
            results['methods']['gradients'] = {}
        
        # 4. Consensus
        logger.info("Finding consensus...")
        consensus = self.find_consensus(results['methods'])
        results['consensus'] = consensus
        
        if consensus:
            logger.info("  Consensus bottlenecks:")
            for feat, score, methods in consensus[:5]:
                logger.info(f"    - {feat}: {score:.4f} (agreed by: {', '.join(methods)})")
        
        return results
    
    def find_consensus(self, methods_results: Dict) -> List[Tuple[str, float, List[str]]]:
        """Find features that multiple methods agree on"""
        from collections import defaultdict
        
        feature_scores = defaultdict(list)
        feature_methods = defaultdict(list)
        
        for method_name, bottlenecks in methods_results.items():
            if bottlenecks:
                for feature, score in bottlenecks.items():
                    feature_scores[feature].append(score)
                    feature_methods[feature].append(method_name)
        
        # Calculate consensus (at least 2 methods agree)
        consensus = []
        for feature, scores in feature_scores.items():
            if len(scores) >= 2:  # At least 2 methods
                avg_score = np.mean(scores)
                consensus.append((feature, avg_score, feature_methods[feature]))
        
        # Sort by average score
        consensus.sort(key=lambda x: x[1], reverse=True)
        
        return consensus
    
    def run_analysis(self, save_results: bool = True):
        """Run complete analysis"""
        # Find test nodes
        test_nodes = self.find_test_nodes(n_per_category=3)
        
        all_results = {}
        
        # Analyze each category
        for category, node_list in test_nodes.items():
            logger.info(f"\n{'='*60}")
            logger.info(f"Analyzing {category.upper()} performers")
            logger.info(f"{'='*60}")
            
            category_results = []
            
            for node_idx in node_list:
                try:
                    results = self.analyze_node(node_idx)
                    category_results.append(results)
                except Exception as e:
                    logger.error(f"Failed to analyze node {node_idx}: {e}")
                    continue
            
            all_results[category] = category_results
        
        # Save results
        if save_results:
            output_path = Path('interpretability_test_results.json')
            with open(output_path, 'w') as f:
                json.dump(all_results, f, indent=2, default=str)
            logger.info(f"\nResults saved to {output_path}")
        
        # Print summary
        self.print_summary(all_results)
        
        return all_results
    
    def print_summary(self, results: Dict):
        """Print analysis summary"""
        logger.info("\n" + "="*60)
        logger.info("ANALYSIS SUMMARY")
        logger.info("="*60)
        
        # Collect all consensus features
        all_features = defaultdict(list)
        
        for category, nodes in results.items():
            logger.info(f"\n{category.upper()} Performers:")
            
            for node_result in nodes:
                if 'consensus' in node_result and node_result['consensus']:
                    for feat, score, methods in node_result['consensus'][:3]:
                        all_features[feat].append((category, score))
                        logger.info(f"  Node {node_result['node_idx']}: {feat} ({score:.3f})")
        
        # Find patterns
        logger.info("\n" + "="*60)
        logger.info("COMMON PATTERNS")
        logger.info("="*60)
        
        for feature, occurrences in all_features.items():
            categories = [occ[0] for occ in occurrences]
            if 'poor' in categories:
                avg_score = np.mean([occ[1] for occ in occurrences])
                logger.info(f"  {feature}: Appears in {len(occurrences)} nodes (avg score: {avg_score:.3f})")


def main():
    """Main execution"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Test interpretability methods')
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='Path to model checkpoint')
    parser.add_argument('--similarity-pt', type=str, required=True,
                       help='Path to similarity graph')
    parser.add_argument('--similarity-npz', type=str, required=True,
                       help='Path to similarity npz')
    parser.add_argument('--features-csv', type=str, required=True,
                       help='Path to features CSV')
    parser.add_argument('--device', type=str, default='cpu',
                       help='Device to use (cpu recommended)')
    parser.add_argument('--max-subgraph', type=int, default=500,
                       help='Maximum subgraph size')
    
    args = parser.parse_args()
    
    # Setup paths
    data_paths = {
        'similarity_pt': args.similarity_pt,
        'similarity_npz': args.similarity_npz,
        'features_csv': args.features_csv
    }
    
    # Run tester
    tester = InterpretabilityTester(
        checkpoint_path=args.checkpoint,
        data_paths=data_paths,
        device=args.device,
        max_subgraph_size=args.max_subgraph
    )
    
    results = tester.run_analysis(save_results=True)
    
    logger.info("\nTest complete!")


if __name__ == "__main__":
    main()