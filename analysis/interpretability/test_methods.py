#!/usr/bin/env python
"""
Standalone script to test interpretability methods on saved GAT model
Dynamically loads all model configuration from training config file
"""

import os
import sys
import torch
import numpy as np
import pandas as pd
from pathlib import Path
import logging
import json
import yaml
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

# Add src to path
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.data.dataset import IOPerformanceGraphDataset
from src.models.gat import create_gat_model
from src.interpretability.attention_analyzer import AttentionAnalyzer
from src.interpretability.gnn_explainer import IOGNNExplainer
from src.interpretability.gradient_methods import GradientAnalyzer, BottleneckIdentifier
from torch_geometric.utils import k_hop_subgraph
from src.utils.visualization import visualize_node_results

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
        config_path: str,
        data_paths: Dict[str, str],
        device: Optional[str] = None,
        max_subgraph_size: Optional[int] = None
    ):
        """
        Args:
            checkpoint_path: Path to saved model checkpoint
            config_path: Path to training config YAML file
            data_paths: Paths to data files (similarity_pt, features_csv, etc.)
            device: Device to use (if None, reads from config or defaults to 'cpu')
            max_subgraph_size: Maximum nodes in subgraph (if None, uses default)
        """
        # Load configuration
        logger.info(f"Loading config from {config_path}")
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        # Set device (priority: argument > config > default)
        if device:
            self.device = torch.device(device)
        elif 'experiment' in self.config and 'device' in self.config['experiment']:
            self.device = torch.device(self.config['experiment']['device'])
        else:
            self.device = torch.device('cpu')
        logger.info(f"Using device: {self.device}")
        
        # Set max subgraph size
        self.max_subgraph_size = max_subgraph_size or 500
        
        # Set data type from config
        dtype_map = {
            'float32': torch.float32,
            'float64': torch.float64
        }
        self.dtype = dtype_map.get(
            self.config.get('model', {}).get('dtype', 'float32'),
            torch.float32
        )
        logger.info(f"Using dtype: {self.dtype}")
        
        # Load model
        logger.info("Loading model...")
        self.model = self._load_model(checkpoint_path)
        
        # Load data
        logger.info("Loading data...")
        self.data = self._load_data(data_paths)
        
        # Get feature names dynamically based on actual data
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
        """Load saved model with dynamic configuration"""
        # Load checkpoint
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        # Log checkpoint structure
        logger.info(f"Checkpoint keys: {checkpoint.keys()}")
        
        # Try to get model config from checkpoint first (if saved)
        if 'model_config' in checkpoint:
            model_config = checkpoint['model_config']
            logger.info("Using model config from checkpoint")
        else:
            # Use config file
            model_config = self.config.get('model', {})
            logger.info("Using model config from config file")
        
        # Detect actual input dimension from saved weights
        if 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
        else:
            state_dict = checkpoint
        
        # Auto-detect number of input features from saved weights
        checkpoint_features = None
        if 'input_proj.weight' in state_dict:
            checkpoint_features = state_dict['input_proj.weight'].shape[1]
            logger.info(f"Detected {checkpoint_features} input features from checkpoint")
        
        # SMART DETECTION: Check if augmentation is already applied
        # If config says feature_augmentation=True and we expect 45 base features
        # but checkpoint has 49, then augmentation is already applied
        base_features = 45  # Your actual base feature count
        feature_augmentation_in_config = model_config.get('feature_augmentation', False)
        
        # Determine if we need to apply augmentation when creating the model
        if feature_augmentation_in_config and checkpoint_features == base_features + 4:
            # Augmentation was applied during training and is baked into checkpoint
            # So we create model WITHOUT augmentation (it's already in the weights)
            use_augmentation = False
            actual_base_features = checkpoint_features  # Use 49 as base
            logger.info(f"Detected pre-augmented features in checkpoint ({checkpoint_features}), "
                    f"disabling feature_augmentation for model creation")
        elif feature_augmentation_in_config and checkpoint_features == base_features:
            # This shouldn't happen if training was done correctly, but handle it
            use_augmentation = True
            actual_base_features = base_features
            logger.warning(f"Config says augmentation=True but checkpoint has base features only")
        else:
            # No augmentation or already handled
            use_augmentation = False
            actual_base_features = checkpoint_features if checkpoint_features else base_features
            logger.info(f"Using {actual_base_features} features, augmentation={use_augmentation}")
        
        # Create model with detected parameters
        model_args = {
            'num_features': actual_base_features,
            'model_type': model_config.get('type', 'standard'),
            'hidden_channels': model_config.get('hidden_channels', 256),
            'num_layers': model_config.get('num_layers', 3),
            'heads': model_config.get('heads', [8, 8, 1]),
            'dropout': model_config.get('dropout', 0.2),
            'residual': model_config.get('residual', True),
            'layer_norm': model_config.get('layer_norm', True),
            'feature_augmentation': use_augmentation,  # Dynamically determined!
            'dtype': self.dtype
        }

        # Add any additional model parameters from config
        exclude_keys = {'type', 'dtype', 'model_type', 'feature_augmentation'}
        for key, value in model_config.items():
            if key not in model_args and key not in exclude_keys:
                model_args[key] = value
        
        logger.info(f"Creating model with config: {model_args}")
        
        # Create model
        model = create_gat_model(**model_args)
        
        # Verify dimensions match before loading
        model_input_dim = model.input_proj.weight.shape[1]
        if model_input_dim != checkpoint_features:
            raise ValueError(f"Model input dimension ({model_input_dim}) doesn't match "
                            f"checkpoint ({checkpoint_features}). Check feature_augmentation setting.")
        
        # Load weights
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)
        
        model = model.to(self.device)
        model.eval()
        
        logger.info(f"Model loaded successfully from {checkpoint_path}")
        
        # Store for later reference
        self.model_config = model_args
        self.checkpoint_features = checkpoint_features
        
        return model    

    def _load_data(self, data_paths: Dict[str, str]):
        """Load graph data with dynamic configuration"""
        # Get data config
        data_config = self.config.get('data', {})
        
        dataset_args = {
            'root': data_config.get('root', './data/processed'),
            'similarity_pt_path': data_paths['similarity_pt'],
            'similarity_npz_path': data_paths['similarity_npz'],
            'features_csv_path': data_paths['features_csv'],
            'use_edge_weights': True,
            'dtype': self.dtype,
            'lazy_load': True
        }
        
        logger.info(f"Loading dataset with args: {dataset_args}")
        
        dataset = IOPerformanceGraphDataset(**dataset_args)
        data = dataset[0]
        
        # Log data statistics
        logger.info(f"Data loaded: {data.num_nodes:,} nodes, {data.num_edges:,} edges")
        logger.info(f"Feature shape: {data.x.shape}")
        logger.info(f"Target shape: {data.y.shape}")
        
        # Now update model if we didn't know feature count
        if hasattr(self, 'model_config') and self.model_config['num_features'] != data.x.shape[1]:
            logger.warning(f"Feature mismatch: Model expects {self.model_config['num_features']}, "
                         f"but data has {data.x.shape[1]}")
        
        # Load splits if available
        if 'splits_path' in data_paths and data_paths['splits_path']:
            logger.info(f"Loading splits from {data_paths['splits_path']}")
            splits = torch.load(data_paths['splits_path'])
            data.train_mask = splits['train_mask']
            data.val_mask = splits['val_mask']
            data.test_mask = splits['test_mask']
        else:
            # Create splits based on config ratios
            logger.info("Creating data splits from config")
            splits = dataset.get_splits(
                train_ratio=data_config.get('train_ratio', 0.6),
                val_ratio=data_config.get('val_ratio', 0.2),
                test_ratio=data_config.get('test_ratio', 0.2),
                seed=self.config.get('experiment', {}).get('seed', 42),
                stratify=data_config.get('stratify', True)
            )
            data.train_mask = splits['train_mask']
            data.val_mask = splits['val_mask']
            data.test_mask = splits['test_mask']
        
        # Check if we need to augment features to match model
        if hasattr(self, 'checkpoint_features') and self.checkpoint_features > data.x.shape[1]:
            logger.info(f"Adding placeholder features from {data.x.shape[1]} to {self.checkpoint_features}")
            # Add zero features as placeholders (they'll be overridden by the model's learned representations)
            num_missing = self.checkpoint_features - data.x.shape[1]
            placeholder_features = torch.zeros(data.x.shape[0], num_missing, dtype=data.x.dtype)
            data.x = torch.cat([data.x, placeholder_features], dim=1)
            logger.info(f"Features augmented to shape: {data.x.shape}")

        # Move to device
        data = data.to(self.device)

        return data
    
    def _get_feature_names(self) -> List[str]:
        """Get feature names - can be customized or loaded from file"""
        # First check if feature names are in config
        if 'feature_names' in self.config.get('data', {}):
            return self.config['data']['feature_names']
        
        # Check if there's a feature names file
        feature_names_path = self.config.get('data', {}).get('feature_names_path')
        if feature_names_path and Path(feature_names_path).exists():
            with open(feature_names_path, 'r') as f:
                return json.load(f)
        
        # Default feature names (you can customize this list)
        default_names = [
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
        
        # Adjust to actual number of features in data
        actual_features = self.data.x.shape[1]
        if len(default_names) < actual_features:
            # Add generic names for extra features
            for i in range(len(default_names), actual_features):
                default_names.append(f'feature_{i}')
        elif len(default_names) > actual_features:
            # Trim to actual number
            default_names = default_names[:actual_features]
        
        logger.info(f"Using {len(default_names)} feature names")
        
        return default_names
    
    def extract_subgraph(self, node_idx: int, num_hops: Optional[int] = None):
        """Extract k-hop subgraph around node"""
        # Get num_hops from config or use default
        if num_hops is None:
            num_hops = self.config.get('interpretability', {}).get('num_hops', 2)
        
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
    
    def find_test_nodes(self, n_per_category: Optional[int] = None) -> Dict[str, List[int]]:
        """Find nodes in different performance categories"""
        # Get n_per_category from config or use default
        if n_per_category is None:
            n_per_category = self.config.get('interpretability', {}).get('n_per_category', 5)
        
        # Use test mask
        if hasattr(self.data, 'test_mask'):
            mask = self.data.test_mask
        else:
            # Fallback: use last portion based on test ratio
            test_ratio = self.config.get('data', {}).get('test_ratio', 0.2)
            n_test = int(test_ratio * self.data.num_nodes)
            mask = torch.zeros(self.data.num_nodes, dtype=torch.bool)
            mask[-n_test:] = True
        
        test_indices = torch.where(mask)[0]
        test_performance = self.data.y[mask]
        
        # Find percentiles (configurable)
        percentiles = self.config.get('interpretability', {}).get('percentiles', [0.25, 0.75])
        perf_low = torch.quantile(test_performance, percentiles[0])
        perf_high = torch.quantile(test_performance, percentiles[1])
        
        # Categorize
        poor_mask = test_performance < perf_low
        medium_mask = (test_performance >= perf_low) & (test_performance <= perf_high)
        good_mask = test_performance > perf_high
        
        # Sample nodes - fix tensor indexing
        categories = {}

        # Get indices for each category
        poor_indices = test_indices[torch.where(poor_mask)[0]]
        medium_indices = test_indices[torch.where(medium_mask)[0]]
        good_indices = test_indices[torch.where(good_mask)[0]]

        categories = {
            'poor': poor_indices[:n_per_category].tolist() if len(poor_indices) > 0 else [],
            'medium': medium_indices[:n_per_category].tolist() if len(medium_indices) > 0 else [],
            'good': good_indices[:n_per_category].tolist() if len(good_indices) > 0 else []
        }
        
        logger.info(f"Selected test nodes:")
        logger.info(f"  Poor performers (<{perf_low:.2f}): {len(categories['poor'])} nodes")
        logger.info(f"  Medium performers: {len(categories['medium'])} nodes")
        logger.info(f"  Good performers (>{perf_high:.2f}): {len(categories['good'])} nodes")
        
        return categories
    
    def analyze_node(self, node_idx: int) -> Dict:
        """Analyze single node with all methods"""
        logger.info(f"\n{'='*60}")
        logger.info(f"Analyzing node {node_idx}")

        # Convert tensor to float before formatting
        performance_value = self.data.y[node_idx].item() if self.data.y[node_idx].numel() == 1 else self.data.y[node_idx].squeeze().item()
        logger.info(f"Performance: {performance_value:.4f}")
        
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
        
        # Check which methods to run from config
        methods_config = self.config.get('interpretability', {}).get('methods', {
            'attention': True,
            'gnn_explainer': True,
            'gradients': True
        })
        
        # 1. Attention Analysis
        if methods_config.get('attention', True):
            logger.info("Running attention analysis...")
            try:
                attention_bottlenecks = self.attention_analyzer.attention_based_bottleneck_detection(
                    subgraph, mapping
                )
                results['methods']['attention'] = attention_bottlenecks
                
                # Log top features
                if attention_bottlenecks:
                    top_k = self.config.get('interpretability', {}).get('top_k_features', 5)
                    top_features = list(attention_bottlenecks.items())[:top_k]
                    logger.info("  Top attention features:")
                    for feat, score in top_features:
                        logger.info(f"    - {feat}: {score:.4f}")
            except Exception as e:
                logger.warning(f"Attention analysis failed: {e}")
                results['methods']['attention'] = {}
        
        # 2. GNNExplainer
        if methods_config.get('gnn_explainer', True):
            logger.info("Running GNNExplainer...")
            try:
                gnn_bottlenecks = self.gnn_explainer.explain_bottleneck_pattern(
                    subgraph, mapping, self.feature_names
                )
                results['methods']['gnn_explainer'] = gnn_bottlenecks
                
                if gnn_bottlenecks:
                    top_k = self.config.get('interpretability', {}).get('top_k_features', 5)
                    top_features = list(gnn_bottlenecks.items())[:top_k]
                    logger.info("  Top GNNExplainer features:")
                    for feat, score in top_features:
                        logger.info(f"    - {feat}: {score:.4f}")
            except Exception as e:
                logger.warning(f"GNNExplainer failed: {e}")
                results['methods']['gnn_explainer'] = {}
        
        # 3. Gradient Methods
        if methods_config.get('gradients', True):
            logger.info("Running gradient analysis...")
            try:
                gradient_bottlenecks = self.gradient_analyzer.integrated_gradients(
                    subgraph, mapping
                )
                results['methods']['gradients'] = gradient_bottlenecks
                
                if gradient_bottlenecks:
                    top_k = self.config.get('interpretability', {}).get('top_k_features', 5)
                    top_features = list(gradient_bottlenecks.items())[:top_k]
                    logger.info("  Top gradient features:")
                    for feat, score in top_features:
                        logger.info(f"    - {feat}: {score:.4f}")
            except Exception as e:
                logger.warning(f"Gradient analysis failed: {e}")
                results['methods']['gradients'] = {}
        
        # 4. Consensus
        min_consensus = self.config.get('interpretability', {}).get('min_consensus_methods', 2)
        logger.info(f"Finding consensus (min {min_consensus} methods)...")
        consensus = self.find_consensus(results['methods'], min_methods=min_consensus)
        results['consensus'] = consensus
        
        if consensus:
            logger.info("  Consensus bottlenecks:")
            top_k = self.config.get('interpretability', {}).get('top_k_consensus', 5)
            for feat, score, methods in consensus[:top_k]:
                logger.info(f"    - {feat}: {score:.4f} (agreed by: {', '.join(methods)})")
        
        return results
    
    def visualize_node(self, node_idx: int, output_dir: str = './figures'):
        """
        Analyze and visualize a single node
        
        Args:
            node_idx: Node index to analyze
            output_dir: Directory to save figures
        """
        # Analyze the node
        results = self.analyze_node(node_idx)
        
        # Generate visualizations
        visualize_node_results(results, output_dir)
        
        return results

    def visualize_specific_nodes(self, node_indices: List[int], output_dir: str = './figures'):
        """
        Analyze and visualize multiple specific nodes
        
        Args:
            node_indices: List of node indices to analyze
            output_dir: Directory to save figures
        """
        import os
        os.makedirs(output_dir, exist_ok=True)
        
        all_results = []
        for node_idx in node_indices:
            logger.info(f"Visualizing node {node_idx}...")
            try:
                results = self.visualize_node(node_idx, output_dir)
                all_results.append(results)
            except Exception as e:
                logger.error(f"Failed to visualize node {node_idx}: {e}")
        
        return all_results
    def find_consensus(self, methods_results: Dict, min_methods: int = 2) -> List[Tuple[str, float, List[str]]]:
        """Find features that multiple methods agree on"""
        from collections import defaultdict
        
        feature_scores = defaultdict(list)
        feature_methods = defaultdict(list)
        
        for method_name, bottlenecks in methods_results.items():
            if bottlenecks:
                for feature, score in bottlenecks.items():
                    feature_scores[feature].append(score)
                    feature_methods[feature].append(method_name)
        
        # Calculate consensus
        consensus = []
        for feature, scores in feature_scores.items():
            if len(scores) >= min_methods:
                avg_score = np.mean(scores)
                consensus.append((feature, avg_score, feature_methods[feature]))
        
        # Sort by average score
        consensus.sort(key=lambda x: x[1], reverse=True)
        
        return consensus
    
    def run_analysis(self, save_results: bool = True, specific_nodes: Optional[List[int]] = None):
        """
        Run complete analysis
        
        Args:
            save_results: Whether to save results to file
            specific_nodes: Optional list of specific node indices to analyze
        """
        all_results = {}
        
        if specific_nodes:
            # Analyze specific nodes
            logger.info(f"Analyzing {len(specific_nodes)} specific nodes")
            specific_results = []
            for node_idx in specific_nodes:
                try:
                    results = self.analyze_node(node_idx)
                    specific_results.append(results)
                except Exception as e:
                    logger.error(f"Failed to analyze node {node_idx}: {e}")
                    continue
            all_results['specific'] = specific_results
        else:
            # Find and analyze test nodes by category
            test_nodes = self.find_test_nodes()
            
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
            output_dir = Path(self.config.get('interpretability', {}).get('output_dir', '.'))
            output_dir.mkdir(parents=True, exist_ok=True)
            output_path = output_dir / 'interpretability_test_results.json'
            
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
        from collections import defaultdict
        all_features = defaultdict(list)
        
        for category, nodes in results.items():
            logger.info(f"\n{category.upper()} Performers:")
            
            for node_result in nodes:
                if 'consensus' in node_result and node_result['consensus']:
                    top_k = self.config.get('interpretability', {}).get('summary_top_k', 3)
                    for feat, score, methods in node_result['consensus'][:top_k]:
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
    
    # Required arguments
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='Path to model checkpoint')
    parser.add_argument('--config', type=str, required=True,
                       help='Path to training config YAML file')
    parser.add_argument('--similarity-pt', type=str, required=True,
                       help='Path to similarity graph')
    parser.add_argument('--similarity-npz', type=str, required=True,
                       help='Path to similarity npz')
    parser.add_argument('--features-csv', type=str, required=True,
                       help='Path to features CSV')
    
    # Optional arguments
    parser.add_argument('--device', type=str,
                       help='Device to use (overrides config)')
    parser.add_argument('--max-subgraph', type=int,
                       help='Maximum subgraph size (overrides default)')
    parser.add_argument('--splits-path', type=str,
                       help='Path to saved data splits')
    parser.add_argument('--specific-nodes', type=int, nargs='+',
                       help='Specific node indices to analyze')
    parser.add_argument('--no-save', action='store_true',
                       help='Do not save results')
            
    parser.add_argument('--visualize-node', type=int,
                       help='Specific node to analyze and visualize')
    parser.add_argument('--visualize-nodes', type=int, nargs='+',
                       help='Multiple nodes to analyze and visualize')
    parser.add_argument('--output-dir', type=str, default='./figures',
                       help='Directory to save visualization figures')
    
    args = parser.parse_args()
    
    # Setup paths
    data_paths = {
        'similarity_pt': args.similarity_pt,
        'similarity_npz': args.similarity_npz,
        'features_csv': args.features_csv
    }
    
    if args.splits_path:
        data_paths['splits_path'] = args.splits_path
    
    # Run tester
    tester = InterpretabilityTester(
        checkpoint_path=args.checkpoint,
        config_path=args.config,
        data_paths=data_paths,
        device=args.device,
        max_subgraph_size=args.max_subgraph
    )
    
    # Run analysis
    # Handle visualization requests
    if args.visualize_node:
        # Single node visualization
        logger.info(f"Visualizing single node {args.visualize_node}")
        results = tester.visualize_node(
            node_idx=args.visualize_node,
            output_dir=args.output_dir
        )
        logger.info(f"Visualization saved to {args.output_dir}")
    
    elif args.visualize_nodes:
        # Multiple nodes visualization
        logger.info(f"Visualizing {len(args.visualize_nodes)} nodes")
        results = tester.visualize_specific_nodes(
            node_indices=args.visualize_nodes,
            output_dir=args.output_dir
        )
        logger.info(f"All visualizations saved to {args.output_dir}")
    
    else:
        # Regular analysis (existing code)
        results = tester.run_analysis(
            save_results=not args.no_save,
            specific_nodes=args.specific_nodes
        )
    
    logger.info("\nTest complete!")

if __name__ == "__main__":
    main()