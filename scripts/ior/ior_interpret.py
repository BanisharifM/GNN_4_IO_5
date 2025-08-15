#!/usr/bin/env python3
"""
IOR Performance Bottleneck Analysis using GNN Interpretability Methods
Analyzes unseen IOR jobs using Attention, GNNExplainer, and Gradient methods
with z-score normalized consensus
"""

import pandas as pd
import torch
import numpy as np
import sys
import os
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import json
import logging
from typing import Dict, List, Tuple, Optional

# Add project to path
sys.path.append('/work/hdd/bdau/mbanisharifdehkordi/GNN_4_IO_5')

from src.models.gat import create_gat_model
from torch_geometric.data import Data
from sklearn.metrics.pairwise import cosine_similarity
import scipy.sparse as sp
from src.interpretability.attention_analyzer import AttentionAnalyzer
from src.interpretability.gnn_explainer import IOGNNExplainer
from src.interpretability.gradient_methods import GradientAnalyzer, BottleneckIdentifier

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class IORInterpretabilityAnalyzer:
    """
    Comprehensive interpretability analysis for IOR performance prediction
    using Attention, GNNExplainer, and Gradient methods with z-score consensus
    """
    
    def __init__(self, 
                 model_checkpoint_path,
                 similarity_graph_path=None,
                 features_csv_path=None,
                 similarity_threshold=0.75,
                 use_cpu=True):
        """
        Initialize analyzer with trained model and graph data
        
        Args:
            model_checkpoint_path: Path to trained GAT model
            similarity_graph_path: Path to similarity matrix
            features_csv_path: Path to training features
            similarity_threshold: Threshold for edge creation
            use_cpu: Force CPU usage
        """
        # Force CPU for memory efficiency
        if use_cpu:
            self.device = torch.device('cpu')
        else:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        self.similarity_threshold = similarity_threshold
        
        # Load model
        self.model, self.checkpoint = self._load_model(model_checkpoint_path)
        
        # Load training data if provided
        self.training_features = None
        self.similarity_matrix = None
        
        if features_csv_path:
            self.training_features = self._load_training_features(features_csv_path)
            logger.info(f"Loaded {len(self.training_features)} training samples")
            
        if similarity_graph_path:
            self.similarity_matrix = self._load_similarity_matrix(similarity_graph_path)
        
        # Define feature names (EXACT names as requested)
        self.feature_names = [
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
        
        # Initialize interpretability analyzers
        self.attention_analyzer = AttentionAnalyzer(self.model, self.feature_names, self.device)
        self.gnn_explainer = IOGNNExplainer(self.model, device=self.device)
        self.gradient_analyzer = GradientAnalyzer(self.model, self.feature_names, self.device)
        self.bottleneck_identifier = BottleneckIdentifier(self.model, self.feature_names, self.device)
    
    def _load_model(self, checkpoint_path):
        """Load trained GAT model"""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        # Detect model configuration from checkpoint
        state_dict = checkpoint['model_state_dict']
        hidden_channels = state_dict['input_proj.weight'].shape[0]
        num_layers = max([int(k.split('.')[1]) for k in state_dict.keys() 
                         if k.startswith('gat_layers.')]) + 1
        
        heads = []
        for i in range(num_layers):
            if f'gat_layers.{i}.gat_conv.att_src' in state_dict:
                n_heads = state_dict[f'gat_layers.{i}.gat_conv.att_src'].shape[1]
                heads.append(n_heads)
        
        logger.info(f"Model configuration: hidden={hidden_channels}, layers={num_layers}, heads={heads}")
        
        # Create model
        model = create_gat_model(
            num_features=49,  # Changed from 45 to 49
            model_type='standard',
            hidden_channels=hidden_channels,
            num_layers=num_layers,
            heads=heads,
            dropout=0.1,
            edge_dim=1,
            residual=True,
            layer_norm=True,
            feature_augmentation=False,  # Changed from True to False
            pool_type='mean',
            dtype=torch.float32
        )
        
        model.load_state_dict(state_dict)
        model.eval()
        model = model.to(self.device)
        
        return model, checkpoint
    
    def _load_training_features(self, features_path):
        """Load training features"""
        df = pd.read_csv(features_path)
        if 'tag' in df.columns:
            df = df.drop('tag', axis=1)
        return df.values.astype(np.float32)
    
    def _load_similarity_matrix(self, graph_path):
        """Load similarity matrix from .npz file"""
        if os.path.exists(graph_path):
            logger.info(f"Loading similarity matrix from {graph_path}")
            similarity_matrix = sp.load_npz(graph_path)
            logger.info(f"Similarity matrix shape: {similarity_matrix.shape}")
            return similarity_matrix
        return None
    
    def create_subgraph_for_analysis(self, new_features, k_neighbors=100, subgraph_size=500):
        """
        Create subgraph for interpretability analysis
        
        Args:
            new_features: Features of new sample [45]
            k_neighbors: Number of nearest neighbors
            subgraph_size: Maximum size of subgraph
        """
        if self.training_features is None:
            # Single node graph
            features = new_features.reshape(1, -1)
            edge_index = torch.tensor([[0], [0]], dtype=torch.long)
            edge_attr = torch.tensor([[1.0]], dtype=torch.float32)
            return features, edge_index, edge_attr, 0
        
        # Calculate similarities
        new_features = new_features.reshape(1, -1)
        similarities = cosine_similarity(new_features, self.training_features)[0]
        
        # Find top k neighbors
        top_k = min(subgraph_size, len(similarities))
        top_indices = np.argpartition(similarities, -top_k)[-top_k:]
        top_indices = top_indices[np.argsort(similarities[top_indices])[::-1]]
        
        # Filter by threshold
        valid_mask = similarities[top_indices] >= self.similarity_threshold
        selected_indices = top_indices[valid_mask][:k_neighbors]
        
        if len(selected_indices) == 0:
            logger.info(f"No neighbors above threshold {self.similarity_threshold}, using top {k_neighbors}")
            selected_indices = top_indices[:k_neighbors]
        
        logger.info(f"Selected {len(selected_indices)} neighbors for analysis")
        
        # Create subgraph
        subgraph_features = np.vstack([
            self.training_features[selected_indices],
            new_features
        ])
        new_node_idx = len(selected_indices)
        
        # Create edges
        edges = []
        weights = []
        
        for i, neighbor_idx in enumerate(selected_indices):
            edges.append([new_node_idx, i])
            edges.append([i, new_node_idx])
            weights.append(similarities[neighbor_idx])
            weights.append(similarities[neighbor_idx])
        
        # Add edges between neighbors if similarity matrix available
        if self.similarity_matrix is not None and len(selected_indices) > 1:
            submatrix = self.similarity_matrix[selected_indices][:, selected_indices]
            if sp.issparse(submatrix):
                submatrix_edges = sp.find(submatrix)
                for i, j, w in zip(submatrix_edges[0], submatrix_edges[1], submatrix_edges[2]):
                    if i < j and w >= self.similarity_threshold:
                        edges.append([i, j])
                        edges.append([j, i])
                        weights.append(w)
                        weights.append(w)
        
        if len(edges) == 0:
            edges = [[new_node_idx, new_node_idx]]
            weights = [1.0]
        
        edge_index = torch.tensor(edges, dtype=torch.long).t()
        edge_attr = torch.tensor(weights, dtype=torch.float32).unsqueeze(1)
        
        logger.info(f"Created subgraph with {len(subgraph_features)} nodes and {edge_index.shape[1]} edges")
        
        return subgraph_features, edge_index, edge_attr, new_node_idx
    
    def analyze_with_all_methods(self, features_path):
        """
        Analyze IOR job using all three interpretability methods
        
        Args:
            features_path: Path to CSV with IOR features
            
        Returns:
            Dictionary with results from all methods
        """
        # Load new sample
        new_data = pd.read_csv(features_path)
        new_features = new_data.iloc[0, :-1].values  # Exclude tag
        actual_tag = new_data.iloc[0, -1]
        
        # Create subgraph
        logger.info("\n=== Creating subgraph for interpretability analysis ===")
        subgraph_features, edge_index, edge_attr, new_node_idx = self.create_subgraph_for_analysis(
            new_features, k_neighbors=100, subgraph_size=500
        )
        
        # Convert to tensor
        features_tensor = torch.FloatTensor(subgraph_features).to(self.device)
        
        # Apply feature augmentation if needed
        if features_tensor.shape[1] == 45:
            # Add augmentation features if we have base features
            feat_mean = features_tensor.mean(dim=1, keepdim=True)
            feat_std = features_tensor.std(dim=1, keepdim=True)
            feat_min = features_tensor.min(dim=1, keepdim=True)[0]
            feat_max = features_tensor.max(dim=1, keepdim=True)[0]
            features_tensor = torch.cat([
                features_tensor, feat_mean, feat_std, feat_min, feat_max
            ], dim=1)
        
        # Create graph data
        data = Data(
            x=features_tensor,
            edge_index=edge_index.to(self.device),
            edge_attr=edge_attr.to(self.device)
        )
        
        # Make prediction
        with torch.no_grad():
            # Forward pass through model
            x = self.model.input_proj(data.x)
            
            for i, gat_layer in enumerate(self.model.gat_layers):
                residual = x
                x, _ = gat_layer(x, data.edge_index, data.edge_attr)
                
                if self.model.residual and i < len(self.model.residual_projs):
                    residual = self.model.residual_projs[i](residual)
                    x = x + residual
                
                if self.model.layer_norm and i < len(self.model.layer_norms):
                    x = self.model.layer_norms[i](x)
                
                if i < self.model.num_layers - 1:
                    x = torch.nn.functional.elu(x)
            
            node_features = x[new_node_idx].unsqueeze(0)
            prediction = self.model.predictor(node_features).item()
        
        predicted_bandwidth = 10**prediction - 1
        actual_bandwidth = 10**actual_tag - 1
        
        logger.info(f"\n=== Performance Prediction ===")
        logger.info(f"Predicted: {predicted_bandwidth:.2f} MB/s")
        logger.info(f"Actual: {actual_bandwidth:.2f} MB/s")
        logger.info(f"Error: {abs(predicted_bandwidth - actual_bandwidth):.2f} MB/s")
        
        # Run all three interpretability methods
        results = {
            'prediction': predicted_bandwidth,
            'actual': actual_bandwidth,
            'methods': {}
        }
        
        # 1. Attention Analysis
        logger.info("\n=== Running Attention Analysis ===")
        try:
            attention_scores = self.attention_analyzer.attention_based_bottleneck_detection(
                data, new_node_idx
            )
            results['methods']['attention'] = attention_scores
            logger.info(f"Found {len(attention_scores)} important features via attention")
        except Exception as e:
            logger.error(f"Attention analysis failed: {e}")
            results['methods']['attention'] = {}
        
        # 2. GNNExplainer
        logger.info("\n=== Running GNNExplainer ===")
        try:
            gnn_scores = self.gnn_explainer.explain_bottleneck_pattern(
                data, new_node_idx, self.feature_names
            )
            results['methods']['gnn_explainer'] = gnn_scores
            logger.info(f"Found {len(gnn_scores)} important features via GNNExplainer")
        except Exception as e:
            logger.error(f"GNNExplainer failed: {e}")
            results['methods']['gnn_explainer'] = {}
        
        # 3. Gradient Methods (Integrated Gradients)
        logger.info("\n=== Running Gradient Analysis ===")
        try:
            gradient_scores = self.gradient_analyzer.integrated_gradients(
                data, new_node_idx
            )
            results['methods']['gradients'] = gradient_scores
            logger.info(f"Found {len(gradient_scores)} important features via gradients")
        except Exception as e:
            logger.error(f"Gradient analysis failed: {e}")
            results['methods']['gradients'] = {}
        
        # Calculate z-score normalized consensus
        logger.info("\n=== Calculating Z-Score Normalized Consensus ===")
        consensus_scores = self.calculate_zscore_consensus(results['methods'])
        results['consensus'] = consensus_scores
        
        return results
    
    def calculate_zscore_consensus(self, methods_results):
        """
        Calculate consensus using z-score normalization
        Each method's scores are z-normalized, then averaged
        
        Args:
            methods_results: Dictionary with results from each method
            
        Returns:
            Dictionary with consensus scores for each feature
        """
        # Collect all features and their scores from each method
        feature_scores = {}
        
        # For each method, z-normalize the scores
        normalized_scores = {}
        
        for method_name, scores in methods_results.items():
            if not scores:
                continue
                
            # Get all scores for this method
            method_scores = list(scores.values())
            
            if len(method_scores) > 1:
                # Calculate z-scores
                mean_score = np.mean(method_scores)
                std_score = np.std(method_scores)
                
                if std_score > 0:
                    # Z-normalize
                    normalized_scores[method_name] = {}
                    for feature, score in scores.items():
                        z_score = (score - mean_score) / std_score
                        normalized_scores[method_name][feature] = z_score
                else:
                    # If all scores are the same, use 0
                    normalized_scores[method_name] = {feature: 0 for feature in scores}
            else:
                # Single score, assign z-score of 0
                normalized_scores[method_name] = {feature: 0 for feature in scores}
        
        # Calculate consensus: average z-scores across methods
        all_features = set()
        for method_scores in normalized_scores.values():
            all_features.update(method_scores.keys())
        
        consensus = {}
        for feature in all_features:
            z_scores = []
            for method_name, method_scores in normalized_scores.items():
                if feature in method_scores:
                    z_scores.append(method_scores[feature])
            
            if z_scores:
                # Average z-score across methods that detected this feature
                consensus[feature] = np.mean(z_scores)
        
        # Sort by consensus score
        consensus = dict(sorted(consensus.items(), key=lambda x: x[1], reverse=True))
        
        logger.info(f"Consensus found {len(consensus)} features")
        logger.info("Top 5 consensus features:")
        for i, (feature, score) in enumerate(list(consensus.items())[:5], 1):
            logger.info(f"  {i}. {feature}: {score:.3f}")
        
        return consensus
    
    def visualize_results(self, results, output_dir='./interpretability_figures'):
        """
        Create separate visualizations for each method and consensus
        
        Args:
            results: Results dictionary from analyze_with_all_methods
            output_dir: Directory to save figures
        """
        os.makedirs(output_dir, exist_ok=True)
        
        # Set style for better-looking plots
        plt.style.use('seaborn-v0_8-darkgrid')
        
        # 1. Visualize Attention Analysis
        if 'attention' in results['methods'] and results['methods']['attention']:
            self._visualize_method(
                results['methods']['attention'],
                'Attention Analysis - Feature Importance',
                os.path.join(output_dir, 'attention_analysis.png'),
                color='blue'
            )
        
        # 2. Visualize GNNExplainer
        if 'gnn_explainer' in results['methods'] and results['methods']['gnn_explainer']:
            self._visualize_method(
                results['methods']['gnn_explainer'],
                'GNNExplainer - Feature Importance',
                os.path.join(output_dir, 'gnn_explainer_analysis.png'),
                color='green'
            )
        
        # 3. Visualize Gradient Methods
        if 'gradients' in results['methods'] and results['methods']['gradients']:
            self._visualize_method(
                results['methods']['gradients'],
                'Integrated Gradients - Feature Importance',
                os.path.join(output_dir, 'gradient_analysis.png'),
                color='red'
            )
        
        # 4. Visualize Z-Score Consensus
        if 'consensus' in results and results['consensus']:
            self._visualize_consensus(
                results['consensus'],
                'Z-Score Normalized Consensus',
                os.path.join(output_dir, 'consensus_analysis.png')
            )
        
        logger.info(f"All visualizations saved to {output_dir}")
    
    def _visualize_method(self, scores, title, save_path, color='blue', top_k=15):
        """
        Visualize scores for a single method
        
        Args:
            scores: Dictionary of feature scores
            title: Plot title
            save_path: Path to save figure
            color: Bar color
            top_k: Number of top features to show
        """
        # Sort and select top features
        sorted_features = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:top_k]
        
        if not sorted_features:
            logger.warning(f"No features to visualize for {title}")
            return
        
        features = [item[0] for item in sorted_features]
        values = [item[1] for item in sorted_features]
        
        # Create figure
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # Create horizontal bar chart
        bars = ax.barh(range(len(features)), values, color=color, alpha=0.7)
        
        # Customize plot
        ax.set_yticks(range(len(features)))
        ax.set_yticklabels(features, fontsize=10)
        ax.set_xlabel('Importance Score', fontsize=12)
        ax.set_title(title, fontsize=14, fontweight='bold')
        
        # Add value labels on bars
        for i, (bar, value) in enumerate(zip(bars, values)):
            ax.text(value, bar.get_y() + bar.get_height()/2, 
                   f'{value:.3f}', ha='left', va='center', fontsize=9)
        
        # Add grid for better readability
        ax.grid(True, axis='x', alpha=0.3)
        
        # Adjust layout
        plt.tight_layout()
        
        # Save figure
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Saved {title} to {save_path}")
    
    def _visualize_consensus(self, consensus_scores, title, save_path, top_k=15):
        """
        Visualize z-score consensus with special formatting
        
        Args:
            consensus_scores: Dictionary of consensus scores
            title: Plot title
            save_path: Path to save figure
            top_k: Number of top features to show
        """
        # Sort and select top features
        sorted_features = sorted(consensus_scores.items(), key=lambda x: x[1], reverse=True)[:top_k]
        
        if not sorted_features:
            logger.warning("No consensus features to visualize")
            return
        
        features = [item[0] for item in sorted_features]
        values = [item[1] for item in sorted_features]
        
        # Create figure with larger size for consensus
        fig, ax = plt.subplots(figsize=(14, 10))
        
        # Create color gradient based on z-score
        norm = plt.Normalize(vmin=min(values), vmax=max(values))
        colors = plt.cm.RdYlGn_r(norm(values))
        
        # Create horizontal bar chart
        bars = ax.barh(range(len(features)), values, color=colors, alpha=0.8)
        
        # Customize plot
        ax.set_yticks(range(len(features)))
        ax.set_yticklabels(features, fontsize=11, fontweight='bold')
        ax.set_xlabel('Z-Score Normalized Consensus Score', fontsize=12)
        ax.set_title(title, fontsize=16, fontweight='bold')
        
        # Add value labels on bars
        for i, (bar, value) in enumerate(zip(bars, values)):
            ax.text(value, bar.get_y() + bar.get_height()/2, 
                   f'{value:.3f}', ha='left', va='center', fontsize=10, fontweight='bold')
        
        # Add a vertical line at z=0 for reference
        ax.axvline(x=0, color='black', linestyle='--', alpha=0.5, linewidth=1)
        
        # Add grid
        ax.grid(True, axis='x', alpha=0.3)
        
        # Add colorbar to show z-score scale
        sm = plt.cm.ScalarMappable(cmap=plt.cm.RdYlGn_r, norm=norm)
        sm.set_array([])
        cbar = plt.colorbar(sm, ax=ax, pad=0.01)
        cbar.set_label('Z-Score', fontsize=10)
        
        # Adjust layout
        plt.tight_layout()
        
        # Save figure
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Saved {title} to {save_path}")
    
    def generate_bottleneck_report(self, results, save_path='bottleneck_report.json'):
        """
        Generate comprehensive bottleneck report
        
        Args:
            results: Results from analysis
            save_path: Path to save report
        """
        report = {
            'performance': {
                'predicted_mbps': results['prediction'],
                'actual_mbps': results['actual'],
                'error_mbps': abs(results['prediction'] - results['actual']),
                'relative_error_percent': abs(results['prediction'] - results['actual']) / results['actual'] * 100
            },
            'bottlenecks': {
                'attention': {},
                'gnn_explainer': {},
                'gradients': {},
                'consensus': {}
            }
        }
        
        # Add top features from each method
        for method in ['attention', 'gnn_explainer', 'gradients']:
            if method in results['methods'] and results['methods'][method]:
                sorted_features = sorted(results['methods'][method].items(), 
                                       key=lambda x: x[1], reverse=True)[:10]
                report['bottlenecks'][method] = dict(sorted_features)
        
        # Add consensus
        if 'consensus' in results:
            sorted_consensus = sorted(results['consensus'].items(), 
                                    key=lambda x: x[1], reverse=True)[:10]
            report['bottlenecks']['consensus'] = dict(sorted_consensus)
        
        # Identify primary bottleneck
        if results['consensus']:
            top_feature = list(results['consensus'].keys())[0]
            report['primary_bottleneck'] = {
                'feature': top_feature,
                'consensus_score': results['consensus'][top_feature],
                'recommendation': self._get_recommendation(top_feature)
            }
        
        # Save report
        with open(save_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        logger.info(f"Bottleneck report saved to {save_path}")
        
        return report
    
    def _get_recommendation(self, feature_name):
        """Get recommendation based on bottleneck feature"""
        recommendations = {
            'POSIX_SIZE_WRITE_100_1K': 'Increase write buffer size to at least 1MB',
            'POSIX_SIZE_WRITE_0_100': 'Avoid very small writes, batch operations',
            'POSIX_SIZE_WRITE_1K_10K': 'Increase write size to 100KB or larger',
            'POSIX_SEEKS': 'Reduce random access, use sequential I/O patterns',
            'POSIX_FILE_NOT_ALIGNED': 'Align I/O operations to file system block boundaries',
            'POSIX_MEM_NOT_ALIGNED': 'Align memory buffers for better performance',
            'POSIX_RW_SWITCHES': 'Reduce switching between reads and writes',
            'LUSTRE_STRIPE_SIZE': 'Adjust Lustre stripe size for workload',
            'LUSTRE_STRIPE_WIDTH': 'Optimize Lustre stripe count',
            'POSIX_OPENS': 'Reduce number of file open operations'
        }
        
        return recommendations.get(feature_name, 'Optimize I/O pattern for better performance')


def main():
    """Main execution function"""
    # Paths
    model_checkpoint = '/work/hdd/bdau/mbanisharifdehkordi/GNN_4_IO_5/data/1M/best_model.pt'
    
    # Data directory
    data_dir = '/work/hdd/bdau/mbanisharifdehkordi/GNN_4_IO_5/data/1M'
    
    # Look for similarity matrix
    similarity_matrix = None
    possible_npz_paths = [
        os.path.join(data_dir, 'similarity_output_0.75', 'similarity_matrix.npz'),
        os.path.join(data_dir, 'similarity_matrix.npz'),
        os.path.join(data_dir, 'similarity_graph.npz'),
    ]
    
    for path in possible_npz_paths:
        if os.path.exists(path):
            similarity_matrix = path
            logger.info(f"Found similarity matrix: {path}")
            break
    
    # Training features
    training_features = os.path.join(data_dir, 'aiio_sample_1000000_normalized.csv')
    if not os.path.exists(training_features):
        training_features = None
        logger.warning("Training features not found")
    
    # IOR test sample
    test_features = '/work/hdd/bdau/mbanisharifdehkordi/GNN_4_IO_5/darshan_features_ior_normalized.csv'
    
    # Initialize analyzer
    logger.info("\n" + "="*70)
    logger.info("INITIALIZING IOR INTERPRETABILITY ANALYZER")
    logger.info("="*70)
    
    analyzer = IORInterpretabilityAnalyzer(
        model_checkpoint_path=model_checkpoint,
        similarity_graph_path=similarity_matrix,
        features_csv_path=training_features,
        similarity_threshold=0.75,
        use_cpu=True
    )
    
    # Run comprehensive analysis
    logger.info("\n" + "="*70)
    logger.info("RUNNING INTERPRETABILITY ANALYSIS")
    logger.info("="*70)
    
    results = analyzer.analyze_with_all_methods(test_features)
    
    # Generate visualizations
    logger.info("\n" + "="*70)
    logger.info("GENERATING VISUALIZATIONS")
    logger.info("="*70)
    
    analyzer.visualize_results(results)
    
    # Generate report
    logger.info("\n" + "="*70)
    logger.info("GENERATING BOTTLENECK REPORT")
    logger.info("="*70)
    
    report = analyzer.generate_bottleneck_report(results)
    
    # Print summary
    logger.info("\n" + "="*70)
    logger.info("ANALYSIS SUMMARY")
    logger.info("="*70)
    logger.info(f"Performance: {results['prediction']:.2f} MB/s (predicted) vs {results['actual']:.2f} MB/s (actual)")
    logger.info(f"Primary Bottleneck: {report.get('primary_bottleneck', {}).get('feature', 'Unknown')}")
    logger.info(f"Recommendation: {report.get('primary_bottleneck', {}).get('recommendation', 'N/A')}")
    
    logger.info("\n" + "="*70)
    logger.info("ANALYSIS COMPLETE!")
    logger.info("="*70)
    logger.info("Outputs:")
    logger.info("  - Visualizations: ./interpretability_figures/")
    logger.info("  - Report: bottleneck_report.json")


if __name__ == "__main__":
    main()