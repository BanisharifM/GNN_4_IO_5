#!/usr/bin/env python3
"""
IOR Performance Bottleneck Analysis using Integrated Gradients
With detailed logging and actual feature values in report
"""

import pandas as pd
import torch
import numpy as np
import sys
import os
from pathlib import Path
import json
import logging
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

# Add project to path
sys.path.append('/work/hdd/bdau/mbanisharifdehkordi/GNN_4_IO_5')

from src.models.gat import create_gat_model
from torch_geometric.data import Data
from sklearn.metrics.pairwise import cosine_similarity
import scipy.sparse as sp
from src.interpretability.gradient_methods import GradientAnalyzer, BottleneckIdentifier

# Setup enhanced logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger(__name__)


class IORInterpretabilityAnalyzer:
    """
    Interpretability analysis using Integrated Gradients
    """
    
    def __init__(self, 
                 model_checkpoint_path,
                 similarity_graph_path=None,
                 features_csv_path=None,
                 similarity_threshold=0.75,
                 use_cpu=True):
        """
        Initialize analyzer with trained model and graph data
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
            logger.info(f"✓ Loaded {len(self.training_features):,} training samples")
            
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
        
        # Initialize gradient analyzer only
        self.gradient_analyzer = GradientAnalyzer(self.model, self.feature_names, self.device)
        
        # Store for analysis
        self.normalized_scores = {}
        self.actual_feature_values = {}
    
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
            num_features=49,
            model_type='standard',
            hidden_channels=hidden_channels,
            num_layers=num_layers,
            heads=heads,
            dropout=0.1,
            edge_dim=1,
            residual=True,
            layer_norm=True,
            feature_augmentation=False,
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
        """Create subgraph for interpretability analysis"""
        if self.training_features is None:
            features = new_features.reshape(1, -1)
            edge_index = torch.tensor([[0], [0]], dtype=torch.long)
            edge_attr = torch.tensor([[1.0]], dtype=torch.float32)
            return features, edge_index, edge_attr, 0
        
        new_features = new_features.reshape(1, -1)
        similarities = cosine_similarity(new_features, self.training_features)[0]
        
        top_k = min(subgraph_size, len(similarities))
        top_indices = np.argpartition(similarities, -top_k)[-top_k:]
        top_indices = top_indices[np.argsort(similarities[top_indices])[::-1]]
        
        valid_mask = similarities[top_indices] >= self.similarity_threshold
        selected_indices = top_indices[valid_mask][:k_neighbors]
        
        if len(selected_indices) == 0:
            logger.info(f"No neighbors above threshold {self.similarity_threshold}, using top {k_neighbors}")
            selected_indices = top_indices[:k_neighbors]
        
        logger.info(f"Selected {len(selected_indices)} neighbors for analysis")
        
        subgraph_features = np.vstack([
            self.training_features[selected_indices],
            new_features
        ])
        new_node_idx = len(selected_indices)
        
        edges = []
        weights = []
        
        for i, neighbor_idx in enumerate(selected_indices):
            edges.append([new_node_idx, i])
            edges.append([i, new_node_idx])
            weights.append(similarities[neighbor_idx])
            weights.append(similarities[neighbor_idx])
        
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
    
    def analyze_with_gradient_method(self, features_path):
        """
        Analyze IOR job using Integrated Gradients with enhanced logging
        """
        # Load new sample
        new_data = pd.read_csv(features_path)
        new_features = new_data.iloc[0, :-1].values
        actual_tag = new_data.iloc[0, -1]
        
        # Store actual feature values
        for i, feature_name in enumerate(self.feature_names):
            if i < len(new_features):
                self.actual_feature_values[feature_name] = float(new_features[i])
        
        # Create subgraph
        logger.info("\n" + "="*70)
        logger.info("🔍 CREATING SUBGRAPH FOR ANALYSIS")
        logger.info("="*70)
        subgraph_features, edge_index, edge_attr, new_node_idx = self.create_subgraph_for_analysis(
            new_features, k_neighbors=100, subgraph_size=500
        )
        
        # Convert to tensor
        features_tensor = torch.FloatTensor(subgraph_features).to(self.device)
        
        # Apply feature augmentation if needed
        if features_tensor.shape[1] == 45:
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
        
        logger.info("\n" + "="*70)
        logger.info("📈 PERFORMANCE PREDICTION")
        logger.info("="*70)
        logger.info(f"Predicted: {predicted_bandwidth:.2f} MB/s")
        logger.info(f"Actual: {actual_bandwidth:.2f} MB/s")
        logger.info(f"Error: {abs(predicted_bandwidth - actual_bandwidth):.2f} MB/s")
        logger.info(f"Relative Error: {abs(predicted_bandwidth - actual_bandwidth) / actual_bandwidth * 100:.1f}%")
        
        # Run Gradient Analysis
        results = {
            'prediction': predicted_bandwidth,
            'actual': actual_bandwidth,
            'gradient_scores': {}
        }
        
        # Gradient Methods (Integrated Gradients)
        logger.info("\n" + "="*70)
        logger.info("📊 GRADIENT ANALYSIS (INTEGRATED GRADIENTS)")
        logger.info("="*70)
        try:
            gradient_scores = self.gradient_analyzer.integrated_gradients(
                data, new_node_idx
            )
            results['gradient_scores'] = gradient_scores
            
            # Log raw scores (top 10 positive)
            if gradient_scores:
                logger.info("\nRaw Gradient Scores (Top 10 Positive - Most Important for High Performance):")
                sorted_grad_pos = sorted(gradient_scores.items(), key=lambda x: x[1], reverse=True)[:10]
                for i, (feat, score) in enumerate(sorted_grad_pos, 1):
                    actual_val = self.actual_feature_values.get(feat, 0)
                    logger.info(f"  {i:2d}. {feat:30s}: Score={score:8.4f}, Actual Value={actual_val:8.2f}")
                
                # Log top 10 negative scores
                logger.info("\nRaw Gradient Scores (Top 10 Negative - Most Detrimental to Performance):")
                sorted_grad_neg = sorted(gradient_scores.items(), key=lambda x: x[1])[:10]
                for i, (feat, score) in enumerate(sorted_grad_neg, 1):
                    actual_val = self.actual_feature_values.get(feat, 0)
                    logger.info(f"  {i:2d}. {feat:30s}: Score={score:8.4f}, Actual Value={actual_val:8.2f}")
            else:
                logger.info("  No features detected via gradients")
                
        except Exception as e:
            logger.error(f"Gradient analysis failed: {e}")
            results['gradient_scores'] = {}
        
        # Calculate z-score normalization
        self.calculate_zscore_normalization(results['gradient_scores'])
        
        return results
    
    def calculate_zscore_normalization(self, gradient_scores):
        """
        Calculate z-score normalization for gradient scores
        """
        if not gradient_scores:
            return {}
        
        logger.info("\n" + "="*70)
        logger.info("🔬 Z-SCORE NORMALIZATION")
        logger.info("="*70)
        
        method_scores = list(gradient_scores.values())
        
        if len(method_scores) > 1:
            mean_score = np.mean(method_scores)
            std_score = np.std(method_scores)
            
            logger.info(f"Mean = {mean_score:.4f}, Std = {std_score:.4f}")
            
            if std_score > 0:
                self.normalized_scores = {}
                
                logger.info(f"\nZ-normalized scores (Top 10):")
                sorted_items = sorted(gradient_scores.items(), key=lambda x: x[1], reverse=True)
                
                for feature, score in sorted_items:
                    z_score = (score - mean_score) / std_score
                    self.normalized_scores[feature] = z_score
                
                # Log top 10 with interpretation
                for feature, score in sorted_items[:10]:
                    z_score = self.normalized_scores[feature]
                    actual_val = self.actual_feature_values.get(feature, 0)
                    if z_score > 1.5:
                        interpretation = "very high importance"
                    elif z_score > 0.5:
                        interpretation = "high importance"
                    elif z_score > 0:
                        interpretation = "medium importance"
                    elif z_score > -0.5:
                        interpretation = "low importance"
                    else:
                        interpretation = "very low importance"
                    logger.info(f"  {feature:30s}: Z={z_score:+7.3f} ({interpretation}), Actual={actual_val:8.2f}")
            else:
                self.normalized_scores = {feature: 0 for feature in gradient_scores}
        else:
            self.normalized_scores = {feature: 0 for feature in gradient_scores}
        
        return self.normalized_scores
    
    def generate_bottleneck_report(self, results, save_path='bottleneck_report.json'):
        """Generate comprehensive bottleneck report"""
        report = {
            'performance': {
                'predicted_mbps': results['prediction'],
                'actual_mbps': results['actual'],
                'error_mbps': abs(results['prediction'] - results['actual']),
                'relative_error_percent': abs(results['prediction'] - results['actual']) / results['actual'] * 100
            },
            'gradient_analysis': {
                'raw_scores': {},
                'normalized_scores': {},
                'top_10_positive': {},
                'top_10_negative': {},
                'actual_feature_values': self.actual_feature_values
            }
        }
        
        # Add all gradient scores
        if results['gradient_scores']:
            # Raw scores
            report['gradient_analysis']['raw_scores'] = results['gradient_scores']
            
            # Normalized scores
            report['gradient_analysis']['normalized_scores'] = self.normalized_scores
            
            # Top 10 positive (most important for high performance)
            sorted_positive = sorted(results['gradient_scores'].items(), 
                                   key=lambda x: x[1], reverse=True)[:10]
            for feat, score in sorted_positive:
                report['gradient_analysis']['top_10_positive'][feat] = {
                    'score': score,
                    'normalized_score': self.normalized_scores.get(feat, 0),
                    'actual_value': self.actual_feature_values.get(feat, 0)
                }
            
            # Top 10 negative (most detrimental to performance)
            sorted_negative = sorted(results['gradient_scores'].items(), 
                                   key=lambda x: x[1])[:10]
            for feat, score in sorted_negative:
                report['gradient_analysis']['top_10_negative'][feat] = {
                    'score': score,
                    'normalized_score': self.normalized_scores.get(feat, 0),
                    'actual_value': self.actual_feature_values.get(feat, 0)
                }
        
        # Identify primary bottleneck (most negative score)
        if results['gradient_scores']:
            # Get the feature with the most negative impact
            sorted_by_negative = sorted(results['gradient_scores'].items(), 
                                       key=lambda x: x[1])
            if sorted_by_negative:
                top_negative_feature = sorted_by_negative[0][0]
                top_negative_score = sorted_by_negative[0][1]
                
                # Also get the most positive for comparison
                sorted_by_positive = sorted(results['gradient_scores'].items(), 
                                           key=lambda x: x[1], reverse=True)
                top_positive_feature = sorted_by_positive[0][0] if sorted_by_positive else None
                
                report['primary_bottleneck'] = {
                    'negative_impact': {
                        'feature': top_negative_feature,
                        'gradient_score': top_negative_score,
                        'actual_value': self.actual_feature_values.get(top_negative_feature, 0),
                        'recommendation': self._get_recommendation(top_negative_feature)
                    }
                }
                
                if top_positive_feature:
                    report['primary_bottleneck']['positive_impact'] = {
                        'feature': top_positive_feature,
                        'gradient_score': sorted_by_positive[0][1],
                        'actual_value': self.actual_feature_values.get(top_positive_feature, 0),
                        'note': 'This feature contributes most positively to performance'
                    }
        
        # Save report
        with open(save_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        logger.info(f"\n✓ Bottleneck report saved to {save_path}")
        
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
            'POSIX_BYTES_WRITTEN': 'Optimize total data written, consider compression',
            'POSIX_CONSEC_WRITES': 'Improve write sequentiality',
            'LUSTRE_STRIPE_SIZE': 'Adjust Lustre stripe size for workload',
            'LUSTRE_STRIPE_WIDTH': 'Optimize Lustre stripe count',
            'POSIX_OPENS': 'Reduce number of file open operations',
            'POSIX_SIZE_READ_0_100': 'Avoid very small reads, batch operations',
            'POSIX_SIZE_READ_100_1K': 'Increase read buffer size',
            'POSIX_SIZE_READ_1K_10K': 'Increase read size to 100KB or larger'
        }
        
        return recommendations.get(feature_name, 'Optimize I/O pattern for better performance')


def main():
    """Main execution function"""
    # Paths
    model_checkpoint = '/work/hdd/bdau/mbanisharifdehkordi/GNN_4_IO_5/data/1M/best_model.pt'
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
            logger.info(f"✓ Found similarity matrix: {path}")
            break
    
    # Training features
    training_features = os.path.join(data_dir, 'aiio_sample_1000000_normalized.csv')
    if not os.path.exists(training_features):
        training_features = None
        logger.warning("⚠ Training features not found")
    
    # IOR test sample
    test_features = '/work/hdd/bdau/mbanisharifdehkordi/GNN_4_IO_5/darshan_log/darshan_log_E2E/Study2/case3/e2e_combined_parsed.csv'
    
    # Initialize analyzer
    logger.info("\n" + "="*70)
    logger.info("🚀 INITIALIZING IOR INTERPRETABILITY ANALYZER")
    logger.info("="*70)
    
    analyzer = IORInterpretabilityAnalyzer(
        model_checkpoint_path=model_checkpoint,
        similarity_graph_path=similarity_matrix,
        features_csv_path=training_features,
        similarity_threshold=0.75,
        use_cpu=True
    )
    
    # Run gradient analysis
    logger.info("\n" + "="*70)
    logger.info("🔬 RUNNING GRADIENT ANALYSIS")
    logger.info("="*70)
    
    results = analyzer.analyze_with_gradient_method(test_features)
    
    # Generate report
    logger.info("\n" + "="*70)
    logger.info("📝 GENERATING BOTTLENECK REPORT")
    logger.info("="*70)
    
    report = analyzer.generate_bottleneck_report(results)
    
    # Print summary
    logger.info("\n" + "="*70)
    logger.info("📋 ANALYSIS SUMMARY")
    logger.info("="*70)
    logger.info(f"Performance: {results['prediction']:.2f} MB/s (predicted) vs {results['actual']:.2f} MB/s (actual)")
    
    if 'primary_bottleneck' in report and 'negative_impact' in report['primary_bottleneck']:
        neg_impact = report['primary_bottleneck']['negative_impact']
        logger.info(f"Primary Bottleneck (Most Negative Impact): {neg_impact['feature']}")
        logger.info(f"  - Gradient Score: {neg_impact['gradient_score']:.4f}")
        logger.info(f"  - Actual Value: {neg_impact['actual_value']:.2f}")
        logger.info(f"  - Recommendation: {neg_impact['recommendation']}")
    
    if 'primary_bottleneck' in report and 'positive_impact' in report['primary_bottleneck']:
        pos_impact = report['primary_bottleneck']['positive_impact']
        logger.info(f"Best Performing Feature: {pos_impact['feature']}")
        logger.info(f"  - Gradient Score: {pos_impact['gradient_score']:.4f}")
        logger.info(f"  - Actual Value: {pos_impact['actual_value']:.2f}")
    
    logger.info("\n" + "="*70)
    logger.info("✅ ANALYSIS COMPLETE!")
    logger.info("="*70)
    logger.info("Output:")
    logger.info("  📝 Report: bottleneck_report.json")


if __name__ == "__main__":
    main()