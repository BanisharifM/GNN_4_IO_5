#!/usr/bin/env python3
"""
Enhanced IOR Performance Bottleneck Analysis using GNN Interpretability Methods
With detailed logging and professional publication-quality visualizations
"""

import pandas as pd
import torch
import numpy as np
import sys
import os
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch
import seaborn as sns
from scipy import stats
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
from src.interpretability.attention_analyzer import AttentionAnalyzer
from src.interpretability.gnn_explainer import IOGNNExplainer
from src.interpretability.gradient_methods import GradientAnalyzer, BottleneckIdentifier

# Setup enhanced logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger(__name__)

# Set matplotlib parameters for publication quality
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['font.size'] = 10
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans']


class IORInterpretabilityAnalyzer:
    """
    Enhanced interpretability analysis with detailed logging and professional visualizations
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
        
        # Initialize interpretability analyzers
        self.attention_analyzer = AttentionAnalyzer(self.model, self.feature_names, self.device)
        self.gnn_explainer = IOGNNExplainer(self.model, device=self.device)
        self.gradient_analyzer = GradientAnalyzer(self.model, self.feature_names, self.device)
        
        # Store for visualization
        self.normalized_scores = {}
    
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
    
    def analyze_with_all_methods(self, features_path):
        """
        Analyze IOR job using all three interpretability methods with enhanced logging
        """
        # Load new sample
        new_data = pd.read_csv(features_path)
        new_features = new_data.iloc[0, :-1].values
        actual_tag = new_data.iloc[0, -1]
        
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
        
        # Run all three interpretability methods
        results = {
            'prediction': predicted_bandwidth,
            'actual': actual_bandwidth,
            'methods': {}
        }
        
        # 1. Attention Analysis
        logger.info("\n" + "="*70)
        logger.info("📊 ATTENTION ANALYSIS")
        logger.info("="*70)
        try:
            attention_scores = self.attention_analyzer.attention_based_bottleneck_detection(
                data, new_node_idx, threshold=0.001
            )
            
            # If no scores, try with even lower threshold
            # If still no scores, extract raw attention weights as fallback
            if not attention_scores:
                logger.info("  Using fallback: extracting raw attention weights")
                try:
                    att_info = self.attention_analyzer.extract_attention_weights(data, new_node_idx)
                    if len(att_info['outgoing_attention']) > 0:
                        attention = att_info['outgoing_attention'].cpu()
                        if attention.dim() > 1:
                            attention = attention.mean(dim=1)
                        attention = attention.numpy()
                        
                        # Use top 10 attention weights as feature importance
                        attention_scores = {}
                        num_features = min(len(attention), len(self.feature_names))
                        
                        # Normalize attention weights
                        attention_normalized = attention[:num_features] / (attention[:num_features].sum() + 1e-10)
                        
                        # Assign to features based on attention strength
                        sorted_indices = np.argsort(attention_normalized)[::-1][:10]
                        for idx in sorted_indices:
                            if idx < len(self.feature_names):
                                attention_scores[self.feature_names[idx]] = float(attention_normalized[idx])
                        
                        logger.info(f"  Extracted {len(attention_scores)} features from raw attention")
                except Exception as e:
                    logger.warning(f"  Fallback attention extraction failed: {e}")
                    attention_scores = {}
            
            results['methods']['attention'] = attention_scores
            
            # Log raw scores
            if attention_scores:
                logger.info("Raw Attention Scores (Top 10):")
                sorted_att = sorted(attention_scores.items(), key=lambda x: x[1], reverse=True)[:10]
                for i, (feat, score) in enumerate(sorted_att, 1):
                    logger.info(f"  {i:2d}. {feat:30s}: {score:8.4f}")
            else:
                logger.info("  No features detected via attention")
                
        except Exception as e:
            logger.error(f"Attention analysis failed: {e}")
            results['methods']['attention'] = {}
        
        # 2. GNNExplainer
        logger.info("\n" + "="*70)
        logger.info("📊 GNNEXPLAINER ANALYSIS")
        logger.info("="*70)
        try:
            # Try with lower threshold if needed
            self.gnn_explainer.feature_mask_threshold = 0.01
            gnn_scores = self.gnn_explainer.explain_bottleneck_pattern(
                data, new_node_idx, self.feature_names
            )
            
            results['methods']['gnn_explainer'] = gnn_scores
            
            # Log raw scores
            if gnn_scores:
                logger.info("Raw GNNExplainer Scores (Top 10):")
                sorted_gnn = sorted(gnn_scores.items(), key=lambda x: x[1], reverse=True)[:10]
                for i, (feat, score) in enumerate(sorted_gnn, 1):
                    logger.info(f"  {i:2d}. {feat:30s}: {score:8.4f}")
            else:
                logger.info("  No features detected via GNNExplainer")
                
        except Exception as e:
            logger.error(f"GNNExplainer failed: {e}")
            results['methods']['gnn_explainer'] = {}
        
        # 3. Gradient Methods
        logger.info("\n" + "="*70)
        logger.info("📊 GRADIENT ANALYSIS (INTEGRATED GRADIENTS)")
        logger.info("="*70)
        try:
            gradient_scores = self.gradient_analyzer.integrated_gradients(
                data, new_node_idx
            )
            results['methods']['gradients'] = gradient_scores
            
            # Log raw scores
            if gradient_scores:
                logger.info("Raw Gradient Scores (Top 10):")
                sorted_grad = sorted(gradient_scores.items(), key=lambda x: x[1], reverse=True)[:10]
                for i, (feat, score) in enumerate(sorted_grad, 1):
                    logger.info(f"  {i:2d}. {feat:30s}: {score:8.4f}")
            else:
                logger.info("  No features detected via gradients")
                
        except Exception as e:
            logger.error(f"Gradient analysis failed: {e}")
            results['methods']['gradients'] = {}
        
        # Calculate z-score normalized consensus
        consensus_scores = self.calculate_zscore_consensus(results['methods'])
        results['consensus'] = consensus_scores
        
        return results
    
    def calculate_zscore_consensus(self, methods_results):
        """
        Calculate consensus using z-score normalization with detailed step-by-step logging
        """
        logger.info("\n" + "="*70)
        logger.info("🔬 Z-SCORE NORMALIZATION AND CONSENSUS CALCULATION")
        logger.info("="*70)
        
        # Step 1: Log raw scores
        logger.info("\n📊 Step 1: Raw Scores from Each Method")
        logger.info("-" * 50)
        
        for method_name, scores in methods_results.items():
            if scores:
                logger.info(f"\n{method_name.upper().replace('_', ' ')}:")
                sorted_scores = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:5]
                for feat, score in sorted_scores:
                    logger.info(f"  - {feat:30s}: {score:8.4f}")
        
        # Step 2: Z-normalize each method
        logger.info("\n📊 Step 2: Z-Score Normalization")
        logger.info("-" * 50)
        logger.info("Formula: z = (x - mean) / std")
        
        normalized_scores = {}
        self.normalized_scores = {}  # Store for visualization
        
        for method_name, scores in methods_results.items():
            if not scores:
                continue
            
            method_scores = list(scores.values())
            
            if len(method_scores) > 1:
                mean_score = np.mean(method_scores)
                std_score = np.std(method_scores)
                
                logger.info(f"\n{method_name.upper().replace('_', ' ')}:")
                logger.info(f"  Mean = {mean_score:.4f}, Std = {std_score:.4f}")
                
                if std_score > 0:
                    normalized_scores[method_name] = {}
                    self.normalized_scores[method_name] = {}
                    
                    # Show top features with z-scores
                    logger.info(f"  Z-normalized scores (Top 5):")
                    sorted_items = sorted(scores.items(), key=lambda x: x[1], reverse=True)
                    
                    for feature, score in sorted_items:
                        z_score = (score - mean_score) / std_score
                        normalized_scores[method_name][feature] = z_score
                        self.normalized_scores[method_name][feature] = z_score
                    
                    # Log top 5 with interpretation
                    for feature, score in sorted_items[:5]:
                        z_score = normalized_scores[method_name][feature]
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
                        logger.info(f"    - {feature:30s}: {z_score:+7.3f} ({interpretation})")
                else:
                    normalized_scores[method_name] = {feature: 0 for feature in scores}
                    self.normalized_scores[method_name] = {feature: 0 for feature in scores}
            else:
                normalized_scores[method_name] = {feature: 0 for feature in scores}
                self.normalized_scores[method_name] = {feature: 0 for feature in scores}
        
        # Step 3: Calculate consensus
        logger.info("\n📊 Step 3: Equal Weight Consensus (1/3 each method)")
        logger.info("-" * 50)
        logger.info("Formula: Consensus = (1/3 × Attention_Z) + (1/3 × GNN_Z) + (1/3 × Gradient_Z)")
        
        all_features = set()
        for method_scores in normalized_scores.values():
            all_features.update(method_scores.keys())
        
        consensus = {}
        feature_contributions = {}
        
        for feature in all_features:
            z_scores = []
            contributors = []
            
            for method_name, method_scores in normalized_scores.items():
                if feature in method_scores:
                    z_score = method_scores[feature]
                    z_scores.append(z_score)
                    method_short = method_name.replace('gnn_explainer', 'GNN').replace('gradients', 'Grad').replace('attention', 'Att')
                    contributors.append(f"{method_short}({z_score:+.2f})")
            
            if z_scores:
                consensus[feature] = np.mean(z_scores)
                feature_contributions[feature] = contributors
        
        # Sort by consensus score
        consensus = dict(sorted(consensus.items(), key=lambda x: x[1], reverse=True))
        
        # Log final consensus rankings
        logger.info("\n🏆 Final Consensus Rankings:")
        for i, (feature, score) in enumerate(list(consensus.items())[:10], 1):
            contributors = " + ".join(feature_contributions[feature])
            num_methods = len(feature_contributions[feature])
            if num_methods > 1:
                consensus_strength = "STRONG (multiple methods agree)"
            else:
                consensus_strength = "WEAK (single method)"
            
            logger.info(f"Rank {i:2d}: {feature:30s} → {score:+7.3f}")
            logger.info(f"         Contributing methods: {contributors}")
            logger.info(f"         Consensus strength: {consensus_strength}")
            if i < 10:
                logger.info("")
        
        return consensus
    
    def create_professional_visualizations(self, results, output_dir='./interpretability_figures'):
        """
        Create multiple types of professional publication-quality visualizations
        """
        os.makedirs(output_dir, exist_ok=True)
        
        # For each method, create multiple chart types
        for method_name in ['attention', 'gnn_explainer', 'gradients']:
            if method_name in results['methods'] and results['methods'][method_name]:
                scores = results['methods'][method_name]
                
                # 1. Horizontal Bar Chart (Classic Academic Style)
                self._create_horizontal_bar_professional(
                    scores,
                    method_name,
                    f"{output_dir}/{method_name}_barh.png"
                )
                
                # 2. Lollipop Chart (Modern Style)
                self._create_lollipop_chart(
                    scores,
                    method_name,
                    f"{output_dir}/{method_name}_lollipop.png"
                )
                
                # 3. Heatmap Style
                self._create_heatmap_single(
                    scores,
                    method_name,
                    f"{output_dir}/{method_name}_heatmap.png"
                )
                
                # Create normalized versions if available
                if method_name in self.normalized_scores:
                    norm_scores = self.normalized_scores[method_name]
                    
                    # Normalized bar chart
                    self._create_diverging_bar(
                        norm_scores,
                        f"{method_name} (Z-Score Normalized)",
                        f"{output_dir}/{method_name}_normalized.png"
                    )
        
        # Create consensus visualizations
        if 'consensus' in results and results['consensus']:
            # 1. Consensus diverging bar
            self._create_consensus_diverging(
                results['consensus'],
                f"{output_dir}/consensus_diverging.png"
            )
            
            # 2. Consensus dot plot
            self._create_dot_plot(
                results,
                f"{output_dir}/consensus_comparison.png"
            )
            
            # 3. Consensus heatmap
            self._create_method_comparison_heatmap(
                self.normalized_scores,
                f"{output_dir}/method_comparison_heatmap.png"
            )

            # 4. Consensus contribution chart
            self._create_consensus_stacked_contribution(
                results,
                f"{output_dir}/consensus_contribution_chart.png"
            )
        
            # 5. Consensus flow diagram
            self._create_consensus_flow_diagram(
                results,
                f"{output_dir}/consensus_flow_diagram.png"
            )

        logger.info(f"\n✓ All visualizations saved to {output_dir}")
    
    def _create_horizontal_bar_professional(self, scores, method_name, save_path, top_k=15):
        """Create professional horizontal bar chart"""
        sorted_features = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:top_k]
        
        if not sorted_features:
            return
        
        features = [item[0] for item in sorted_features]
        values = [item[1] for item in sorted_features]
        
        # Create figure with clean style
        fig, ax = plt.subplots(figsize=(10, 8))
        fig.patch.set_facecolor('white')
        ax.set_facecolor('white')
        
        # Color palette
        colors = plt.cm.viridis(np.linspace(0.3, 0.9, len(features)))
        
        # Create bars
        bars = ax.barh(range(len(features)), values, color=colors, edgecolor='black', linewidth=0.5)
        
        # Customize
        ax.set_yticks(range(len(features)))
        ax.set_yticklabels(features, fontsize=9)
        ax.set_xlabel('Importance Score', fontsize=11, fontweight='bold')
        ax.set_title(f'{method_name.replace("_", " ").title()} - Feature Importance', 
                    fontsize=12, fontweight='bold', pad=20)
        
        # Add value labels
        for i, (bar, value) in enumerate(zip(bars, values)):
            ax.text(value + max(values)*0.01, bar.get_y() + bar.get_height()/2, 
                   f'{value:.3f}', ha='left', va='center', fontsize=8)
        
        # Clean grid
        ax.grid(True, axis='x', alpha=0.2, linestyle='--')
        ax.set_axisbelow(True)
        
        # Remove top and right spines
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
    
    def _create_lollipop_chart(self, scores, method_name, save_path, top_k=15):
        """Create modern lollipop chart"""
        sorted_features = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:top_k]
        
        if not sorted_features:
            return
        
        features = [item[0] for item in sorted_features]
        values = [item[1] for item in sorted_features]
        
        fig, ax = plt.subplots(figsize=(10, 8))
        fig.patch.set_facecolor('white')
        
        # Create lollipops
        y_positions = range(len(features))
        
        # Lines
        for y, val in zip(y_positions, values):
            ax.plot([0, val], [y, y], color='gray', alpha=0.5, linewidth=1.5)
        
        # Dots
        colors = plt.cm.RdYlBu_r(np.linspace(0.2, 0.8, len(features)))
        ax.scatter(values, y_positions, color=colors, s=100, edgecolor='black', 
                  linewidth=0.5, zorder=3)
        
        # Customize
        ax.set_yticks(y_positions)
        ax.set_yticklabels(features, fontsize=9)
        ax.set_xlabel('Importance Score', fontsize=11, fontweight='bold')
        ax.set_title(f'{method_name.replace("_", " ").title()} - Lollipop Chart', 
                    fontsize=12, fontweight='bold', pad=20)
        
        # Add value labels
        for y, val in zip(y_positions, values):
            ax.text(val + max(values)*0.02, y, f'{val:.3f}', 
                   va='center', fontsize=8)
        
        # Clean style
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_visible(False)
        ax.grid(True, axis='x', alpha=0.2, linestyle='--')
        ax.set_axisbelow(True)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
    
    def _create_heatmap_single(self, scores, method_name, save_path, top_k=15):
        """Create single-method heatmap"""
        sorted_features = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:top_k]
        
        if not sorted_features:
            return
        
        features = [item[0] for item in sorted_features]
        values = [[item[1]] for item in sorted_features]
        
        fig, ax = plt.subplots(figsize=(4, 10))
        
        # Create heatmap
        im = ax.imshow(values, cmap='YlOrRd', aspect='auto')
        
        # Customize
        ax.set_yticks(range(len(features)))
        ax.set_yticklabels(features, fontsize=9)
        ax.set_xticks([0])
        ax.set_xticklabels(['Importance'], fontsize=10)
        ax.set_title(f'{method_name.replace("_", " ").title()} - Heatmap', 
                    fontsize=12, fontweight='bold', pad=20)
        
        # Add text annotations
        for i, val in enumerate(values):
            ax.text(0, i, f'{val[0]:.3f}', ha='center', va='center', 
                   color='white' if val[0] > np.mean([v[0] for v in values]) else 'black',
                   fontsize=9, fontweight='bold')
        
        # Colorbar
        cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label('Score', fontsize=10)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
    
    def _create_diverging_bar(self, scores, title, save_path, top_k=15):
        """Create diverging bar chart for z-scores"""
        sorted_features = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:top_k]
        
        if not sorted_features:
            return
        
        features = [item[0] for item in sorted_features]
        values = [item[1] for item in sorted_features]
        
        fig, ax = plt.subplots(figsize=(10, 8))
        fig.patch.set_facecolor('white')
        
        # Colors based on positive/negative
        colors = ['#2E7D32' if v > 0 else '#C62828' for v in values]
        
        # Create bars
        bars = ax.barh(range(len(features)), values, color=colors, alpha=0.7, 
                      edgecolor='black', linewidth=0.5)
        
        # Customize
        ax.set_yticks(range(len(features)))
        ax.set_yticklabels(features, fontsize=9)
        ax.set_xlabel('Z-Score', fontsize=11, fontweight='bold')
        ax.set_title(title, fontsize=12, fontweight='bold', pad=20)
        
        # Add center line
        ax.axvline(x=0, color='black', linestyle='-', linewidth=1)
        
        # Add value labels
        for bar, val in zip(bars, values):
            x_pos = val + (0.05 if val > 0 else -0.05)
            ha = 'left' if val > 0 else 'right'
            ax.text(x_pos, bar.get_y() + bar.get_height()/2, 
                   f'{val:.2f}', ha=ha, va='center', fontsize=8, fontweight='bold')
        
        # Grid
        ax.grid(True, axis='x', alpha=0.2, linestyle='--')
        ax.set_axisbelow(True)
        
        # Remove spines
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
    
    def _create_consensus_diverging(self, consensus_scores, save_path, top_k=15):
        """Create special consensus visualization with diverging bars"""
        sorted_features = sorted(consensus_scores.items(), key=lambda x: x[1], reverse=True)[:top_k]
        
        if not sorted_features:
            return
        
        features = [item[0] for item in sorted_features]
        values = [item[1] for item in sorted_features]
        
        fig, ax = plt.subplots(figsize=(12, 9))
        fig.patch.set_facecolor('#F5F5F5')
        ax.set_facecolor('white')
        
        # Create gradient colors
        norm = plt.Normalize(vmin=min(values), vmax=max(values))
        colors = plt.cm.RdYlGn_r(norm(values))
        
        # Create bars with border
        bars = ax.barh(range(len(features)), values, color=colors, 
                      edgecolor='black', linewidth=1.5, alpha=0.9)
        
        # Customize
        ax.set_yticks(range(len(features)))
        ax.set_yticklabels(features, fontsize=10, fontweight='bold')
        ax.set_xlabel('Z-Score Normalized Consensus', fontsize=12, fontweight='bold')
        ax.set_title('🏆 Consensus Feature Importance Rankings', 
                    fontsize=14, fontweight='bold', pad=20)
        
        # Add center line
        ax.axvline(x=0, color='black', linestyle='--', linewidth=1.5, alpha=0.5)
        
        # Add value labels with rank
        for i, (bar, val) in enumerate(zip(bars, values)):
            # Rank label
            ax.text(-0.5, bar.get_y() + bar.get_height()/2, 
                   f'#{i+1}', ha='right', va='center', fontsize=10, 
                   fontweight='bold', color='#333333')
            # Value label
            x_pos = val + max(abs(min(values)), max(values))*0.02
            ax.text(x_pos, bar.get_y() + bar.get_height()/2, 
                   f'{val:.3f}', ha='left', va='center', fontsize=9, fontweight='bold')
        
        # Grid
        ax.grid(True, axis='x', alpha=0.3, linestyle='--', color='gray')
        ax.set_axisbelow(True)
        
        # Style
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_linewidth(2)
        ax.spines['bottom'].set_linewidth(2)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='#F5F5F5')
        plt.close()
    
    def _create_dot_plot(self, results, save_path, top_k=10):
        """Create dot plot comparing all methods"""
        # Get top consensus features
        if 'consensus' not in results:
            return
        
        top_features = list(results['consensus'].keys())[:top_k]
        
        fig, ax = plt.subplots(figsize=(12, 8))
        fig.patch.set_facecolor('white')
        
        methods = ['attention', 'gnn_explainer', 'gradients']
        method_colors = {'attention': '#1f77b4', 'gnn_explainer': '#ff7f0e', 'gradients': '#2ca02c'}
        
        y_positions = list(range(len(top_features)))
        
        for method in methods:
            if method in self.normalized_scores:
                scores = self.normalized_scores[method]
                x_values = []
                y_values = []
                
                for i, feature in enumerate(top_features):
                    if feature in scores:
                        x_values.append(scores[feature])
                        y_values.append(i)
                
                ax.scatter(x_values, y_values, label=method.replace('_', ' ').title(), 
                          color=method_colors[method], s=100, alpha=0.7, 
                          edgecolor='black', linewidth=0.5)
        
        # Customize
        ax.set_yticks(y_positions)
        ax.set_yticklabels(top_features, fontsize=9)
        ax.set_xlabel('Z-Score', fontsize=11, fontweight='bold')
        ax.set_title('Method Comparison - Top Features', fontsize=12, fontweight='bold', pad=20)
        
        # Add zero line
        ax.axvline(x=0, color='black', linestyle='--', alpha=0.5)
        
        # Legend
        ax.legend(loc='best', framealpha=0.9)
        
        # Grid and style
        ax.grid(True, alpha=0.2, linestyle='--')
        ax.set_axisbelow(True)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
    
    def _create_method_comparison_heatmap(self, normalized_scores, save_path, top_k=15):
        """Create heatmap comparing all methods"""
        if not normalized_scores:
            return
        
        # Get all features that appear in consensus
        all_features = set()
        for method_scores in normalized_scores.values():
            all_features.update(method_scores.keys())
        
        # Sort by total absolute z-score
        feature_totals = {}
        for feature in all_features:
            total = 0
            for method_scores in normalized_scores.values():
                if feature in method_scores:
                    total += abs(method_scores[feature])
            feature_totals[feature] = total
        
        top_features = sorted(feature_totals.keys(), 
                             key=lambda x: feature_totals[x], reverse=True)[:top_k]
        
        # Create matrix
        methods = list(normalized_scores.keys())
        matrix = []
        
        for feature in top_features:
            row = []
            for method in methods:
                if feature in normalized_scores[method]:
                    row.append(normalized_scores[method][feature])
                else:
                    row.append(0)
            matrix.append(row)
        
        matrix = np.array(matrix)
        
        # Create figure
        fig, ax = plt.subplots(figsize=(8, 10))
        
        # Create heatmap
        im = ax.imshow(matrix, cmap='RdBu_r', aspect='auto', vmin=-2, vmax=2)
        
        # Set ticks
        ax.set_xticks(range(len(methods)))
        ax.set_xticklabels([m.replace('_', ' ').title() for m in methods], fontsize=10)
        ax.set_yticks(range(len(top_features)))
        ax.set_yticklabels(top_features, fontsize=9)
        
        # Add text annotations
        for i in range(len(top_features)):
            for j in range(len(methods)):
                value = matrix[i, j]
                if value != 0:
                    color = 'white' if abs(value) > 1 else 'black'
                    ax.text(j, i, f'{value:.2f}', ha='center', va='center', 
                           color=color, fontsize=8, fontweight='bold')
        
        # Title and labels
        ax.set_title('Method Comparison Heatmap (Z-Scores)', fontsize=12, fontweight='bold', pad=20)
        
        # Colorbar
        cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label('Z-Score', fontsize=10)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
    
    def generate_bottleneck_report(self, results, save_path='bottleneck_report.json'):
        """Generate comprehensive bottleneck report"""
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
        
        logger.info(f"✓ Bottleneck report saved to {save_path}")
        
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
            'POSIX_OPENS': 'Reduce number of file open operations'
        }
        
        return recommendations.get(feature_name, 'Optimize I/O pattern for better performance')
    
    def _create_consensus_stacked_contribution(self, results, save_path, top_k=10):
        """Create stacked bar chart showing individual method contributions and agreement strength"""
        
        if 'consensus' not in results or not results['consensus']:
            return
        
        # Get top consensus features
        top_features = list(results['consensus'].keys())[:top_k]
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8), gridspec_kw={'width_ratios': [3, 1]})
        fig.patch.set_facecolor('white')
        
        # Prepare data for stacking
        methods = ['attention', 'gnn_explainer', 'gradients']
        method_colors = {
            'attention': '#3498db',      # Blue
            'gnn_explainer': '#2ecc71',  # Green  
            'gradients': '#e74c3c'       # Red
        }
        
        # Collect contributions for each feature
        feature_contributions = []
        agreement_counts = []
        
        for feature in top_features:
            contributions = {}
            count = 0
            
            for method in methods:
                if method in self.normalized_scores and feature in self.normalized_scores[method]:
                    z_score = self.normalized_scores[method][feature]
                    contributions[method] = z_score
                    count += 1
                else:
                    contributions[method] = 0
            
            feature_contributions.append(contributions)
            agreement_counts.append(count)
        
        # Create stacked bars (left plot)
        y_positions = np.arange(len(top_features))
        bar_height = 0.6
        
        # Calculate cumulative positions for stacking
        for method in methods:
            values = [fc.get(method, 0) for fc in feature_contributions]
            colors = [method_colors[method] if v != 0 else 'lightgray' for v in values]
            
            # Plot positive and negative separately
            pos_values = [max(0, v) for v in values]
            neg_values = [min(0, v) for v in values]
            
            if method == methods[0]:
                pos_left = np.zeros(len(top_features))
                neg_left = np.zeros(len(top_features))
            
            # Positive bars
            bars_pos = ax1.barh(y_positions, pos_values, bar_height, 
                            left=pos_left, color=method_colors[method],
                            alpha=0.8, label=method.replace('_', ' ').title())
            
            # Negative bars
            bars_neg = ax1.barh(y_positions, neg_values, bar_height,
                            left=neg_left, color=method_colors[method],
                            alpha=0.8)
            
            # Update positions
            pos_left += np.array(pos_values)
            neg_left += np.array(neg_values)
            
            # Add value labels inside bars
            for i, (pos_val, neg_val) in enumerate(zip(pos_values, neg_values)):
                if pos_val > 0.1:  # Only show if significant
                    ax1.text(pos_left[i] - pos_val/2, i, f'{pos_val:.1f}',
                            ha='center', va='center', color='white', fontsize=8, fontweight='bold')
                if neg_val < -0.1:
                    ax1.text(neg_left[i] - neg_val/2, i, f'{neg_val:.1f}',
                            ha='center', va='center', color='white', fontsize=8, fontweight='bold')
        
        # Add center line
        ax1.axvline(x=0, color='black', linestyle='-', linewidth=1.5)
        
        # Add consensus scores at the end
        for i, (feature, score) in enumerate(list(results['consensus'].items())[:top_k]):
            ax1.text(max(pos_left[i], abs(neg_left[i])) + 0.1, i,
                    f'Σ = {score:.2f}', ha='left', va='center', fontsize=10, fontweight='bold')
        
        # Customize left plot
        ax1.set_yticks(y_positions)
        ax1.set_yticklabels(top_features, fontsize=10)
        ax1.set_xlabel('Z-Score Contribution', fontsize=11, fontweight='bold')
        ax1.set_title('Method Contributions to Consensus', fontsize=13, fontweight='bold')
        ax1.legend(loc='upper right', framealpha=0.9)
        ax1.grid(True, axis='x', alpha=0.2, linestyle='--')
        ax1.spines['top'].set_visible(False)
        ax1.spines['right'].set_visible(False)
        
        # Agreement strength visualization (right plot)
        agreement_colors = []
        agreement_labels = []
        
        for count in agreement_counts:
            if count == 3:
                agreement_colors.append('#27ae60')  # Green - strong
                agreement_labels.append('✓✓✓')
            elif count == 2:
                agreement_colors.append('#f39c12')  # Orange - medium
                agreement_labels.append('✓✓')
            else:
                agreement_colors.append('#e74c3c')  # Red - weak
                agreement_labels.append('✓')
        
        bars = ax2.barh(y_positions, agreement_counts, bar_height,
                        color=agreement_colors, alpha=0.8)
        
        # Add agreement indicators
        for i, (bar, label, count) in enumerate(zip(bars, agreement_labels, agreement_counts)):
            ax2.text(bar.get_width()/2, bar.get_y() + bar.get_height()/2,
                    f'{count}\n{label}', ha='center', va='center',
                    fontsize=10, fontweight='bold', color='white')
        
        # Customize right plot
        ax2.set_yticks([])
        ax2.set_xlim(0, 3.5)
        ax2.set_xticks([1, 2, 3])
        ax2.set_xticklabels(['1', '2', '3'])
        ax2.set_xlabel('# Methods', fontsize=11, fontweight='bold')
        ax2.set_title('Agreement\nStrength', fontsize=13, fontweight='bold')
        ax2.spines['top'].set_visible(False)
        ax2.spines['right'].set_visible(False)
        ax2.spines['left'].set_visible(False)
        
        # Add legend for agreement strength
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor='#27ae60', alpha=0.8, label='Strong (3 methods)'),
            Patch(facecolor='#f39c12', alpha=0.8, label='Medium (2 methods)'),
            Patch(facecolor='#e74c3c', alpha=0.8, label='Weak (1 method)')
        ]
        ax2.legend(handles=legend_elements, loc='lower right', fontsize=8)
        
        # plt.suptitle('🏆 Consensus Analysis with Method Agreement', fontsize=14, fontweight='bold', y=1.02)
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        
        logger.info(f"Saved consensus contribution chart to {save_path}")    

    def _create_consensus_flow_diagram(self, results, save_path, top_k=5):
        """Create a flow diagram showing method contributions flowing into consensus"""
        
        if 'consensus' not in results:
            return
            
        fig, ax = plt.subplots(figsize=(14, 10))
        fig.patch.set_facecolor('#f8f9fa')
        ax.set_facecolor('white')
        
        top_features = list(results['consensus'].keys())[:top_k]
        
        # Define positions
        y_positions = np.linspace(0.8, 0.2, len(top_features))
        method_x = 0.2
        consensus_x = 0.8
        
        methods = ['attention', 'gnn_explainer', 'gradients']
        method_colors = {'attention': '#3498db', 'gnn_explainer': '#2ecc71', 'gradients': '#e74c3c'}
        
        for i, feature in enumerate(top_features):
            y_pos = y_positions[i]
            
            # Draw feature name in center
            ax.text(0.5, y_pos, feature, ha='center', va='center',
                    fontsize=11, fontweight='bold', bbox=dict(boxstyle="round,pad=0.3", 
                    facecolor='white', edgecolor='gray'))
            
            # Draw connections from methods
            for j, method in enumerate(methods):
                if method in self.normalized_scores and feature in self.normalized_scores[method]:
                    z_score = self.normalized_scores[method][feature]
                    
                    # Line thickness based on absolute z-score
                    linewidth = min(abs(z_score) * 2, 8)
                    alpha = min(0.3 + abs(z_score) * 0.2, 0.9)
                    
                    # Draw curved line
                    ax.annotate('', xy=(0.45, y_pos), xytext=(method_x, y_pos + (j-1)*0.05),
                            arrowprops=dict(arrowstyle='-', lw=linewidth,
                                            color=method_colors[method], alpha=alpha,
                                            connectionstyle="arc3,rad=0.3"))
                    
                    # Add z-score label
                    ax.text(method_x - 0.05, y_pos + (j-1)*0.05, f'{z_score:+.1f}',
                        ha='right', va='center', fontsize=8, color=method_colors[method],
                        fontweight='bold')
            
            # Draw arrow to consensus
            consensus_score = results['consensus'][feature]
            ax.annotate('', xy=(consensus_x, y_pos), xytext=(0.55, y_pos),
                    arrowprops=dict(arrowstyle='->', lw=3, color='black', alpha=0.7))
            
            # Add consensus score
            ax.text(consensus_x + 0.05, y_pos, f'Σ = {consensus_score:.2f}',
                ha='left', va='center', fontsize=11, fontweight='bold')
        
        # Add method labels
        for j, method in enumerate(methods):
            ax.text(method_x - 0.1, 0.9, method.replace('_', '\n').title(),
                ha='center', va='center', fontsize=10, color=method_colors[method],
                fontweight='bold')
        
        # Add titles
        ax.text(0.5, 0.95, 'Features', ha='center', fontsize=12, fontweight='bold')
        ax.text(consensus_x + 0.05, 0.95, 'Consensus', ha='center', fontsize=12, fontweight='bold')
        
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis('off')
        
        plt.title('Method Contributions Flow to Consensus', fontsize=14, fontweight='bold', pad=20)
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='#f8f9fa')
        plt.close()

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
    test_features = '/work/hdd/bdau/mbanisharifdehkordi/GNN_4_IO_5/darshan_features_ior_normalized_baseline_read.csv'
    
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
    
    # Run comprehensive analysis
    logger.info("\n" + "="*70)
    logger.info("🔬 RUNNING INTERPRETABILITY ANALYSIS")
    logger.info("="*70)
    
    results = analyzer.analyze_with_all_methods(test_features)
    
    # Generate visualizations
    logger.info("\n" + "="*70)
    logger.info("📊 GENERATING PROFESSIONAL VISUALIZATIONS")
    logger.info("="*70)
    
    analyzer.create_professional_visualizations(results)
    
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
    logger.info(f"Primary Bottleneck: {report.get('primary_bottleneck', {}).get('feature', 'Unknown')}")
    logger.info(f"Recommendation: {report.get('primary_bottleneck', {}).get('recommendation', 'N/A')}")
    
    logger.info("\n" + "="*70)
    logger.info("✅ ANALYSIS COMPLETE!")
    logger.info("="*70)
    logger.info("Outputs:")
    logger.info("  📊 Visualizations: ./interpretability_figures/")
    logger.info("  📝 Report: bottleneck_report.json")
    logger.info("\nVisualization types created for each method:")
    logger.info("  • Horizontal bar chart (classic academic style)")
    logger.info("  • Lollipop chart (modern style)")
    logger.info("  • Heatmap (intensity visualization)")
    logger.info("  • Diverging bar chart (z-score normalized)")
    logger.info("\nConsensus visualizations:")
    logger.info("  • Consensus diverging bar with rankings")
    logger.info("  • Method comparison dot plot")
    logger.info("  • Method comparison heatmap")


if __name__ == "__main__":
    main()