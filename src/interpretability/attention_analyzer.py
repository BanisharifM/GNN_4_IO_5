"""
Attention weight analysis for identifying important neighbors and patterns
"""

import torch
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict
import logging
import networkx as nx
from pathlib import Path

logger = logging.getLogger(__name__)


class AttentionAnalyzer:
    """
    Analyze GAT attention weights to understand I/O performance patterns
    """
    
    def __init__(
        self,
        model,
        feature_names: List[str],
        device: torch.device = None
    ):
        """
        Args:
            model: Trained GAT model
            feature_names: List of I/O counter names
            device: Computing device
        """
        self.model = model
        self.feature_names = feature_names
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Storage for attention patterns
        self.attention_cache = {}
        self.neighbor_importance = defaultdict(list)
        
    @torch.no_grad()
    def extract_attention_weights(
        self,
        data,
        node_idx: int,
        layer: int = -1
    ) -> Dict[str, torch.Tensor]:
        """
        Extract attention weights for a specific node
        
        Args:
            data: Graph data
            node_idx: Target node index
            layer: Which GAT layer to analyze (-1 for last)
        
        Returns:
            Dictionary with attention information
        """
        self.model.eval()
        
        # Move data to device
        if data.x.device != self.device:
            data = data.to(self.device)
        
        # Forward pass with attention
        _, attention_weights = self.model(
            data.x, data.edge_index, data.edge_attr,
            return_attention=True
        )
        
        if layer == -1:
            layer = len(attention_weights) - 1
        
        # Get attention for specific layer
        edge_index, attention = attention_weights[layer]
        
        # Find edges involving target node
        source_mask = edge_index[0] == node_idx
        target_mask = edge_index[1] == node_idx
        
        # Outgoing attention (node -> neighbors)
        outgoing_edges = edge_index[:, source_mask]
        outgoing_attention = attention[source_mask]
        
        # Incoming attention (neighbors -> node)
        incoming_edges = edge_index[:, target_mask]
        incoming_attention = attention[target_mask]
        
        return {
            'node_idx': node_idx,
            'layer': layer,
            'outgoing_neighbors': outgoing_edges[1],
            'outgoing_attention': outgoing_attention,
            'incoming_neighbors': incoming_edges[0],
            'incoming_attention': incoming_attention,
            'prediction': self.model(data.x, data.edge_index, data.edge_attr)[node_idx]
        }
    
    def analyze_attention_patterns(
        self,
        data,
        node_indices: List[int],
        layer: int = -1
    ) -> pd.DataFrame:
        """
        Analyze attention patterns for multiple nodes
        
        Args:
            data: Graph data
            node_indices: List of nodes to analyze
            layer: Which layer to analyze
        
        Returns:
            DataFrame with attention statistics
        """
        results = []
        
        for node_idx in node_indices:
            att_info = self.extract_attention_weights(data, node_idx, layer)
            
            # Calculate statistics
            if len(att_info['outgoing_attention']) > 0:
                out_att = att_info['outgoing_attention'].cpu()
                
                # Multi-head attention: average across heads
                if out_att.dim() > 1:
                    out_att = out_att.mean(dim=1)
                
                results.append({
                    'node_idx': node_idx,
                    'num_neighbors': len(att_info['outgoing_neighbors']),
                    'max_attention': out_att.max().item(),
                    'min_attention': out_att.min().item(),
                    'mean_attention': out_att.mean().item(),
                    'std_attention': out_att.std().item(),
                    'entropy': self._calculate_entropy(out_att),
                    'top1_neighbor': att_info['outgoing_neighbors'][out_att.argmax()].item(),
                    'top1_attention': out_att.max().item(),
                    'prediction': att_info['prediction'].item(),
                    'actual': data.y[node_idx].item() if data.y is not None else None
                })
        
        df = pd.DataFrame(results)
        
        # Add prediction error if we have ground truth
        if 'actual' in df.columns and df['actual'].notna().all():
            df['error'] = df['prediction'] - df['actual']
            df['abs_error'] = df['error'].abs()
        
        return df
    
    def identify_influential_neighbors(
        self,
        data,
        node_idx: int,
        top_k: int = 10
    ) -> pd.DataFrame:
        """
        Identify most influential neighbors for a node's prediction
        
        Args:
            data: Graph data
            node_idx: Target node
            top_k: Number of top neighbors to return
        
        Returns:
            DataFrame with influential neighbors
        """
        att_info = self.extract_attention_weights(data, node_idx)
        
        if len(att_info['outgoing_attention']) == 0:
            return pd.DataFrame()
        
        # Get attention weights
        neighbors = att_info['outgoing_neighbors'].cpu().numpy()
        attention = att_info['outgoing_attention'].cpu()
        
        # Handle multi-head attention
        if attention.dim() > 1:
            attention = attention.mean(dim=1)
        
        attention = attention.numpy()
        
        # Sort by attention
        sorted_idx = np.argsort(attention)[::-1][:top_k]
        
        # Create DataFrame
        influential = []
        for idx in sorted_idx:
            neighbor_idx = neighbors[idx]
            influential.append({
                'neighbor_idx': neighbor_idx,
                'attention_weight': attention[idx],
                'neighbor_performance': data.y[neighbor_idx].item() if data.y is not None else None,
                'similarity': data.edge_attr[idx].item() if data.edge_attr is not None else None
            })
        
        df = pd.DataFrame(influential)
        
        # Add feature differences
        if data.x is not None:
            target_features = data.x[node_idx].cpu().numpy()
            for _, row in df.iterrows():
                neighbor_features = data.x[int(row['neighbor_idx'])].cpu().numpy()
                feature_diff = np.abs(target_features - neighbor_features)
                
                # Find top differing features
                top_diff_idx = np.argsort(feature_diff)[::-1][:5]
                row['top_diff_features'] = [self.feature_names[i] for i in top_diff_idx]
                row['top_diff_values'] = feature_diff[top_diff_idx].tolist()
        
        return df
    
    def attention_based_bottleneck_detection(
        self,
        data,
        node_idx: int,
        threshold: float = 0.01
    ) -> Dict[str, float]:
        """
        Detect I/O bottlenecks based on attention-weighted neighbor analysis
        
        Args:
            data: Graph data
            node_idx: Target node
            threshold: Minimum attention weight to consider
        
        Returns:
            Dictionary of feature importance scores
        """
        att_info = self.extract_attention_weights(data, node_idx)
        
        if len(att_info['outgoing_attention']) == 0:
            return {}
        
        neighbors = att_info['outgoing_neighbors'].cpu().numpy()
        attention = att_info['outgoing_attention'].cpu()
        
        # Handle multi-head attention
        if attention.dim() > 1:
            attention = attention.mean(dim=1)
        attention = attention.numpy()
        
        # Filter by threshold
        mask = attention > threshold
        important_neighbors = neighbors[mask]
        important_attention = attention[mask]
        
        # Normalize attention
        important_attention = important_attention / important_attention.sum()
        
        # Calculate weighted feature importance
        feature_importance = np.zeros(len(self.feature_names))
        
        target_features = data.x[node_idx].cpu().numpy()
        target_performance = data.y[node_idx].item() if data.y is not None else 0
        
        for neighbor_idx, att_weight in zip(important_neighbors, important_attention):
            neighbor_features = data.x[neighbor_idx].cpu().numpy()
            neighbor_performance = data.y[neighbor_idx].item() if data.y is not None else 0
            
            # If neighbor has worse performance, its different features might be bottlenecks
            if neighbor_performance < target_performance:
                feature_diff = np.abs(target_features - neighbor_features)
                # Ensure all are 1D arrays
                feature_diff = feature_diff.flatten() if feature_diff.ndim > 1 else feature_diff
                # Calculate importance contribution
                importance_contrib = feature_diff * float(att_weight) * float(target_performance - neighbor_performance)
                feature_importance = feature_importance + importance_contrib
        
        # Normalize and create dictionary
        if feature_importance.sum() > 0:
            feature_importance = feature_importance / feature_importance.sum()
        
        bottlenecks = {
            self.feature_names[i]: feature_importance[i]
            for i in range(len(self.feature_names))
            if feature_importance[i] > 0.01  # Only significant features
        }
        
        # Sort by importance
        bottlenecks = dict(sorted(bottlenecks.items(), key=lambda x: x[1], reverse=True))
        
        return bottlenecks
    
    def visualize_attention_graph(
        self,
        data,
        node_idx: int,
        save_path: Optional[str] = None,
        layout: str = 'spring'
    ):
        """
        Visualize attention weights as a graph
        
        Args:
            data: Graph data
            node_idx: Central node
            save_path: Path to save figure
            layout: Graph layout algorithm
        """
        att_info = self.extract_attention_weights(data, node_idx)
        
        if len(att_info['outgoing_attention']) == 0:
            logger.warning(f"No attention weights for node {node_idx}")
            return
        
        # Create graph
        G = nx.DiGraph()
        
        # Add central node
        G.add_node(node_idx, color='red', size=1000)
        
        # Add neighbors and edges
        neighbors = att_info['outgoing_neighbors'].cpu().numpy()
        attention = att_info['outgoing_attention'].cpu()
        
        if attention.dim() > 1:
            attention = attention.mean(dim=1)
        attention = attention.numpy()
        
        for neighbor, att_weight in zip(neighbors, attention):
            G.add_node(neighbor, color='lightblue', size=500)
            G.add_edge(node_idx, neighbor, weight=att_weight, width=att_weight*10)
        
        # Layout
        if layout == 'spring':
            pos = nx.spring_layout(G, k=2, iterations=50)
        elif layout == 'circular':
            pos = nx.circular_layout(G)
        else:
            pos = nx.kamada_kawai_layout(G)
        
        # Plot
        plt.figure(figsize=(12, 8))
        
        # Draw nodes
        node_colors = [G.nodes[node].get('color', 'lightblue') for node in G.nodes()]
        node_sizes = [G.nodes[node].get('size', 500) for node in G.nodes()]
        
        nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=node_sizes, alpha=0.9)
        
        # Draw edges with varying width based on attention
        edges = G.edges()
        weights = [G[u][v]['width'] for u, v in edges]
        
        nx.draw_networkx_edges(G, pos, width=weights, alpha=0.6, edge_color='gray', arrows=True)
        
        # Labels
        labels = {node: str(node) for node in G.nodes()}
        nx.draw_networkx_labels(G, pos, labels, font_size=10)
        
        # Edge labels (attention weights)
        edge_labels = {(u, v): f"{G[u][v]['weight']:.3f}" for u, v in G.edges()}
        nx.draw_networkx_edge_labels(G, pos, edge_labels, font_size=8)
        
        plt.title(f'Attention Graph for Node {node_idx}')
        plt.axis('off')
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            logger.info(f"Saved attention graph to {save_path}")
        
        plt.show()
    
    def attention_heatmap(
        self,
        data,
        node_indices: List[int],
        layer: int = -1,
        save_path: Optional[str] = None
    ):
        """
        Create heatmap of attention patterns
        
        Args:
            data: Graph data
            node_indices: Nodes to analyze
            layer: Which layer
            save_path: Save path
        """
        # Collect attention matrices
        attention_matrix = []
        
        for node_idx in node_indices:
            att_info = self.extract_attention_weights(data, node_idx, layer)
            
            if len(att_info['outgoing_attention']) > 0:
                att = att_info['outgoing_attention'].cpu()
                if att.dim() > 1:
                    att = att.mean(dim=1)
                
                # Pad to fixed size (50 neighbors)
                att_padded = torch.zeros(50)
                att_padded[:min(50, len(att))] = att[:50]
                attention_matrix.append(att_padded.numpy())
        
        if not attention_matrix:
            logger.warning("No attention weights to visualize")
            return
        
        attention_matrix = np.array(attention_matrix)
        
        # Plot heatmap
        plt.figure(figsize=(12, 8))
        sns.heatmap(
            attention_matrix,
            cmap='YlOrRd',
            xticklabels=range(50),
            yticklabels=node_indices,
            cbar_kws={'label': 'Attention Weight'}
        )
        
        plt.xlabel('Neighbor Index')
        plt.ylabel('Node Index')
        plt.title(f'Attention Patterns (Layer {layer})')
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            logger.info(f"Saved attention heatmap to {save_path}")
        
        plt.show()
    
    def _calculate_entropy(self, attention: torch.Tensor) -> float:
        """Calculate entropy of attention distribution"""
        attention = attention.float()
        entropy = -(attention * torch.log(attention + 1e-10)).sum()
        return entropy.item()