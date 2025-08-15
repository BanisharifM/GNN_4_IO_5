"""
GNNExplainer implementation for identifying important subgraphs
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional, Tuple, Dict, List
from torch_geometric.explain import Explainer, GNNExplainer as PyGExplainer
import matplotlib.pyplot as plt
import networkx as nx
import logging

logger = logging.getLogger(__name__)


class IOGNNExplainer:
    """
    GNNExplainer adapted for I/O performance prediction
    """
    
    def __init__(
        self,
        model: nn.Module,
        epochs: int = 200,
        lr: float = 0.01,
        edge_mask_threshold: float = 0.3,
        feature_mask_threshold: float = 0.1,
        device: torch.device = None
    ):
        """
        Args:
            model: Trained GAT model
            epochs: Training epochs for explanation
            lr: Learning rate
            edge_mask_threshold: Threshold for edge importance
            feature_mask_threshold: Threshold for feature importance
            device: Computing device
        """
        self.model = model
        self.epochs = epochs
        self.lr = lr
        self.edge_mask_threshold = edge_mask_threshold
        self.feature_mask_threshold = feature_mask_threshold
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Initialize PyG explainer
        self.explainer = Explainer(
            model=model,
            algorithm=PyGExplainer(epochs=epochs, lr=lr),
            explanation_type='model',
            node_mask_type='attributes',
            edge_mask_type='object',
            model_config=dict(
                mode='regression',
                task_level='node',
                return_type='raw'
            )
        )
    
    def explain_node(
        self,
        data,
        node_idx: int,
        num_hops: int = 2
    ) -> Dict[str, torch.Tensor]:
        """
        Explain prediction for a specific node
        
        Args:
            data: Graph data
            node_idx: Target node
            num_hops: Number of hops for subgraph
        
        Returns:
            Dictionary with edge and feature masks
        """
        # Move data to device
        if data.x.device != self.device:
            data = data.to(self.device)
        
        # Get k-hop subgraph
        subset, edge_index, mapping, edge_mask = self._k_hop_subgraph(
            node_idx, num_hops, data.edge_index, relabel_nodes=True,
            num_nodes=data.x.size(0)
        )
        
        # Extract subgraph features
        x_subset = data.x[subset]
        edge_attr_subset = data.edge_attr[edge_mask] if data.edge_attr is not None else None
        
        # Target node in subgraph
        target_idx = mapping
        
        # Get explanation
        explanation = self.explainer(
            x=x_subset,
            edge_index=edge_index,
            index=target_idx,
            edge_attr=edge_attr_subset
        )
        
        # Extract masks
        edge_mask = explanation.edge_mask
        node_mask = explanation.node_mask
        
        # Apply thresholds
        important_edges = edge_mask > self.edge_mask_threshold
        important_features = node_mask > self.feature_mask_threshold if node_mask is not None else None
        
        return {
            'node_idx': node_idx,
            'subset': subset,
            'edge_index': edge_index,
            'edge_mask': edge_mask,
            'node_mask': node_mask,
            'important_edges': important_edges,
            'important_features': important_features,
            'prediction': self.model(data.x, data.edge_index, data.edge_attr)[node_idx]
        }
    
    def explain_bottleneck_pattern(
        self,
        data,
        node_idx: int,
        feature_names: List[str]
    ) -> Dict[str, float]:
        """
        Identify I/O bottleneck patterns using GNNExplainer
        
        Args:
            data: Graph data
            node_idx: Target node
            feature_names: List of feature names
        
        Returns:
            Dictionary of feature importance scores
        """
        explanation = self.explain_node(data, node_idx)
        
        bottlenecks = {}
        
        # Feature-level bottlenecks
        if explanation['node_mask'] is not None:
            feature_importance = explanation['node_mask'][0].cpu().numpy()  # First node is target
            
            for i, importance in enumerate(feature_importance):
                if importance > self.feature_mask_threshold and i < len(feature_names):
                    bottlenecks[feature_names[i]] = float(importance)
        
        # Sort by importance
        bottlenecks = dict(sorted(bottlenecks.items(), key=lambda x: x[1], reverse=True))
        
        return bottlenecks
    
    def visualize_explanation(
        self,
        data,
        node_idx: int,
        feature_names: List[str],
        save_path: Optional[str] = None
    ):
        """
        Visualize subgraph explanation
        
        Args:
            data: Graph data
            node_idx: Target node
            feature_names: Feature names
            save_path: Save path
        """
        explanation = self.explain_node(data, node_idx)
        
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        
        # Subgraph visualization
        ax = axes[0]
        self._plot_subgraph(
            explanation['edge_index'],
            explanation['edge_mask'],
            explanation['subset'],
            node_idx,
            ax
        )
        ax.set_title(f'Important Subgraph for Node {node_idx}')
        
        # Feature importance
        ax = axes[1]
        if explanation['node_mask'] is not None:
            feature_importance = explanation['node_mask'][0].cpu().numpy()
            top_k = 15
            top_indices = np.argsort(feature_importance)[::-1][:top_k]
            
            top_features = [feature_names[i] for i in top_indices if i < len(feature_names)]
            top_importance = feature_importance[top_indices[:len(top_features)]]
            
            ax.barh(range(len(top_features)), top_importance)
            ax.set_yticks(range(len(top_features)))
            ax.set_yticklabels(top_features)
            ax.set_xlabel('Importance Score')
            ax.set_title('Top Important Features')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            logger.info(f"Saved explanation to {save_path}")
        
        plt.show()
    
    def _k_hop_subgraph(
        self,
        node_idx: int,
        num_hops: int,
        edge_index: torch.Tensor,
        relabel_nodes: bool = False,
        num_nodes: Optional[int] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Extract k-hop subgraph"""
        from torch_geometric.utils import k_hop_subgraph
        
        subset, edge_index, mapping, edge_mask = k_hop_subgraph(
            node_idx, num_hops, edge_index,
            relabel_nodes=relabel_nodes,
            num_nodes=num_nodes
        )
        
        return subset, edge_index, mapping, edge_mask
    
    def _plot_subgraph(
        self,
        edge_index: torch.Tensor,
        edge_mask: torch.Tensor,
        subset: torch.Tensor,
        center_node: int,
        ax
    ):
        """Plot subgraph with edge importance"""
        # Create networkx graph
        G = nx.Graph()
        
        # Add nodes
        for i, node in enumerate(subset.cpu().numpy()):
            G.add_node(i, label=str(node))
        
        # Add edges with weights
        edge_index = edge_index.cpu().numpy()
        edge_mask = edge_mask.cpu().numpy()
        
        for i, (src, dst) in enumerate(edge_index.T):
            G.add_edge(src, dst, weight=edge_mask[i])
        
        # Layout
        pos = nx.spring_layout(G, k=2, iterations=50)
        
        # Draw nodes
        node_colors = ['red' if node == 0 else 'lightblue' for node in G.nodes()]
        nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=500, ax=ax)
        
        # Draw edges with varying width
        edges = G.edges()
        weights = [G[u][v]['weight'] * 5 for u, v in edges]
        
        nx.draw_networkx_edges(G, pos, width=weights, alpha=0.6, ax=ax)
        
        # Labels
        labels = nx.get_node_attributes(G, 'label')
        nx.draw_networkx_labels(G, pos, labels, font_size=10, ax=ax)
        
        ax.axis('off')