#!/usr/bin/env python
"""
Subgraph builder for creating local graph structures around new jobs
"""

import numpy as np
import torch
from torch_geometric.data import Data
from typing import Dict, List, Tuple, Optional
import logging

logger = logging.getLogger(__name__)


class SubgraphBuilder:
    """Build subgraphs for new jobs with their neighbors"""
    
    def __init__(
        self,
        edge_construction_method: str = 'knn',
        similarity_threshold: float = 0.8,
        max_edges_per_node: int = 10
    ):
        """
        Initialize subgraph builder
        
        Args:
            edge_construction_method: 'knn', 'threshold', or 'combined'
            similarity_threshold: Threshold for edge creation
            max_edges_per_node: Maximum edges per node
        """
        self.edge_construction_method = edge_construction_method
        self.similarity_threshold = similarity_threshold
        self.max_edges_per_node = max_edges_per_node
    
    def build_subgraph(
        self,
        query_features: np.ndarray,
        neighbor_features: np.ndarray,
        neighbor_indices: np.ndarray,
        neighbor_similarities: np.ndarray,
        include_neighbor_edges: bool = True
    ) -> Data:
        """
        Build a PyG Data object for the subgraph
        
        Args:
            query_features: Features of the new job (1D array)
            neighbor_features: Features of neighbor jobs (2D array)
            neighbor_indices: Indices of neighbors in training data
            neighbor_similarities: Similarity scores to query
            include_neighbor_edges: Whether to add edges between neighbors
        
        Returns:
            PyG Data object representing the subgraph
        """
        n_neighbors = len(neighbor_features)
        
        # Combine features: query node first, then neighbors
        all_features = np.vstack([query_features.reshape(1, -1), neighbor_features])
        x = torch.tensor(all_features, dtype=torch.float32)
        
        # Build edges
        edge_index, edge_weight = self._construct_edges(
            query_features,
            neighbor_features,
            neighbor_similarities,
            include_neighbor_edges
        )
        
        # Create node masks
        query_mask = torch.zeros(n_neighbors + 1, dtype=torch.bool)
        query_mask[0] = True  # First node is the query
        
        # Create Data object
        data = Data(
            x=x,
            edge_index=edge_index,
            edge_attr=edge_weight.unsqueeze(-1) if edge_weight is not None else None,
            query_mask=query_mask,
            num_nodes=n_neighbors + 1,
            neighbor_indices=torch.tensor(neighbor_indices, dtype=torch.long)
        )
        
        return data
    
    def _construct_edges(
        self,
        query_features: np.ndarray,
        neighbor_features: np.ndarray,
        neighbor_similarities: np.ndarray,
        include_neighbor_edges: bool
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Construct edges for the subgraph
        
        Returns:
            edge_index: (2, E) tensor of edges
            edge_weight: (E,) tensor of edge weights (similarities)
        """
        edges = []
        weights = []
        
        # Add edges from query to neighbors
        for i, sim in enumerate(neighbor_similarities):
            if self.edge_construction_method == 'threshold':
                if sim >= self.similarity_threshold:
                    edges.append([0, i + 1])  # Query to neighbor
                    edges.append([i + 1, 0])  # Bidirectional
                    weights.extend([sim, sim])
            else:  # 'knn' or 'combined'
                if i < self.max_edges_per_node:
                    edges.append([0, i + 1])
                    edges.append([i + 1, 0])
                    weights.extend([sim, sim])
        
        # Add edges between neighbors
        if include_neighbor_edges:
            from sklearn.metrics.pairwise import cosine_similarity
            
            # Compute pairwise similarities between neighbors
            neighbor_sims = cosine_similarity(neighbor_features)
            
            for i in range(len(neighbor_features)):
                # Get top-k similar neighbors for each neighbor
                sims_to_others = neighbor_sims[i]
                sims_to_others[i] = -1  # Exclude self
                
                if self.edge_construction_method == 'threshold':
                    valid_neighbors = np.where(sims_to_others >= self.similarity_threshold)[0]
                else:
                    # Get top-k
                    k = min(self.max_edges_per_node, len(neighbor_features) - 1)
                    valid_neighbors = np.argsort(sims_to_others)[-k:]
                
                for j in valid_neighbors:
                    if sims_to_others[j] > 0:
                        edges.append([i + 1, j + 1])  # +1 because query is node 0
                        weights.append(sims_to_others[j])
        
        if not edges:
            # Create at least one edge to avoid empty graph
            edges = [[0, 1], [1, 0]]
            weights = [neighbor_similarities[0], neighbor_similarities[0]]
        
        edge_index = torch.tensor(edges, dtype=torch.long).t()
        edge_weight = torch.tensor(weights, dtype=torch.float32)
        
        return edge_index, edge_weight
    
    def build_batch_subgraphs(
        self,
        query_features_batch: np.ndarray,
        neighbor_features_batch: List[np.ndarray],
        neighbor_indices_batch: List[np.ndarray],
        neighbor_similarities_batch: List[np.ndarray]
    ) -> List[Data]:
        """
        Build subgraphs for a batch of queries
        
        Args:
            query_features_batch: Features of multiple new jobs
            neighbor_features_batch: List of neighbor features for each query
            neighbor_indices_batch: List of neighbor indices for each query
            neighbor_similarities_batch: List of similarity scores for each query
        
        Returns:
            List of PyG Data objects
        """
        subgraphs = []
        
        for i in range(len(query_features_batch)):
            subgraph = self.build_subgraph(
                query_features_batch[i],
                neighbor_features_batch[i],
                neighbor_indices_batch[i],
                neighbor_similarities_batch[i]
            )
            subgraphs.append(subgraph)
        
        return subgraphs
    
    def add_global_context(
        self,
        subgraph: Data,
        global_features: Optional[np.ndarray] = None
    ) -> Data:
        """
        Add global context node to subgraph (e.g., system-wide statistics)
        
        Args:
            subgraph: Existing subgraph
            global_features: Optional global context features
        
        Returns:
            Updated subgraph with global context
        """
        if global_features is None:
            # Use mean of all node features as global context
            global_features = subgraph.x.mean(dim=0, keepdim=True)
        else:
            global_features = torch.tensor(global_features, dtype=torch.float32).unsqueeze(0)
        
        # Add global node
        subgraph.x = torch.cat([subgraph.x, global_features], dim=0)
        
        # Connect global node to all other nodes
        n_nodes = subgraph.num_nodes
        global_edges = []
        
        for i in range(n_nodes):
            global_edges.append([i, n_nodes])  # Node to global
            global_edges.append([n_nodes, i])  # Global to node
        
        global_edge_index = torch.tensor(global_edges, dtype=torch.long).t()
        subgraph.edge_index = torch.cat([subgraph.edge_index, global_edge_index], dim=1)
        
        # Update edge weights if present
        if subgraph.edge_attr is not None:
            # Use uniform weights for global connections
            global_weights = torch.ones(len(global_edges), 1) * 0.5
            subgraph.edge_attr = torch.cat([subgraph.edge_attr, global_weights], dim=0)
        
        subgraph.num_nodes = n_nodes + 1
        
        return subgraph