"""
Attention-based interpretability method
"""
import torch
import numpy as np
from torch_geometric.data import Data
from typing import Dict, Optional
import logging
from .base import InterpretabilityMethod

logger = logging.getLogger(__name__)


class AttentionMethod(InterpretabilityMethod):
    """Attention-based feature importance analysis"""
    
    def analyze(self, 
                data: Data, 
                node_idx: int,
                threshold: float = 0.001,
                use_fallback: bool = True) -> Dict[str, float]:
        """
        Analyze feature importance using attention weights
        
        Args:
            data: Graph data
            node_idx: Node to analyze
            threshold: Minimum attention weight threshold
            use_fallback: Whether to use fallback method if primary fails
            
        Returns:
            Feature importance scores
        """
        logger.debug(f"Running attention analysis for node {node_idx}")
        
        # Try primary method
        scores = self._extract_attention_scores(data, node_idx, threshold)
        
        # Use fallback if needed
        if not scores and use_fallback:
            logger.info("Using fallback attention extraction")
            scores = self._fallback_attention_extraction(data, node_idx)
        
        return scores
    
    def _extract_attention_scores(self, 
                                  data: Data, 
                                  node_idx: int,
                                  threshold: float) -> Dict[str, float]:
        """Extract attention-based feature importance"""
        scores = {}
        
        try:
            with torch.no_grad():
                x = self.model.input_proj(data.x)
                
                # Accumulate feature importance across layers
                accumulated_importance = torch.zeros(data.x.shape[1], device=data.x.device)
                
                # Get attention weights from each layer
                for i, gat_layer in enumerate(self.model.gat_layers):
                    # Forward pass through GAT layer
                    x_out, (edge_index_out, attention_weights) = gat_layer(
                        x, data.edge_index, data.edge_attr, return_attention_weights=True
                    )
                    
                    # Find edges where node_idx is the target
                    mask = edge_index_out[1] == node_idx
                    if mask.any():
                        source_nodes = edge_index_out[0][mask]
                        node_attention = attention_weights[mask]
                        
                        # Get features for analysis
                        target_features = data.x[node_idx]
                        source_features = data.x[source_nodes]
                        
                        # Calculate feature importance based on attention-weighted differences
                        for j, source_idx in enumerate(source_nodes):
                            # Feature difference weighted by attention
                            feature_diff = torch.abs(source_features[j] - target_features)
                            # Average attention across heads if multi-head
                            att_weight = node_attention[j].mean() if node_attention[j].dim() > 0 else node_attention[j]
                            accumulated_importance += feature_diff * att_weight
                    
                    # Update x for next layer
                    x = x_out
                
                # Normalize accumulated importance
                if accumulated_importance.sum() > 0:
                    accumulated_importance = accumulated_importance / accumulated_importance.sum()
                    
                    # Convert to dictionary
                    importance_np = accumulated_importance.cpu().numpy()
                    for j, feat_name in enumerate(self.feature_names):
                        if j < len(importance_np) and importance_np[j] > threshold:
                            scores[feat_name] = float(importance_np[j])
                    
        except Exception as e:
            logger.warning(f"Attention extraction failed: {e}")
            
        return scores
    
    def _fallback_attention_extraction(self, 
                                      data: Data, 
                                      node_idx: int) -> Dict[str, float]:
        """Fallback method using raw attention patterns"""
        scores = {}
        
        try:
            with torch.no_grad():
                # Get node embedding after all layers
                x = self.model.input_proj(data.x)
                
                for gat_layer in self.model.gat_layers[:-1]:
                    x, _ = gat_layer(x, data.edge_index, data.edge_attr)
                
                # Use final layer attention
                x_final, (edge_index_out, att_weights) = self.model.gat_layers[-1](
                    x, data.edge_index, data.edge_attr, return_attention_weights=True
                )
                
                # Extract attention for target node
                target_mask = edge_index_out[1] == node_idx
                if target_mask.any():
                    source_nodes = edge_index_out[0][target_mask]
                    target_attention = att_weights[target_mask].cpu()
                    
                    # Average across heads if needed
                    if target_attention.dim() > 1:
                        target_attention = target_attention.mean(dim=1)
                    
                    # Get features
                    source_features = data.x[source_nodes].cpu()
                    target_features = data.x[node_idx].cpu()
                    
                    # Calculate feature importance
                    feature_scores = torch.zeros(data.x.shape[1])
                    for i, att_weight in enumerate(target_attention):
                        feature_diff = torch.abs(source_features[i] - target_features)
                        feature_scores += feature_diff * att_weight
                    
                    # Normalize
                    if feature_scores.sum() > 0:
                        feature_scores = feature_scores / feature_scores.sum()
                        
                        # Get top features
                        top_indices = torch.argsort(feature_scores, descending=True)[:10]
                        
                        for idx in top_indices:
                            if idx < len(self.feature_names) and feature_scores[idx] > 0:
                                scores[self.feature_names[idx]] = float(feature_scores[idx].item())
                                
        except Exception as e:
            logger.warning(f"Fallback attention extraction failed: {e}")
            
        return scores