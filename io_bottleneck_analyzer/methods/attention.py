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
                
                # Get attention weights from each layer
                for i, gat_layer in enumerate(self.model.gat_layers):
                    # Forward pass through GAT layer
                    x_out, (edge_index_out, attention_weights) = gat_layer(
                        x, data.edge_index, data.edge_attr, return_attention_weights=True
                    )
                    
                    # Find edges where node_idx is the target
                    mask = edge_index_out[1] == node_idx
                    if mask.any():
                        node_attention = attention_weights[mask].mean(dim=0)
                        
                        # Convert to feature importance
                        if node_attention.numel() > 0:
                            attention_values = node_attention.cpu().numpy()
                            
                            # Map to features
                            for j, feat_name in enumerate(self.feature_names[:len(attention_values)]):
                                if attention_values[j] > threshold:
                                    scores[feat_name] = float(attention_values[j])
                    
                    # Update x for next layer
                    x = x_out
                    
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
                x_final, (_, att_weights) = self.model.gat_layers[-1](
                    x, data.edge_index, data.edge_attr, return_attention_weights=True
                )
                
                # Extract attention for target node
                target_mask = data.edge_index[1] == node_idx
                if target_mask.any():
                    target_attention = att_weights[target_mask].cpu()
                    
                    # Average across heads if needed
                    if target_attention.dim() > 1:
                        target_attention = target_attention.mean(dim=1)
                    
                    attention_np = target_attention.numpy()
                    
                    # Create uniform scores for top features
                    num_features = min(len(attention_np), len(self.feature_names))
                    if num_features > 0:
                        attention_normalized = attention_np[:num_features] / (attention_np[:num_features].sum() + 1e-10)
                        sorted_indices = np.argsort(attention_normalized)[::-1][:10]
                        
                        for idx in sorted_indices:
                            if idx < len(self.feature_names):
                                scores[self.feature_names[idx]] = float(attention_normalized[idx])
                                
        except Exception as e:
            logger.warning(f"Fallback attention extraction failed: {e}")
            
        return scores