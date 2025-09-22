"""
Attention-based interpretability method
"""
import torch
import numpy as np
from torch_geometric.data import Data
from typing import Dict, Optional, List
import logging
from .base import InterpretabilityMethod

logger = logging.getLogger(__name__)


class AttentionMethod(InterpretabilityMethod):
    """Attention-based feature importance analysis"""
    
    def analyze(self, 
                data: Data, 
                node_idx: int,
                threshold: float = 0.001,
                use_fallback: bool = True,
                filter_irrelevant: bool = False) -> Dict[str, float]:
        """
        Analyze feature importance using attention weights
        
        Args:
            data: Graph data
            node_idx: Node to analyze
            threshold: Minimum attention weight threshold
            use_fallback: Whether to use fallback method if primary fails
            filter_irrelevant: Whether to filter out irrelevant features before analysis
            
        Returns:
            Feature importance scores
        """
        logger.debug(f"Running attention analysis for node {node_idx}")
        
        # Get feature mask if filtering is enabled
        feature_mask = None
        if filter_irrelevant:
            feature_mask = self._get_relevant_feature_mask(data, node_idx)
            logger.debug(f"Feature filtering enabled: {feature_mask.sum().item()}/{len(feature_mask)} features active")
        
        # Try primary method
        scores = self._extract_attention_scores(data, node_idx, threshold, feature_mask)
        
        # Use fallback if needed
        if not scores and use_fallback:
            logger.info("Using fallback attention extraction")
            scores = self._fallback_attention_extraction(data, node_idx, feature_mask)
        
        return scores
    
    def _get_relevant_feature_mask(self, data: Data, node_idx: int) -> torch.Tensor:
        """Create a mask for relevant features based on I/O activity"""
        
        # Initialize mask with all features active
        mask = torch.ones(len(self.feature_names), dtype=torch.bool)
        
        # Get target node features
        target_features = data.x[node_idx].cpu().numpy()
        
        # Create feature index mapping
        feature_indices = {name: i for i, name in enumerate(self.feature_names)}
        
        # Check for read/write activity
        has_reads = feature_indices.get('POSIX_READS') and target_features[feature_indices['POSIX_READS']] > 0
        has_writes = feature_indices.get('POSIX_WRITES') and target_features[feature_indices['POSIX_WRITES']] > 0
        
        # Filter features based on I/O patterns
        for i, feat_name in enumerate(self.feature_names):
            # Filter read-related features if no reads
            if not has_reads and any(keyword in feat_name.upper() for keyword in ['_READ', 'CONSEC_READS', 'SEQ_READS']):
                mask[i] = False
                
            # Filter write-related features if no writes
            if not has_writes and any(keyword in feat_name.upper() for keyword in ['_WRITE', 'CONSEC_WRITES', 'SEQ_WRITES']):
                mask[i] = False
            
            # Filter stride/access features if their counts are zero
            if 'STRIDE' in feat_name and '_COUNT' not in feat_name:
                # Check if corresponding count is zero
                stride_num = feat_name.split('STRIDE')[1].split('_')[0]
                count_name = f'POSIX_STRIDE{stride_num}_COUNT'
                if count_name in feature_indices:
                    if target_features[feature_indices[count_name]] == 0:
                        mask[i] = False
                        
            if 'ACCESS' in feat_name and '_COUNT' not in feat_name:
                # Check if corresponding count is zero
                access_num = feat_name.split('ACCESS')[1].split('_')[0]
                count_name = f'POSIX_ACCESS{access_num}_COUNT'
                if count_name in feature_indices:
                    if target_features[feature_indices[count_name]] == 0:
                        mask[i] = False
            
            # Also mask the COUNT features themselves if they're zero
            if '_COUNT' in feat_name and target_features[i] == 0:
                mask[i] = False
        
        return mask.to(data.x.device)
    
    def _extract_attention_scores(self, 
                                data: Data, 
                                node_idx: int,
                                threshold: float,
                                feature_mask: Optional[torch.Tensor] = None) -> Dict[str, float]:
        """Extract attention-based feature importance"""
        scores = {}
        
        try:
            with torch.no_grad():
                # Handle augmented features (graph builder adds 4 extra features: 45 -> 49)
                if feature_mask is not None:
                    if data.x.shape[1] > len(feature_mask):
                        extended_mask = torch.ones(data.x.shape[1], dtype=torch.bool, device=feature_mask.device)
                        extended_mask[:len(feature_mask)] = feature_mask
                        feature_mask = extended_mask
                    
                    masked_x = data.x.clone()
                    masked_x[:, ~feature_mask] = 0
                    x = self.model.input_proj(masked_x)
                else:
                    x = self.model.input_proj(data.x)
                
                # Accumulate feature importance only for original features
                num_original_features = len(self.feature_names)  # Should be 45
                accumulated_importance = torch.zeros(num_original_features, device=data.x.device)
                
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
                            # Feature difference weighted by attention (only first 45 features)
                            feature_diff = torch.abs(source_features[j][:num_original_features] - 
                                                target_features[:num_original_features])
                            
                            # Apply feature mask to differences if provided
                            if feature_mask is not None:
                                feature_diff = feature_diff * feature_mask[:num_original_features].float()
                            
                            # Average attention across heads if multi-head
                            att_weight = node_attention[j].mean() if node_attention[j].dim() > 0 else node_attention[j]
                            accumulated_importance += feature_diff * att_weight
                    
                    # Update x for next layer
                    x = x_out
                
                # Apply feature mask to final importance
                if feature_mask is not None:
                    accumulated_importance = accumulated_importance * feature_mask[:num_original_features].float()
                
                # Normalize accumulated importance
                if accumulated_importance.sum() > 0:
                    accumulated_importance = accumulated_importance / accumulated_importance.sum()
                    
                    # Convert to dictionary
                    importance_np = accumulated_importance.cpu().numpy()
                    for j, feat_name in enumerate(self.feature_names):
                        if j < len(importance_np) and importance_np[j] > threshold:
                            # Only include if not masked
                            if feature_mask is None or (j < len(feature_mask) and feature_mask[j]):
                                scores[feat_name] = float(importance_np[j])
                    
        except Exception as e:
            logger.warning(f"Attention extraction failed: {e}")
            
        return scores

    def _fallback_attention_extraction(self, 
                                    data: Data, 
                                    node_idx: int,
                                    feature_mask: Optional[torch.Tensor] = None) -> Dict[str, float]:
        """Fallback method using raw attention patterns"""
        scores = {}
        
        try:
            with torch.no_grad():
                # Handle augmented features
                if feature_mask is not None:
                    if data.x.shape[1] > len(feature_mask):
                        extended_mask = torch.ones(data.x.shape[1], dtype=torch.bool, device=feature_mask.device)
                        extended_mask[:len(feature_mask)] = feature_mask
                        feature_mask = extended_mask
                    
                    masked_x = data.x.clone()
                    masked_x[:, ~feature_mask] = 0
                    x = self.model.input_proj(masked_x)
                else:
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
                    
                    # Calculate feature importance only for original features
                    num_original_features = len(self.feature_names)  # Should be 45
                    feature_scores = torch.zeros(num_original_features)
                    
                    for i, att_weight in enumerate(target_attention):
                        # Only use first 45 features
                        feature_diff = torch.abs(source_features[i][:num_original_features] - 
                                            target_features[:num_original_features])
                        # Apply feature mask if provided
                        if feature_mask is not None:
                            feature_diff = feature_diff * feature_mask[:num_original_features].cpu().float()
                        feature_scores += feature_diff * att_weight
                    
                    # Apply final mask
                    if feature_mask is not None:
                        feature_scores = feature_scores * feature_mask[:num_original_features].cpu().float()
                    
                    # Normalize
                    if feature_scores.sum() > 0:
                        feature_scores = feature_scores / feature_scores.sum()
                        
                        # Get top features
                        top_indices = torch.argsort(feature_scores, descending=True)[:10]
                        
                        for idx in top_indices:
                            if idx < len(self.feature_names) and feature_scores[idx] > 0:
                                # Only include if not masked
                                if feature_mask is None or (idx < len(feature_mask) and feature_mask[idx]):
                                    scores[self.feature_names[idx]] = float(feature_scores[idx].item())
                                
        except Exception as e:
            logger.warning(f"Fallback attention extraction failed: {e}")
            
        return scores