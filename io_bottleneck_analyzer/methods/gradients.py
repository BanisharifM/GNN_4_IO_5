"""
Gradient-based interpretability method with feature filtering and improvements
"""
import torch
import torch.nn.functional as F
import numpy as np
from torch_geometric.data import Data
from typing import Dict, Optional, List, Tuple
import logging
from .base import InterpretabilityMethod

logger = logging.getLogger(__name__)


class GradientMethod(InterpretabilityMethod):
    """Gradient-based feature importance analysis"""
    
    def analyze(self, 
                data: Data, 
                node_idx: int,
                method: str = 'integrated_gradients',
                n_steps: int = 50,
                filter_irrelevant: bool = False) -> Dict[str, float]:
        """
        Analyze feature importance using gradients
        
        Args:
            data: Graph data  
            node_idx: Node to analyze
            method: Type of gradient method ('vanilla' or 'integrated_gradients')
            n_steps: Number of steps for integrated gradients
            filter_irrelevant: Whether to filter irrelevant features
            
        Returns:
            Feature importance scores
        """
        logger.debug(f"Running {method} gradient analysis for node {node_idx}")
        
        # Get relevant feature mask if filtering enabled
        relevance_mask = None
        if filter_irrelevant:
            relevance_mask = self._get_relevant_feature_mask(data, node_idx)
            logger.debug(f"Feature filtering enabled: {relevance_mask.sum().item()}/{len(relevance_mask)} features active")
        
        if method == 'integrated_gradients':
            scores = self._integrated_gradients(data, node_idx, n_steps, relevance_mask)
        else:
            scores = self._vanilla_gradients(data, node_idx, relevance_mask)
            
        return scores
    
    def _get_relevant_feature_mask(self, data: Data, node_idx: int) -> torch.Tensor:
        """Create a mask for relevant features based on I/O activity"""
        
        # Initialize mask for original features only
        num_original_features = len(self.feature_names)  # 45
        mask = torch.ones(num_original_features, dtype=torch.bool)
        
        # Get target node features (only first 45, ignore augmented)
        target_features = data.x[node_idx][:num_original_features].cpu().numpy()
        
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
                stride_num = feat_name.split('STRIDE')[1].split('_')[0]
                count_name = f'POSIX_STRIDE{stride_num}_COUNT'
                if count_name in feature_indices and target_features[feature_indices[count_name]] == 0:
                    mask[i] = False
                    
            if 'ACCESS' in feat_name and '_COUNT' not in feat_name:
                access_num = feat_name.split('ACCESS')[1].split('_')[0]
                count_name = f'POSIX_ACCESS{access_num}_COUNT'
                if count_name in feature_indices and target_features[feature_indices[count_name]] == 0:
                    mask[i] = False
            
            # Also mask COUNT features themselves if they're zero
            if '_COUNT' in feat_name and target_features[i] == 0:
                mask[i] = False
        
        return mask.to(data.x.device)
    
    def _apply_feature_mask(self, data: Data, mask: Optional[torch.Tensor]) -> Data:
        """Apply feature mask to data"""
        if mask is None:
            return data
        
        # Extend mask for augmented features
        num_total_features = data.x.shape[1]
        if num_total_features > len(mask):
            extended_mask = torch.ones(num_total_features, dtype=torch.bool, device=mask.device)
            extended_mask[:len(mask)] = mask
            mask = extended_mask
        
        # Create masked data
        masked_x = data.x.clone()
        masked_x[:, ~mask] = 0  # Zero out irrelevant features
        
        return Data(x=masked_x, edge_index=data.edge_index, edge_attr=data.edge_attr)
    
    def _vanilla_gradients(self, 
                          data: Data, 
                          node_idx: int,
                          relevance_mask: Optional[torch.Tensor] = None) -> Dict[str, float]:
        """Compute vanilla gradients"""
        scores = {}
        
        # Apply feature mask
        if relevance_mask is not None:
            data = self._apply_feature_mask(data, relevance_mask)
        
        # Enable gradients
        data.x.requires_grad_(True)
        
        # Forward pass
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
                x = F.elu(x)
        
        node_features = x[node_idx].unsqueeze(0)
        prediction = self.model.predictor(node_features)
        
        # Backward pass
        prediction.backward()
        
        # Get gradients for target node (only original features)
        num_original_features = len(self.feature_names)
        if data.x.grad is not None:
            gradients = data.x.grad[node_idx][:num_original_features].cpu().numpy()
            
            # Apply mask to gradients
            if relevance_mask is not None:
                gradients = gradients * relevance_mask.cpu().numpy()
            
            # Clip extreme gradients
            gradients = np.clip(gradients, -1e6, 1e6)
            
            # Convert to feature importance (filtering out zeros)
            for i, feat_name in enumerate(self.feature_names):
                if i < len(gradients) and abs(gradients[i]) > 1e-8:  # Filter near-zero
                    if relevance_mask is None or relevance_mask[i]:
                        scores[feat_name] = float(abs(gradients[i]))
        
        data.x.requires_grad_(False)
        
        return scores
    
    def _integrated_gradients(self, 
                             data: Data, 
                             node_idx: int,
                             n_steps: int,
                             relevance_mask: Optional[torch.Tensor] = None) -> Dict[str, float]:
        """Compute integrated gradients with improvements"""
        
        # Apply feature mask
        if relevance_mask is not None:
            data = self._apply_feature_mask(data, relevance_mask)
        
        # Create improved baseline (mean of non-zero values)
        baseline = torch.zeros_like(data.x)
        for i in range(data.x.shape[1]):
            non_zero = data.x[:, i][data.x[:, i] != 0]
            if len(non_zero) > 0:
                baseline[:, i] = non_zero.mean()
        
        # Apply mask to baseline
        if relevance_mask is not None:
            num_total_features = baseline.shape[1]
            if num_total_features > len(relevance_mask):
                extended_mask = torch.ones(num_total_features, dtype=torch.bool, device=relevance_mask.device)
                extended_mask[:len(relevance_mask)] = relevance_mask
                relevance_mask_extended = extended_mask
            else:
                relevance_mask_extended = relevance_mask
            baseline[:, ~relevance_mask_extended] = 0
        
        # Generate interpolated inputs
        alphas = torch.linspace(0, 1, n_steps + 1, device=self.device)
        
        num_original_features = len(self.feature_names)
        integrated_grads = torch.zeros(num_original_features, device=self.device)
        
        for alpha in alphas[1:]:  # Skip alpha=0
            # Interpolate between baseline and input
            interpolated_x = baseline + alpha * (data.x - baseline)
            interpolated_x.requires_grad_(True)
            
            # Create interpolated data
            interpolated_data = Data(
                x=interpolated_x,
                edge_index=data.edge_index,
                edge_attr=data.edge_attr
            )
            
            # Forward pass
            x = self.model.input_proj(interpolated_data.x)
            
            for i, gat_layer in enumerate(self.model.gat_layers):
                residual = x
                x, _ = gat_layer(x, data.edge_index, data.edge_attr)
                
                if self.model.residual and i < len(self.model.residual_projs):
                    residual = self.model.residual_projs[i](residual)
                    x = x + residual
                    
                if self.model.layer_norm and i < len(self.model.layer_norms):
                    x = self.model.layer_norms[i](x)
                    
                if i < self.model.num_layers - 1:
                    x = F.elu(x)
            
            node_features = x[node_idx].unsqueeze(0)
            prediction = self.model.predictor(node_features)
            
            # Backward pass
            prediction.backward()
            
            # Accumulate gradients (only for original features)
            if interpolated_data.x.grad is not None:
                grads = interpolated_data.x.grad[node_idx][:num_original_features]
                integrated_grads += grads / n_steps
        
        # Multiply by input difference (only original features)
        input_diff = (data.x[node_idx][:num_original_features] - baseline[node_idx][:num_original_features])
        integrated_grads *= input_diff
        
        # Apply relevance mask
        if relevance_mask is not None:
            integrated_grads = integrated_grads * relevance_mask.float()
        
        # Convert to scores with gradient clipping
        scores = {}
        grads_np = integrated_grads.cpu().numpy()
        grads_np = np.clip(grads_np, -1e6, 1e6)  # Clip extreme values
        
        # Normalize gradients (optional but helpful)
        if np.max(np.abs(grads_np)) > 0:
            grads_np = grads_np / np.max(np.abs(grads_np))  # Normalize to [-1, 1]
        
        for i, feat_name in enumerate(self.feature_names):
            if i < len(grads_np) and abs(grads_np[i]) > 1e-8:  # Filter near-zero
                if relevance_mask is None or relevance_mask[i]:
                    scores[feat_name] = float(abs(grads_np[i]))
        
        return scores