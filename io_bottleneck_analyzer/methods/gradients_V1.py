"""
Gradient-based interpretability method
"""
import torch
import torch.nn.functional as F
import numpy as np
from torch_geometric.data import Data
from typing import Dict, Optional
import logging
from .base import InterpretabilityMethod

logger = logging.getLogger(__name__)


class GradientMethod(InterpretabilityMethod):
    """Gradient-based feature importance analysis"""
    
    def analyze(self, 
                data: Data, 
                node_idx: int,
                method: str = 'integrated_gradients',
                n_steps: int = 50) -> Dict[str, float]:
        """
        Analyze feature importance using gradients
        
        Args:
            data: Graph data  
            node_idx: Node to analyze
            method: Type of gradient method ('vanilla' or 'integrated_gradients')
            n_steps: Number of steps for integrated gradients
            
        Returns:
            Feature importance scores
        """
        logger.debug(f"Running {method} gradient analysis for node {node_idx}")
        
        if method == 'integrated_gradients':
            scores = self._integrated_gradients(data, node_idx, n_steps)
        else:
            scores = self._vanilla_gradients(data, node_idx)
            
        return scores
    
    def _vanilla_gradients(self, data: Data, node_idx: int) -> Dict[str, float]:
        """Compute vanilla gradients"""
        scores = {}
        
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
        
        # Get gradients for target node
        if data.x.grad is not None:
            gradients = data.x.grad[node_idx].cpu().numpy()
            
            # Convert to feature importance
            for i, feat_name in enumerate(self.feature_names):
                if i < len(gradients):
                    scores[feat_name] = float(abs(gradients[i]))
        
        data.x.requires_grad_(False)
        
        return scores
    
    def _integrated_gradients(self, 
                             data: Data, 
                             node_idx: int,
                             n_steps: int) -> Dict[str, float]:
        """Compute integrated gradients"""
        # Create baseline (zeros)
        baseline = torch.zeros_like(data.x)
        
        # Generate interpolated inputs
        alphas = torch.linspace(0, 1, n_steps + 1, device=self.device)
        
        integrated_grads = torch.zeros_like(data.x[node_idx])
        
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
            
            # Accumulate gradients
            if interpolated_data.x.grad is not None:
                integrated_grads += interpolated_data.x.grad[node_idx] / n_steps
        
        # Multiply by input difference
        integrated_grads *= (data.x[node_idx] - baseline[node_idx])
        
        # Convert to scores
        scores = {}
        grads_np = integrated_grads.cpu().numpy()
        
        for i, feat_name in enumerate(self.feature_names):
            if i < len(grads_np):
                scores[feat_name] = float(abs(grads_np[i]))
        
        return scores