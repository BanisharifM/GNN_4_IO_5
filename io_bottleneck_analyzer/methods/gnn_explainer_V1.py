"""
GNNExplainer interpretability method
"""
import torch
import torch.nn.functional as F
import numpy as np
import random
from torch_geometric.data import Data
from typing import Dict, Optional
import logging
from .base import InterpretabilityMethod

logger = logging.getLogger(__name__)


class GNNExplainerMethod(InterpretabilityMethod):
    """GNNExplainer for feature importance analysis"""
    
    def __init__(self, 
                 model: torch.nn.Module,
                 feature_names: list,
                 device: str = 'cpu',
                 num_epochs: int = 200,
                 lr: float = 0.01):
        """
        Initialize GNNExplainer
        
        Args:
            model: Trained GAT model
            feature_names: List of feature names
            device: Device to run on
            num_epochs: Number of optimization epochs
            lr: Learning rate for mask optimization
        """
        super().__init__(model, feature_names, device)
        self.num_epochs = num_epochs
        self.lr = lr
        
    def analyze(self, 
                data: Data, 
                node_idx: int,
                feature_mask_threshold: float = 0.01,
                seed: int = 42) -> Dict[str, float]:
        """
        Analyze feature importance using GNNExplainer
        
        Args:
            data: Graph data
            node_idx: Node to analyze
            feature_mask_threshold: Threshold for feature importance
            seed: Random seed for reproducibility
            
        Returns:
            Feature importance scores
        """
        logger.debug(f"Running GNNExplainer for node {node_idx}")
        
        # Set random seeds for reproducibility
        self._set_random_seeds(seed)
        
        # Get original prediction
        original_pred = self._get_prediction(data, node_idx)
        
        # Learn feature mask using perturbation
        feature_importance = self._learn_feature_importance(data, node_idx, original_pred)
        
        # Convert to scores
        scores = self._importance_to_scores(feature_importance, feature_mask_threshold)
        
        return scores
    
    def _set_random_seeds(self, seed: int):
        """Set random seeds for reproducibility"""
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
    
    def _get_prediction(self, data: Data, node_idx: int) -> torch.Tensor:
        """Get model prediction for node"""
        self.model.eval()
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
                    x = F.elu(x)
            
            node_features = x[node_idx].unsqueeze(0)
            prediction = self.model.predictor(node_features)
            
        return prediction
    
    def _learn_feature_importance(self, 
                                 data: Data, 
                                 node_idx: int, 
                                 original_pred: torch.Tensor) -> torch.Tensor:
        """Learn feature importance through optimization"""
        num_features = data.x.shape[1]
        
        # Initialize with random values to break symmetry
        feature_mask = torch.nn.Parameter(
            torch.randn(num_features, device=self.device) * 0.1
        )
        
        optimizer = torch.optim.Adam([feature_mask], lr=self.lr)
        
        # Get baseline prediction (with zeros or mean values)
        baseline_x = torch.zeros_like(data.x)
        # Use mean of non-zero features as baseline
        for i in range(num_features):
            non_zero = data.x[:, i][data.x[:, i] != 0]
            if len(non_zero) > 0:
                baseline_x[:, i] = non_zero.mean()
        
        baseline_data = Data(
            x=baseline_x,
            edge_index=data.edge_index,
            edge_attr=data.edge_attr
        )
        baseline_pred = self._get_prediction(baseline_data, node_idx)
        
        best_mask = None
        best_loss = float('inf')
        
        for epoch in range(self.num_epochs):
            optimizer.zero_grad()
            
            # Compute soft mask
            soft_mask = torch.sigmoid(feature_mask)
            
            # Interpolate between baseline and original features
            masked_x = baseline_x + (data.x - baseline_x) * soft_mask.unsqueeze(0)
            
            masked_data = Data(
                x=masked_x,
                edge_index=data.edge_index,
                edge_attr=data.edge_attr
            )
            
            # Get prediction with masked features
            masked_pred = self._get_prediction(masked_data, node_idx)
            
            # Loss: maintain prediction while minimizing mask
            pred_loss = F.mse_loss(masked_pred, original_pred)
            
            # L1 regularization to encourage sparsity
            mask_loss = soft_mask.sum() * 0.01
            
            # Entropy regularization for decisive masks
            entropy = -soft_mask * torch.log(soft_mask + 1e-8) - (1 - soft_mask) * torch.log(1 - soft_mask + 1e-8)
            entropy_loss = entropy.sum() * 0.01
            
            total_loss = pred_loss + mask_loss + entropy_loss
            
            # Track best mask
            if total_loss.item() < best_loss:
                best_loss = total_loss.item()
                best_mask = soft_mask.detach().clone()
            
            total_loss.backward()
            optimizer.step()
            
            if epoch % 50 == 0:
                logger.debug(f"Epoch {epoch}: Loss = {total_loss.item():.4f}, "
                           f"Pred = {pred_loss.item():.4f}, "
                           f"Mask = {mask_loss.item():.4f}")
        
        return best_mask if best_mask is not None else torch.sigmoid(feature_mask).detach()
    
    def _importance_to_scores(self, 
                            feature_importance: torch.Tensor,
                            threshold: float) -> Dict[str, float]:
        """Convert importance values to scores"""
        scores = {}
        importance_values = feature_importance.cpu().numpy()
        
        # Normalize to [0, 1] if needed
        if importance_values.max() > 0:
            importance_values = importance_values / importance_values.max()
        
        # Create scores for features above threshold
        for i, feat_name in enumerate(self.feature_names):
            if i < len(importance_values) and importance_values[i] > threshold:
                scores[feat_name] = float(importance_values[i])
        
        return scores