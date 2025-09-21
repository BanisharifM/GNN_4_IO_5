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
        
        # Learn feature mask
        feature_mask = self._learn_mask(data, node_idx, original_pred)
        
        # Convert mask to scores
        scores = self._mask_to_scores(feature_mask, feature_mask_threshold)
        
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
    
    def _learn_mask(self, 
                   data: Data, 
                   node_idx: int, 
                   original_pred: torch.Tensor) -> torch.Tensor:
        """Learn feature importance mask"""
        # Initialize feature mask
        num_features = data.x.shape[1]
        feature_mask = torch.nn.Parameter(
            torch.ones(num_features, device=self.device) * 0.5
        )
        
        optimizer = torch.optim.Adam([feature_mask], lr=self.lr)
        
        for epoch in range(self.num_epochs):
            optimizer.zero_grad()
            
            # Apply mask to features
            masked_x = data.x * torch.sigmoid(feature_mask)
            masked_data = Data(
                x=masked_x,
                edge_index=data.edge_index,
                edge_attr=data.edge_attr
            )
            
            # Get prediction with masked features
            masked_pred = self._get_prediction(masked_data, node_idx)
            
            # Compute loss
            pred_loss = F.mse_loss(masked_pred, original_pred)
            size_loss = torch.sigmoid(feature_mask).sum() * 0.01
            entropy_loss = -torch.sigmoid(feature_mask) * torch.log(torch.sigmoid(feature_mask) + 1e-8)
            entropy_loss = entropy_loss.sum() * 0.1
            
            loss = pred_loss + size_loss + entropy_loss
            
            loss.backward()
            optimizer.step()
            
            if epoch % 50 == 0:
                logger.debug(f"Epoch {epoch}: Loss = {loss.item():.4f}")
        
        return torch.sigmoid(feature_mask).detach()
    
    def _mask_to_scores(self, 
                       feature_mask: torch.Tensor,
                       threshold: float) -> Dict[str, float]:
        """Convert learned mask to feature scores"""
        scores = {}
        mask_values = feature_mask.cpu().numpy()
        
        # Only keep features above threshold
        for i, feat_name in enumerate(self.feature_names):
            if i < len(mask_values) and mask_values[i] > threshold:
                scores[feat_name] = float(mask_values[i])
        
        return scores