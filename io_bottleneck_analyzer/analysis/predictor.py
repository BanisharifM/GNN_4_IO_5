"""
Performance predictor for I/O jobs
"""
import torch
import torch.nn.functional as F
from torch_geometric.data import Data
import logging
from typing import Tuple

logger = logging.getLogger(__name__)


class PerformancePredictor:
    """Handles performance prediction for I/O jobs"""
    
    def __init__(self, model: torch.nn.Module, device: str = 'cpu'):
        """
        Initialize predictor
        
        Args:
            model: Trained GAT model
            device: Device for computation
        """
        self.model = model
        self.device = torch.device(device)
        self.model.eval()
        
    def predict(self, data: Data, node_idx: int) -> Tuple[float, float]:
        """
        Predict performance for a node
        
        Args:
            data: Graph data
            node_idx: Index of node to predict
            
        Returns:
            Tuple of (predicted_log_performance, predicted_mbps)
        """
        with torch.no_grad():
            # Forward pass through model
            x = self.model.input_proj(data.x)
            
            for i, gat_layer in enumerate(self.model.gat_layers):
                residual = x
                x, _ = gat_layer(x, data.edge_index, data.edge_attr)
                
                # Handle residual connections
                if self.model.residual and i < len(self.model.residual_projs):
                    residual = self.model.residual_projs[i](residual)
                    x = x + residual
                
                # Layer normalization
                if self.model.layer_norm and i < len(self.model.layer_norms):
                    x = self.model.layer_norms[i](x)
                
                # Activation
                if i < self.model.num_layers - 1:
                    x = F.elu(x)
            
            # Get prediction for target node
            node_features = x[node_idx].unsqueeze(0)
            log_prediction = self.model.predictor(node_features).item()
            
            # Convert to MB/s
            mbps_prediction = 10**log_prediction - 1
            
            logger.debug(f"Predicted performance: {mbps_prediction:.2f} MB/s")
            
        return log_prediction, mbps_prediction
    
    def compute_error(self, 
                     predicted_mbps: float, 
                     actual_mbps: float) -> dict:
        """
        Compute prediction errors
        
        Args:
            predicted_mbps: Predicted performance in MB/s
            actual_mbps: Actual performance in MB/s
            
        Returns:
            Dictionary of error metrics
        """
        absolute_error = abs(predicted_mbps - actual_mbps)
        relative_error = absolute_error / actual_mbps * 100 if actual_mbps > 0 else 0
        
        return {
            'absolute_error_mbps': absolute_error,
            'relative_error_percent': relative_error,
            'predicted_mbps': predicted_mbps,
            'actual_mbps': actual_mbps
        }