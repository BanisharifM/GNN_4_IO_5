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
                seed: int = 42,
                filter_irrelevant: bool = False) -> Dict[str, float]:
        """
        Analyze feature importance using GNNExplainer
        
        Args:
            data: Graph data
            node_idx: Node to analyze
            feature_mask_threshold: Threshold for feature importance
            seed: Random seed for reproducibility
            filter_irrelevant: Whether to filter irrelevant features
            
        Returns:
            Feature importance scores
        """
        logger.debug(f"Running GNNExplainer for node {node_idx}")
        
        # Set random seeds for reproducibility
        self._set_random_seeds(seed)
        
        # Get relevant feature mask if filtering enabled
        relevance_mask = None
        if filter_irrelevant:
            relevance_mask = self._get_relevant_feature_mask(data, node_idx)
            logger.debug(f"Feature filtering enabled: {relevance_mask.sum().item()}/{len(relevance_mask)} features active")
        
        # Get original prediction
        original_pred = self._get_prediction(data, node_idx)
        
        # Learn feature mask using perturbation
        feature_importance = self._learn_feature_importance(data, node_idx, original_pred, relevance_mask)
        
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

    def _get_relevant_feature_mask(self, data: Data, node_idx: int) -> torch.Tensor:
        """Create a mask for relevant features based on I/O activity"""
        
        # Initialize mask with all features active
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
                if count_name in feature_indices:
                    if target_features[feature_indices[count_name]] == 0:
                        mask[i] = False
                        
            if 'ACCESS' in feat_name and '_COUNT' not in feat_name:
                access_num = feat_name.split('ACCESS')[1].split('_')[0]
                count_name = f'POSIX_ACCESS{access_num}_COUNT'
                if count_name in feature_indices:
                    if target_features[feature_indices[count_name]] == 0:
                        mask[i] = False
            
            # mask the COUNT features themselves if they're zero
            if '_COUNT' in feat_name and target_features[i] == 0:
                mask[i] = False
        
        return mask.to(data.x.device)

    def _learn_feature_importance(self, 
                                data: Data, 
                                node_idx: int, 
                                original_pred: torch.Tensor,
                                relevance_mask: Optional[torch.Tensor] = None,
                                temperature: float = 0.5) -> torch.Tensor:
        """Learn feature importance through optimization"""
        
        # Handle augmented features (45 -> 49)
        num_original_features = len(self.feature_names)
        num_total_features = data.x.shape[1]
        
        # Extend relevance mask if needed
        if relevance_mask is not None and num_total_features > len(relevance_mask):
            extended_mask = torch.ones(num_total_features, dtype=torch.bool, device=relevance_mask.device)
            extended_mask[:len(relevance_mask)] = relevance_mask
            # Keep augmented features active
            relevance_mask = extended_mask
        
        # Initialize feature mask
        if relevance_mask is not None:
            # Initialize irrelevant features to large negative values (will sigmoid to ~0)
            initial_values = torch.where(
                relevance_mask,
                torch.randn(num_total_features, device=self.device) * 0.1,  # Normal init for relevant
                torch.ones(num_total_features, device=self.device) * -10    # Large negative for irrelevant
            )
            feature_mask = torch.nn.Parameter(initial_values)
        else:
            feature_mask = torch.nn.Parameter(
                torch.randn(num_total_features, device=self.device) * 0.1
            )
        
        optimizer = torch.optim.Adam([feature_mask], lr=self.lr)
        
        # Get baseline prediction
        baseline_x = torch.zeros_like(data.x)
        for i in range(num_total_features):
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
            soft_mask = torch.sigmoid(feature_mask / temperature)
            
            # Force irrelevant features to stay at 0
            if relevance_mask is not None:
                soft_mask = soft_mask * relevance_mask.float()
            
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
            
            # L1 regularization only on relevant features
            if relevance_mask is not None:
                mask_loss = (soft_mask * relevance_mask.float()).sum() * 0.1
            else:
                mask_loss = soft_mask.sum() * 0.1
            
            # Entropy regularization only for relevant features
            if relevance_mask is not None:
                relevant_mask = soft_mask[relevance_mask]
                if len(relevant_mask) > 0:
                    entropy = -relevant_mask * torch.log(relevant_mask + 1e-8) - \
                            (1 - relevant_mask) * torch.log(1 - relevant_mask + 1e-8)
                    entropy_loss = entropy.sum() * 0.01
                else:
                    entropy_loss = 0
            else:
                entropy = -soft_mask * torch.log(soft_mask + 1e-8) - \
                        (1 - soft_mask) * torch.log(1 - soft_mask + 1e-8)
                entropy_loss = entropy.sum() * 0.01
            
            total_loss = pred_loss + mask_loss + entropy_loss
            
            # Track best mask (only first 45 features)
            if total_loss.item() < best_loss:
                best_loss = total_loss.item()
                best_mask = soft_mask[:num_original_features].detach().clone()
            
            total_loss.backward()
            
            # Zero out gradients for irrelevant features to prevent updates
            if relevance_mask is not None:
                feature_mask.grad = feature_mask.grad * relevance_mask.float()
            
            optimizer.step()
            
            if epoch % 50 == 0:
                logger.debug(f"Epoch {epoch}: Loss = {total_loss.item():.4f}, "
                        f"Pred = {pred_loss.item():.4f}, "
                        f"Mask = {mask_loss.item():.4f}")
        
        # Return only original features (first 45)
        if best_mask is not None:
            return best_mask
        else:
            final_mask = torch.sigmoid(feature_mask / temperature)[:num_original_features]
            if relevance_mask is not None:
                final_mask = final_mask * relevance_mask[:num_original_features].float()
            return final_mask.detach()

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