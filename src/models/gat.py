"""
Graph Attention Network (GAT) for I/O Performance Prediction
Optimized for high-similarity graphs with interpretability
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv, global_mean_pool, global_max_pool, global_add_pool
from torch_geometric.nn.norm import BatchNorm, LayerNorm
from typing import Optional, Tuple, List, Dict
import logging

logger = logging.getLogger(__name__)


class EnhancedGATConv(nn.Module):
    """
    Enhanced GAT layer with edge weights and float64 support
    """
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        heads: int = 8,
        dropout: float = 0.1,
        edge_dim: int = 1,
        add_self_loops: bool = True,
        bias: bool = True,
        dtype: torch.dtype = torch.float64
    ):
        super().__init__()
        
        self.gat_conv = GATConv(
            in_channels=in_channels,
            out_channels=out_channels,
            heads=heads,
            dropout=dropout,
            edge_dim=edge_dim,
            add_self_loops=add_self_loops,
            bias=bias,
        )
        
        # Convert to float64
        self.gat_conv = self.gat_conv.double()
        
        # Store attention weights for interpretability
        self.attention_weights = None
        
    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: Optional[torch.Tensor] = None,
        return_attention_weights: bool = False
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        """
        Forward pass with optional attention weight extraction
        """
        # Ensure float64
        x = x.double()
        if edge_attr is not None:
            edge_attr = edge_attr.double()
        
        # Forward pass with attention weights
        out, attention = self.gat_conv(
            x, edge_index, edge_attr,
            return_attention_weights=True
        )
        
        # Store attention weights for analysis
        if return_attention_weights:
            self.attention_weights = attention
        
        return out, attention if return_attention_weights else (out, None)


class IOPerformanceGAT(nn.Module):
    """
    Multi-head GAT for I/O performance prediction
    Designed for high-similarity graphs with interpretability
    """
    
    def __init__(
        self,
        num_features: int,
        hidden_channels: int = 256,
        num_layers: int = 3,
        heads: List[int] = [8, 8, 1],
        dropout: float = 0.2,
        edge_dim: int = 1,
        pool_type: str = 'mean',
        residual: bool = True,
        layer_norm: bool = True,
        feature_augmentation: bool = True,
        dtype: torch.dtype = torch.float64
    ):
        """
        Args:
            num_features: Number of input features (45 for I/O counters)
            hidden_channels: Hidden dimension size
            num_layers: Number of GAT layers
            heads: Number of attention heads per layer
            dropout: Dropout rate
            edge_dim: Edge feature dimension (1 for similarity scores)
            pool_type: Global pooling type ('mean', 'max', 'add')
            residual: Use residual connections
            layer_norm: Use layer normalization
            feature_augmentation: Add statistical features
            dtype: Data type (float64 for precision)
        """
        super().__init__()
        
        self.num_features = num_features
        self.hidden_channels = hidden_channels
        self.num_layers = num_layers
        self.heads = heads if len(heads) == num_layers else [heads[0]] * num_layers
        self.dropout = dropout
        self.residual = residual
        self.layer_norm = layer_norm
        self.feature_augmentation = feature_augmentation
        self.dtype = dtype
        
        # Input projection with feature augmentation
        aug_features = num_features
        if feature_augmentation:
            # Add statistical features (mean, std, min, max of features)
            aug_features = num_features + 4
        
        self.input_proj = nn.Linear(aug_features, hidden_channels, dtype=dtype)
        
        # GAT layers
        self.gat_layers = nn.ModuleList()
        self.layer_norms = nn.ModuleList() if layer_norm else None
        self.residual_projs = nn.ModuleList() if residual else None
        
        for i in range(num_layers):
            in_channels = hidden_channels * (self.heads[i-1] if i > 0 else 1)
            out_channels = hidden_channels
            
            # GAT layer
            self.gat_layers.append(
                EnhancedGATConv(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    heads=self.heads[i],
                    dropout=dropout if i < num_layers - 1 else 0,  # No dropout on last layer
                    edge_dim=edge_dim,
                    dtype=dtype
                )
            )
            
            # Layer normalization
            if layer_norm:
                norm_channels = out_channels * self.heads[i]
                self.layer_norms.append(LayerNorm(norm_channels))
            
            # Residual projection if dimensions don't match
            if residual and in_channels != out_channels * self.heads[i]:
                self.residual_projs.append(
                    nn.Linear(in_channels, out_channels * self.heads[i], dtype=dtype)
                )
            elif residual:
                self.residual_projs.append(nn.Identity())
        
        # Global pooling
        self.pool_type = pool_type
        if pool_type == 'mean':
            self.global_pool = global_mean_pool
        elif pool_type == 'max':
            self.global_pool = global_max_pool
        elif pool_type == 'add':
            self.global_pool = global_add_pool
        else:
            raise ValueError(f"Unknown pool type: {pool_type}")
        
        # Output layers
        final_channels = hidden_channels * self.heads[-1]
        
        # Multi-layer predictor for complex patterns
        self.predictor = nn.Sequential(
            nn.Linear(final_channels, hidden_channels, dtype=dtype),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_channels, hidden_channels // 2, dtype=dtype),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_channels // 2, 1, dtype=dtype)
        )
        
        # Store attention weights for all layers
        self.all_attention_weights = []
        
        # Initialize weights
        self.reset_parameters()
        
    def reset_parameters(self):
        """Initialize model parameters"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
    
    def augment_features(self, x: torch.Tensor) -> torch.Tensor:
        """
        Augment features with statistical information
        """
        if not self.feature_augmentation:
            return x
        
        # Calculate statistics for each node
        x_mean = x.mean(dim=1, keepdim=True)
        x_std = x.std(dim=1, keepdim=True)
        x_min = x.min(dim=1, keepdim=True)[0]
        x_max = x.max(dim=1, keepdim=True)[0]
        
        # Concatenate
        x_aug = torch.cat([x, x_mean, x_std, x_min, x_max], dim=1)
        
        return x_aug
    
    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: Optional[torch.Tensor] = None,
        batch: Optional[torch.Tensor] = None,
        return_attention: bool = False
    ) -> Tuple[torch.Tensor, Optional[List[Tuple]]]:
        """
        Forward pass
        
        Args:
            x: Node features [N, F]
            edge_index: Edge indices [2, E]
            edge_attr: Edge weights [E, 1]
            batch: Batch indices for mini-batch training
            return_attention: Whether to return attention weights
        
        Returns:
            predictions: Performance predictions [N, 1] or [B, 1] if batched
            attention_weights: List of attention weights per layer (optional)
        """
        # Ensure float64
        x = x.double()
        if edge_attr is not None:
            edge_attr = edge_attr.double()
        
        # Feature augmentation
        x = self.augment_features(x)
        
        # Input projection
        x = self.input_proj(x)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        
        # Store attention weights
        attention_weights = []
        
        # GAT layers
        for i in range(self.num_layers):
            identity = x
            
            # GAT convolution
            x, att = self.gat_layers[i](
                x, edge_index, edge_attr,
                return_attention_weights=return_attention
            )
            
            if return_attention:
                attention_weights.append(att)
            
            # Activation (except last layer)
            if i < self.num_layers - 1:
                x = F.elu(x)
            
            # Layer normalization
            if self.layer_norm:
                x = self.layer_norms[i](x)
            
            # Residual connection
            if self.residual:
                x = x + self.residual_projs[i](identity)
            
            # Dropout (except last layer)
            if i < self.num_layers - 1:
                x = F.dropout(x, p=self.dropout, training=self.training)
        
        # Store all attention weights for analysis
        if return_attention:
            self.all_attention_weights = attention_weights
        
        # Global pooling for graph-level prediction
        if batch is not None:
            x = self.global_pool(x, batch)
        
        # Final prediction
        out = self.predictor(x)
        
        if return_attention:
            return out, attention_weights
        else:
            return out        
    
    def get_attention_weights(self) -> List[Tuple]:
        """
        Get stored attention weights for interpretability
        """
        return self.all_attention_weights
    
    def get_feature_importance(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: Optional[torch.Tensor] = None,
        target_idx: Optional[int] = None
    ) -> torch.Tensor:
        """
        Calculate feature importance using gradients
        
        Args:
            x: Node features
            edge_index: Edge indices
            edge_attr: Edge weights
            target_idx: Target node index for explanation
        
        Returns:
            Feature importance scores [F]
        """
        x = x.double()
        if edge_attr is not None:
            edge_attr = edge_attr.double()
        
        # Enable gradients for input
        x.requires_grad = True
        
        # Forward pass
        out = self.forward(x, edge_index, edge_attr)
        
        # Select target output
        if target_idx is not None:
            target_out = out[target_idx]
        else:
            target_out = out.mean()
        
        # Backward pass
        self.zero_grad()
        target_out.backward()
        
        # Get gradients
        feature_importance = x.grad.abs().mean(dim=0)
        
        # Normalize
        feature_importance = feature_importance / feature_importance.sum()
        
        return feature_importance


class LightweightGAT(IOPerformanceGAT):
    """
    Lightweight version of GAT for faster training/inference
    """
    
    def __init__(self, num_features: int, **kwargs):
        # Override with lighter configuration
        super().__init__(
            num_features=num_features,
            hidden_channels=128,  # Smaller hidden dimension
            num_layers=2,  # Fewer layers
            heads=[4, 1],  # Fewer attention heads
            dropout=0.1,  # Less dropout
            residual=True,
            layer_norm=False,  # Skip layer norm for speed
            feature_augmentation=False,  # No augmentation
            **kwargs
        )

def create_gat_model(
    num_features: int,
    model_type: str = 'standard',
    **kwargs
) -> IOPerformanceGAT:
    """
    Factory function to create GAT models
    
    Args:
        num_features: Number of input features
        model_type: 'standard', 'lightweight', or 'deep'
        **kwargs: Additional arguments for the model
    
    Returns:
        GAT model instance
    """
    if model_type == 'lightweight':
        return LightweightGAT(num_features, **kwargs)
    elif model_type == 'deep':
        # Deep model defaults (will be overridden by config)
        deep_defaults = {
            'hidden_channels': 512,
            'num_layers': 4,
            'heads': [16, 8, 4, 1]
        }
        # Update defaults with provided kwargs
        deep_defaults.update(kwargs)
        return IOPerformanceGAT(num_features, **deep_defaults)
    else:  # standard
        return IOPerformanceGAT(num_features, **kwargs)

# def create_gat_model(
#     num_features: int,
#     model_type: str = 'standard',
#     **kwargs
# ) -> IOPerformanceGAT:
#     """
#     Factory function to create GAT models
    
#     Args:
#         num_features: Number of input features
#         model_type: 'standard', 'lightweight', or 'deep'
#         **kwargs: Additional arguments for the model
    
#     Returns:
#         GAT model instance
#     """
#     if model_type == 'lightweight':
#         return LightweightGAT(num_features, **kwargs)
#     elif model_type == 'deep':
#         return IOPerformanceGAT(
#             num_features,
#             hidden_channels=512,
#             num_layers=4,
#             heads=[16, 8, 4, 1],
#             **kwargs
#         )
#     else:  # standard
#         return IOPerformanceGAT(num_features, **kwargs)