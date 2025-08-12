"""
Custom layers and components for enhanced GAT
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn.inits import glorot, zeros
from torch_geometric.utils import softmax
from typing import Optional, Tuple
import math


class FeatureWiseAttention(nn.Module):
    """
    Feature-wise attention mechanism to identify important I/O counters
    """
    
    def __init__(
        self,
        num_features: int,
        hidden_dim: int = 64,
        temperature: float = 1.0,
        dtype: torch.dtype = torch.float64
    ):
        super().__init__()
        
        self.num_features = num_features
        self.temperature = temperature
        
        # Attention network
        self.attention_net = nn.Sequential(
            nn.Linear(num_features, hidden_dim, dtype=dtype),
            nn.Tanh(),
            nn.Linear(hidden_dim, num_features, dtype=dtype)
        )
        
        self.reset_parameters()
    
    def reset_parameters(self):
        for module in self.attention_net.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Apply feature-wise attention
        
        Args:
            x: Input features [N, F]
        
        Returns:
            attended_features: Weighted features [N, F]
            attention_weights: Feature attention weights [N, F]
        """
        # Calculate attention scores
        attention_scores = self.attention_net(x)
        
        # Apply temperature scaling
        attention_scores = attention_scores / self.temperature
        
        # Softmax over features
        attention_weights = F.softmax(attention_scores, dim=-1)
        
        # Apply attention
        attended_features = x * attention_weights
        
        return attended_features, attention_weights


class EdgeGateLayer(nn.Module):
    """
    Edge gating mechanism to handle high-similarity edges
    """
    
    def __init__(
        self,
        in_channels: int,
        edge_dim: int = 1,
        dtype: torch.dtype = torch.float64
    ):
        super().__init__()
        
        self.in_channels = in_channels
        self.edge_dim = edge_dim
        
        # Edge gate network
        self.edge_gate = nn.Sequential(
            nn.Linear(2 * in_channels + edge_dim, in_channels, dtype=dtype),
            nn.ReLU(),
            nn.Linear(in_channels, 1, dtype=dtype),
            nn.Sigmoid()
        )
        
        self.reset_parameters()
    
    def reset_parameters(self):
        for module in self.edge_gate.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)
    
    def forward(
        self,
        x_i: torch.Tensor,
        x_j: torch.Tensor,
        edge_attr: torch.Tensor
    ) -> torch.Tensor:
        """
        Calculate edge gates
        
        Args:
            x_i: Source node features [E, F]
            x_j: Target node features [E, F]
            edge_attr: Edge attributes [E, D]
        
        Returns:
            gates: Edge gates [E, 1]
        """
        # Concatenate source, target, and edge features
        edge_input = torch.cat([x_i, x_j, edge_attr], dim=-1)
        
        # Calculate gates
        gates = self.edge_gate(edge_input)
        
        return gates


class AdaptivePooling(nn.Module):
    """
    Adaptive pooling that combines multiple pooling strategies
    """
    
    def __init__(
        self,
        in_channels: int,
        dtype: torch.dtype = torch.float64
    ):
        super().__init__()
        
        # Learnable weights for different pooling strategies
        self.pool_weights = nn.Parameter(torch.ones(3, dtype=dtype))
        
        # Projection after concatenation
        self.proj = nn.Linear(in_channels * 3, in_channels, dtype=dtype)
        
        self.reset_parameters()
    
    def reset_parameters(self):
        nn.init.ones_(self.pool_weights)
        nn.init.xavier_uniform_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)
    
    def forward(
        self,
        x: torch.Tensor,
        batch: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Apply adaptive pooling
        
        Args:
            x: Node features [N, F]
            batch: Batch indices [N]
        
        Returns:
            pooled: Pooled features [B, F] or [1, F]
        """
        from torch_geometric.nn import global_mean_pool, global_max_pool, global_add_pool
        
        # Apply different pooling strategies
        if batch is not None:
            mean_pool = global_mean_pool(x, batch)
            max_pool = global_max_pool(x, batch)
            sum_pool = global_add_pool(x, batch)
        else:
            mean_pool = x.mean(dim=0, keepdim=True)
            max_pool = x.max(dim=0, keepdim=True)[0]
            sum_pool = x.sum(dim=0, keepdim=True)
        
        # Normalize pool weights
        weights = F.softmax(self.pool_weights, dim=0)
        
        # Weighted combination
        pooled = torch.cat([
            mean_pool * weights[0],
            max_pool * weights[1],
            sum_pool * weights[2]
        ], dim=-1)
        
        # Project back to original dimension
        pooled = self.proj(pooled)
        
        return pooled


class ResidualGATBlock(nn.Module):
    """
    Residual GAT block with skip connections
    """
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        heads: int = 8,
        dropout: float = 0.1,
        edge_dim: int = 1,
        dtype: torch.dtype = torch.float64
    ):
        super().__init__()
        
        from .gat import EnhancedGATConv
        
        # GAT layer
        self.gat = EnhancedGATConv(
            in_channels=in_channels,
            out_channels=out_channels,
            heads=heads,
            dropout=dropout,
            edge_dim=edge_dim,
            dtype=dtype
        )
        
        # Projection for residual connection
        self.residual_proj = nn.Linear(
            in_channels,
            out_channels * heads,
            dtype=dtype
        ) if in_channels != out_channels * heads else nn.Identity()
        
        # Normalization
        self.norm = nn.LayerNorm(out_channels * heads)
        
        # Activation
        self.activation = nn.ELU()
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
    
    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass with residual connection
        """
        # Store input for residual
        identity = self.residual_proj(x)
        
        # GAT convolution
        out, _ = self.gat(x, edge_index, edge_attr)
        
        # Add residual
        out = out + identity
        
        # Normalization
        out = self.norm(out)
        
        # Activation and dropout
        out = self.activation(out)
        out = self.dropout(out)
        
        return out