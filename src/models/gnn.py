"""
GNN models for I/O performance prediction.

This module implements Graph Neural Network models for I/O performance prediction,
including the TabGNN approach that combines GNN embeddings with traditional tabular models.
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv, global_mean_pool
from typing import List, Dict, Tuple, Optional, Union, Any
import logging
import numpy as np
import json

# Set up logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

class GNNBase(nn.Module):
    """
    Base class for GNN models.
    """
    
    def __init__(
        self, 
        in_channels: int,
        hidden_channels: int,
        out_channels: int,
        num_layers: int,
        dropout: float = 0.0,
        model_type: str = 'gcn'
    ):
        """
        Initialize GNN base model.
        
        Args:
            in_channels (int): Number of input features
            hidden_channels (int): Number of hidden features
            out_channels (int): Number of output features
            num_layers (int): Number of GNN layers
            dropout (float): Dropout probability
            model_type (str): Type of GNN ('gcn' or 'gat')
        """
        super(GNNBase, self).__init__()
        
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.out_channels = out_channels
        self.num_layers = num_layers
        self.dropout = dropout
        self.model_type = model_type
        
        # Create GNN layers
        self.convs = nn.ModuleList()
        
        # Input layer
        if model_type == 'gcn':
            self.convs.append(GCNConv(in_channels, hidden_channels))
        elif model_type == 'gat':
            self.convs.append(GATConv(in_channels, hidden_channels))
        else:
            raise ValueError(f"Unsupported GNN type: {model_type}")
        
        # Hidden layers
        for _ in range(num_layers - 2):
            if model_type == 'gcn':
                self.convs.append(GCNConv(hidden_channels, hidden_channels))
            elif model_type == 'gat':
                self.convs.append(GATConv(hidden_channels, hidden_channels))
        
        # Output layer
        if model_type == 'gcn':
            self.convs.append(GCNConv(hidden_channels, out_channels))
        elif model_type == 'gat':
            self.convs.append(GATConv(hidden_channels, out_channels))
        
        logger.info(f"Initialized {model_type.upper()} model with {num_layers} layers")
    
    def forward(
        self, 
        x: torch.Tensor, 
        edge_index: torch.Tensor,
        batch: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x (torch.Tensor): Node features
            edge_index (torch.Tensor): Edge indices
            batch (torch.Tensor, optional): Batch assignment for nodes
            
        Returns:
            torch.Tensor: Node embeddings
        """
        # Apply GNN layers
        for i, conv in enumerate(self.convs[:-1]):
            x = conv(x, edge_index)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        
        # Apply final layer
        x = self.convs[-1](x, edge_index)
        
        return x

class TabGNNRegressor(nn.Module):
    """
    TabGNN model for regression tasks.
    """
    
    def __init__(
        self, 
        in_channels: int,
        hidden_channels: int,
        gnn_out_channels: int,
        mlp_hidden_channels: List[int],
        num_layers: int,
        num_graph_types: int = 1,
        dropout: float = 0.0,
        model_type: str = 'gcn'
    ):
        """
        Initialize TabGNN regressor.
        
        Args:
            in_channels (int): Number of input features
            hidden_channels (int): Number of hidden features
            gnn_out_channels (int): Number of GNN output features
            mlp_hidden_channels (List[int]): List of hidden channel sizes for MLP
            num_layers (int): Number of GNN layers
            num_graph_types (int): Number of graph types (multiplex graphs)
            dropout (float): Dropout probability
            model_type (str): Type of GNN ('gcn' or 'gat')
        """
        super(TabGNNRegressor, self).__init__()
        
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.gnn_out_channels = gnn_out_channels
        self.mlp_hidden_channels = mlp_hidden_channels
        self.num_layers = num_layers
        self.num_graph_types = num_graph_types
        self.dropout = dropout
        self.model_type = model_type
        
        # Create GNN models for each graph type
        self.gnn_models = nn.ModuleList()
        for _ in range(num_graph_types):
            self.gnn_models.append(
                GNNBase(
                    in_channels=in_channels,
                    hidden_channels=hidden_channels,
                    out_channels=gnn_out_channels,
                    num_layers=num_layers,
                    dropout=dropout,
                    model_type=model_type
                )
            )
        
        # Create MLP for final prediction
        mlp_layers = []
        
        # Input layer
        mlp_input_size = in_channels + (gnn_out_channels * num_graph_types)
        mlp_layers.append(nn.Linear(mlp_input_size, mlp_hidden_channels[0]))
        mlp_layers.append(nn.ReLU())
        mlp_layers.append(nn.Dropout(dropout))
        
        # Hidden layers
        for i in range(len(mlp_hidden_channels) - 1):
            mlp_layers.append(nn.Linear(mlp_hidden_channels[i], mlp_hidden_channels[i+1]))
            mlp_layers.append(nn.ReLU())
            mlp_layers.append(nn.Dropout(dropout))
        
        # Output layer
        mlp_layers.append(nn.Linear(mlp_hidden_channels[-1], 1))
        
        self.mlp = nn.Sequential(*mlp_layers)
        
        logger.info(f"Initialized TabGNN regressor with {num_graph_types} graph types")
    
    def extract_embeddings(
        self, 
        x: torch.Tensor, 
        edge_indices: List[torch.Tensor],
        batch: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Extract embeddings from GNN models.
        
        Args:
            x (torch.Tensor): Node features
            edge_indices (List[torch.Tensor]): List of edge indices for each graph type
            batch (torch.Tensor, optional): Batch assignment for nodes
            
        Returns:
            torch.Tensor: Extracted embeddings
        """
        # Apply GNN models to each graph type
        embeddings = []
        
        for i, gnn_model in enumerate(self.gnn_models):
            if i < len(edge_indices):
                emb = gnn_model(x, edge_indices[i], batch)
                embeddings.append(emb)
            else:
                logger.warning(f"Edge indices not provided for graph type {i}")
        
        # Concatenate embeddings
        if embeddings:
            embeddings = torch.cat(embeddings, dim=1)
        else:
            embeddings = torch.zeros((x.size(0), 0), device=x.device)
        
        return embeddings
    
    def forward(
        self, 
        x: torch.Tensor, 
        edge_indices: List[torch.Tensor],
        batch: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x (torch.Tensor): Node features
            edge_indices (List[torch.Tensor]): List of edge indices for each graph type
            batch (torch.Tensor, optional): Batch assignment for nodes
            
        Returns:
            torch.Tensor: Predictions
        """
        # Extract embeddings
        embeddings = self.extract_embeddings(x, edge_indices, batch)
        
        # Concatenate with original features
        combined = torch.cat([x, embeddings], dim=1)
        
        # Apply MLP
        out = self.mlp(combined)
        
        return out
    
    def save_checkpoint(
        self, 
        checkpoint_dir: str,
        epoch: int = 0,
        optimizer: Optional[torch.optim.Optimizer] = None,
        train_loss: float = 0.0,
        val_loss: float = 0.0,
        filename: str = "model_checkpoint.pt"
    ) -> str:
        """
        Save model checkpoint.
        
        Args:
            checkpoint_dir (str): Directory to save checkpoint
            epoch (int): Current epoch
            optimizer (torch.optim.Optimizer, optional): Optimizer
            train_loss (float): Training loss
            val_loss (float): Validation loss
            filename (str): Filename for checkpoint
            
        Returns:
            str: Path to saved checkpoint
        """
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        # Create checkpoint
        checkpoint = {
            'model_state_dict': self.state_dict(),
            'epoch': epoch,
            'train_loss': train_loss,
            'val_loss': val_loss,
            'model_config': {
                'in_channels': self.in_channels,
                'hidden_channels': self.hidden_channels,
                'gnn_out_channels': self.gnn_out_channels,
                'mlp_hidden_channels': self.mlp_hidden_channels,
                'num_layers': self.num_layers,
                'num_graph_types': self.num_graph_types,
                'dropout': self.dropout,
                'model_type': self.model_type
            }
        }
        
        if optimizer is not None:
            checkpoint['optimizer_state_dict'] = optimizer.state_dict()
        
        # Save checkpoint
        checkpoint_path = os.path.join(checkpoint_dir, filename)
        torch.save(checkpoint, checkpoint_path)
        
        logger.info(f"Model checkpoint saved to {checkpoint_path}")
        
        return checkpoint_path
    
    @classmethod
    def load_checkpoint(
        cls, 
        checkpoint_path: str,
        device: Optional[torch.device] = None
    ) -> 'TabGNNRegressor':
        """
        Load model from checkpoint.
        
        Args:
            checkpoint_path (str): Path to checkpoint
            device (torch.device, optional): Device to load model to
            
        Returns:
            TabGNNRegressor: Loaded model
        """
        # Load checkpoint
        if device is None:
            checkpoint = torch.load(checkpoint_path, map_location=torch.device('cpu'))
        else:
            checkpoint = torch.load(checkpoint_path, map_location=device)
        
        # Create model
        model = cls(
            in_channels=checkpoint['model_config']['in_channels'],
            hidden_channels=checkpoint['model_config']['hidden_channels'],
            gnn_out_channels=checkpoint['model_config']['gnn_out_channels'],
            mlp_hidden_channels=checkpoint['model_config']['mlp_hidden_channels'],
            num_layers=checkpoint['model_config']['num_layers'],
            num_graph_types=checkpoint['model_config']['num_graph_types'],
            dropout=checkpoint['model_config']['dropout'],
            model_type=checkpoint['model_config']['model_type']
        )
        
        # Load state dict
        model.load_state_dict(checkpoint['model_state_dict'])
        
        if device is not None:
            model = model.to(device)
        
        logger.info(f"Model loaded from {checkpoint_path}")
        
        return model

class MultiplexGNN(nn.Module):
    """
    Multiplex GNN model for I/O performance prediction.
    """
    
    def __init__(
        self, 
        in_channels: int,
        hidden_channels: int,
        out_channels: int,
        num_layers: int,
        num_graph_types: int,
        dropout: float = 0.0,
        model_type: str = 'gcn',
        combine_method: str = 'concat'
    ):
        """
        Initialize multiplex GNN model.
        
        Args:
            in_channels (int): Number of input features
            hidden_channels (int): Number of hidden features
            out_channels (int): Number of output features
            num_layers (int): Number of GNN layers
            num_graph_types (int): Number of graph types (multiplex graphs)
            dropout (float): Dropout probability
            model_type (str): Type of GNN ('gcn' or 'gat')
            combine_method (str): Method to combine embeddings ('concat', 'sum', 'mean')
        """
        super(MultiplexGNN, self).__init__()
        
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.out_channels = out_channels
        self.num_layers = num_layers
        self.num_graph_types = num_graph_types
        self.dropout = dropout
        self.model_type = model_type
        self.combine_method = combine_method
        
        # Create GNN models for each graph type
        self.gnn_models = nn.ModuleList()
        for _ in range(num_graph_types):
            self.gnn_models.append(
                GNNBase(
                    in_channels=in_channels,
                    hidden_channels=hidden_channels,
                    out_channels=out_channels,
                    num_layers=num_layers,
                    dropout=dropout,
                    model_type=model_type
                )
            )
        
        # Create output layer
        if combine_method == 'concat':
            self.out_layer = nn.Linear(out_channels * num_graph_types, out_channels)
        else:
            self.out_layer = nn.Linear(out_channels, out_channels)
        
        logger.info(f"Initialized multiplex GNN model with {num_graph_types} graph types")
    
    def forward(
        self, 
        x: torch.Tensor, 
        edge_indices: List[torch.Tensor],
        batch: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x (torch.Tensor): Node features
            edge_indices (List[torch.Tensor]): List of edge indices for each graph type
            batch (torch.Tensor, optional): Batch assignment for nodes
            
        Returns:
            torch.Tensor: Node embeddings
        """
        # Apply GNN models to each graph type
        embeddings = []
        
        for i, gnn_model in enumerate(self.gnn_models):
            if i < len(edge_indices):
                emb = gnn_model(x, edge_indices[i], batch)
                embeddings.append(emb)
            else:
                logger.warning(f"Edge indices not provided for graph type {i}")
        
        # Combine embeddings
        if self.combine_method == 'concat':
            if embeddings:
                combined = torch.cat(embeddings, dim=1)
            else:
                combined = torch.zeros((x.size(0), 0), device=x.device)
        elif self.combine_method == 'sum':
            if embeddings:
                combined = torch.stack(embeddings).sum(dim=0)
            else:
                combined = torch.zeros((x.size(0), self.out_channels), device=x.device)
        elif self.combine_method == 'mean':
            if embeddings:
                combined = torch.stack(embeddings).mean(dim=0)
            else:
                combined = torch.zeros((x.size(0), self.out_channels), device=x.device)
        else:
            raise ValueError(f"Unsupported combine method: {self.combine_method}")
        
        # Apply output layer
        out = self.out_layer(combined)
        
        return out

class RayTuneableGNN(nn.Module):
    """
    GNN model with tunable hyperparameters for Ray Tune.
    """
    
    def __init__(
        self, 
        in_channels: int,
        hidden_channels: int,
        out_channels: int,
        num_layers: int,
        dropout: float = 0.0,
        model_type: str = 'gcn',
        **kwargs
    ):
        """
        Initialize Ray Tuneable GNN model.
        
        Args:
            in_channels (int): Number of input features
            hidden_channels (int): Number of hidden features
            out_channels (int): Number of output features
            num_layers (int): Number of GNN layers
            dropout (float): Dropout probability
            model_type (str): Type of GNN ('gcn' or 'gat')
            **kwargs: Additional hyperparameters
        """
        super(RayTuneableGNN, self).__init__()
        
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.out_channels = out_channels
        self.num_layers = num_layers
        self.dropout = dropout
        self.model_type = model_type
        self.kwargs = kwargs
        
        # Create GNN model
        self.gnn = GNNBase(
            in_channels=in_channels,
            hidden_channels=hidden_channels,
            out_channels=out_channels,
            num_layers=num_layers,
            dropout=dropout,
            model_type=model_type
        )
        
        # Create MLP for final prediction
        self.mlp = nn.Sequential(
            nn.Linear(out_channels, hidden_channels),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_channels, 1)
        )
        
        logger.info(f"Initialized Ray Tuneable GNN model")
    
    def forward(
        self, 
        x: torch.Tensor, 
        edge_index: torch.Tensor,
        batch: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x (torch.Tensor): Node features
            edge_index (torch.Tensor): Edge indices
            batch (torch.Tensor, optional): Batch assignment for nodes
            
        Returns:
            torch.Tensor: Predictions
        """
        # Apply GNN
        x = self.gnn(x, edge_index, batch)
        
        # Apply global pooling if batch is provided
        if batch is not None:
            x = global_mean_pool(x, batch)
        
        # Apply MLP
        out = self.mlp(x)
        
        return out
    
    def get_config(self) -> Dict[str, Any]:
        """
        Get model configuration.
        
        Returns:
            Dict[str, Any]: Model configuration
        """
        config = {
            'in_channels': self.in_channels,
            'hidden_channels': self.hidden_channels,
            'out_channels': self.out_channels,
            'num_layers': self.num_layers,
            'dropout': self.dropout,
            'model_type': self.model_type
        }
        
        # Add additional hyperparameters
        config.update(self.kwargs)
        
        return config
    
    @classmethod
    def from_config(
        cls, 
        config: Dict[str, Any]
    ) -> 'RayTuneableGNN':
        """
        Create model from configuration.
        
        Args:
            config (Dict[str, Any]): Model configuration
            
        Returns:
            RayTuneableGNN: Created model
        """
        # Extract required parameters
        in_channels = config.pop('in_channels')
        hidden_channels = config.pop('hidden_channels')
        out_channels = config.pop('out_channels')
        num_layers = config.pop('num_layers')
        dropout = config.pop('dropout')
        model_type = config.pop('model_type')
        
        # Create model
        model = cls(
            in_channels=in_channels,
            hidden_channels=hidden_channels,
            out_channels=out_channels,
            num_layers=num_layers,
            dropout=dropout,
            model_type=model_type,
            **config
        )
        
        return model
