"""
Base model interface for all models in GNN4_IO_4.

This module defines the base interface that all models should implement,
ensuring consistency across different model types.
"""

from abc import ABC, abstractmethod
import numpy as np
from typing import Dict, Any, Optional, Union, List, Tuple
import os
import logging

logger = logging.getLogger(__name__)

class BaseModel(ABC):
    """Base class for all models."""
    
    @abstractmethod
    def fit(self, X, y, **kwargs):
        """
        Train the model.
        
        Args:
            X: Input features
            y: Target values
            **kwargs: Additional arguments for specific model implementations
            
        Returns:
            self: The trained model
        """
        pass
    
    @abstractmethod
    def predict(self, X, **kwargs):
        """
        Make predictions using the trained model.
        
        Args:
            X: Input features
            **kwargs: Additional arguments for specific model implementations
            
        Returns:
            np.ndarray: Predictions
        """
        pass
    
    @abstractmethod
    def evaluate(self, X, y, **kwargs) -> Dict[str, float]:
        """
        Evaluate the model performance.
        
        Args:
            X: Input features
            y: Target values
            **kwargs: Additional arguments for specific model implementations
            
        Returns:
            Dict[str, float]: Dictionary of evaluation metrics
        """
        pass
    
    @abstractmethod
    def save(self, path: str, **kwargs):
        """
        Save the model to disk.
        
        Args:
            path: Path to save the model
            **kwargs: Additional arguments for specific model implementations
        """
        pass
    
    @abstractmethod
    def load(self, path: str, **kwargs):
        """
        Load the model from disk.
        
        Args:
            path: Path to load the model from
            **kwargs: Additional arguments for specific model implementations
            
        Returns:
            self: The loaded model
        """
        pass
    
    def get_params(self) -> Dict[str, Any]:
        """
        Get model parameters.
        
        Returns:
            Dict[str, Any]: Dictionary of model parameters
        """
        return self.__dict__
    
    def set_params(self, **params) -> 'BaseModel':
        """
        Set model parameters.
        
        Args:
            **params: Parameters to set
            
        Returns:
            self: The model with updated parameters
        """
        for key, value in params.items():
            setattr(self, key, value)
        return self


class TabularModel(BaseModel):
    """Base class for tabular models."""
    
    def evaluate(self, X, y, **kwargs) -> Dict[str, float]:
        """
        Evaluate the model performance.
        
        Args:
            X: Input features
            y: Target values
            **kwargs: Additional arguments for specific model implementations
            
        Returns:
            Dict[str, float]: Dictionary of evaluation metrics
        """
        from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
        
        # Make predictions
        y_pred = self.predict(X)
        
        # Calculate metrics
        mse = mean_squared_error(y, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y, y_pred)
        r2 = r2_score(y, y_pred)
        
        return {
            'mse': mse,
            'rmse': rmse,
            'mae': mae,
            'r2': r2
        }


class GNNModel(BaseModel):
    """Base class for GNN models."""
    
    @abstractmethod
    def forward(self, x, edge_index, edge_attr=None, batch=None):
        """
        Forward pass of the GNN model.
        
        Args:
            x: Node features
            edge_index: Graph connectivity
            edge_attr: Edge features
            batch: Batch assignment for nodes
            
        Returns:
            torch.Tensor: Output of the model
        """
        pass
    
    def evaluate(self, X, edge_index, y, edge_attr=None, batch=None, **kwargs) -> Dict[str, float]:
        """
        Evaluate the model performance.
        
        Args:
            X: Node features
            edge_index: Graph connectivity
            y: Target values
            edge_attr: Edge features
            batch: Batch assignment for nodes
            **kwargs: Additional arguments for specific model implementations
            
        Returns:
            Dict[str, float]: Dictionary of evaluation metrics
        """
        import torch
        from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
        
        # Set model to evaluation mode
        self.eval()
        
        # Make predictions
        with torch.no_grad():
            y_pred = self.forward(X, edge_index, edge_attr, batch)
            
            # Convert to numpy for metric calculation
            if isinstance(y_pred, torch.Tensor):
                y_pred = y_pred.cpu().numpy()
            if isinstance(y, torch.Tensor):
                y = y.cpu().numpy()
        
        # Calculate metrics
        mse = mean_squared_error(y, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y, y_pred)
        r2 = r2_score(y, y_pred)
        
        return {
            'mse': mse,
            'rmse': rmse,
            'mae': mae,
            'r2': r2
        }


class TabGNNModel(BaseModel):
    """Base class for TabGNN models that combine GNN and tabular approaches."""
    
    def __init__(self, gnn_model: GNNModel, tabular_model: TabularModel):
        """
        Initialize TabGNN model.
        
        Args:
            gnn_model: GNN model component
            tabular_model: Tabular model component
        """
        self.gnn_model = gnn_model
        self.tabular_model = tabular_model
    
    @abstractmethod
    def extract_embeddings(self, X, edge_index, edge_attr=None, batch=None):
        """
        Extract embeddings from GNN model.
        
        Args:
            X: Node features
            edge_index: Graph connectivity
            edge_attr: Edge features
            batch: Batch assignment for nodes
            
        Returns:
            np.ndarray: Extracted embeddings
        """
        pass
    
    @abstractmethod
    def combine_features(self, original_features, embeddings):
        """
        Combine original features with GNN embeddings.
        
        Args:
            original_features: Original tabular features
            embeddings: GNN embeddings
            
        Returns:
            np.ndarray: Combined features
        """
        pass
