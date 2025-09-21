"""
Base class for interpretability methods
"""
from abc import ABC, abstractmethod
from typing import Dict, Optional
import torch
from torch_geometric.data import Data
import logging

logger = logging.getLogger(__name__)


class InterpretabilityMethod(ABC):
    """Base class for all interpretability methods"""
    
    def __init__(self, 
                 model: torch.nn.Module,
                 feature_names: list,
                 device: str = 'cpu'):
        """
        Initialize interpretability method
        
        Args:
            model: Trained GAT model
            feature_names: List of feature names
            device: Device to run computations on
        """
        self.model = model
        self.feature_names = feature_names
        self.device = torch.device(device)
        self.model.eval()
        
    @abstractmethod
    def analyze(self, 
                data: Data, 
                node_idx: int,
                **kwargs) -> Dict[str, float]:
        """
        Analyze feature importance for a specific node
        
        Args:
            data: Graph data
            node_idx: Index of node to analyze
            **kwargs: Method-specific parameters
            
        Returns:
            Dictionary mapping feature names to importance scores
        """
        pass
    
    def _filter_zero_features(self, 
                             scores: Dict[str, float],
                             features: torch.Tensor) -> Dict[str, float]:
        """
        Remove scores for features that are zero
        
        Args:
            scores: Feature importance scores
            features: Original feature values
            
        Returns:
            Filtered scores
        """
        features_np = features.cpu().numpy()
        filtered_scores = {}
        
        for i, (feature_name, score) in enumerate(scores.items()):
            if i < len(features_np) and features_np[i] != 0:
                filtered_scores[feature_name] = score
                
        return filtered_scores
    
    def _normalize_scores(self, scores: Dict[str, float]) -> Dict[str, float]:
        """
        Normalize scores to sum to 1
        
        Args:
            scores: Raw scores
            
        Returns:
            Normalized scores
        """
        if not scores:
            return scores
            
        total = sum(abs(s) for s in scores.values())
        if total > 0:
            return {k: v/total for k, v in scores.items()}
        return scores