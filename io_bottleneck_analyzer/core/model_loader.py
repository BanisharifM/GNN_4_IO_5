"""
Model loader for GAT models
"""
import torch
import logging
from pathlib import Path
from typing import Dict, Tuple, Optional

# Import from existing src
import sys
sys.path.append(str(Path(__file__).parent.parent.parent))
from src.models.gat import create_gat_model

logger = logging.getLogger(__name__)


class ModelLoader:
    """Handles loading and configuration of GAT models"""
    
    def __init__(self, device: str = 'cpu'):
        """
        Initialize model loader
        
        Args:
            device: Device to load model on ('cpu' or 'cuda')
        """
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        logger.info(f"Model loader initialized with device: {self.device}")
    
    def load_checkpoint(self, checkpoint_path: str) -> Tuple[torch.nn.Module, Dict]:
        """
        Load model from checkpoint
        
        Args:
            checkpoint_path: Path to model checkpoint
            
        Returns:
            Tuple of (model, checkpoint_dict)
        """
        checkpoint_path = Path(checkpoint_path)
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Model checkpoint not found: {checkpoint_path}")
        
        logger.info(f"Loading model from: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        # Detect model configuration
        config = self._extract_model_config(checkpoint['model_state_dict'])
        logger.info(f"Detected model config: hidden={config['hidden_channels']}, "
                   f"layers={config['num_layers']}, heads={config['heads']}")
        
        # Create model
        model = self._create_model(config)
        
        # Load weights
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        model = model.to(self.device)
        
        logger.info("Model loaded successfully")
        return model, checkpoint
    
    def _extract_model_config(self, state_dict: Dict) -> Dict:
        """
        Extract model configuration from state dict
        
        Args:
            state_dict: Model state dictionary
            
        Returns:
            Configuration dictionary
        """
        # Get hidden channels from input projection
        hidden_channels = state_dict['input_proj.weight'].shape[0]
        
        # Get number of layers
        num_layers = max([int(k.split('.')[1]) for k in state_dict.keys() 
                         if k.startswith('gat_layers.')]) + 1
        
        # Get attention heads per layer
        heads = []
        for i in range(num_layers):
            if f'gat_layers.{i}.gat_conv.att_src' in state_dict:
                n_heads = state_dict[f'gat_layers.{i}.gat_conv.att_src'].shape[1]
                heads.append(n_heads)
        
        return {
            'hidden_channels': hidden_channels,
            'num_layers': num_layers,
            'heads': heads
        }
    
    def _create_model(self, config: Dict) -> torch.nn.Module:
        """
        Create model from configuration
        
        Args:
            config: Model configuration
            
        Returns:
            Created model
        """
        return create_gat_model(
            num_features=49,  # 45 features + 4 augmented
            model_type='standard',
            hidden_channels=config['hidden_channels'],
            num_layers=config['num_layers'],
            heads=config['heads'],
            dropout=0.1,
            edge_dim=1,
            residual=True,
            layer_norm=True,
            feature_augmentation=False,
            pool_type='mean',
            dtype=torch.float32
        )