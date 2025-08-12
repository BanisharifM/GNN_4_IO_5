"""
Data loaders for full-batch and mini-batch training
Automatically selects best strategy based on available memory
"""

import torch
from torch_geometric.loader import NeighborLoader, DataLoader
from torch_geometric.data import Data
import logging
import psutil
import GPUtil
from typing import Optional, Union, Tuple

logger = logging.getLogger(__name__)


class AdaptiveDataLoader:
    """
    Automatically selects between full-batch and mini-batch loading
    based on available GPU memory
    """
    
    def __init__(
        self,
        dataset: Data,
        device: torch.device = None,
        force_mode: Optional[str] = None,
        batch_size: int = 2048,
        num_neighbors: list = [25, 10],  # Neighbors per layer
        num_workers: int = 4,
        pin_memory: bool = True
    ):
        """
        Args:
            dataset: PyG Data object
            device: Target device (cuda/cpu)
            force_mode: Force 'full' or 'mini' batch mode
            batch_size: Batch size for mini-batch training
            num_neighbors: Number of neighbors to sample per layer
            num_workers: Number of data loading workers
            pin_memory: Pin memory for faster GPU transfer
        """
        self.dataset = dataset
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.batch_size = batch_size
        self.num_neighbors = num_neighbors
        self.num_workers = num_workers
        self.pin_memory = pin_memory and self.device.type == 'cuda'
        
        # Determine optimal loading strategy
        if force_mode:
            self.mode = force_mode
        else:
            self.mode = self._determine_mode()
        
        logger.info(f"Using {self.mode}-batch data loading")
        
    def _determine_mode(self) -> str:
        """
        Determine whether to use full-batch or mini-batch loading
        based on available memory
        """
        if self.device.type == 'cpu':
            # Check RAM
            available_ram = psutil.virtual_memory().available / 1e9  # GB
            estimated_usage = self._estimate_memory_usage()
            
            if estimated_usage < available_ram * 0.7:  # Use 70% threshold
                return 'full'
            else:
                return 'mini'
        else:
            # Check GPU memory
            try:
                gpus = GPUtil.getGPUs()
                if gpus:
                    available_gpu_mem = gpus[0].memoryFree / 1000  # GB
                    estimated_usage = self._estimate_memory_usage()
                    
                    # Conservative estimate - use 60% of available GPU memory
                    if estimated_usage < available_gpu_mem * 0.6:
                        return 'full'
            except:
                logger.warning("Could not determine GPU memory, defaulting to mini-batch")
            
            return 'mini'
    
    def _estimate_memory_usage(self) -> float:
        """
        Estimate memory usage for full-batch training (in GB)
        """
        data = self.dataset.data if hasattr(self.dataset, 'data') else self.dataset
        
        # Calculate sizes
        num_nodes = data.num_nodes
        num_edges = data.edge_index.shape[1]
        num_features = data.x.shape[1]
        
        # Estimate memory (float64 = 8 bytes)
        bytes_per_element = 8
        
        # Features and targets
        feature_memory = num_nodes * num_features * bytes_per_element
        target_memory = num_nodes * bytes_per_element
        
        # Graph structure
        edge_index_memory = 2 * num_edges * 8  # long type
        edge_weight_memory = num_edges * bytes_per_element if data.edge_attr is not None else 0
        
        # Model and gradients (rough estimate)
        model_memory = 100 * 1e6  # 100MB for model
        
        # Intermediate activations (rough estimate for GAT)
        # Assume 8 attention heads, 3 layers, hidden_dim=256
        activation_memory = num_nodes * 256 * 8 * 3 * bytes_per_element
        attention_memory = num_edges * 8 * 3 * bytes_per_element  # Attention scores
        
        total_memory = (
            feature_memory + target_memory + edge_index_memory + 
            edge_weight_memory + model_memory + activation_memory + attention_memory
        ) / 1e9  # Convert to GB
        
        logger.info(f"Estimated memory usage: {total_memory:.2f} GB")
        return total_memory
    
    def get_loader(self, masks: dict = None) -> Union[DataLoader, dict]:
        """
        Get appropriate data loader based on mode
        
        Args:
            masks: Dictionary with train_mask, val_mask, test_mask
        
        Returns:
            DataLoader for full-batch or dict of loaders for mini-batch
        """
        data = self.dataset.data if hasattr(self.dataset, 'data') else self.dataset
        
        if self.mode == 'full':
            return self._get_full_batch_loader(data, masks)
        else:
            return self._get_mini_batch_loaders(data, masks)
    
    def _get_full_batch_loader(self, data: Data, masks: dict = None) -> Data:
        """
        Prepare data for full-batch training
        """
        # Add masks to data
        if masks:
            data.train_mask = masks.get('train_mask')
            data.val_mask = masks.get('val_mask')
            data.test_mask = masks.get('test_mask')
        
        # Move to device if specified
        if self.device.type == 'cuda':
            data = data.to(self.device)
        
        logger.info(f"Full-batch mode: {data.num_nodes:,} nodes, {data.edge_index.shape[1]:,} edges")
        
        return data
    
    def _get_mini_batch_loaders(self, data: Data, masks: dict = None) -> dict:
        """
        Create mini-batch loaders using neighbor sampling
        """
        loaders = {}
        
        # Training loader
        if masks and 'train_mask' in masks:
            train_indices = masks['train_mask'].nonzero().squeeze()
            loaders['train'] = NeighborLoader(
                data,
                num_neighbors=self.num_neighbors,
                batch_size=self.batch_size,
                input_nodes=train_indices,
                shuffle=True,
                num_workers=self.num_workers,
                pin_memory=self.pin_memory
            )
            logger.info(f"Train loader: {len(train_indices):,} nodes, "
                       f"{len(loaders['train'])} batches")
        
        # Validation loader
        if masks and 'val_mask' in masks:
            val_indices = masks['val_mask'].nonzero().squeeze()
            loaders['val'] = NeighborLoader(
                data,
                num_neighbors=self.num_neighbors,
                batch_size=self.batch_size * 2,  # Larger batch for evaluation
                input_nodes=val_indices,
                shuffle=False,
                num_workers=self.num_workers,
                pin_memory=self.pin_memory
            )
            logger.info(f"Val loader: {len(val_indices):,} nodes, "
                       f"{len(loaders['val'])} batches")
        
        # Test loader
        if masks and 'test_mask' in masks:
            test_indices = masks['test_mask'].nonzero().squeeze()
            loaders['test'] = NeighborLoader(
                data,
                num_neighbors=self.num_neighbors,
                batch_size=self.batch_size * 2,
                input_nodes=test_indices,
                shuffle=False,
                num_workers=self.num_workers,
                pin_memory=self.pin_memory
            )
            logger.info(f"Test loader: {len(test_indices):,} nodes, "
                       f"{len(loaders['test'])} batches")
        
        return loaders
    
    def get_memory_stats(self):
        """Get current memory statistics"""
        stats = {}
        
        if self.device.type == 'cuda':
            torch.cuda.synchronize()
            stats['gpu_allocated'] = torch.cuda.memory_allocated() / 1e9
            stats['gpu_reserved'] = torch.cuda.memory_reserved() / 1e9
            
            try:
                gpus = GPUtil.getGPUs()
                if gpus:
                    stats['gpu_total'] = gpus[0].memoryTotal / 1000
                    stats['gpu_free'] = gpus[0].memoryFree / 1000
                    stats['gpu_used'] = gpus[0].memoryUsed / 1000
            except:
                pass
        
        stats['ram_available'] = psutil.virtual_memory().available / 1e9
        stats['ram_used'] = psutil.virtual_memory().used / 1e9
        stats['ram_total'] = psutil.virtual_memory().total / 1e9
        
        logger.info(f"Memory stats: {stats}")
        return stats