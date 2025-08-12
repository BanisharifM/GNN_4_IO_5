"""
I/O Performance Graph Dataset for GNN Training
Handles 100K nodes with 5M edges efficiently with lazy loading
"""

import torch
import numpy as np
import pandas as pd
from torch_geometric.data import Data, InMemoryDataset
from typing import Optional, Tuple, Dict, Union
import logging
from pathlib import Path
import gc
import warnings
warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class IOPerformanceGraphDataset(InMemoryDataset):
    """
    PyTorch Geometric dataset for I/O performance prediction
    Optimized for large graphs with float64 precision
    """
    
    def __init__(
        self,
        root: str,
        similarity_pt_path: str,
        similarity_npz_path: str,
        features_csv_path: str,
        transform=None,
        pre_transform=None,
        pre_filter=None,
        use_edge_weights: bool = True,
        dtype: torch.dtype = torch.float64,
        lazy_load: bool = True
    ):
        """
        Args:
            root: Root directory for processed data
            similarity_pt_path: Path to similarity graph PyTorch file
            similarity_npz_path: Path to similarity graph NumPy file
            features_csv_path: Path to normalized features CSV
            use_edge_weights: Whether to use similarity scores as edge weights
            dtype: Data type (default float64 for precision)
            lazy_load: Whether to use lazy loading for memory efficiency
        """
        self.similarity_pt_path = Path(similarity_pt_path)
        self.similarity_npz_path = Path(similarity_npz_path)
        self.features_csv_path = Path(features_csv_path)
        self.use_edge_weights = use_edge_weights
        self.dtype = dtype
        self.lazy_load = lazy_load
        
        # Cache for lazy loading
        self._edge_index = None
        self._edge_weight = None
        self._node_features = None
        self._targets = None
        self._num_nodes = None
        
        super().__init__(root, transform, pre_transform, pre_filter)
        self.data, self.slices = torch.load(self.processed_paths[0])
        
    @property
    def raw_file_names(self):
        """Files needed for processing"""
        return []
    
    @property
    def processed_file_names(self):
        """Processed data file"""
        return ['io_graph_data.pt']
    
    def download(self):
        """No download needed - using local files"""
        pass
    
    def _load_similarity_graph(self) -> Tuple[torch.Tensor, Optional[torch.Tensor], int]:
        """
        Load similarity graph with memory optimization
        Returns: edge_index, edge_weight, num_nodes
        """
        logger.info("Loading similarity graph...")
        
        if self.lazy_load and self._edge_index is not None:
            return self._edge_index, self._edge_weight, self._num_nodes
        
        try:
            # Load PyTorch file (preferred for consistency)
            graph_data = torch.load(self.similarity_pt_path, map_location='cpu')
            
            edge_index = graph_data['edge_index'].to(torch.long)
            edge_weight = None
            
            if self.use_edge_weights and 'edge_weight' in graph_data:
                edge_weight = graph_data['edge_weight'].to(self.dtype)
                logger.info(f"Edge weights loaded: shape {edge_weight.shape}, "
                          f"range [{edge_weight.min():.4f}, {edge_weight.max():.4f}]")
            
            num_nodes = graph_data['num_nodes']
            
            # Validate edge index
            assert edge_index.shape[0] == 2, f"Edge index should have shape [2, E], got {edge_index.shape}"
            assert edge_index.max() < num_nodes, f"Edge index max {edge_index.max()} >= num_nodes {num_nodes}"
            assert edge_index.min() >= 0, f"Edge index contains negative values"
            
            logger.info(f"Graph loaded: {num_nodes:,} nodes, {edge_index.shape[1]:,} edges")
            
            if self.lazy_load:
                self._edge_index = edge_index
                self._edge_weight = edge_weight
                self._num_nodes = num_nodes
            
            return edge_index, edge_weight, num_nodes
            
        except Exception as e:
            logger.error(f"Error loading similarity graph: {e}")
            raise
    
    def _load_node_features(self, num_nodes: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Load node features and targets with memory optimization
        Returns: features, targets
        """
        logger.info("Loading node features...")
        
        if self.lazy_load and self._node_features is not None:
            return self._node_features, self._targets
        
        try:
            # First try to load from NPZ (contains preprocessed features)
            if self.similarity_npz_path.exists():
                npz_data = np.load(self.similarity_npz_path)
                
                if 'features' in npz_data and 'target' in npz_data:
                    features = torch.tensor(npz_data['features'], dtype=self.dtype)
                    targets = torch.tensor(npz_data['target'], dtype=self.dtype)
                    
                    logger.info(f"Features loaded from NPZ: shape {features.shape}")
                    
                    if self.lazy_load:
                        self._node_features = features
                        self._targets = targets
                    
                    return features, targets
            
            # Fallback to CSV
            logger.info(f"Loading features from CSV: {self.features_csv_path}")
            
            # Use chunks for memory efficiency
            df = pd.read_csv(self.features_csv_path, dtype=np.float64)
            
            # Separate features and target
            if 'tag' in df.columns:
                targets = torch.tensor(df['tag'].values, dtype=self.dtype)
                features = torch.tensor(df.drop('tag', axis=1).values, dtype=self.dtype)
            else:
                # If no tag column, assume last column is target
                targets = torch.tensor(df.iloc[:, -1].values, dtype=self.dtype)
                features = torch.tensor(df.iloc[:, :-1].values, dtype=self.dtype)
            
            # Validate dimensions
            assert features.shape[0] == num_nodes, \
                f"Feature rows {features.shape[0]} != num_nodes {num_nodes}"
            assert targets.shape[0] == num_nodes, \
                f"Target rows {targets.shape[0]} != num_nodes {num_nodes}"
            
            logger.info(f"Features shape: {features.shape}, Targets shape: {targets.shape}")
            logger.info(f"Features range: [{features.min():.4f}, {features.max():.4f}]")
            logger.info(f"Targets range: [{targets.min():.4f}, {targets.max():.4f}]")
            
            if self.lazy_load:
                self._node_features = features
                self._targets = targets
            
            # Clean up DataFrame
            del df
            gc.collect()
            
            return features, targets
            
        except Exception as e:
            logger.error(f"Error loading node features: {e}")
            raise
    
    def process(self):
        """Process raw data and save"""
        logger.info("Processing I/O performance graph dataset...")
        
        # Load graph structure
        edge_index, edge_weight, num_nodes = self._load_similarity_graph()
        
        # Load node features and targets
        features, targets = self._load_node_features(num_nodes)
        
        # Create PyG Data object
        data = Data(
            x=features,
            edge_index=edge_index,
            edge_attr=edge_weight.unsqueeze(-1) if edge_weight is not None else None,
            y=targets.unsqueeze(-1) if targets.dim() == 1 else targets
        )
        
        # Add metadata
        data.num_nodes = num_nodes
        data.num_edges = edge_index.shape[1]
        data.num_features = features.shape[1]
        
        # Calculate additional graph statistics
        data = self._add_graph_statistics(data)
        
        # Apply pre-filter if specified
        if self.pre_filter is not None:
            data = self.pre_filter(data)
        
        # Apply pre-transform if specified
        if self.pre_transform is not None:
            data = self.pre_transform(data)
        
        # Save processed data
        logger.info(f"Saving processed data to {self.processed_paths[0]}")
        torch.save(self.collate([data]), self.processed_paths[0])
        
    def _add_graph_statistics(self, data: Data) -> Data:
        """Add useful graph statistics for analysis"""
        edge_index = data.edge_index
        
        # Calculate degree for each node
        row, col = edge_index
        degree = torch.zeros(data.num_nodes, dtype=torch.long)
        degree.scatter_add_(0, row, torch.ones_like(row))
        
        data.degree = degree
        data.avg_degree = degree.float().mean().item()
        data.max_degree = degree.max().item()
        data.min_degree = degree.min().item()
        
        logger.info(f"Graph statistics - Avg degree: {data.avg_degree:.2f}, "
                   f"Min: {data.min_degree}, Max: {data.max_degree}")
        
        return data
    
    def get_splits(
        self, 
        train_ratio: float = 0.6,
        val_ratio: float = 0.2,
        test_ratio: float = 0.2,
        seed: int = 42,
        stratify: bool = True
    ) -> Dict[str, torch.Tensor]:
        """
        Create train/val/test splits with optional stratification
        
        Args:
            train_ratio: Fraction for training
            val_ratio: Fraction for validation
            test_ratio: Fraction for testing
            seed: Random seed for reproducibility
            stratify: Whether to stratify by target values
        
        Returns:
            Dictionary with train_mask, val_mask, test_mask
        """
        torch.manual_seed(seed)
        np.random.seed(seed)
        
        num_nodes = self.data.num_nodes
        indices = np.arange(num_nodes)
        
        if stratify:
            # Stratified split based on performance quartiles
            y = self.data.y.squeeze().numpy()
            quartiles = np.percentile(y, [25, 50, 75])
            
            # Assign each node to a stratum
            strata = np.digitize(y, quartiles)
            
            train_indices = []
            val_indices = []
            test_indices = []
            
            # Sample from each stratum
            for stratum in np.unique(strata):
                stratum_indices = indices[strata == stratum]
                np.random.shuffle(stratum_indices)
                
                n = len(stratum_indices)
                n_train = int(n * train_ratio)
                n_val = int(n * val_ratio)
                
                train_indices.extend(stratum_indices[:n_train])
                val_indices.extend(stratum_indices[n_train:n_train + n_val])
                test_indices.extend(stratum_indices[n_train + n_val:])
        else:
            # Random split
            np.random.shuffle(indices)
            n_train = int(num_nodes * train_ratio)
            n_val = int(num_nodes * val_ratio)
            
            train_indices = indices[:n_train]
            val_indices = indices[n_train:n_train + n_val]
            test_indices = indices[n_train + n_val:]
        
        # Create masks
        train_mask = torch.zeros(num_nodes, dtype=torch.bool)
        val_mask = torch.zeros(num_nodes, dtype=torch.bool)
        test_mask = torch.zeros(num_nodes, dtype=torch.bool)
        
        train_mask[train_indices] = True
        val_mask[val_indices] = True
        test_mask[test_indices] = True
        
        logger.info(f"Data splits - Train: {train_mask.sum():,}, "
                   f"Val: {val_mask.sum():,}, Test: {test_mask.sum():,}")
        
        return {
            'train_mask': train_mask,
            'val_mask': val_mask,
            'test_mask': test_mask
        }
    
    def get_memory_usage(self) -> Dict[str, float]:
        """Calculate memory usage of dataset components"""
        memory_usage = {}
        
        if self._edge_index is not None:
            memory_usage['edge_index'] = self._edge_index.element_size() * self._edge_index.nelement() / 1e9
        
        if self._edge_weight is not None:
            memory_usage['edge_weight'] = self._edge_weight.element_size() * self._edge_weight.nelement() / 1e9
        
        if self._node_features is not None:
            memory_usage['features'] = self._node_features.element_size() * self._node_features.nelement() / 1e9
        
        if self._targets is not None:
            memory_usage['targets'] = self._targets.element_size() * self._targets.nelement() / 1e9
        
        memory_usage['total'] = sum(memory_usage.values())
        
        logger.info(f"Memory usage (GB): {memory_usage}")
        return memory_usage


# Convenience function for quick loading
def load_io_dataset(
    root: str = './data/processed',
    similarity_pt_path: str = None,
    similarity_npz_path: str = None,
    features_csv_path: str = None,
    **kwargs
) -> IOPerformanceGraphDataset:
    """
    Convenience function to load the I/O performance dataset
    """
    return IOPerformanceGraphDataset(
        root=root,
        similarity_pt_path=similarity_pt_path,
        similarity_npz_path=similarity_npz_path,
        features_csv_path=features_csv_path,
        **kwargs
    )