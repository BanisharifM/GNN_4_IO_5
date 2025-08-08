"""
Data module for GNN4_IO_4.

This module provides utilities for data processing, graph construction,
and PyTorch Geometric data creation for I/O performance prediction.
"""

import os
import pandas as pd
import numpy as np
import torch
from torch_geometric.data import Data
from typing import Dict, List, Tuple, Optional, Union, Any
import logging
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import pickle
import json

# Set up logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

class GraphConstructor:
    """
    Base class for graph construction.
    """
    
    def __init__(self):
        """Initialize graph constructor."""
        pass
    
    def construct_graphs(self, data: pd.DataFrame) -> Dict[str, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Construct graphs from data.
        
        Args:
            data (pd.DataFrame): Input data
            
        Returns:
            Dict[str, Tuple[torch.Tensor, torch.Tensor]]: Dictionary mapping graph names to (edge_index, edge_attr) tuples
        """
        raise NotImplementedError("Subclasses must implement construct_graphs")
    
    def create_pyg_data(
        self, 
        data: pd.DataFrame, 
        target_column: Optional[str] = None
    ) -> Dict[str, Data]:
        """
        Create PyTorch Geometric Data objects.
        
        Args:
            data (pd.DataFrame): Input data
            target_column (str, optional): Target column for prediction
            
        Returns:
            Dict[str, Data]: Dictionary mapping graph names to PyG Data objects
        """
        raise NotImplementedError("Subclasses must implement create_pyg_data")
    
    def create_combined_pyg_data(
        self, 
        data: pd.DataFrame, 
        target_column: Optional[str] = None
    ) -> Data:
        """
        Create combined PyTorch Geometric Data object.
        
        Args:
            data (pd.DataFrame): Input data
            target_column (str, optional): Target column for prediction
            
        Returns:
            Data: Combined PyG Data object
        """
        raise NotImplementedError("Subclasses must implement create_combined_pyg_data")

class FeatureSimilarityGraphConstructor(GraphConstructor):
    """
    Graph constructor based on feature similarity.
    """
    
    def __init__(
        self, 
        feature: str,
        similarity_threshold: float = 0.1,
        similarity_metric: str = "cosine",
        max_edges_per_node: Optional[int] = None
    ):
        """
        Initialize feature similarity graph constructor.
        
        Args:
            feature (str): Feature to use for similarity calculation
            similarity_threshold (float): Threshold for similarity
            similarity_metric (str): Similarity metric to use ('cosine' or 'euclidean')
            max_edges_per_node (int, optional): Maximum number of edges per node
        """
        super().__init__()
        
        self.feature = feature
        self.similarity_threshold = similarity_threshold
        self.similarity_metric = similarity_metric
        self.max_edges_per_node = max_edges_per_node
        
        logger.info(f"Initialized feature similarity graph constructor for {feature}")
    
    def calculate_similarity(
        self, 
        data: pd.DataFrame
    ) -> np.ndarray:
        """
        Calculate similarity matrix.
        
        Args:
            data (pd.DataFrame): Input data
            
        Returns:
            np.ndarray: Similarity matrix
        """
        if self.feature not in data.columns:
            logger.warning(f"Feature {self.feature} not found in data")
            return np.zeros((len(data), len(data)))
        
        # Extract feature values
        feature_values = data[self.feature].values.reshape(-1, 1)
        
        # Calculate similarity
        if self.similarity_metric == "cosine":
            # Normalize feature values
            norms = np.linalg.norm(feature_values, axis=1, keepdims=True)
            norms[norms == 0] = 1.0  # Avoid division by zero
            normalized_values = feature_values / norms
            
            # Calculate cosine similarity
            similarity = normalized_values @ normalized_values.T
        elif self.similarity_metric == "euclidean":
            # Calculate pairwise distances
            distances = np.zeros((len(data), len(data)))
            for i in range(len(data)):
                distances[i] = np.abs(feature_values - feature_values[i]).flatten()
            
            # Convert distances to similarities
            max_distance = np.max(distances)
            if max_distance > 0:
                similarity = 1.0 - distances / max_distance
            else:
                similarity = np.ones((len(data), len(data)))
        else:
            raise ValueError(f"Unsupported similarity metric: {self.similarity_metric}")
        
        return similarity
    
    def construct_graphs(
        self, 
        data: pd.DataFrame
    ) -> Dict[str, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Construct graphs from data.
        
        Args:
            data (pd.DataFrame): Input data
            
        Returns:
            Dict[str, Tuple[torch.Tensor, torch.Tensor]]: Dictionary mapping graph names to (edge_index, edge_attr) tuples
        """
        if self.feature not in data.columns:
            logger.warning(f"Feature {self.feature} not found in data, skipping")
            return {}
        
        # Calculate similarity
        similarity = self.calculate_similarity(data)
        
        # Create edges
        edge_index = []
        edge_attr = []
        
        for i in range(len(data)):
            # Find similar nodes
            similar_indices = np.where(similarity[i] >= self.similarity_threshold)[0]
            
            # Remove self-loop
            similar_indices = similar_indices[similar_indices != i]
            
            # Limit number of edges per node
            if self.max_edges_per_node is not None and len(similar_indices) > self.max_edges_per_node:
                # Sort by similarity
                sorted_indices = np.argsort(similarity[i][similar_indices])[::-1]
                similar_indices = similar_indices[sorted_indices[:self.max_edges_per_node]]
            
            # Add edges
            for j in similar_indices:
                edge_index.append([i, j])
                edge_attr.append([similarity[i, j]])
        
        # Convert to tensors
        if edge_index:
            edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
            edge_attr = torch.tensor(edge_attr, dtype=torch.float)
        else:
            # Create empty tensors
            edge_index = torch.zeros((2, 0), dtype=torch.long)
            edge_attr = torch.zeros((0, 1), dtype=torch.float)
        
        return {self.feature: (edge_index, edge_attr)}
    
    def create_pyg_data(
        self, 
        data: pd.DataFrame, 
        target_column: Optional[str] = None
    ) -> Dict[str, Data]:
        """
        Create PyTorch Geometric Data objects.
        
        Args:
            data (pd.DataFrame): Input data
            target_column (str, optional): Target column for prediction
            
        Returns:
            Dict[str, Data]: Dictionary mapping graph names to PyG Data objects
        """
        # Construct graphs
        graphs = self.construct_graphs(data)
        
        # Create PyG Data objects
        pyg_data_dict = {}
        
        for graph_name, (edge_index, edge_attr) in graphs.items():
            # Create node features
            x = torch.tensor(data.drop(columns=[target_column] if target_column else []).values, dtype=torch.float)
            
            # Create target
            y = None
            if target_column and target_column in data.columns:
                y = torch.tensor(data[target_column].values, dtype=torch.float)
            
            # Create PyG Data object
            pyg_data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y)
            
            pyg_data_dict[graph_name] = pyg_data
        
        return pyg_data_dict
    
    def create_combined_pyg_data(
        self, 
        data: pd.DataFrame, 
        target_column: Optional[str] = None
    ) -> Data:
        """
        Create combined PyTorch Geometric Data object.
        
        Args:
            data (pd.DataFrame): Input data
            target_column (str, optional): Target column for prediction
            
        Returns:
            Data: Combined PyG Data object
        """
        # Create PyG Data objects
        pyg_data_dict = self.create_pyg_data(data, target_column)
        
        if not pyg_data_dict:
            # Create a simple Data object without edges
            x = torch.tensor(data.drop(columns=[target_column] if target_column else []).values, dtype=torch.float)
            
            y = None
            if target_column and target_column in data.columns:
                y = torch.tensor(data[target_column].values, dtype=torch.float)
            
            return Data(x=x, edge_index=torch.zeros((2, 0), dtype=torch.long), y=y)
        
        # Get first PyG Data object
        combined_data = list(pyg_data_dict.values())[0]
        
        return combined_data

class MultiplexGraphConstructor(GraphConstructor):
    """
    Multiplex graph constructor based on multiple features.
    """
    
    def __init__(
        self, 
        important_features: List[str],
        similarity_thresholds: Dict[str, float] = None,
        similarity_metric: str = "cosine",
        max_edges_per_node: Optional[int] = None
    ):
        """
        Initialize multiplex graph constructor.
        
        Args:
            important_features (List[str]): List of important features for graph construction
            similarity_thresholds (Dict[str, float], optional): Thresholds for similarity for each feature
            similarity_metric (str): Similarity metric to use ('cosine' or 'euclidean')
            max_edges_per_node (int, optional): Maximum number of edges per node
        """
        super().__init__()
        
        self.important_features = important_features
        
        if similarity_thresholds is None:
            self.similarity_thresholds = {feature: 0.1 for feature in important_features}
        else:
            self.similarity_thresholds = similarity_thresholds
        
        self.similarity_metric = similarity_metric
        self.max_edges_per_node = max_edges_per_node
        
        # Create feature similarity graph constructors
        self.graph_constructors = {}
        for feature in important_features:
            threshold = self.similarity_thresholds.get(feature, 0.1)
            self.graph_constructors[feature] = FeatureSimilarityGraphConstructor(
                feature=feature,
                similarity_threshold=threshold,
                similarity_metric=similarity_metric,
                max_edges_per_node=max_edges_per_node
            )
        
        logger.info(f"Initialized multiplex graph constructor for {len(important_features)} features")
    
    def construct_graphs(
        self, 
        data: pd.DataFrame
    ) -> Dict[str, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Construct multiplex graphs from data.
        
        Args:
            data (pd.DataFrame): Input data
            
        Returns:
            Dict[str, Tuple[torch.Tensor, torch.Tensor]]: Dictionary mapping graph names to (edge_index, edge_attr) tuples
        """
        logger.info(f"Constructing multiplex graphs for {len(self.important_features)} features")
        
        # Construct graphs for each feature
        multiplex_graphs = {}
        
        for feature, graph_constructor in self.graph_constructors.items():
            feature_graphs = graph_constructor.construct_graphs(data)
            multiplex_graphs.update(feature_graphs)
        
        logger.info(f"Created {len(multiplex_graphs)} multiplex graphs")
        
        return multiplex_graphs
    
    def create_combined_graph(
        self, 
        multiplex_graphs: Dict[str, Tuple[torch.Tensor, torch.Tensor]]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Create combined graph from multiplex graphs.
        
        Args:
            multiplex_graphs (Dict[str, Tuple[torch.Tensor, torch.Tensor]]): Dictionary mapping graph names to (edge_index, edge_attr) tuples
            
        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Combined (edge_index, edge_attr)
        """
        logger.info("Creating combined graph from multiplex graphs")
        
        if not multiplex_graphs:
            raise ValueError("No multiplex graphs provided")
        
        # Combine edge indices and attributes
        combined_edge_indices = []
        combined_edge_attrs = []
        
        for graph_name, (edge_index, edge_attr) in multiplex_graphs.items():
            combined_edge_indices.append(edge_index)
            
            # Add graph type as a feature
            graph_type = torch.ones((edge_attr.shape[0], 1), dtype=torch.float) * len(combined_edge_indices)
            edge_attr_with_type = torch.cat([edge_attr, graph_type], dim=1)
            
            combined_edge_attrs.append(edge_attr_with_type)
        
        # Concatenate edge indices and attributes
        combined_edge_index = torch.cat(combined_edge_indices, dim=1) if combined_edge_indices else torch.zeros((2, 0), dtype=torch.long)
        combined_edge_attr = torch.cat(combined_edge_attrs, dim=0) if combined_edge_attrs else torch.zeros((0, 2), dtype=torch.float)
        
        return combined_edge_index, combined_edge_attr
    
    def create_pyg_data(
        self, 
        data: pd.DataFrame, 
        target_column: Optional[str] = None
    ) -> Dict[str, Data]:
        """
        Create PyTorch Geometric Data objects.
        
        Args:
            data (pd.DataFrame): Input data
            target_column (str, optional): Target column for prediction
            
        Returns:
            Dict[str, Data]: Dictionary mapping graph names to PyG Data objects
        """
        logger.info("Creating PyTorch Geometric Data objects")
        
        # Construct multiplex graphs
        multiplex_graphs = self.construct_graphs(data)
        
        # Create combined graph
        try:
            combined_edge_index, combined_edge_attr = self.create_combined_graph(multiplex_graphs)
        except ValueError:
            # Create a simple Data object without edges
            x = torch.tensor(data.drop(columns=[target_column] if target_column else []).values, dtype=torch.float)
            
            y = None
            if target_column and target_column in data.columns:
                y = torch.tensor(data[target_column].values, dtype=torch.float)
            
            return {"combined": Data(x=x, edge_index=torch.zeros((2, 0), dtype=torch.long), y=y)}
        
        # Create node features
        x = torch.tensor(data.drop(columns=[target_column] if target_column else []).values, dtype=torch.float)
        
        # Create target
        y = None
        if target_column and target_column in data.columns:
            y = torch.tensor(data[target_column].values, dtype=torch.float)
        
        # Create PyG Data object
        combined_data = Data(x=x, edge_index=combined_edge_index, edge_attr=combined_edge_attr, y=y)
        
        return {"combined": combined_data}
    
    def create_combined_pyg_data(
        self, 
        data: pd.DataFrame, 
        target_column: Optional[str] = None
    ) -> Data:
        """
        Create combined PyTorch Geometric Data object.
        
        Args:
            data (pd.DataFrame): Input data
            target_column (str, optional): Target column for prediction
            
        Returns:
            Data: Combined PyG Data object
        """
        logger.info("Creating combined PyTorch Geometric Data object")
        
        # Create PyG Data objects
        pyg_data_dict = self.create_pyg_data(data, target_column)
        
        # Get combined PyG Data object
        combined_data = pyg_data_dict["combined"]
        
        return combined_data

class IODataProcessor:
    """
    Data processor for I/O counter data.
    """
    
    def __init__(
        self, 
        data_path: str,
        important_features: List[str] = None,
        similarity_thresholds: Dict[str, float] = None,
        similarity_metric: str = "cosine",
        max_edges_per_node: Optional[int] = None,
        precomputed_similarity_path: Optional[str] = None
    ):
        """
        Initialize I/O data processor.
        
        Args:
            data_path (str): Path to data CSV file
            important_features (List[str], optional): List of important features for graph construction
            similarity_thresholds (Dict[str, float], optional): Thresholds for similarity for each feature
            similarity_metric (str): Similarity metric to use ('cosine' or 'euclidean')
            max_edges_per_node (int, optional): Maximum number of edges per node
        """
        self.data_path = data_path
        self.important_features = important_features
        self.similarity_thresholds = similarity_thresholds
        self.similarity_metric = similarity_metric
        self.max_edges_per_node = max_edges_per_node

        self.precomputed_similarity_path = precomputed_similarity_path
        
        self.data = None
        self.scaler = None
        self.graph_constructor = None

        self._sim_cache = None # Cache for precomputed similarity dict

        self._combined_data: Optional[Data] = None
        
        logger.info(f"Initialized I/O data processor for {data_path}")
    
    def load_data(self) -> pd.DataFrame:
        """
        Load data from CSV file.
        
        Returns:
            pd.DataFrame: Loaded data
        """
        logger.info(f"Loading data from {self.data_path}")
        
        # Load data
        self.data = pd.read_csv(self.data_path)
        
        logger.info(f"Loaded data with shape {self.data.shape}")
        
        return self.data
    
    def preprocess_data(self) -> pd.DataFrame:
        """
        Preprocess data.
        
        Returns:
            pd.DataFrame: Preprocessed data
        """
        logger.info("Preprocessing data")
        
        if self.data is None:
            self.load_data()
        
        # If a pre-computed similarity file is supplied, nothing else to do
        if self.precomputed_similarity_path and os.path.exists(self.precomputed_similarity_path):
            logger.info("Pre-computed similarity file found – skipping per-feature graph constructor creation.")
            return self.data

        # Create graph constructor
        if self.graph_constructor is None and self.important_features is not None:
            self.graph_constructor = MultiplexGraphConstructor(
                important_features=self.important_features,
                similarity_thresholds=self.similarity_thresholds,
                similarity_metric=self.similarity_metric,
                max_edges_per_node=self.max_edges_per_node
            )
        
        return self.data
    
    def construct_multiplex_graphs(self) -> Dict[str, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Construct multiplex graphs.
        
        Returns:
            Dict[str, Tuple[torch.Tensor, torch.Tensor]]: Dictionary mapping graph names to (edge_index, edge_attr) tuples
        """
        logger.info("Constructing multiplex graphs")

        if self.precomputed_similarity_path and os.path.exists(self.precomputed_similarity_path):
            logger.info(f"Loading precomputed similarity from {self.precomputed_similarity_path}")

            if self._sim_cache is None:
                self._sim_cache = torch.load(self.precomputed_similarity_path)
            else:
                logger.info("Using cached similarity dictionary")
            sim_dict = self._sim_cache

            # Case A: dict[str(feature)] -> dict[int node] -> List[(int dst, float sim)]
            if isinstance(sim_dict, dict) and len(sim_dict) > 0 and all(isinstance(k, str) for k in sim_dict.keys()):
                out: Dict[str, Tuple[torch.Tensor, torch.Tensor]] = {}
                # Preserve order if important_features provided
                feature_keys = [f for f in (self.important_features or list(sim_dict.keys())) if f in sim_dict]
                for feat in feature_keys:
                    ei, ea = [], []
                    for src, neighbors in sim_dict[feat].items():
                        for dst, sim in neighbors:
                            ei.append([src, dst])
                            ea.append([sim])
                    if ei:
                        ei = torch.tensor(ei, dtype=torch.long).t().contiguous()
                        ea = torch.tensor(ea, dtype=torch.float)
                    else:
                        ei = torch.zeros((2, 0), dtype=torch.long)
                        ea = torch.zeros((0, 1), dtype=torch.float)
                    out[feat] = (ei, ea)
                return out

            # Case B: legacy “combined” dict[int] -> List[(int dst, float sim)]
            ei, ea = [], []
            for src, neighbors in sim_dict.items():
                for dst, sim in neighbors:
                    ei.append([src, dst]); ea.append([sim])
            if ei:
                ei = torch.tensor(ei, dtype=torch.long).t().contiguous()
                ea = torch.tensor(ea, dtype=torch.float)
            else:
                ei = torch.zeros((2, 0), dtype=torch.long)
                ea = torch.zeros((0, 1), dtype=torch.float)
            return {"combined": (ei, ea)}
        
        if self.data is None:
            self.load_data()
        
        if self.graph_constructor is None:
            if self.important_features is None:
                raise ValueError("important_features must be provided")
            
            self.graph_constructor = MultiplexGraphConstructor(
                important_features=self.important_features,
                similarity_thresholds=self.similarity_thresholds,
                similarity_metric=self.similarity_metric,
                max_edges_per_node=self.max_edges_per_node
            )
        
        # Construct multiplex graphs
        multiplex_graphs = self.graph_constructor.construct_graphs(self.data)
        
        return multiplex_graphs
    
    def create_combined_pyg_data(self, target_column: Optional[str] = None) -> Data:
        # Cached?
        if self._combined_data is not None:
            return self._combined_data

        if self.data is None:
            self.load_data()

        # Build multiplex graphs (from precomputed or on-the-fly)
        multiplex_graphs = self.construct_multiplex_graphs()

        # Choose deterministic feature order
        if self.important_features:
            graph_keys = [k for k in self.important_features if k in multiplex_graphs]
        else:
            graph_keys = list(multiplex_graphs.keys())

        # Edge lists
        edge_indices_list = [multiplex_graphs[k][0] for k in graph_keys]
        edge_attrs_list   = [multiplex_graphs[k][1] for k in graph_keys]

        # Combined edge_index (backward-compat for any code still reading data.edge_index)
        if len(edge_indices_list) > 0:
            combined_edge_index = torch.cat(
                [ei for ei in edge_indices_list if ei.numel() > 0],
                dim=1
            ) if any(ei.numel() > 0 for ei in edge_indices_list) else torch.zeros((2, 0), dtype=torch.long)
        else:
            combined_edge_index = torch.zeros((2, 0), dtype=torch.long)

        # Node features / target
        x = torch.tensor(self.data.drop(columns=[target_column]).values, dtype=torch.float)
        y = torch.tensor(self.data[target_column].values, dtype=torch.float)

        # Build Data with multiplex attributes
        data = Data(x=x, y=y)
        data.edge_indices = edge_indices_list        # <- list[Tensor], one per graph type
        data.edge_attrs   = edge_attrs_list          # <- list[Tensor], aligned with edge_indices
        data.edge_index   = combined_edge_index      # <- keep for legacy use

        self._combined_data = data
        return data

    def train_val_test_split(
        self, 
        data: Data,
        train_ratio: float = 0.7,
        val_ratio: float = 0.15,
        test_ratio: float = 0.15,
        random_state: int = 42
    ) -> Data:
        """
        Split data into train, validation, and test sets.
        
        Args:
            data (Data): PyTorch Geometric Data object
            train_ratio (float): Ratio of training data
            val_ratio (float): Ratio of validation data
            test_ratio (float): Ratio of test data
            random_state (int): Random state for reproducibility
            
        Returns:
            Data: Data object with train, val, and test masks
        """
        logger.info("Splitting data into train, validation, and test sets")
        
        # Check ratios
        if train_ratio + val_ratio + test_ratio != 1.0:
            logger.warning("Ratios do not sum to 1.0, normalizing")
            total = train_ratio + val_ratio + test_ratio
            train_ratio /= total
            val_ratio /= total
            test_ratio /= total
        
        # Create indices
        indices = np.arange(data.x.shape[0])
        
        # Split indices
        train_indices, test_indices = train_test_split(
            indices, test_size=test_ratio, random_state=random_state
        )
        
        train_indices, val_indices = train_test_split(
            train_indices, test_size=val_ratio / (train_ratio + val_ratio), random_state=random_state
        )
        
        # Create masks
        train_mask = torch.zeros(data.x.shape[0], dtype=torch.bool)
        val_mask = torch.zeros(data.x.shape[0], dtype=torch.bool)
        test_mask = torch.zeros(data.x.shape[0], dtype=torch.bool)
        
        train_mask[train_indices] = True
        val_mask[val_indices] = True
        test_mask[test_indices] = True
        
        # Add masks to data
        data.train_mask = train_mask
        data.val_mask = val_mask
        data.test_mask = test_mask
        
        logger.info(f"Train: {train_mask.sum().item()}, Val: {val_mask.sum().item()}, Test: {test_mask.sum().item()}")
        
        return data
    
    def save_processed_data(
        self, 
        output_dir: str,
        target_column: Optional[str] = None
    ):
        """
        Save processed data.
        
        Args:
            output_dir (str): Output directory
            target_column (str, optional): Target column for prediction
        """
        logger.info(f"Saving processed data to {output_dir}")
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Save data
        self.data.to_csv(os.path.join(output_dir, "data.csv"), index=False)
        
        # Save multiplex graphs only if no precomputed file is provided
        if not self.precomputed_similarity_path:
            logger.info("No precomputed similarity path provided. Constructing multiplex graphs.")
            multiplex_graphs = self.construct_multiplex_graphs()

            for graph_name, (edge_index, edge_attr) in multiplex_graphs.items():
                torch.save(edge_index, os.path.join(output_dir, f"{graph_name}_edge_index.pt"))
                torch.save(edge_attr, os.path.join(output_dir, f"{graph_name}_edge_attr.pt"))
        else:
            logger.info("Precomputed similarity detected. Skipping multiplex graph construction.")
        
        # Save combined PyG data
        if self._combined_data is None:
            self._combined_data = self.create_combined_pyg_data(target_column)
        combined_data = self._combined_data
        torch.save(combined_data, os.path.join(output_dir, "combined_data.pt"))

        # Save configuration
        config = {
            "data_path": self.data_path,
            "important_features": self.important_features,
            "similarity_thresholds": self.similarity_thresholds,
            "similarity_metric": self.similarity_metric,
            "max_edges_per_node": self.max_edges_per_node
        }
        
        with open(os.path.join(output_dir, "config.json"), "w") as f:
            json.dump(config, f, indent=4)
        
        logger.info(f"Saved processed data to {output_dir}")
    
    @classmethod
    def load_processed_data(cls, input_dir: str) -> Tuple['IODataProcessor', Data]:
        """
        Load processed data.
        
        Args:
            input_dir (str): Input directory
            
        Returns:
            Tuple[IODataProcessor, Data]: Data processor and combined PyG data
        """
        logger.info(f"Loading processed data from {input_dir}")
        
        # Load configuration
        with open(os.path.join(input_dir, "config.json"), "r") as f:
            config = json.load(f)
        
        # Create data processor
        processor = cls(
            data_path=config["data_path"],
            important_features=config["important_features"],
            similarity_thresholds=config["similarity_thresholds"],
            similarity_metric=config["similarity_metric"],
            max_edges_per_node=config["max_edges_per_node"]
        )
        
        # Load data
        processor.data = pd.read_csv(os.path.join(input_dir, "data.csv"))
        
        # Load combined PyG data
        combined_data = torch.load(os.path.join(input_dir, "combined_data.pt"))
        
        logger.info(f"Loaded processed data from {input_dir}")
        
        return processor, combined_data
