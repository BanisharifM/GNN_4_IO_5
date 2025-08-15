#!/usr/bin/env python
"""
Job processor for complete pipeline: from Darshan log to bottleneck diagnosis
"""

import torch
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union
import logging
import yaml
import json
from torch_geometric.data import Data

logger = logging.getLogger(__name__)


class JobProcessor:
    """Process new jobs through the complete pipeline"""
    
    def __init__(
        self,
        model_checkpoint: str,
        model_config: str,
        similarity_path: str,
        features_csv_path: str,
        device: Optional[str] = None
    ):
        """
        Initialize job processor
        
        Args:
            model_checkpoint: Path to trained GAT model
            model_config: Path to model config YAML
            similarity_path: Path to precomputed similarity graph
            features_csv_path: Path to training features CSV
            device: Device to use ('cuda' or 'cpu')
        """
        self.model_checkpoint = Path(model_checkpoint)
        self.config_path = Path(model_config)
        
        # Load config
        with open(self.config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        # Set device
        if device:
            self.device = torch.device(device)
        elif torch.cuda.is_available():
            self.device = torch.device('cuda')
        else:
            self.device = torch.device('cpu')
        
        logger.info(f"Using device: {self.device}")
        
        # Initialize components
        self._init_neighbor_finder(similarity_path, features_csv_path)
        self._init_subgraph_builder()
        self._load_model()
        self._init_interpretability()
    
    def _init_neighbor_finder(self, similarity_path: str, features_csv_path: str):
        """Initialize neighbor finder"""
        from .neighbor_finder import NeighborFinder
        
        self.neighbor_finder = NeighborFinder(
            similarity_path=similarity_path,
            features_csv_path=features_csv_path,
            similarity_format='pt'
        )
        logger.info(f"Initialized neighbor finder with {self.neighbor_finder.num_nodes:,} nodes")
    
    def _init_subgraph_builder(self):
        """Initialize subgraph builder"""
        from .subgraph_builder import SubgraphBuilder
        
        self.subgraph_builder = SubgraphBuilder(
            edge_construction_method='knn',
            similarity_threshold=0.75,
            max_edges_per_node=10
        )
        logger.info("Initialized subgraph builder")
    
    def _load_model(self):
        """Load trained GAT model"""
        import sys
        from pathlib import Path
        
        # Add src to path for model imports
        project_root = Path(__file__).parent.parent.parent
        sys.path.insert(0, str(project_root))
        
        from src.models.gat import create_gat_model
        
        # Load checkpoint
        checkpoint = torch.load(self.model_checkpoint, map_location=self.device)
        logger.info(f"Loaded checkpoint from {self.model_checkpoint}")
        
        # Get model config
        model_config = self.config.get('model', {})
        
        # Create model with correct parameters
        # Your model expects 49 features (45 base + 4 augmented)
        self.model = create_gat_model(
            num_features=49,  # Important: 49 features after augmentation
            model_type=model_config.get('type', 'standard'),
            hidden_channels=model_config.get('hidden_channels', 256),
            num_layers=model_config.get('num_layers', 3),
            heads=model_config.get('heads', [8, 8, 1]),
            dropout=model_config.get('dropout', 0.2),
            residual=model_config.get('residual', True),
            layer_norm=model_config.get('layer_norm', True),
            feature_augmentation=False,  # We'll augment manually
            dtype=torch.float32
        )
        
        # Load weights
        if 'model_state_dict' in checkpoint:
            self.model.load_state_dict(checkpoint['model_state_dict'])
        else:
            self.model.load_state_dict(checkpoint)
        
        self.model = self.model.to(self.device)
        self.model.eval()
        
        logger.info("Model loaded and set to evaluation mode")
    
    def _init_interpretability(self):
        """Initialize interpretability analyzers"""
        import sys
        from pathlib import Path
        
        project_root = Path(__file__).parent.parent.parent
        sys.path.insert(0, str(project_root))
        
        from src.interpretability.attention_analyzer import AttentionAnalyzer
        from src.interpretability.gradient_methods import GradientAnalyzer
        
        # Get feature names
        self.feature_names = self.neighbor_finder.feature_names
        # Add augmented feature names
        self.feature_names_augmented = self.feature_names + [
            'node_degree', 'clustering_coef', 'betweenness', 'closeness'
        ]
        
        self.attention_analyzer = AttentionAnalyzer(
            self.model, self.feature_names_augmented, self.device
        )
        
        self.gradient_analyzer = GradientAnalyzer(
            self.model, self.feature_names_augmented, self.device
        )
        
        logger.info("Initialized interpretability analyzers")
    
    def augment_features(self, features: torch.Tensor) -> torch.Tensor:
        """
        Augment features from 45 to 49 dimensions
        
        Args:
            features: Original features (N, 45)
        
        Returns:
            Augmented features (N, 49)
        """
        if features.shape[1] == 45:
            # Add 4 graph-based features
            # For now, using placeholders - in practice, compute from graph
            n_nodes = features.shape[0]
            augmented = torch.zeros(n_nodes, 4, device=features.device)
            
            # Could compute actual graph features here:
            # augmented[:, 0] = node_degrees / max_degree
            # augmented[:, 1] = clustering_coefficients
            # augmented[:, 2] = betweenness_centrality
            # augmented[:, 3] = closeness_centrality
            
            return torch.cat([features, augmented], dim=1)
        
        return features
    
    def process_new_job(
        self,
        job_features: np.ndarray,
        k_neighbors: int = 50,
        return_details: bool = True
    ) -> Dict:
        """
        Complete pipeline for processing a new job
        
        Args:
            job_features: Features of new job (45 dimensions)
            k_neighbors: Number of neighbors to find
            return_details: Whether to return detailed analysis
        
        Returns:
            Dictionary with predictions and bottleneck analysis
        """
        logger.info("Processing new job...")
        
        # 1. Find neighbors
        logger.info(f"Finding {k_neighbors} nearest neighbors...")
        similarities, indices = self.neighbor_finder.find_neighbors_for_new_job(
            job_features, k=k_neighbors
        )
        
        # 2. Build subgraph (use top 20 for actual subgraph)
        logger.info("Building subgraph...")
        subgraph_k = min(20, k_neighbors)
        neighbor_features = self.neighbor_finder.training_features[indices[:subgraph_k]]
        
        subgraph = self.subgraph_builder.build_subgraph(
            query_features=job_features,
            neighbor_features=neighbor_features,
            neighbor_indices=indices[:subgraph_k],
            neighbor_similarities=similarities[:subgraph_k],
            include_neighbor_edges=True
        )
        
        # 3. Augment features for model
        logger.info("Augmenting features (45 -> 49)...")
        subgraph.x = self.augment_features(subgraph.x)
        
        # Move to device
        subgraph = subgraph.to(self.device)
        
        # 4. Run model inference
        logger.info("Running model inference...")
        with torch.no_grad():
            prediction = self.model(subgraph.x, subgraph.edge_index)
            
            # Get prediction for query node (index 0)
            query_prediction = prediction[0].item()
        
        # 5. Analyze bottlenecks
        logger.info("Analyzing bottlenecks...")
        bottlenecks = self.identify_bottlenecks(subgraph)
        
        # 6. Compile results
        results = {
            'predicted_performance': query_prediction,
            'actual_performance': None,  # Will be set if known
            'num_neighbors': len(indices),
            'neighbor_indices': indices.tolist(),
            'neighbor_similarities': similarities.tolist(),
            'bottlenecks': bottlenecks
        }
        
        if return_details:
            # Add neighbor performance statistics
            neighbor_perfs = self.neighbor_finder.training_performance[indices]
            results['neighbor_stats'] = {
                'mean': float(np.mean(neighbor_perfs)),
                'std': float(np.std(neighbor_perfs)),
                'min': float(np.min(neighbor_perfs)),
                'max': float(np.max(neighbor_perfs))
            }
        
        return results
    
    def identify_bottlenecks(self, subgraph: Data) -> Dict:
        """
        Identify bottlenecks using multiple interpretability methods
        
        Args:
            subgraph: Subgraph with query node at index 0
        
        Returns:
            Dictionary with bottleneck analysis
        """
        bottlenecks = {}
        
        # 1. Attention-based analysis
        try:
            attention_scores = self.attention_analyzer.analyze_node_importance(
                subgraph, target_node_idx=0
            )
            bottlenecks['attention'] = attention_scores
        except Exception as e:
            logger.warning(f"Attention analysis failed: {e}")
            bottlenecks['attention'] = {}
        
        # 2. Gradient-based analysis
        try:
            gradient_scores = self.gradient_analyzer.integrated_gradients(
                subgraph, target_node_idx=0
            )
            bottlenecks['gradients'] = gradient_scores
        except Exception as e:
            logger.warning(f"Gradient analysis failed: {e}")
            bottlenecks['gradients'] = {}
        
        # 3. Find consensus bottlenecks
        bottlenecks['consensus'] = self._find_consensus_bottlenecks(bottlenecks)
        
        return bottlenecks
    
    def _find_consensus_bottlenecks(
        self,
        bottlenecks: Dict,
        top_k: int = 10
    ) -> List[Tuple[str, float]]:
        """Find features that multiple methods agree are bottlenecks"""
        from collections import defaultdict
        
        feature_scores = defaultdict(list)
        
        # Aggregate scores from different methods
        for method, scores in bottlenecks.items():
            if method != 'consensus' and scores:
                for feature, score in scores.items():
                    feature_scores[feature].append(score)
        
        # Calculate consensus (average score across methods)
        consensus = []
        for feature, scores in feature_scores.items():
            if len(scores) >= 1:  # At least one method identified it
                avg_score = np.mean(scores)
                consensus.append((feature, avg_score))
        
        # Sort by score and return top-k
        consensus.sort(key=lambda x: x[1], reverse=True)
        
        return consensus[:top_k]
    
    def generate_recommendations(self, bottlenecks: Dict) -> List[str]:
        """
        Generate recommendations based on identified bottlenecks
        
        Args:
            bottlenecks: Dictionary with bottleneck analysis
        
        Returns:
            List of recommendation strings
        """
        recommendations = []
        
        # Get top bottlenecks
        if 'consensus' in bottlenecks:
            top_bottlenecks = bottlenecks['consensus'][:5]
        else:
            top_bottlenecks = []
        
        # Generate recommendations based on bottleneck patterns
        for feature, score in top_bottlenecks:
            if score > 0.5:  # Significant bottleneck
                rec = self._get_recommendation_for_feature(feature)
                if rec:
                    recommendations.append(rec)
        
        return recommendations
    
    def _get_recommendation_for_feature(self, feature: str) -> Optional[str]:
        """Get specific recommendation for a bottleneck feature"""
        
        recommendations_map = {
            'POSIX_SIZE_WRITE_0_100': "Small writes detected. Consider buffering writes or using larger transfer sizes.",
            'POSIX_SIZE_WRITE_100_1K': "Small writes detected. Increase write buffer size to at least 1MB.",
            'POSIX_SIZE_READ_0_100': "Small reads detected. Consider prefetching or larger read operations.",
            'POSIX_MEM_NOT_ALIGNED': "Memory alignment issues. Ensure buffers are properly aligned.",
            'POSIX_FILE_NOT_ALIGNED': "File alignment issues. Align I/O operations to file system block boundaries.",
            'LUSTRE_STRIPE_WIDTH': "Consider adjusting Lustre stripe width for better parallelism.",
            'LUSTRE_STRIPE_SIZE': "Lustre stripe size may be suboptimal. Try 1MB or 4MB stripe size.",
            'POSIX_SEQ_READS': "Sequential read pattern detected but may be inefficient.",
            'POSIX_SEQ_WRITES': "Sequential write pattern could benefit from larger buffer sizes.",
            'POSIX_RW_SWITCHES': "Frequent read/write switches. Consider separating read and write phases.",
            'POSIX_OPENS': "High number of file opens. Consider file pooling or keeping files open longer."
        }
        
        return recommendations_map.get(feature)