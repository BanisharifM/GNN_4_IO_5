"""
Main analyzer orchestrator
"""
import logging
from pathlib import Path
from typing import Dict, Optional, Tuple
import sys

# Add parent path for imports
sys.path.append(str(Path(__file__).parent.parent.parent))

from ..config import (
    POSIX_FEATURES, 
    SIMILARITY_THRESHOLD,
    K_NEIGHBORS,
    SUBGRAPH_SIZE,
    ATTENTION_THRESHOLD,
    GNN_EXPLAINER_THRESHOLD,
    GNN_EXPLAINER_NUM_EPOCHS,
    RANDOM_SEED
)
from ..core import ModelLoader, DataLoader, GraphBuilder
from ..methods import AttentionMethod, GNNExplainerMethod, GradientMethod
from .predictor import PerformancePredictor
from .consensus import ConsensusBuilder

logger = logging.getLogger(__name__)


class BottleneckAnalyzer:
    """Main orchestrator for bottleneck analysis"""
    
    def __init__(self,
                 model_path: str,
                 data_dir: str,
                 similarity_matrix_path: Optional[str] = None,
                 training_features_path: Optional[str] = None,
                 device: str = 'cpu'):
        """
        Initialize bottleneck analyzer
        
        Args:
            model_path: Path to trained model
            data_dir: Directory containing data files
            similarity_matrix_path: Optional path to similarity matrix
            training_features_path: Optional path to training features
            device: Device to use for computation
        """
        self.device = device
        
        # Initialize components
        logger.info("Initializing Bottleneck Analyzer")
        
        # Load model
        self.model_loader = ModelLoader(device)
        self.model, self.checkpoint = self.model_loader.load_checkpoint(model_path)
        
        # Load data
        self.data_loader = DataLoader()
        if training_features_path:
            self.training_features = self.data_loader.load_training_features(training_features_path)
        else:
            self.training_features = None
            
        if similarity_matrix_path:
            self.similarity_matrix = self.data_loader.load_similarity_matrix(similarity_matrix_path)
        else:
            self.similarity_matrix = None
        
        # Initialize other components
        self.graph_builder = GraphBuilder(
            similarity_threshold=SIMILARITY_THRESHOLD,
            device=device
        )
        self.predictor = PerformancePredictor(self.model, device)
        self.consensus_builder = ConsensusBuilder()
        
        # Initialize interpretability methods
        self.attention_method = AttentionMethod(self.model, POSIX_FEATURES, device)
        self.gnn_explainer = GNNExplainerMethod(
            self.model, POSIX_FEATURES, device,
            num_epochs=GNN_EXPLAINER_NUM_EPOCHS
        )
        self.gradient_method = GradientMethod(self.model, POSIX_FEATURES, device)
        
        logger.info("Bottleneck Analyzer initialized successfully")
    
    def analyze(self, test_file: str) -> Dict:
        """
        Analyze bottlenecks for a test sample
        
        Args:
            test_file: Path to test CSV file
            
        Returns:
            Analysis results dictionary
        """
        logger.info("\n" + "="*70)
        logger.info("RUNNING BOTTLENECK ANALYSIS")
        logger.info("="*70)
        
        # Load test sample
        test_features, actual_tag = self.data_loader.load_test_sample(test_file)
        
        # Create subgraph
        logger.info("\nCreating analysis subgraph...")
        data, node_idx = self.graph_builder.create_subgraph(
            test_features,
            self.training_features,
            self.similarity_matrix,
            k_neighbors=K_NEIGHBORS,
            max_subgraph_size=SUBGRAPH_SIZE
        )
        
        # Predict performance
        logger.info("\nPredicting performance...")
        log_pred, mbps_pred = self.predictor.predict(data, node_idx)
        actual_mbps = 10**actual_tag - 1
        
        error_metrics = self.predictor.compute_error(mbps_pred, actual_mbps)
        
        logger.info(f"Predicted: {mbps_pred:.2f} MB/s")
        logger.info(f"Actual: {actual_mbps:.2f} MB/s")
        logger.info(f"Error: {error_metrics['absolute_error_mbps']:.2f} MB/s "
                   f"({error_metrics['relative_error_percent']:.1f}%)")
        
        # Run interpretability methods
        methods_results = {}
        
        # Attention analysis
        logger.info("\nRunning attention analysis...")
        methods_results['attention'] = self.attention_method.analyze(
            data, node_idx, threshold=ATTENTION_THRESHOLD, filter_irrelevant=True
        )
        
        # GNNExplainer analysis
        logger.info("Running GNNExplainer analysis...")
        methods_results['gnn_explainer'] = self.gnn_explainer.analyze(
            data, node_idx, 
            feature_mask_threshold=GNN_EXPLAINER_THRESHOLD,
            seed=RANDOM_SEED,
            filter_irrelevant=True
        )
        
        # Gradient analysis
        logger.info("Running gradient analysis...")
        methods_results['gradients'] = self.gradient_method.analyze(
            data, node_idx, method='integrated_gradients', filter_irrelevant=True
        )
        
        # Build consensus
        logger.info("\nBuilding consensus...")
        consensus = self.consensus_builder.calculate_consensus(methods_results)
        
        # Compile results
        results = {
            'test_file': test_file,
            'performance': {
                'predicted_mbps': mbps_pred,
                'actual_mbps': actual_mbps,
                'error_metrics': error_metrics
            },
            'methods': methods_results,
            'consensus': consensus,
            'normalized_scores': self.consensus_builder.get_normalized_scores(),
            'feature_contributions': self.consensus_builder.get_feature_contributions()
        }
        
        logger.info("\nAnalysis complete!")
        
        return results