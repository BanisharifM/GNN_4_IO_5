"""
Data loader for training features and similarity matrices
"""
import pandas as pd
import numpy as np
import scipy.sparse as sp
import logging
from pathlib import Path
from typing import Optional, Tuple

logger = logging.getLogger(__name__)


class DataLoader:
    """Handles loading of training data and similarity matrices"""
    
    def __init__(self):
        """Initialize data loader"""
        self.training_features = None
        self.similarity_matrix = None
        self.feature_names = None
    
    def load_training_features(self, features_path: str) -> np.ndarray:
        """
        Load training features from CSV
        
        Args:
            features_path: Path to features CSV file
            
        Returns:
            Numpy array of features
        """
        features_path = Path(features_path)
        if not features_path.exists():
            raise FileNotFoundError(f"Training features not found: {features_path}")
        
        logger.info(f"Loading training features from: {features_path}")
        df = pd.read_csv(features_path)
        
        # Store feature names
        self.feature_names = [col for col in df.columns if col != 'tag']
        
        # Remove tag column if present
        if 'tag' in df.columns:
            df = df.drop('tag', axis=1)
        
        self.training_features = df.values.astype(np.float32)
        logger.info(f"Loaded {len(self.training_features):,} training samples "
                   f"with {self.training_features.shape[1]} features")
        
        return self.training_features
    
    def load_similarity_matrix(self, matrix_path: str) -> sp.spmatrix:
        """
        Load similarity matrix from npz file
        
        Args:
            matrix_path: Path to similarity matrix npz file
            
        Returns:
            Sparse similarity matrix
        """
        matrix_path = Path(matrix_path)
        if not matrix_path.exists():
            raise FileNotFoundError(f"Similarity matrix not found: {matrix_path}")
        
        logger.info(f"Loading similarity matrix from: {matrix_path}")
        self.similarity_matrix = sp.load_npz(matrix_path)
        logger.info(f"Loaded similarity matrix with shape: {self.similarity_matrix.shape}")
        
        return self.similarity_matrix
    
    def load_test_sample(self, test_path: str) -> Tuple[np.ndarray, float]:
        """
        Load test sample from CSV
        
        Args:
            test_path: Path to test sample CSV
            
        Returns:
            Tuple of (features, actual_tag)
        """
        test_path = Path(test_path)
        if not test_path.exists():
            raise FileNotFoundError(f"Test sample not found: {test_path}")
        
        logger.info(f"Loading test sample from: {test_path}")
        df = pd.read_csv(test_path)
        
        # Extract first row
        features = df.iloc[0, :-1].values.astype(np.float32)
        actual_tag = df.iloc[0, -1]
        
        logger.info(f"Loaded test sample with {len(features)} features")
        return features, actual_tag
    
    def validate_features(self, features: np.ndarray) -> bool:
        """
        Validate that features match expected format
        
        Args:
            features: Features array to validate
            
        Returns:
            True if valid
        """
        expected_features = 45  # Base POSIX features
        if features.shape[-1] not in [expected_features, expected_features + 4]:
            logger.warning(f"Unexpected feature count: {features.shape[-1]}")
            return False
        return True