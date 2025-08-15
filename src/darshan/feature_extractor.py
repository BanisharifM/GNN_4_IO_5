#!/usr/bin/env python
"""
Feature extraction from Darshan logs matching AIIO paper features
"""

import numpy as np
import logging
from typing import Dict, List, Optional
from pathlib import Path

logger = logging.getLogger(__name__)


class FeatureExtractor:
    """Extract and normalize features for GNN model"""
    
    # Define the 45 features used in training (excluding 'tag')
    FEATURE_NAMES = [
        'nprocs', 'POSIX_OPENS', 'LUSTRE_STRIPE_SIZE', 'LUSTRE_STRIPE_WIDTH',
        'POSIX_FILENOS', 'POSIX_MEM_ALIGNMENT', 'POSIX_FILE_ALIGNMENT',
        'POSIX_READS', 'POSIX_WRITES', 'POSIX_SEEKS', 'POSIX_STATS',
        'POSIX_BYTES_READ', 'POSIX_BYTES_WRITTEN', 'POSIX_CONSEC_READS',
        'POSIX_CONSEC_WRITES', 'POSIX_SEQ_READS', 'POSIX_SEQ_WRITES',
        'POSIX_RW_SWITCHES', 'POSIX_MEM_NOT_ALIGNED', 'POSIX_FILE_NOT_ALIGNED',
        'POSIX_SIZE_READ_0_100', 'POSIX_SIZE_READ_100_1K', 'POSIX_SIZE_READ_1K_10K',
        'POSIX_SIZE_READ_100K_1M', 'POSIX_SIZE_WRITE_0_100', 'POSIX_SIZE_WRITE_100_1K',
        'POSIX_SIZE_WRITE_1K_10K', 'POSIX_SIZE_WRITE_10K_100K', 'POSIX_SIZE_WRITE_100K_1M',
        'POSIX_STRIDE1_STRIDE', 'POSIX_STRIDE2_STRIDE', 'POSIX_STRIDE3_STRIDE',
        'POSIX_STRIDE4_STRIDE', 'POSIX_STRIDE1_COUNT', 'POSIX_STRIDE2_COUNT',
        'POSIX_STRIDE3_COUNT', 'POSIX_STRIDE4_COUNT', 'POSIX_ACCESS1_ACCESS',
        'POSIX_ACCESS2_ACCESS', 'POSIX_ACCESS3_ACCESS', 'POSIX_ACCESS4_ACCESS',
        'POSIX_ACCESS1_COUNT', 'POSIX_ACCESS2_COUNT', 'POSIX_ACCESS3_COUNT',
        'POSIX_ACCESS4_COUNT'
    ]
    
    def __init__(self, normalization_stats: Optional[Dict] = None):
        """
        Initialize feature extractor
        
        Args:
            normalization_stats: Pre-computed normalization statistics from training data
                                Should contain 'mean' and 'std' for each feature
        """
        self.normalization_stats = normalization_stats
    
    def extract_features(self, 
                        posix_counters: Dict[str, float],
                        lustre_info: Dict[str, float],
                        job_info: Dict) -> np.ndarray:
        """
        Extract feature vector matching training data format
        
        Args:
            posix_counters: POSIX counters from Darshan
            lustre_info: Lustre stripe information
            job_info: Job-level information
        
        Returns:
            Feature vector of shape (45,)
        """
        features = np.zeros(len(self.FEATURE_NAMES))
        
        for i, feature_name in enumerate(self.FEATURE_NAMES):
            if feature_name == 'nprocs':
                features[i] = job_info.get('nprocs', 1)
            elif feature_name.startswith('LUSTRE_'):
                features[i] = lustre_info.get(feature_name, 0)
            else:
                # POSIX counter
                features[i] = posix_counters.get(feature_name, 0)
        
        return features
    
    def calculate_performance_tag(self, 
                                 posix_counters: Dict[str, float],
                                 runtime: float) -> float:
        """
        Calculate performance 'tag' using AIIO paper formula
        
        From AIIO paper:
        Performance = Total_Bytes_Transferred / Runtime (in MB/s)
        Then apply log10 transformation
        
        Args:
            posix_counters: POSIX counters including bytes read/written
            runtime: Job runtime in seconds
        
        Returns:
            Log10-transformed performance value
        """
        if runtime <= 0:
            logger.warning("Invalid runtime, setting performance to 0")
            return 0.0
        
        # Calculate total bytes transferred
        bytes_read = posix_counters.get('POSIX_BYTES_READ', 0)
        bytes_written = posix_counters.get('POSIX_BYTES_WRITTEN', 0)
        total_bytes = bytes_read + bytes_written
        
        # Convert to MB/s
        performance_mbs = (total_bytes / (1024 * 1024)) / runtime
        
        # Apply log10 transformation (matching training data)
        # Add small epsilon to avoid log(0)
        epsilon = 1e-10
        performance_log = np.log10(performance_mbs + epsilon)
        
        logger.info(f"Performance: {performance_mbs:.2f} MB/s, Log10: {performance_log:.4f}")
        
        return performance_log
    
    def normalize_features(self, features: np.ndarray) -> np.ndarray:
        """
        Apply log10 normalization to features (matching training preprocessing)
        
        Args:
            features: Raw feature vector
        
        Returns:
            Normalized feature vector
        """
        # Apply log10 transformation with small epsilon to avoid log(0)
        epsilon = 1e-10
        normalized = np.log10(features + epsilon)
        
        # Handle special case for POSIX_FILENOS which can be very large
        # (as you discovered in your previous work)
        filenos_idx = self.FEATURE_NAMES.index('POSIX_FILENOS')
        if normalized[filenos_idx] > 10:  # Cap at reasonable value
            logger.warning(f"Large POSIX_FILENOS detected: {features[filenos_idx]}, capping normalized value")
            normalized[filenos_idx] = 10.0
        
        return normalized
    
    def extract_from_darshan_log(self, 
                                darshan_data: Dict,
                                include_performance: bool = True) -> Dict:
        """
        Complete feature extraction pipeline from parsed Darshan data
        
        Args:
            darshan_data: Parsed Darshan data from DarshanParser
            include_performance: Whether to calculate performance tag
        
        Returns:
            Dictionary with:
                - 'features': normalized feature vector
                - 'features_raw': raw feature values
                - 'performance': performance tag (if requested)
                - 'metadata': job metadata
        """
        from .parser import DarshanParser
        parser = DarshanParser()
        
        # Extract components
        posix_counters = parser.extract_posix_counters(darshan_data)
        lustre_info = parser.extract_lustre_info(darshan_data)
        job_info = parser.extract_job_info(darshan_data)
        
        # Extract raw features
        features_raw = self.extract_features(posix_counters, lustre_info, job_info)
        
        # Normalize features
        features_normalized = self.normalize_features(features_raw)
        
        result = {
            'features': features_normalized,
            'features_raw': features_raw,
            'feature_names': self.FEATURE_NAMES,
            'metadata': {
                'jobid': job_info.get('jobid', 'unknown'),
                'nprocs': job_info.get('nprocs', 1),
                'runtime': job_info.get('runtime', 0)
            }
        }
        
        # Calculate performance if requested
        if include_performance:
            performance = self.calculate_performance_tag(
                posix_counters, 
                job_info.get('runtime', 1)
            )
            result['performance'] = performance
            result['performance_raw'] = 10 ** performance  # Actual MB/s
        
        return result
    
    def process_darshan_log_file(self, 
                                log_path: str,
                                darshan_parser_path: Optional[str] = None) -> Dict:
        """
        Complete pipeline: parse log file and extract features
        
        Args:
            log_path: Path to Darshan log file
            darshan_parser_path: Optional path to darshan-parser executable
        
        Returns:
            Feature dictionary ready for model inference
        """
        from .parser import DarshanParser
        
        # Parse log
        parser = DarshanParser(darshan_parser_path)
        darshan_data = parser.parse_darshan_log(log_path)
        
        # Extract features
        return self.extract_from_darshan_log(darshan_data)


# Utility function for batch processing
def process_multiple_logs(log_paths: List[str], 
                        output_csv: Optional[str] = None) -> List[Dict]:
    """
    Process multiple Darshan logs
    
    Args:
        log_paths: List of paths to Darshan logs
        output_csv: Optional path to save features as CSV
    
    Returns:
        List of feature dictionaries
    """
    import pandas as pd
    
    extractor = FeatureExtractor()
    results = []
    
    for log_path in log_paths:
        try:
            logger.info(f"Processing: {log_path}")
            result = extractor.process_darshan_log_file(log_path)
            results.append(result)
        except Exception as e:
            logger.error(f"Failed to process {log_path}: {e}")
            continue
    
    # Save to CSV if requested
    if output_csv and results:
        # Convert to DataFrame
        rows = []
        for result in results:
            row = dict(zip(result['feature_names'], result['features']))
            row['performance'] = result.get('performance', np.nan)
            # row['jobid'] = result['metadata']['jobid']
            rows.append(row)
        
        df = pd.DataFrame(rows)
        df.to_csv(output_csv, index=False)
        logger.info(f"Saved features to {output_csv}")
    
    return results