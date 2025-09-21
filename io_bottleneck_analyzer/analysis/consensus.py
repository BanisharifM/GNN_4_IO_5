"""
Consensus builder for combining multiple interpretability methods
"""
import numpy as np
import logging
from typing import Dict, List, Tuple

logger = logging.getLogger(__name__)


class ConsensusBuilder:
    """Builds consensus from multiple interpretability methods"""
    
    def __init__(self):
        """Initialize consensus builder"""
        self.normalized_scores = {}
        self.feature_contributions = {}
        
    def calculate_consensus(self, 
                          methods_results: Dict[str, Dict[str, float]],
                          verbose: bool = True) -> Dict[str, float]:
        """
        Calculate consensus using z-score normalization
        
        Args:
            methods_results: Results from each method
            verbose: Whether to log detailed information
            
        Returns:
            Consensus scores for features
        """
        if verbose:
            logger.info("\n" + "="*70)
            logger.info("Z-SCORE NORMALIZATION AND CONSENSUS CALCULATION")
            logger.info("="*70)
            self._log_raw_scores(methods_results)
        
        # Z-normalize each method
        normalized_scores = self._normalize_scores(methods_results, verbose)
        
        # Calculate consensus
        consensus = self._compute_consensus(normalized_scores, verbose)
        
        return consensus
    
    def _log_raw_scores(self, methods_results: Dict[str, Dict[str, float]]):
        """Log raw scores from each method"""
        logger.info("\nStep 1: Raw Scores from Each Method")
        logger.info("-" * 50)
        
        for method_name, scores in methods_results.items():
            if scores:
                logger.info(f"\n{method_name.upper().replace('_', ' ')}:")
                sorted_scores = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:5]
                for feat, score in sorted_scores:
                    logger.info(f"  - {feat:30s}: {score:8.4f}")
    
    def _normalize_scores(self, 
                         methods_results: Dict[str, Dict[str, float]],
                         verbose: bool) -> Dict[str, Dict[str, float]]:
        """Normalize scores using z-score normalization"""
        if verbose:
            logger.info("\nStep 2: Z-Score Normalization")
            logger.info("-" * 50)
            logger.info("Formula: z = (x - mean) / std")
        
        normalized_scores = {}
        self.normalized_scores = {}
        
        for method_name, scores in methods_results.items():
            if not scores:
                continue
            
            method_scores = list(scores.values())
            
            if len(method_scores) > 1:
                mean_score = np.mean(method_scores)
                std_score = np.std(method_scores)
                
                if verbose:
                    logger.info(f"\n{method_name.upper().replace('_', ' ')}:")
                    logger.info(f"  Mean = {mean_score:.4f}, Std = {std_score:.4f}")
                
                if std_score > 0:
                    normalized_scores[method_name] = {}
                    self.normalized_scores[method_name] = {}
                    
                    # Normalize all scores
                    for feature, score in scores.items():
                        z_score = (score - mean_score) / std_score
                        normalized_scores[method_name][feature] = z_score
                        self.normalized_scores[method_name][feature] = z_score
                    
                    # Log top normalized scores
                    if verbose:
                        logger.info(f"  Z-normalized scores (Top 5):")
                        sorted_items = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:5]
                        for feature, _ in sorted_items:
                            z_score = normalized_scores[method_name][feature]
                            interpretation = self._interpret_z_score(z_score)
                            logger.info(f"    - {feature:30s}: {z_score:+7.3f} ({interpretation})")
                else:
                    # Zero std - all scores are the same
                    normalized_scores[method_name] = {feature: 0 for feature in scores}
                    self.normalized_scores[method_name] = normalized_scores[method_name].copy()
            else:
                # Single score - can't normalize
                normalized_scores[method_name] = {feature: 0 for feature in scores}
                self.normalized_scores[method_name] = normalized_scores[method_name].copy()
        
        return normalized_scores
    
    def _compute_consensus(self, 
                          normalized_scores: Dict[str, Dict[str, float]],
                          verbose: bool) -> Dict[str, float]:
        """Compute final consensus scores"""
        if verbose:
            logger.info("\nStep 3: Equal Weight Consensus")
            logger.info("-" * 50)
            num_methods = len(normalized_scores)
            weight = 1.0 / num_methods if num_methods > 0 else 0
            logger.info(f"Formula: Consensus = {weight:.2f} × " + 
                       " + ".join([f"{m.replace('_', ' ')}" for m in normalized_scores.keys()]))
        
        # Get all features
        all_features = set()
        for method_scores in normalized_scores.values():
            all_features.update(method_scores.keys())
        
        # Calculate consensus
        consensus = {}
        self.feature_contributions = {}
        
        for feature in all_features:
            z_scores = []
            contributors = []
            
            for method_name, method_scores in normalized_scores.items():
                if feature in method_scores:
                    z_score = method_scores[feature]
                    z_scores.append(z_score)
                    method_short = self._shorten_method_name(method_name)
                    contributors.append(f"{method_short}({z_score:+.2f})")
            
            if z_scores:
                consensus[feature] = np.mean(z_scores)
                self.feature_contributions[feature] = contributors
        
        # Sort by consensus score
        consensus = dict(sorted(consensus.items(), key=lambda x: x[1], reverse=True))
        
        # Log final rankings
        if verbose:
            self._log_consensus_rankings(consensus)
        
        return consensus
    
    def _log_consensus_rankings(self, consensus: Dict[str, float]):
        """Log final consensus rankings"""
        logger.info("\nFinal Consensus Rankings:")
        
        for i, (feature, score) in enumerate(list(consensus.items())[:10], 1):
            contributors = " + ".join(self.feature_contributions[feature])
            num_methods = len(self.feature_contributions[feature])
            
            if num_methods > 1:
                strength = "STRONG (multiple methods agree)"
            else:
                strength = "WEAK (single method)"
            
            logger.info(f"Rank {i:2d}: {feature:30s} → {score:+7.3f}")
            logger.info(f"         Contributing methods: {contributors}")
            logger.info(f"         Consensus strength: {strength}")
            
            if i < 10:
                logger.info("")
    
    def _interpret_z_score(self, z_score: float) -> str:
        """Interpret z-score value"""
        if z_score > 1.5:
            return "very high importance"
        elif z_score > 0.5:
            return "high importance"
        elif z_score > 0:
            return "medium importance"
        elif z_score > -0.5:
            return "low importance"
        else:
            return "very low importance"
    
    def _shorten_method_name(self, method_name: str) -> str:
        """Shorten method name for display"""
        replacements = {
            'gnn_explainer': 'GNN',
            'gradients': 'Grad',
            'attention': 'Att'
        }
        for old, new in replacements.items():
            method_name = method_name.replace(old, new)
        return method_name
    
    def get_normalized_scores(self) -> Dict[str, Dict[str, float]]:
        """Get normalized scores for all methods"""
        return self.normalized_scores
    
    def get_feature_contributions(self) -> Dict[str, List[str]]:
        """Get feature contributions for consensus"""
        return self.feature_contributions