"""
Recommendation engine for I/O bottlenecks
"""
import logging
from typing import List, Dict, Tuple

logger = logging.getLogger(__name__)


class RecommendationEngine:
    """Generates recommendations for identified bottlenecks"""
    
    def __init__(self):
        """Initialize recommendation engine"""
        # Import from config to avoid circular imports
        from ..config.feature_definitions import BOTTLENECK_RECOMMENDATIONS
        self.recommendations = BOTTLENECK_RECOMMENDATIONS
        
        # Priority levels for different bottleneck types
        self.priority_levels = {
            'critical': [
                'POSIX_SIZE_WRITE_0_100',
                'POSIX_SIZE_READ_0_100',
                'POSIX_FILE_NOT_ALIGNED',
                'POSIX_SEEKS'
            ],
            'high': [
                'POSIX_SIZE_WRITE_100_1K',
                'POSIX_SIZE_WRITE_1K_10K',
                'POSIX_MEM_NOT_ALIGNED',
                'POSIX_RW_SWITCHES',
                'LUSTRE_STRIPE_SIZE'
            ],
            'medium': [
                'POSIX_OPENS',
                'POSIX_BYTES_WRITTEN',
                'POSIX_STRIDE1_STRIDE',
                'LUSTRE_STRIPE_WIDTH'
            ],
            'low': [
                'POSIX_CONSEC_WRITES',
                'POSIX_SEQ_WRITES',
                'POSIX_STATS'
            ]
        }
    
    def get_recommendation(self, feature: str) -> str:
        """
        Get recommendation for a specific feature
        
        Args:
            feature: Feature name
            
        Returns:
            Recommendation string
        """
        return self.recommendations.get(
            feature,
            self.recommendations['default']
        )
    
    def get_recommendations_for_bottlenecks(self, 
                                           bottlenecks: List[str],
                                           max_recommendations: int = 5) -> List[Dict]:
        """
        Get recommendations for multiple bottlenecks
        
        Args:
            bottlenecks: List of bottleneck feature names
            max_recommendations: Maximum number of recommendations
            
        Returns:
            List of recommendation dictionaries
        """
        recommendations = []
        
        for feature in bottlenecks[:max_recommendations]:
            priority = self.get_priority(feature)
            recommendation = {
                'feature': feature,
                'priority': priority,
                'recommendation': self.get_recommendation(feature),
                'category': self.get_category(feature)
            }
            recommendations.append(recommendation)
        
        # Sort by priority
        priority_order = {'critical': 0, 'high': 1, 'medium': 2, 'low': 3, 'unknown': 4}
        recommendations.sort(key=lambda x: priority_order.get(x['priority'], 4))
        
        return recommendations
    
    def get_priority(self, feature: str) -> str:
        """
        Get priority level for a bottleneck
        
        Args:
            feature: Feature name
            
        Returns:
            Priority level string
        """
        for priority, features in self.priority_levels.items():
            if feature in features:
                return priority
        return 'unknown'
    
    def get_category(self, feature: str) -> str:
        """
        Get category for a feature
        
        Args:
            feature: Feature name
            
        Returns:
            Category string
        """
        if 'SIZE_WRITE' in feature:
            return 'Write Size'
        elif 'SIZE_READ' in feature:
            return 'Read Size'
        elif 'STRIPE' in feature or 'LUSTRE' in feature:
            return 'File System Configuration'
        elif 'ALIGNED' in feature:
            return 'Alignment'
        elif 'SEQ' in feature or 'CONSEC' in feature:
            return 'Access Pattern'
        elif 'STRIDE' in feature:
            return 'Stride Pattern'
        elif 'BYTES' in feature:
            return 'Data Volume'
        elif feature in ['POSIX_OPENS', 'POSIX_STATS', 'POSIX_FILENOS']:
            return 'File Operations'
        else:
            return 'Other'
    
    def generate_optimization_plan(self, 
                                  recommendations: List[Dict]) -> List[Tuple[str, List[str]]]:
        """
        Generate optimization plan grouped by category
        
        Args:
            recommendations: List of recommendation dictionaries
            
        Returns:
            List of (category, recommendations) tuples
        """
        # Group by category
        plan = {}
        for rec in recommendations:
            category = rec['category']
            if category not in plan:
                plan[category] = []
            plan[category].append(rec['recommendation'])
        
        # Sort categories by importance
        category_order = [
            'Write Size', 'Read Size', 'Alignment', 
            'Access Pattern', 'File System Configuration',
            'Stride Pattern', 'File Operations', 'Data Volume', 'Other'
        ]
        
        sorted_plan = []
        for category in category_order:
            if category in plan:
                sorted_plan.append((category, plan[category]))
        
        return sorted_plan
    
    def estimate_impact(self, feature: str, score: float) -> str:
        """
        Estimate potential performance impact of fixing bottleneck
        
        Args:
            feature: Feature name
            score: Consensus score
            
        Returns:
            Impact estimate string
        """
        priority = self.get_priority(feature)
        
        # Estimate based on priority and score
        if priority == 'critical' and score > 1.0:
            return "Very High (>50% improvement possible)"
        elif priority == 'critical' or (priority == 'high' and score > 1.0):
            return "High (20-50% improvement possible)"
        elif priority == 'high' or (priority == 'medium' and score > 0.5):
            return "Medium (10-20% improvement possible)"
        elif priority == 'medium' or score > 0:
            return "Low (5-10% improvement possible)"
        else:
            return "Minimal (<5% improvement expected)"