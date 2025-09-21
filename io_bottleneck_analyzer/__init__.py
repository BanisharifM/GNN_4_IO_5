"""
IO Bottleneck Analyzer

A production-ready tool for identifying I/O performance bottlenecks
using Graph Neural Networks and interpretability methods.
"""

__version__ = '1.0.0'

# Import main components for easy access
from .analysis import BottleneckAnalyzer
from .reporting import ReportGenerator, RecommendationEngine
from .config import POSIX_FEATURES, get_recommendation

__all__ = [
    'BottleneckAnalyzer',
    'ReportGenerator', 
    'RecommendationEngine',
    'POSIX_FEATURES',
    'get_recommendation'
]