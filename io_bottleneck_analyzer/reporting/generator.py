"""
Report generator for bottleneck analysis results
"""
import json
import logging
from pathlib import Path
from typing import Dict, Optional
from datetime import datetime

logger = logging.getLogger(__name__)


class ReportGenerator:
    """Generates reports from bottleneck analysis results"""
    
    def __init__(self, output_dir: str = './results'):
        """
        Initialize report generator
        
        Args:
            output_dir: Directory to save reports
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
    def generate_report(self, 
                       results: Dict,
                       save_full_scores: bool = True) -> Dict:
        """
        Generate comprehensive bottleneck report
        
        Args:
            results: Analysis results
            save_full_scores: Whether to save full scores report
            
        Returns:
            Generated report dictionary
        """
        # Import recommendations here to avoid circular imports
        from .recommendations import RecommendationEngine
        rec_engine = RecommendationEngine()
        
        # Build main report
        report = {
            'metadata': {
                'timestamp': datetime.now().isoformat(),
                'test_file': results.get('test_file', 'unknown'),
                'analysis_version': '1.0'
            },
            'performance': {
                'predicted_mbps': results['performance']['predicted_mbps'],
                'actual_mbps': results['performance']['actual_mbps'],
                'absolute_error_mbps': results['performance']['error_metrics']['absolute_error_mbps'],
                'relative_error_percent': results['performance']['error_metrics']['relative_error_percent']
            },
            'bottlenecks': {
                'attention': self._get_top_features(results['methods'].get('attention', {})),
                'gnn_explainer': self._get_top_features(results['methods'].get('gnn_explainer', {})),
                'gradients': self._get_top_features(results['methods'].get('gradients', {})),
                'consensus': self._get_top_features(results.get('consensus', {}))
            },
            'normalized_scores': {
                'attention': self._get_top_features(
                    results.get('normalized_scores', {}).get('attention', {})
                ),
                'gnn_explainer': self._get_top_features(
                    results.get('normalized_scores', {}).get('gnn_explainer', {})
                ),
                'gradients': self._get_top_features(
                    results.get('normalized_scores', {}).get('gradients', {})
                )
            }
        }
        
        # Add primary bottleneck and recommendations
        if results.get('consensus'):
            top_bottlenecks = list(results['consensus'].keys())[:5]
            report['primary_bottleneck'] = {
                'feature': top_bottlenecks[0] if top_bottlenecks else 'unknown',
                'consensus_score': results['consensus'][top_bottlenecks[0]] if top_bottlenecks else 0,
                'recommendation': rec_engine.get_recommendation(top_bottlenecks[0]) if top_bottlenecks else ''
            }
            
            # Add recommendations for top 5 bottlenecks
            report['recommendations'] = rec_engine.get_recommendations_for_bottlenecks(top_bottlenecks)
        
        # Save main report
        report_path = self._generate_filename('bottleneck_report.json')
        self.save_json(report, report_path)
        logger.info(f"Main report saved to: {report_path}")
        
        # Generate full scores report if requested
        if save_full_scores:
            full_report = self._generate_full_scores_report(results)
            full_report_path = self._generate_filename('bottleneck_report_full_scores.json')
            self.save_json(full_report, full_report_path)
            logger.info(f"Full scores report saved to: {full_report_path}")
        
        return report
    
    def _generate_full_scores_report(self, results: Dict) -> Dict:
        """Generate report with all scores"""
        return {
            'metadata': {
                'timestamp': datetime.now().isoformat(),
                'test_file': results.get('test_file', 'unknown'),
                'analysis_version': '1.0'
            },
            'performance': results['performance'],
            'all_raw_scores': {
                'attention': results['methods'].get('attention', {}),
                'gnn_explainer': results['methods'].get('gnn_explainer', {}),
                'gradients': results['methods'].get('gradients', {})
            },
            'all_normalized_scores': results.get('normalized_scores', {}),
            'consensus_all': results.get('consensus', {}),
            'feature_contributions': results.get('feature_contributions', {})
        }
    
    def _get_top_features(self, 
                         scores: Dict[str, float], 
                         top_k: int = 10) -> Dict[str, float]:
        """Get top-k features by score"""
        if not scores:
            return {}
        sorted_items = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:top_k]
        return dict(sorted_items)
    
    def _generate_filename(self, base_name: str) -> Path:
        """Generate unique filename with timestamp"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        name_parts = base_name.split('.')
        name_parts[0] = f"{name_parts[0]}_{timestamp}"
        return self.output_dir / '.'.join(name_parts)
    
    def save_json(self, data: Dict, filepath: Path):
        """Save dictionary as JSON file"""
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2, default=str)
    
    def generate_summary(self, report: Dict) -> str:
        """
        Generate text summary of the report
        
        Args:
            report: Generated report dictionary
            
        Returns:
            Text summary
        """
        summary_lines = [
            "=" * 70,
            "I/O BOTTLENECK ANALYSIS SUMMARY",
            "=" * 70,
            "",
            "Performance Metrics:",
            f"  Predicted: {report['performance']['predicted_mbps']:.2f} MB/s",
            f"  Actual: {report['performance']['actual_mbps']:.2f} MB/s",
            f"  Error: {report['performance']['absolute_error_mbps']:.2f} MB/s "
            f"({report['performance']['relative_error_percent']:.1f}%)",
            "",
            "Primary Bottleneck:",
            f"  Feature: {report['primary_bottleneck']['feature']}",
            f"  Consensus Score: {report['primary_bottleneck']['consensus_score']:.3f}",
            f"  Recommendation: {report['primary_bottleneck']['recommendation']}",
            "",
            "Top 5 Bottlenecks (Consensus):"
        ]
        
        for i, (feature, score) in enumerate(report['bottlenecks']['consensus'].items(), 1):
            if i > 5:
                break
            summary_lines.append(f"  {i}. {feature}: {score:.3f}")
        
        summary_lines.extend([
            "",
            "=" * 70
        ])
        
        return '\n'.join(summary_lines)