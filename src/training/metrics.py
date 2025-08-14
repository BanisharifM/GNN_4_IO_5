"""
Evaluation metrics for comparing with AIIO
"""

import torch
import numpy as np
from typing import Dict, List, Tuple, Optional
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import logging

logger = logging.getLogger(__name__)


class PerformanceMetrics:
    """
    Calculate and track performance metrics for I/O prediction
    """
    
    def __init__(self, denormalize: bool = False, log_scale: bool = True):
        """
        Args:
            denormalize: Whether to denormalize predictions
            log_scale: Whether targets are in log scale
        """
        self.denormalize = denormalize
        self.log_scale = log_scale
        
        # AIIO reported metrics for comparison
        self.aiio_results = {
            'XGBoost': {'rmse': 0.5634, 'prediction_func': 0.5634, 'diagnosis_func': 0.2604},
            'LightGBM': {'rmse': 0.2632, 'prediction_func': 0.2632, 'diagnosis_func': 0.2599},
            'CatBoost': {'rmse': 0.2686, 'prediction_func': 0.2686, 'diagnosis_func': 0.2637},
            'MLP': {'rmse': 0.5416, 'prediction_func': 0.5416, 'diagnosis_func': 0.4611},
            'TabNet': {'rmse': 0.3078, 'prediction_func': 0.3078, 'diagnosis_func': 0.3077},
            'Closest_Method': {'rmse': 0.1860},
            'Average_Method': {'rmse': 0.2405}
        }
    
    def calculate_metrics(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
        return_numpy: bool = True
    ) -> Dict[str, float]:
        """
        Calculate all metrics
        
        Args:
            predictions: Model predictions
            targets: Ground truth values
            return_numpy: Convert to numpy for sklearn metrics
        
        Returns:
            Dictionary of metrics
        """
        # Ensure CPU and numpy for sklearn
        if isinstance(predictions, torch.Tensor):
            predictions = predictions.detach().cpu().squeeze()
            targets = targets.detach().cpu().squeeze()
        
        if return_numpy:
            predictions = predictions.numpy()
            targets = targets.numpy()
        
        # Denormalize if needed
        if self.denormalize and self.log_scale:
            predictions = np.exp(predictions) - 1
            targets = np.exp(targets) - 1
        
        # Calculate metrics
        metrics = {
            'rmse': np.sqrt(mean_squared_error(targets, predictions)),
            'mae': mean_absolute_error(targets, predictions),
            'r2': r2_score(targets, predictions),
            'mape': self.calculate_mape(targets, predictions),
            'max_error': np.max(np.abs(targets - predictions))
        }
        
        # Add normalized metrics if denormalized
        if self.denormalize:
            # Also calculate on log scale for comparison with AIIO
            log_pred = np.log(predictions + 1)
            log_target = np.log(targets + 1)
            metrics['rmse_log'] = np.sqrt(mean_squared_error(log_target, log_pred))
            metrics['mae_log'] = mean_absolute_error(log_target, log_pred)
        
        return metrics
    
    def calculate_mape(
        self,
        targets: np.ndarray,
        predictions: np.ndarray,
        epsilon: float = 1e-10
    ) -> float:
        """
        Calculate Mean Absolute Percentage Error
        """
        # Avoid division by zero
        mask = np.abs(targets) > epsilon
        if not np.any(mask):
            return 0.0
        
        mape = np.mean(np.abs((targets[mask] - predictions[mask]) / targets[mask])) * 100
        return mape
    
    def compare_with_aiio(
        self,
        our_metrics: Dict[str, float]
    ) -> Dict[str, Dict[str, float]]:
        """
        Compare our results with AIIO baselines
        
        Args:
            our_metrics: Our model's metrics
        
        Returns:
            Comparison dictionary
        """
        comparison = {}
        our_rmse = our_metrics.get('rmse_log', our_metrics['rmse'])
        
        for method, aiio_metrics in self.aiio_results.items():
            if 'rmse' in aiio_metrics:
                aiio_rmse = aiio_metrics['rmse']
                improvement = (aiio_rmse - our_rmse) / aiio_rmse * 100
                
                comparison[method] = {
                    'aiio_rmse': aiio_rmse,
                    'our_rmse': our_rmse,
                    'improvement_percent': improvement,
                    'is_better': our_rmse < aiio_rmse
                }
        
        return comparison
    
    def print_comparison(
        self,
        our_metrics: Dict[str, float]
    ):
        """
        Print formatted comparison with AIIO
        """
        comparison = self.compare_with_aiio(our_metrics)
        
        print("\n" + "="*60)
        print("COMPARISON WITH AIIO METHODS")
        print("="*60)
        
        for method, comp in comparison.items():
            status = "✅" if comp['is_better'] else "❌"
            print(f"{status} {method:15} | AIIO: {comp['aiio_rmse']:.4f} | "
                  f"Ours: {comp['our_rmse']:.4f} | "
                  f"Improvement: {comp['improvement_percent']:+.1f}%")
        
        print("="*60)
        
        # Summary
        better_count = sum(1 for c in comparison.values() if c['is_better'])
        print(f"\nSummary: Better than {better_count}/{len(comparison)} AIIO methods")
        
        if 'Closest_Method' in comparison:
            if comparison['Closest_Method']['is_better']:
                print("🎉 Achieved better performance than AIIO's best method!")
        
        return comparison


class MetricTracker:
    """
    Track metrics over training epochs
    """
    
    def __init__(self):
        self.metrics = {}
        self.best_metrics = {}
        self.best_epoch = {}
    
    def update(
        self,
        epoch: int,
        train_metrics: Dict[str, float],
        val_metrics: Dict[str, float],
        test_metrics: Optional[Dict[str, float]] = None
    ):
        """
        Update metrics for current epoch
        """
        # Store metrics
        for key, value in train_metrics.items():
            self._add_metric(f'train_{key}', value)
        
        for key, value in val_metrics.items():
            self._add_metric(f'val_{key}', value)
        
        if test_metrics:
            for key, value in test_metrics.items():
                self._add_metric(f'test_{key}', value)
        
        # Update best metrics
        val_rmse = val_metrics.get('rmse', float('inf'))
        if 'best_val_rmse' not in self.best_metrics or val_rmse < self.best_metrics['best_val_rmse']:
            self.best_metrics['best_val_rmse'] = val_rmse
            self.best_epoch['val_rmse'] = epoch
            
            # Store all metrics at best epoch
            for key in val_metrics:
                self.best_metrics[f'best_val_{key}'] = val_metrics[key]
            for key in train_metrics:
                self.best_metrics[f'best_train_{key}'] = train_metrics[key]
    
    def _add_metric(self, key: str, value: float):
        """Add metric value"""
        if key not in self.metrics:
            self.metrics[key] = []
        self.metrics[key].append(value)
    
    def get_best(self) -> Dict[str, float]:
        """Get best metrics"""
        return self.best_metrics
    
    def get_history(self) -> Dict[str, List[float]]:
        """Get full history"""
        return self.metrics
    
    def plot_metrics(
        self,
        save_path: Optional[str] = None,
        show_aiio: bool = True
    ):
        """
        Plot training curves
        """
        import matplotlib.pyplot as plt
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # RMSE plot
        ax = axes[0, 0]
        epochs = range(len(self.metrics.get('train_rmse', [])))
        
        if 'train_rmse' in self.metrics:
            ax.plot(epochs, self.metrics['train_rmse'], label='Train', linewidth=2)
        if 'val_rmse' in self.metrics:
            ax.plot(epochs, self.metrics['val_rmse'], label='Val', linewidth=2)
        
        # Add AIIO baselines
        if show_aiio:
            ax.axhline(y=0.2632, color='r', linestyle='--', label='AIIO LightGBM', alpha=0.7)
            ax.axhline(y=0.1860, color='g', linestyle='--', label='AIIO Best', alpha=0.7)
        
        ax.set_xlabel('Epoch')
        ax.set_ylabel('RMSE')
        ax.set_title('RMSE over Training')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # MAE plot
        ax = axes[0, 1]
        if 'train_mae' in self.metrics:
            ax.plot(epochs, self.metrics['train_mae'], label='Train', linewidth=2)
        if 'val_mae' in self.metrics:
            ax.plot(epochs, self.metrics['val_mae'], label='Val', linewidth=2)
        
        ax.set_xlabel('Epoch')
        ax.set_ylabel('MAE')
        ax.set_title('MAE over Training')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # R² plot
        ax = axes[1, 0]
        if 'val_r2' in self.metrics:
            ax.plot(epochs, self.metrics['val_r2'], label='Val R²', linewidth=2, color='purple')
        
        ax.set_xlabel('Epoch')
        ax.set_ylabel('R² Score')
        ax.set_title('R² Score over Training')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Loss plot
        ax = axes[1, 1]
        if 'train_loss' in self.metrics:
            ax.plot(epochs, self.metrics['train_loss'], label='Train Loss', linewidth=2, color='orange')
            ax.set_xlabel('Epoch')
            ax.set_ylabel('Loss')
            ax.set_title('Training Loss')
            ax.legend()
            ax.grid(True, alpha=0.3)
        elif 'lr' in self.metrics:
            ax.plot(epochs, self.metrics['lr'], label='Learning Rate', linewidth=2, color='orange')
            ax.set_xlabel('Epoch')
            ax.set_ylabel('Learning Rate')
            ax.set_title('Learning Rate Schedule')
            ax.set_yscale('log')
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            logger.info(f"Saved training curves to {save_path}")
        
        plt.show()