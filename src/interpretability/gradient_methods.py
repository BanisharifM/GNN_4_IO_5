"""
Gradient-based methods for feature importance
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Optional, Tuple
import matplotlib.pyplot as plt
import seaborn as sns
import logging
import pandas as pd

from .attention_analyzer import AttentionAnalyzer
from .gnn_explainer import IOGNNExplainer
from collections import defaultdict

logger = logging.getLogger(__name__)


class GradientAnalyzer:
    """
    Gradient-based feature importance methods for GNNs
    """
    
    def __init__(
        self,
        model: nn.Module,
        feature_names: List[str],
        device: torch.device = None
    ):
        """
        Args:
            model: Trained model
            feature_names: List of feature names
            device: Computing device
        """
        self.model = model
        self.feature_names = feature_names
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
    def vanilla_gradients(
        self,
        data,
        node_idx: int
    ) -> Dict[str, float]:
        """
        Compute vanilla gradients for feature importance
        
        Args:
            data: Graph data
            node_idx: Target node
        
        Returns:
            Feature importance scores
        """
        # Ensure model is in eval mode
        self.model.eval()
        
        # Move data to device
        if data.x.device != self.device:
            data = data.to(self.device)
        
        # Enable gradients
        x = data.x.clone().requires_grad_(True)
        
        # Forward pass
        output = self.model(x, data.edge_index, data.edge_attr)
        
        # Get prediction for target node
        if output.dim() > 1 and node_idx is not None:
            target_output = output[node_idx]
        else:
            target_output = output.mean()
        
        # Backward pass
        self.model.zero_grad()
        target_output.backward()
        
        # Get gradients for target node
        gradients = x.grad[node_idx].abs().cpu().numpy()
        
        # Normalize
        gradients = gradients / (np.sum(gradients) + 1e-10)
        
        # Create importance dictionary
        # importance = {
        #     self.feature_names[i]: float(gradients[i])
        #     for i in range(min(len(gradients), len(self.feature_names)))
        # }

        # Create importance dictionary - ensure scalar conversion
        importance = {
            self.feature_names[i]: float(importance_scores[i].item() if hasattr(importance_scores[i], 'item') else importance_scores[i])
            for i in range(min(len(importance_scores), len(self.feature_names)))
        }
        
        return importance
    
    def integrated_gradients(
        self,
        data,
        node_idx: int,
        baseline: Optional[torch.Tensor] = None,
        steps: int = 50
    ) -> Dict[str, float]:
        """
        Compute Integrated Gradients for feature importance
        
        Args:
            data: Graph data
            node_idx: Target node
            baseline: Baseline input (zeros by default)
            steps: Number of integration steps
        
        Returns:
            Feature importance scores
        """
        self.model.eval()
        
        # Move data to device
        if data.x.device != self.device:
            data = data.to(self.device)
        
        # Set baseline
        if baseline is None:
            baseline = torch.zeros_like(data.x)
        
        # Generate interpolated inputs
        alphas = torch.linspace(0, 1, steps).to(self.device)
        
        # Accumulate gradients
        integrated_grads = torch.zeros_like(data.x[node_idx])
        
        for alpha in alphas:
            # Interpolate input
            interpolated = baseline + alpha * (data.x - baseline)
            interpolated.requires_grad_(True)
            
            # Forward pass
            output = self.model(interpolated, data.edge_index, data.edge_attr)
            
            # Get target output
            if output.dim() > 1 and node_idx is not None:
                target_output = output[node_idx]
            else:
                target_output = output.mean()
            
            # Backward pass
            self.model.zero_grad()
            target_output.backward()
            
            # Accumulate gradients
            integrated_grads += interpolated.grad[node_idx] / steps
        
        # Multiply by input difference
        integrated_grads *= (data.x[node_idx] - baseline[node_idx])
        
        # Take absolute value and normalize
        importance_scores = integrated_grads.abs().cpu().numpy()
        importance_scores = importance_scores / (np.sum(importance_scores) + 1e-10)
        
        # Create importance dictionary
        # Create importance dictionary - ensure scalar conversion
        importance = {
            self.feature_names[i]: float(importance_scores[i].item() if hasattr(importance_scores[i], 'item') else importance_scores[i])
            for i in range(min(len(importance_scores), len(self.feature_names)))
        }
        
        return importance
    
    def smooth_grad(
        self,
        data,
        node_idx: int,
        n_samples: int = 50,
        noise_level: float = 0.1
    ) -> Dict[str, float]:
        """
        SmoothGrad: Average gradients over noisy inputs
        
        Args:
            data: Graph data
            node_idx: Target node
            n_samples: Number of noisy samples
            noise_level: Standard deviation of noise
        
        Returns:
            Feature importance scores
        """
        self.model.eval()
        
        # Move data to device
        if data.x.device != self.device:
            data = data.to(self.device)
        
        # Accumulate gradients
        accumulated_grads = torch.zeros_like(data.x[node_idx])
        
        for _ in range(n_samples):
            # Add noise to input
            noise = torch.randn_like(data.x) * noise_level
            noisy_x = data.x + noise
            noisy_x.requires_grad_(True)
            
            # Forward pass
            output = self.model(noisy_x, data.edge_index, data.edge_attr)
            
            # Get target output
            if output.dim() > 1 and node_idx is not None:
                target_output = output[node_idx]
            else:
                target_output = output.mean()
            
            # Backward pass
            self.model.zero_grad()
            target_output.backward()
            
            # Accumulate gradients
            accumulated_grads += noisy_x.grad[node_idx].abs()
        
        # Average gradients
        avg_grads = accumulated_grads / n_samples
        
        # Normalize
        importance_scores = avg_grads.cpu().numpy()
        importance_scores = importance_scores / (np.sum(importance_scores) + 1e-10)
        
        # Create importance dictionary
        # importance = {
        #     self.feature_names[i]: float(importance_scores[i])
        #     for i in range(min(len(importance_scores), len(self.feature_names)))
        # }

        # Create importance dictionary - ensure scalar conversion
        importance = {
            self.feature_names[i]: float(importance_scores[i].item() if hasattr(importance_scores[i], 'item') else importance_scores[i])
            for i in range(min(len(importance_scores), len(self.feature_names)))
        }

        return importance
    
    def compare_methods(
        self,
        data,
        node_idx: int
    ) -> pd.DataFrame:
        """
        Compare different gradient methods
        
        Args:
            data: Graph data
            node_idx: Target node
        
        Returns:
            DataFrame comparing methods
        """
        import pandas as pd
        
        # Compute importance with different methods
        vanilla = self.vanilla_gradients(data, node_idx)
        integrated = self.integrated_gradients(data, node_idx)
        smooth = self.smooth_grad(data, node_idx)
        
        # Create DataFrame
        df_data = []
        for feature in self.feature_names:
            df_data.append({
                'feature': feature,
                'vanilla_gradient': vanilla.get(feature, 0),
                'integrated_gradient': integrated.get(feature, 0),
                'smooth_grad': smooth.get(feature, 0)
            })
        
        df = pd.DataFrame(df_data)
        
        # Add average and std
        df['average'] = df[['vanilla_gradient', 'integrated_gradient', 'smooth_grad']].mean(axis=1)
        df['std'] = df[['vanilla_gradient', 'integrated_gradient', 'smooth_grad']].std(axis=1)
        
        # Sort by average importance
        df = df.sort_values('average', ascending=False)
        
        return df
    
    def visualize_feature_importance(
        self,
        importance: Dict[str, float],
        top_k: int = 20,
        title: str = "Feature Importance",
        save_path: Optional[str] = None
    ):
        """
        Visualize feature importance scores
        
        Args:
            importance: Feature importance dictionary
            top_k: Number of top features to show
            title: Plot title
            save_path: Save path
        """
        # Sort and select top features
        sorted_features = sorted(importance.items(), key=lambda x: x[1], reverse=True)[:top_k]
        
        if not sorted_features:
            logger.warning("No features to visualize")
            return
        
        features, scores = zip(*sorted_features)
        
        # Create plot
        plt.figure(figsize=(10, 8))
        
        # Color based on positive/negative impact
        colors = ['red' if s > np.mean(scores) else 'blue' for s in scores]
        
        plt.barh(range(len(features)), scores, color=colors, alpha=0.7)
        plt.yticks(range(len(features)), features)
        plt.xlabel('Importance Score')
        plt.title(title)
        
        # Add value labels
        for i, (feat, score) in enumerate(sorted_features):
            plt.text(score, i, f'{score:.3f}', ha='left', va='center')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            logger.info(f"Saved feature importance to {save_path}")
        
        plt.show()
    
    def identify_bottlenecks(
        self,
        importance: Dict[str, float],
        threshold: float = 0.05
    ) -> List[str]:
        """
        Identify bottleneck features based on importance scores
        
        Args:
            importance: Feature importance scores
            threshold: Minimum importance to be considered bottleneck
        
        Returns:
            List of bottleneck feature names
        """
        bottlenecks = [
            feature for feature, score in importance.items()
            if score > threshold
        ]
        
        # Sort by importance
        bottlenecks = sorted(
            bottlenecks,
            key=lambda x: importance[x],
            reverse=True
        )
        
        return bottlenecks


class BottleneckIdentifier:
    """
    Combine all interpretability methods to identify I/O bottlenecks
    """
    
    def __init__(
        self,
        model: nn.Module,
        feature_names: List[str],
        device: torch.device = None
    ):
        """
        Args:
            model: Trained GAT model
            feature_names: List of feature names
            device: Computing device
        """
        self.model = model
        self.feature_names = feature_names
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Initialize analyzers
        self.attention_analyzer = AttentionAnalyzer(model, feature_names, device)
        self.gnn_explainer = IOGNNExplainer(model, device=device)
        self.gradient_analyzer = GradientAnalyzer(model, feature_names, device)
    
    def comprehensive_analysis(
        self,
        data,
        node_idx: int
    ) -> Dict[str, Dict[str, float]]:
        """
        Perform comprehensive bottleneck analysis using all methods
        
        Args:
            data: Graph data
            node_idx: Target node
        
        Returns:
            Dictionary with results from all methods
        """
        results = {}
        
        # Attention-based analysis
        try:
            results['attention'] = self.attention_analyzer.attention_based_bottleneck_detection(
                data, node_idx
            )
        except Exception as e:
            logger.warning(f"Attention analysis failed: {e}")
            results['attention'] = {}
        
        # GNNExplainer analysis
        try:
            results['gnn_explainer'] = self.gnn_explainer.explain_bottleneck_pattern(
                data, node_idx, self.feature_names
            )
        except Exception as e:
            logger.warning(f"GNNExplainer analysis failed: {e}")
            results['gnn_explainer'] = {}
        
        # Gradient-based analysis
        try:
            results['integrated_gradients'] = self.gradient_analyzer.integrated_gradients(
                data, node_idx
            )
        except Exception as e:
            logger.warning(f"Gradient analysis failed: {e}")
            results['integrated_gradients'] = {}
        
        return results
    
    def consensus_bottlenecks(
        self,
        results: Dict[str, Dict[str, float]],
        min_methods: int = 2
    ) -> List[Tuple[str, float]]:
        """
        Find consensus bottlenecks across methods
        
        Args:
            results: Results from all methods
            min_methods: Minimum methods that must agree
        
        Returns:
            List of (feature, average_importance) tuples
        """
        # Count appearances and scores
        feature_scores = defaultdict(list)
        
        for method_results in results.values():
            for feature, score in method_results.items():
                feature_scores[feature].append(score)
        
        # Filter by minimum methods
        consensus = []
        for feature, scores in feature_scores.items():
            if len(scores) >= min_methods:
                avg_score = np.mean(scores)
                consensus.append((feature, avg_score))
        
        # Sort by average importance
        consensus = sorted(consensus, key=lambda x: x[1], reverse=True)
        
        return consensus