"""
visualization.py - Generate publication-quality visualizations for GNN interpretability results
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import pandas as pd
from typing import Dict, List, Tuple, Optional
from matplotlib.patches import FancyBboxPatch, Rectangle
from matplotlib.gridspec import GridSpec
import matplotlib.patches as patches
from matplotlib_venn import venn3, venn3_circles
import warnings
warnings.filterwarnings('ignore')

# Set publication-quality defaults
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans']
plt.rcParams['font.size'] = 10
plt.rcParams['axes.labelsize'] = 11
plt.rcParams['axes.titlesize'] = 12
plt.rcParams['xtick.labelsize'] = 9
plt.rcParams['ytick.labelsize'] = 9
plt.rcParams['legend.fontsize'] = 9
plt.rcParams['figure.dpi'] = 150
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['axes.linewidth'] = 0.8


class InterpretabilityVisualizer:
    """Create publication-quality visualizations for GNN interpretability results"""
    
    def __init__(self, node_results: Dict, feature_names: List[str] = None):
        """
        Args:
            node_results: Results dictionary from analyze_node method
            feature_names: Optional list of all feature names
        """
        self.node_results = node_results
        self.node_idx = node_results.get('node_idx')
        self.performance = node_results.get('performance')
        self.methods = node_results.get('methods', {})
        self.consensus = node_results.get('consensus', [])
        self.feature_names = feature_names
        
        # Color schemes
        self.method_colors = {
            'attention': '#3498db',      # Blue
            'gnn_explainer': '#e74c3c',  # Red
            'gradients': '#2ecc71',      # Green
            'consensus': '#34495e'       # Dark gray
        }
        
        self.impact_colors = {
            'positive': '#2ecc71',  # Green for good
            'negative': '#e74c3c',  # Red for bad
            'neutral': '#95a5a6'    # Gray for neutral
        }
    
    def create_main_figure(self, save_path: Optional[str] = None):
        """
        Create the main 4-panel figure for the paper
        
        Args:
            save_path: Path to save the figure
        """
        fig = plt.figure(figsize=(16, 10))
        gs = GridSpec(2, 2, figure=fig, hspace=0.3, wspace=0.3)
        
        # Panel A: Three-method comparison
        ax1 = fig.add_subplot(gs[0, 0])
        self._plot_method_comparison(ax1)
        
        # Panel B: Consensus strength visualization
        ax2 = fig.add_subplot(gs[0, 1])
        self._plot_consensus_strength(ax2)
        
        # Panel C: Method agreement matrix
        ax3 = fig.add_subplot(gs[1, 0])
        self._plot_agreement_matrix(ax3)
        
        # Panel D: Feature contribution waterfall
        ax4 = fig.add_subplot(gs[1, 1])
        self._plot_contribution_waterfall(ax4)
        
        # Main title
        perf_category = self._get_performance_category()
        fig.suptitle(
            f'Multi-Method GNN Interpretability Analysis for Node {self.node_idx} ({perf_category})\n'
            f'Performance Score: {self.performance:.4f}',
            fontsize=14, fontweight='bold', y=1.02
        )
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, bbox_inches='tight', dpi=300)
            print(f"Figure saved to {save_path}")
        
        plt.show()
        return fig
    
    def _plot_method_comparison(self, ax):
        """Panel A: Grouped bar chart comparing all methods"""
        # Prepare data
        all_features = set()
        for method_results in self.methods.values():
            if method_results:
                all_features.update(method_results.keys())
        
        # Get consensus features
        consensus_dict = {feat: score for feat, score, _ in self.consensus[:10]}
        all_features.update(consensus_dict.keys())
        
        # Select top features by consensus
        top_features = list(consensus_dict.keys())[:10]
        if len(top_features) < 10:
            # Add more features from methods
            for feat in all_features:
                if feat not in top_features:
                    top_features.append(feat)
                if len(top_features) >= 10:
                    break
        
        # Create data matrix
        n_features = len(top_features)
        n_methods = 4  # attention, gnn_explainer, gradients, consensus
        
        data = np.zeros((n_features, n_methods))
        
        for i, feat in enumerate(top_features):
            # Attention
            if 'attention' in self.methods and feat in self.methods['attention']:
                data[i, 0] = self.methods['attention'][feat]
            
            # GNN Explainer
            if 'gnn_explainer' in self.methods and feat in self.methods['gnn_explainer']:
                data[i, 1] = self.methods['gnn_explainer'][feat]
            
            # Gradients
            if 'gradients' in self.methods and feat in self.methods['gradients']:
                data[i, 2] = self.methods['gradients'][feat]
            
            # Consensus
            if feat in consensus_dict:
                data[i, 3] = consensus_dict[feat]
        
        # Plot grouped bars
        x = np.arange(n_features)
        width = 0.2
        
        bars1 = ax.barh(x - 1.5*width, data[:, 0], width, 
                       label='Attention', color=self.method_colors['attention'], alpha=0.8)
        bars2 = ax.barh(x - 0.5*width, data[:, 1], width,
                       label='GNNExplainer', color=self.method_colors['gnn_explainer'], alpha=0.8)
        bars3 = ax.barh(x + 0.5*width, data[:, 2], width,
                       label='Gradients', color=self.method_colors['gradients'], alpha=0.8)
        bars4 = ax.barh(x + 1.5*width, data[:, 3], width,
                       label='Consensus', color=self.method_colors['consensus'], alpha=1.0)
        
        # Formatting
        ax.set_yticks(x)
        ax.set_yticklabels([self._format_feature_name(f) for f in top_features])
        ax.set_xlabel('Importance Score')
        ax.set_title('(A) Method Comparison', fontweight='bold', loc='left')
        ax.legend(loc='lower right')
        ax.grid(True, alpha=0.3, axis='x')
        ax.set_xlim(0, max(data.flatten()) * 1.1)
        
        # Add value labels on bars
        for bars in [bars1, bars2, bars3, bars4]:
            for bar in bars:
                width = bar.get_width()
                if width > 0.01:  # Only show if significant
                    ax.text(width, bar.get_y() + bar.get_height()/2,
                           f'{width:.2f}', ha='left', va='center', fontsize=7)
    
    def _plot_consensus_strength(self, ax):
        """Panel B: Venn diagram or bar chart showing consensus"""
        # For simplicity, using a stacked bar chart
        consensus_features = [(feat, score, methods) for feat, score, methods in self.consensus[:8]]
        
        if not consensus_features:
            ax.text(0.5, 0.5, 'No consensus features found', 
                   ha='center', va='center', transform=ax.transAxes)
            ax.set_title('(B) Consensus Strength', fontweight='bold', loc='left')
            return
        
        features = [f[0] for f in consensus_features]
        scores = [f[1] for f in consensus_features]
        methods_detecting = [f[2] for f in consensus_features]
        
        # Create bars
        colors = []
        for methods in methods_detecting:
            if len(methods) == 3:
                colors.append('#27ae60')  # Green for 3/3 consensus
            elif len(methods) == 2:
                colors.append('#f39c12')  # Orange for 2/3 consensus
            else:
                colors.append('#95a5a6')  # Gray for 1/3
        
        bars = ax.barh(range(len(features)), scores, color=colors, alpha=0.8)
        
        # Add method indicators
        for i, (feat, score, methods) in enumerate(consensus_features):
            # Add symbols to show which methods agree
            symbols = []
            if 'attention' in methods:
                symbols.append('A')
            if 'gnn_explainer' in methods:
                symbols.append('G')
            if 'gradients' in methods:
                symbols.append('I')
            
            ax.text(score + 0.01, i, f"[{'/'.join(symbols)}]", 
                   va='center', fontsize=8)
        
        ax.set_yticks(range(len(features)))
        ax.set_yticklabels([self._format_feature_name(f) for f in features])
        ax.set_xlabel('Consensus Score')
        ax.set_title('(B) Consensus Strength', fontweight='bold', loc='left')
        ax.grid(True, alpha=0.3, axis='x')
        
        # Add legend
        legend_elements = [
            mpatches.Patch(color='#27ae60', label='3/3 methods agree', alpha=0.8),
            mpatches.Patch(color='#f39c12', label='2/3 methods agree', alpha=0.8),
            mpatches.Patch(color='#95a5a6', label='1/3 method', alpha=0.8)
        ]
        ax.legend(handles=legend_elements, loc='lower right', fontsize=8)
    
    def _plot_agreement_matrix(self, ax):
        """Panel C: Heatmap showing method agreement"""
        # Get top features
        top_features = []
        for feat, _, _ in self.consensus[:10]:
            top_features.append(feat)
        
        # Create agreement matrix
        methods = ['Attention', 'GNNExplainer', 'Gradients']
        matrix = np.zeros((len(top_features), len(methods)))
        
        for i, feat in enumerate(top_features):
            if 'attention' in self.methods and feat in self.methods['attention']:
                matrix[i, 0] = self.methods['attention'][feat]
            if 'gnn_explainer' in self.methods and feat in self.methods['gnn_explainer']:
                matrix[i, 1] = self.methods['gnn_explainer'][feat]
            if 'gradients' in self.methods and feat in self.methods['gradients']:
                matrix[i, 2] = self.methods['gradients'][feat]
        
        # Plot heatmap
        im = ax.imshow(matrix, cmap='YlOrRd', aspect='auto', vmin=0)
        
        # Set ticks
        ax.set_xticks(np.arange(len(methods)))
        ax.set_yticks(np.arange(len(top_features)))
        ax.set_xticklabels(methods, rotation=45, ha='right')
        ax.set_yticklabels([self._format_feature_name(f) for f in top_features])
        
        # Add text annotations
        for i in range(len(top_features)):
            for j in range(len(methods)):
                if matrix[i, j] > 0:
                    text = ax.text(j, i, f'{matrix[i, j]:.2f}',
                                 ha="center", va="center", color="white" if matrix[i, j] > 0.3 else "black",
                                 fontsize=8)
        
        ax.set_title('(C) Method Agreement Matrix', fontweight='bold', loc='left')
        
        # Add colorbar
        cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.ax.set_ylabel('Importance Score', rotation=270, labelpad=15)
    
    def _plot_contribution_waterfall(self, ax):
        """Panel D: Waterfall chart of feature contributions"""
        # Get top consensus features
        top_consensus = self.consensus[:7] if self.consensus else []
        
        if not top_consensus:
            ax.text(0.5, 0.5, 'No consensus features for waterfall', 
                   ha='center', va='center', transform=ax.transAxes)
            ax.set_title('(D) Feature Contribution Waterfall', fontweight='bold', loc='left')
            return
        
        # Prepare data
        features = [self._format_feature_name(f[0]) for f in top_consensus]
        values = [f[1] for f in top_consensus]
        
        # Normalize to show relative contribution
        total = sum(values)
        if total > 0:
            values = [v/total for v in values]
        
        # Create waterfall
        cumulative = [0]
        for v in values:
            cumulative.append(cumulative[-1] + v)
        
        # Plot bars
        for i, (feat, val) in enumerate(zip(features, values)):
            color = self.impact_colors['negative'] if 'NOT_ALIGNED' in feat or 'SMALL' in feat else self.impact_colors['positive']
            ax.bar(i, val, bottom=cumulative[i], color=color, alpha=0.7, edgecolor='black', linewidth=0.5)
            
            # Add connecting lines
            if i < len(features) - 1:
                ax.plot([i+0.4, i+0.6], [cumulative[i+1], cumulative[i+1]], 'k--', alpha=0.5)
            
            # Add value labels
            ax.text(i, cumulative[i] + val/2, f'{val:.1%}', 
                   ha='center', va='center', fontsize=8, fontweight='bold')
        
        # Add total bar
        ax.bar(len(features), cumulative[-1], color=self.method_colors['consensus'], 
              alpha=0.8, edgecolor='black', linewidth=1)
        ax.text(len(features), cumulative[-1]/2, f'Total\n{cumulative[-1]:.1%}', 
               ha='center', va='center', fontsize=9, fontweight='bold')
        
        # Formatting
        ax.set_xticks(range(len(features) + 1))
        ax.set_xticklabels(features + ['Total'], rotation=45, ha='right')
        ax.set_ylabel('Cumulative Contribution')
        ax.set_title('(D) Feature Contribution Waterfall', fontweight='bold', loc='left')
        ax.grid(True, alpha=0.3, axis='y')
        ax.set_ylim(0, max(cumulative) * 1.1)
    
    def create_single_bar_chart(self, save_path: Optional[str] = None, top_k: int = 15):
        """
        Create a single comprehensive bar chart (AIIO-style)
        
        Args:
            save_path: Path to save the figure
            top_k: Number of top features to show
        """
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Get consensus features
        consensus_features = self.consensus[:top_k] if self.consensus else []
        
        if not consensus_features:
            ax.text(0.5, 0.5, 'No consensus features found', 
                   ha='center', va='center', transform=ax.transAxes)
            plt.show()
            return fig
        
        features = []
        scores = []
        colors = []
        error_bars = []
        markers = []
        
        for feat, score, methods in consensus_features:
            features.append(self._format_feature_name(feat))
            scores.append(score)
            
            # Determine color based on impact
            if any(neg in feat for neg in ['NOT_ALIGNED', 'SMALL', 'ACCESS2', 'ACCESS4']):
                colors.append(self.impact_colors['negative'])
            else:
                colors.append(self.impact_colors['positive'])
            
            # Calculate error bar (variance across methods)
            method_scores = []
            if 'attention' in self.methods and feat in self.methods['attention']:
                method_scores.append(self.methods['attention'][feat])
            if 'gnn_explainer' in self.methods and feat in self.methods['gnn_explainer']:
                method_scores.append(self.methods['gnn_explainer'][feat])
            if 'gradients' in self.methods and feat in self.methods['gradients']:
                method_scores.append(self.methods['gradients'][feat])
            
            if len(method_scores) > 1:
                error_bars.append(np.std(method_scores))
            else:
                error_bars.append(0)
            
            # Markers for number of methods
            markers.append(len(methods))
        
        # Create horizontal bar chart
        y_pos = np.arange(len(features))
        bars = ax.barh(y_pos, scores, color=colors, alpha=0.7, edgecolor='black', linewidth=0.5)
        
        # Add error bars
        ax.errorbar(scores, y_pos, xerr=error_bars, fmt='none', 
                   color='black', alpha=0.5, capsize=3)
        
        # Add method count markers
        for i, (score, num_methods) in enumerate(zip(scores, markers)):
            marker_str = '●' * num_methods
            ax.text(score + max(scores)*0.02, i, marker_str, 
                   va='center', fontsize=10)
        
        # Add value labels
        for i, (score, err) in enumerate(zip(scores, error_bars)):
            ax.text(score/2, i, f'{score:.3f}', 
                   ha='center', va='center', fontsize=9, fontweight='bold')
        
        # Formatting
        ax.set_yticks(y_pos)
        ax.set_yticklabels(features)
        ax.set_xlabel('Consensus Importance Score', fontsize=12)
        ax.set_title(f'GNN Multi-Method Interpretability: Node {self.node_idx} ({self._get_performance_category()})\n'
                    f'Performance Score: {self.performance:.4f}',
                    fontsize=14, fontweight='bold', pad=20)
        ax.grid(True, alpha=0.3, axis='x')
        ax.set_xlim(0, max(scores) * 1.15)
        
        # Add legend
        legend_elements = [
            mpatches.Patch(color=self.impact_colors['negative'], label='Performance degrading', alpha=0.7),
            mpatches.Patch(color=self.impact_colors['positive'], label='Performance enhancing', alpha=0.7),
            plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='black', 
                      markersize=8, label='● = 1 method')
        ]
        ax.legend(handles=legend_elements, loc='lower right')
        
        # Add annotation
        ax.text(0.02, 0.02, f'Consensus from {len(self.methods)} methods (Attention, GNNExplainer, Gradients)',
               transform=ax.transAxes, fontsize=9, style='italic')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, bbox_inches='tight', dpi=300)
            print(f"Figure saved to {save_path}")
        
        plt.show()
        return fig
    
    def create_comparison_figure(self, other_nodes: List[Dict], save_path: Optional[str] = None):
        """
        Create a figure comparing this node with other nodes
        
        Args:
            other_nodes: List of other node results to compare
            save_path: Path to save the figure
        """
        fig, axes = plt.subplots(1, len(other_nodes) + 1, figsize=(5*(len(other_nodes)+1), 8))
        
        if len(other_nodes) == 0:
            axes = [axes]
        
        # Plot this node
        self._plot_single_node_bars(axes[0], self.node_results, 'Current Node')
        
        # Plot other nodes
        for i, node_result in enumerate(other_nodes):
            self._plot_single_node_bars(axes[i+1], node_result, f'Node {node_result["node_idx"]}')
        
        fig.suptitle('Multi-Node Comparison', fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, bbox_inches='tight', dpi=300)
            print(f"Figure saved to {save_path}")
        
        plt.show()
        return fig
    
    def _plot_single_node_bars(self, ax, node_result, title):
        """Helper to plot bars for a single node"""
        consensus = node_result.get('consensus', [])[:10]
        
        if not consensus:
            ax.text(0.5, 0.5, 'No data', ha='center', va='center', transform=ax.transAxes)
            ax.set_title(title)
            return
        
        features = [self._format_feature_name(f[0]) for f in consensus]
        scores = [f[1] for f in consensus]
        
        colors = []
        for feat in [f[0] for f in consensus]:
            if any(neg in feat for neg in ['NOT_ALIGNED', 'SMALL', 'ACCESS2', 'ACCESS4']):
                colors.append(self.impact_colors['negative'])
            else:
                colors.append(self.impact_colors['positive'])
        
        ax.barh(range(len(features)), scores, color=colors, alpha=0.7)
        ax.set_yticks(range(len(features)))
        ax.set_yticklabels(features)
        ax.set_xlabel('Importance Score')
        ax.set_title(f'{title}\nPerf: {node_result.get("performance", 0):.3f}')
        ax.grid(True, alpha=0.3, axis='x')
    
    def _format_feature_name(self, name: str) -> str:
        """Format feature names for display"""
        # Shorten long names
        replacements = {
            'POSIX_': '',
            '_COUNT': '_CNT',
            '_ACCESS': '_ACC',
            'SIZE_READ_': 'RD_',
            'SIZE_WRITE_': 'WR_',
            'LUSTRE_STRIPE_': 'STRIPE_',
            'NOT_ALIGNED': 'MISALIGN',
            'BYTES_WRITTEN': 'BYTES_WR',
            'BYTES_READ': 'BYTES_RD',
            'CONSEC_': 'CONS_'
        }
        
        formatted = name
        for old, new in replacements.items():
            formatted = formatted.replace(old, new)
        
        return formatted
    
    def _get_performance_category(self) -> str:
        """Determine performance category"""
        if self.performance < 1.05:
            return "Poor Performer"
        elif self.performance > 2.57:
            return "Good Performer"
        else:
            return "Medium Performer"


def visualize_node_results(node_results: Dict, output_dir: str = './figures'):
    """
    Main function to generate all visualizations for a node
    
    Args:
        node_results: Results from analyze_node
        output_dir: Directory to save figures
    """
    import os
    os.makedirs(output_dir, exist_ok=True)
    
    visualizer = InterpretabilityVisualizer(node_results)
    
    # Generate main 4-panel figure
    print("Generating main 4-panel figure...")
    visualizer.create_main_figure(
        save_path=os.path.join(output_dir, f'node_{node_results["node_idx"]}_main.pdf')
    )
    
    # Generate single comprehensive bar chart (AIIO-style)
    print("Generating AIIO-style bar chart...")
    visualizer.create_single_bar_chart(
        save_path=os.path.join(output_dir, f'node_{node_results["node_idx"]}_bars.pdf')
    )
    
    print(f"All figures saved to {output_dir}")


# Integration with test_methods.py
def integrate_with_test_methods():
    """
    Example of how to integrate with your test_methods.py
    Add this to your analyze_node method or create a new method
    """
    code = '''
    # In test_methods.py, add this method to InterpretabilityTester class:
    
    def visualize_node(self, node_idx: int, output_dir: str = './figures'):
        """
        Analyze and visualize a single node
        
        Args:
            node_idx: Node index to analyze
            output_dir: Directory to save figures
        """
        # Analyze the node
        results = self.analyze_node(node_idx)
        
        # Import visualization module
        from visualization import visualize_node_results
        
        # Generate visualizations
        visualize_node_results(results, output_dir)
        
        return results
    
    # Usage in main:
    if __name__ == "__main__":
        # ... existing code ...
        
        # For specific node visualization
        if args.visualize_node:
            results = tester.visualize_node(
                node_idx=args.visualize_node,
                output_dir=args.output_dir or './figures'
            )
    '''
    return code


if __name__ == "__main__":

    # Generate visualizations
    visualize_node_results(example_results)



    # Example usage with mock data
    # example_results = {
    #     'node_idx': 25,
    #     'performance': 0.3631,
    #     'methods': {
    #         'attention': {
    #             'POSIX_MEM_NOT_ALIGNED': 0.2950,
    #             'POSIX_ACCESS4_COUNT': 0.2833,
    #             'POSIX_SIZE_READ_1K_10K': 0.1536,
    #             'POSIX_BYTES_READ': 0.1430,
    #             'POSIX_SIZE_READ_100_1K': 0.0428
    #         },
    #         'gnn_explainer': {
    #             'POSIX_SIZE_READ_0_100': 0.5871,
    #             'POSIX_ACCESS2_COUNT': 0.5588,
    #             'POSIX_FILE_NOT_ALIGNED': 0.5547,
    #             'POSIX_OPENS': 0.5536,
    #             'POSIX_SEQ_READS': 0.5437
    #         },
    #         'gradients': {
    #             'nprocs': 0.0129,
    #             'POSIX_OPENS': 0.0515,
    #             'LUSTRE_STRIPE_SIZE': 0.0733,
    #             'LUSTRE_STRIPE_WIDTH': 0.0065,
    #             'POSIX_FILENOS': 0.0000
    #         }
    #     },
    #     'consensus': [
    #         ('POSIX_SIZE_READ_0_100', 0.3074, ['gnn_explainer', 'gradients']),
    #         ('POSIX_OPENS', 0.3026, ['gnn_explainer', 'gradients']),
    #         ('POSIX_FILE_NOT_ALIGNED', 0.2995, ['gnn_explainer', 'gradients']),
    #         ('POSIX_MEM_NOT_ALIGNED', 0.2795, ['attention', 'gnn_explainer', 'gradients']),
    #         ('nprocs', 0.2670, ['gnn_explainer', 'gradients'])
    #     ]
    # }