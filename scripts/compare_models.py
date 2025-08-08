"""
Model comparison script for GNN4_IO_4.

This script compares the performance of different models trained on I/O performance data,
including traditional tabular models and TabGNN models.
"""

import os
import argparse
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Dict, List, Any
import logging
import glob

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Compare model performance for I/O performance prediction")
    
    parser.add_argument("--results_dir", type=str, required=True, help="Directory containing model results")
    parser.add_argument("--output_file", type=str, default="comparison_report.json", help="Output file for comparison report")
    parser.add_argument("--plot_file", type=str, default="model_comparison.png", help="Output file for comparison plot")
    parser.add_argument("--experiment_name", type=str, required=True, help="Name of the experiment (e.g., Experiment3)")

    return parser.parse_args()

def load_metrics(results_dir: str, experiment_name: str) -> Dict[str, Dict[str, float]]:
    """
    Load metrics from model results directories.

    Args:
        results_dir (str): Directory containing model results

    Returns:
        Dict[str, Dict[str, float]]: Dictionary mapping model names to metrics
    """
    metrics = {}

    # New structure: logs/training/all/Experiment4/<model_name>
    experiment_root = os.path.join(results_dir, "all", experiment_name)
    if not os.path.exists(experiment_root):
        logger.error(f"Experiment folder not found: {experiment_root}")
        return {}

    model_dirs = [os.path.join(experiment_root, d) for d in os.listdir(experiment_root)
                  if os.path.isdir(os.path.join(experiment_root, d))]

    for model_dir in model_dirs:
        model_name = os.path.basename(model_dir) 
        if model_name == "all":
            continue

        metrics_file = os.path.join(model_dir, "metrics.json")
        if os.path.exists(metrics_file):
            with open(metrics_file, 'r') as f:
                metrics[model_name] = json.load(f)
                logger.info(f"Loaded metrics for {model_name}: {metrics[model_name]}")
        else:
            logger.warning(f"No metrics file found in {model_dir}")

    return metrics

def create_comparison_table(metrics: Dict[str, Dict[str, float]]) -> pd.DataFrame:
    """
    Create comparison table from metrics.
    
    Args:
        metrics (Dict[str, Dict[str, float]]): Dictionary mapping model names to metrics
        
    Returns:
        pd.DataFrame: Comparison table
    """
    # Create DataFrame
    comparison = pd.DataFrame(index=metrics.keys())
    
    # Add metrics as columns
    for model, model_metrics in metrics.items():
        for metric, value in model_metrics.items():
            comparison.loc[model, metric] = value
    
    # Sort by RMSE (lower is better)
    if 'rmse' in comparison.columns:
        comparison = comparison.sort_values('rmse')
    
    return comparison

def plot_comparison(metrics: Dict[str, Dict[str, float]], output_file_base: str):
    """
    Plot separate bar charts for RMSE, MAE, and R² (excluding tabnet).
    
    Args:
        metrics (Dict[str, Dict[str, float]]): Dictionary mapping model names to metrics
        output_file_base (str): Base path for saving plot files (e.g., "model_comparison")
    """
    metrics_to_plot = ['rmse', 'mae', 'r2']
    models = [m for m in metrics if m != "tabnet"]

    for metric in metrics_to_plot:
        values = [metrics[model].get(metric, 0) for model in models]
        
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.bar(models, values, color='skyblue')
        
        ax.set_title(f'{metric.upper()} Comparison')
        ax.set_ylabel(f'{metric.upper()} Value')
        ax.set_xlabel('Model')
        ax.set_xticks(np.arange(len(models)))
        ax.set_xticklabels(models, rotation=45, ha='right')
        ax.grid(axis='y', linestyle='--', alpha=0.7)
        
        plt.tight_layout()
        output_path = output_file_base.replace(".png", f"_{metric}.png")
        plt.savefig(output_path)
        plt.close()
        
        logger.info(f"Saved {metric.upper()} plot to {output_path}")

def create_comparison_report(metrics: Dict[str, Dict[str, float]], output_file: str):
    """
    Create comparison report and markdown summary.
    """
    # Create comparison table
    comparison = create_comparison_table(metrics)
    
    # Calculate improvement over LightGBM
    if 'lightgbm' in comparison.index and 'rmse' in comparison.columns:
        baseline_rmse = comparison.loc['lightgbm', 'rmse']
        comparison['improvement'] = (baseline_rmse - comparison['rmse']) / baseline_rmse * 100

    # Determine best model
    best_model = comparison.index[0] if not comparison.empty else None
    best_metrics = comparison.iloc[0].to_dict() if not comparison.empty else None

    # Create markdown table
    comparison_md = comparison.reset_index().to_markdown(index=False)

    # Create report dict for JSON
    report = {
        'metrics': metrics,
        'comparison': comparison.to_dict(),
        'best_model': best_model,
        'best_metrics': best_metrics,
        'markdown_table': comparison_md
    }

    # Save markdown report
    md_path = output_file.replace(".json", ".md")
    with open(md_path, "w") as f_md:
        f_md.write("# 🧪 Model Comparison Results\n\n")
        f_md.write(f"🏆 **Best Model:** `{best_model}`  \n\n")
        f_md.write("### 📊 Comparison Table (sorted by RMSE)\n\n")
        f_md.write(comparison_md)

    # Save JSON report
    with open(output_file, 'w') as f_json:
        json.dump(report, f_json, indent=4)

    logger.info(f"Comparison report saved to {output_file}")
    logger.info(f"Markdown summary saved to {md_path}")

    return report

def main():
    """Main function."""
    # Parse arguments
    args = parse_args()
    
    # Load metrics
    metrics = load_metrics(args.results_dir, args.experiment_name)
    
    if not metrics:
        logger.error("No metrics found in results directory")
        return
    
    # Create comparison report
    report = create_comparison_report(metrics, args.output_file)
    
    # Plot comparison
    plot_comparison(metrics, args.plot_file)
    
    # Print summary
    logger.info(f"Best model: {report['best_model']}")
    logger.info(f"Best metrics: {report['best_metrics']}")

if __name__ == "__main__":
    main()
