"""
Test script for GNN4_IO_4 implementation.

This script tests the data processing, graph construction, and model training
components using the sample_train_100.csv dataset.
"""

import os
import sys
import pandas as pd
import numpy as np
import torch
import logging
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import project modules
from src.data import IODataProcessor
from src.models.tabular import LightGBMModel, CatBoostModel, XGBoostModel, MLPModel, TabNetModel

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

def test_data_processing():
    """Test data processing and graph construction."""
    logger.info("Testing data processing and graph construction...")
    
    # Create data processor
    data_path = "/home/ubuntu/upload/sample_train_100.csv"
    
    # Define important features
    important_features = [
        "POSIX_SEQ_WRITES_PERC",
        "POSIX_SIZE_READ_1K_10K_PERC",
        "POSIX_SIZE_READ_10K_100K_PERC",
        "POSIX_unique_bytes_perc",
        "POSIX_SIZE_READ_0_100_PERC",
        "POSIX_SIZE_WRITE_100K_1M_PERC",
        "POSIX_write_only_bytes_perc",
        "POSIX_MEM_NOT_ALIGNED_PERC",
        "POSIX_FILE_NOT_ALIGNED_PERC"
    ]
    
    # Define similarity thresholds
    similarity_thresholds = {feature: 0.5 for feature in important_features}
    
    # Create data processor
    processor = IODataProcessor(
        data_path=data_path,
        important_features=important_features,
        similarity_thresholds=similarity_thresholds
    )
    
    # Load and preprocess data
    data = processor.load_data()
    preprocessed_data = processor.preprocess_data()
    
    # Print data info
    logger.info(f"Data shape: {data.shape}")
    logger.info(f"Data columns: {data.columns.tolist()}")
    
    # Construct multiplex graphs
    multiplex_edges = processor.construct_multiplex_graphs()
    
    # Print graph info
    for feature, (edge_index, edge_attr) in multiplex_edges.items():
        logger.info(f"Graph for {feature}: {edge_index.shape[1]} edges")
    
    # Create PyG data
    target_column = "tag" if "tag" in data.columns else None
    if target_column is None:
        logger.warning("No 'tag' column found in data. Using first column as target.")
        target_column = data.columns[0]
    
    pyg_data = processor.create_combined_pyg_data(target_column=target_column)
    
    # Print PyG data info
    logger.info(f"PyG data: {pyg_data}")
    logger.info(f"Number of nodes: {pyg_data.x.shape[0]}")
    logger.info(f"Node feature dimension: {pyg_data.x.shape[1]}")
    logger.info(f"Number of edges: {pyg_data.edge_index.shape[1]}")
    
    # Split data
    split_data = processor.train_val_test_split(pyg_data)
    
    # Print split info
    logger.info(f"Train samples: {split_data.train_mask.sum().item()}")
    logger.info(f"Validation samples: {split_data.val_mask.sum().item()}")
    logger.info(f"Test samples: {split_data.test_mask.sum().item()}")
    
    # Save processed data
    output_dir = "/home/ubuntu/GNN4_IO_4/test_output/processed_data"
    processor.save_processed_data(output_dir, target_column=target_column)
    
    logger.info(f"Saved processed data to {output_dir}")
    
    return split_data, target_column

def test_tabular_models(split_data, target_column):
    """Test tabular models."""
    logger.info("Testing tabular models...")
    
    # Extract features and target
    X = split_data.x.numpy()
    y = split_data.y.numpy()
    
    # Get train, val, test indices
    train_idx = split_data.train_mask.numpy()
    val_idx = split_data.val_mask.numpy()
    test_idx = split_data.test_mask.numpy()
    
    # Split data
    X_train, y_train = X[train_idx], y[train_idx]
    X_val, y_val = X[val_idx], y[val_idx]
    X_test, y_test = X[test_idx], y[test_idx]
    
    # Create evaluation set
    eval_set = [(X_val, y_val)]
    
    # Test LightGBM
    logger.info("Testing LightGBM model...")
    lgb_model = LightGBMModel(
        n_estimators=100,
        learning_rate=0.05,
        random_state=42
    )
    
    # Train model
    lgb_model.fit(X_train, y_train, eval_set=eval_set)
    
    # Evaluate model
    lgb_metrics = lgb_model.evaluate(X_test, y_test)
    logger.info(f"LightGBM metrics: {lgb_metrics}")
    
    # Save model
    output_dir = "/home/ubuntu/GNN4_IO_4/test_output/models"
    os.makedirs(output_dir, exist_ok=True)
    lgb_model.save(os.path.join(output_dir, "lightgbm_model.joblib"))
    
    # Test CatBoost
    logger.info("Testing CatBoost model...")
    cat_model = CatBoostModel(
        n_estimators=100,
        learning_rate=0.05,
        random_state=42,
        verbose=0
    )
    
    # Train model
    cat_model.fit(X_train, y_train, eval_set=(X_val, y_val))
    
    # Evaluate model
    cat_metrics = cat_model.evaluate(X_test, y_test)
    logger.info(f"CatBoost metrics: {cat_metrics}")
    
    # Save model
    cat_model.save(os.path.join(output_dir, "catboost_model.cbm"))
    
    # Test XGBoost
    logger.info("Testing XGBoost model...")
    xgb_model = XGBoostModel(
        n_estimators=100,
        learning_rate=0.05,
        random_state=42
    )
    
    # Train model
    xgb_model.fit(X_train, y_train, eval_set=eval_set)
    
    # Evaluate model
    xgb_metrics = xgb_model.evaluate(X_test, y_test)
    logger.info(f"XGBoost metrics: {xgb_metrics}")
    
    # Save model
    xgb_model.save(os.path.join(output_dir, "xgboost_model.json"))
    
    # Test MLP
    logger.info("Testing MLP model...")
    mlp_model = MLPModel(
        hidden_layer_sizes=(64, 32),
        max_iter=100,
        random_state=42
    )
    
    # Train model
    mlp_model.fit(X_train, y_train)
    
    # Evaluate model
    mlp_metrics = mlp_model.evaluate(X_test, y_test)
    logger.info(f"MLP metrics: {mlp_metrics}")
    
    # Save model
    mlp_model.save(os.path.join(output_dir, "mlp_model.joblib"))
    
    # Collect all metrics
    all_metrics = {
        'LightGBM': lgb_metrics,
        'CatBoost': cat_metrics,
        'XGBoost': xgb_metrics,
        'MLP': mlp_metrics
    }
    
    # Plot comparison
    plot_model_comparison(all_metrics, output_dir)
    
    return all_metrics

def plot_model_comparison(metrics, output_dir):
    """Plot model comparison."""
    logger.info("Plotting model comparison...")
    
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Extract metrics
    models = list(metrics.keys())
    rmse_values = [metrics[model]['rmse'] for model in models]
    r2_values = [metrics[model]['r2'] for model in models]
    
    # Plot RMSE
    x = np.arange(len(models))
    width = 0.35
    
    ax.bar(x - width/2, rmse_values, width, label='RMSE')
    ax.bar(x + width/2, r2_values, width, label='R²')
    
    ax.set_xlabel('Model')
    ax.set_ylabel('Metric Value')
    ax.set_title('Model Comparison')
    ax.set_xticks(x)
    ax.set_xticklabels(models)
    ax.legend()
    
    # Save figure
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "model_comparison.png"))
    plt.close()
    
    logger.info(f"Saved model comparison plot to {os.path.join(output_dir, 'model_comparison.png')}")

def main():
    """Main function."""
    logger.info("Starting GNN4_IO_4 test script...")
    
    # Create output directory
    output_dir = "/home/ubuntu/GNN4_IO_4/test_output"
    os.makedirs(output_dir, exist_ok=True)
    
    # Test data processing
    split_data, target_column = test_data_processing()
    
    # Test tabular models
    metrics = test_tabular_models(split_data, target_column)
    
    # Save metrics
    import json
    with open(os.path.join(output_dir, "metrics.json"), "w") as f:
        json.dump({k: {m: float(v) for m, v in v.items()} for k, v in metrics.items()}, f, indent=4)
    
    logger.info("Test script completed successfully!")

if __name__ == "__main__":
    main()
