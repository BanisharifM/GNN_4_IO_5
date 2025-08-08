"""
Test script for GNN4_IO_4 implementation with Ray Tune integration.

This script tests the data processing, graph construction, model training,
and Ray Tune optimization using the sample_train_100.csv dataset.
"""

import os
import sys
import pandas as pd
import numpy as np
import torch
import logging
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import json

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import project modules
from src.data import IODataProcessor
from src.models.tabular import LightGBMModel, CatBoostModel, XGBoostModel, MLPModel
from src.models.gnn import TabGNNRegressor
from src.raytune_optimizer import RayTuneOptimizer

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
    os.makedirs(output_dir, exist_ok=True)
    processor.save_processed_data(output_dir, target_column=target_column)
    
    logger.info(f"Saved processed data to {output_dir}")
    
    return processor, split_data, target_column

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
    
    # Save metrics
    with open(os.path.join(output_dir, "tabular_metrics.json"), "w") as f:
        json.dump({k: {m: float(v) for m, v in v.items()} for k, v in all_metrics.items()}, f, indent=4)
    
    # Plot comparison
    plot_model_comparison(all_metrics, output_dir)
    
    return all_metrics

def test_gnn_model(split_data, target_column):
    """Test GNN model."""
    logger.info("Testing GNN model...")
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    
    # Move data to device
    data = split_data.to(device)
    
    # Create model
    model = TabGNNRegressor(
        in_channels=data.x.shape[1],
        hidden_channels=64,
        gnn_out_channels=64,
        mlp_hidden_channels=[64, 32],
        num_layers=2,
        num_graph_types=1,
        model_type="gcn",
        dropout=0.1
    ).to(device)
    
    # Create optimizer
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=0.001,
        weight_decay=0.0001
    )
    
    # Create scheduler
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=0.5,
        patience=5,
        verbose=True
    )
    
    # Training loop
    epochs = 50
    best_val_loss = float('inf')
    best_model_state = None
    patience = 10
    patience_counter = 0
    history = {
        'train_loss': [],
        'val_loss': [],
        'test_loss': []
    }
    
    for epoch in range(epochs):
        # Training
        model.train()
        optimizer.zero_grad()
        
        # Forward pass
        out = model(
            data.x, 
            [data.edge_index],
            batch=None
        )
        
        # Calculate loss
        train_mask = data.train_mask
        train_loss = torch.nn.functional.mse_loss(
            out[train_mask].squeeze(),
            data.y[train_mask]
        )
        
        # Backward pass
        train_loss.backward()
        optimizer.step()
        
        # Validation
        model.eval()
        with torch.no_grad():
            out = model(
                data.x, 
                [data.edge_index],
                batch=None
            )
            
            # Calculate validation loss
            val_mask = data.val_mask
            val_loss = torch.nn.functional.mse_loss(
                out[val_mask].squeeze(),
                data.y[val_mask]
            )
            
            # Calculate test loss
            test_mask = data.test_mask
            test_loss = torch.nn.functional.mse_loss(
                out[test_mask].squeeze(),
                data.y[test_mask]
            )
        
        # Update scheduler
        scheduler.step(val_loss)
        
        # Save history
        history['train_loss'].append(train_loss.item())
        history['val_loss'].append(val_loss.item())
        history['test_loss'].append(test_loss.item())
        
        # Log progress
        logger.info(f"Epoch {epoch+1}/{epochs}: "
                   f"Train Loss: {train_loss.item():.4f}, "
                   f"Val Loss: {val_loss.item():.4f}, "
                   f"Test Loss: {test_loss.item():.4f}")
        
        # Check for early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_state = model.state_dict().copy()
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                logger.info(f"Early stopping at epoch {epoch+1}")
                break
    
    # Load best model
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
    
    # Evaluate model
    model.eval()
    with torch.no_grad():
        out = model(
            data.x, 
            [data.edge_index],
            batch=None
        )
        
        # Calculate test loss
        test_mask = data.test_mask
        test_loss = torch.nn.functional.mse_loss(
            out[test_mask].squeeze(),
            data.y[test_mask]
        )
        
        # Calculate metrics
        y_pred = out[test_mask].squeeze().cpu().numpy()
        y_true = data.y[test_mask].cpu().numpy()
        
        mse = mean_squared_error(y_true, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_true, y_pred)
        r2 = r2_score(y_true, y_pred)
        
        metrics = {
            'mse': mse,
            'rmse': rmse,
            'mae': mae,
            'r2': r2
        }
    
    logger.info(f"GNN metrics: {metrics}")
    
    # Save model
    output_dir = "/home/ubuntu/GNN4_IO_4/test_output/models"
    os.makedirs(output_dir, exist_ok=True)
    model.save_checkpoint(
        checkpoint_dir=output_dir,
        filename="gnn_model.pt"
    )
    
    # Save metrics
    with open(os.path.join(output_dir, "gnn_metrics.json"), "w") as f:
        json.dump({m: float(v) for m, v in metrics.items()}, f, indent=4)
    
    # Plot training history
    plot_training_history(history, os.path.join(output_dir, "gnn_training_history.png"))
    
    return model, metrics, history

def test_raytune_optimization(processor, split_data, target_column):
    """Test Ray Tune optimization."""
    logger.info("Testing Ray Tune optimization...")
    
    # Create output directory
    output_dir = "/home/ubuntu/GNN4_IO_4/test_output/raytune"
    os.makedirs(output_dir, exist_ok=True)
    
    # Create Ray Tune optimizer
    optimizer = RayTuneOptimizer(
        data_processor=processor,
        target_column=target_column,
        output_dir=output_dir,
        num_samples=2,  # Small number for testing
        max_num_epochs=5,  # Small number for testing
        gpus_per_trial=0.5 if torch.cuda.is_available() else 0,
        cpus_per_trial=2
    )
    
    # Test graph construction search space
    graph_space = optimizer.get_graph_construction_search_space()
    logger.info(f"Graph construction search space: {graph_space}")
    
    # Test GNN search space
    gnn_space = optimizer.get_gnn_search_space()
    logger.info(f"GNN search space: {gnn_space}")
    
    # Test tabular search space
    for model_type in ["lightgbm", "catboost", "xgboost", "mlp", "tabnet"]:
        tabular_space = optimizer.get_tabular_search_space(model_type)
        logger.info(f"{model_type} search space: {tabular_space}")
    
    # Save search spaces
    search_spaces = {
        "graph": graph_space,
        "gnn": gnn_space,
        "tabular": {
            model_type: optimizer.get_tabular_search_space(model_type)
            for model_type in ["lightgbm", "catboost", "xgboost", "mlp", "tabnet"]
        }
    }
    
    with open(os.path.join(output_dir, "search_spaces.json"), "w") as f:
        # Convert Ray Tune objects to strings for JSON serialization
        def convert_tune_to_str(obj):
            if isinstance(obj, dict):
                return {k: convert_tune_to_str(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_tune_to_str(v) for v in obj]
            else:
                return str(obj)
        
        json.dump(convert_tune_to_str(search_spaces), f, indent=4)
    
    logger.info(f"Search spaces saved to {os.path.join(output_dir, 'search_spaces.json')}")
    
    # Note: We don't actually run the optimization here as it would take too long
    # and consume too many resources for a simple test
    logger.info("Ray Tune optimization test completed (without actual optimization runs)")

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

def plot_training_history(history, output_path):
    """Plot training history."""
    logger.info("Plotting training history...")
    
    plt.figure(figsize=(10, 6))
    plt.plot(history['train_loss'], label='Train Loss')
    plt.plot(history['val_loss'], label='Validation Loss')
    plt.plot(history['test_loss'], label='Test Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training History')
    plt.legend()
    plt.grid(True)
    plt.savefig(output_path)
    plt.close()
    
    logger.info(f"Saved training history plot to {output_path}")

def main():
    """Main function."""
    logger.info("Starting GNN4_IO_4 test script...")
    
    # Create output directory
    output_dir = "/home/ubuntu/GNN4_IO_4/test_output"
    os.makedirs(output_dir, exist_ok=True)
    
    try:
        # Test data processing
        processor, split_data, target_column = test_data_processing()
        
        # Test tabular models
        tabular_metrics = test_tabular_models(split_data, target_column)
        
        # Test GNN model
        gnn_model, gnn_metrics, gnn_history = test_gnn_model(split_data, target_column)
        
        # Test Ray Tune optimization
        test_raytune_optimization(processor, split_data, target_column)
        
        # Compare all models
        all_metrics = {
            **tabular_metrics,
            'GNN': gnn_metrics
        }
        
        # Save combined metrics
        with open(os.path.join(output_dir, "all_metrics.json"), "w") as f:
            json.dump({k: {m: float(v) for m, v in v.items()} for k, v in all_metrics.items()}, f, indent=4)
        
        # Plot combined comparison
        plot_model_comparison(all_metrics, output_dir)
        
        logger.info("Test script completed successfully!")
        
    except Exception as e:
        logger.error(f"Error in test script: {e}", exc_info=True)
        raise

if __name__ == "__main__":
    main()
