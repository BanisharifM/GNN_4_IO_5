"""
Script to train models with best hyperparameters from Ray Tune optimization.
"""

import os
import argparse
import json
import torch
import numpy as np
import pandas as pd
import logging
from typing import Dict, Any

# Add parent directory to path
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import project modules
from src.data import IODataProcessor
from src.models.gnn import TabGNNRegressor
from src.models.tabular import (
    LightGBMModel, CatBoostModel, XGBoostModel, 
    MLPModel, TabNetModel, TabGNNTabularModel
)

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Train models with best hyperparameters")
    
    parser.add_argument("--data_path", type=str, required=True, help="Path to input data CSV file")
    parser.add_argument("--output_dir", type=str, default="./output", help="Directory to save outputs")
    parser.add_argument("--target_column", type=str, required=True, help="Target column for prediction")
    parser.add_argument("--config_path", type=str, required=True, help="Path to best configurations JSON file")
    parser.add_argument("--device", type=str, default=None, help="Device to use (cuda or cpu, default: auto-detect)")
    
    return parser.parse_args()

def train_gnn_with_config(
    data,
    config: Dict[str, Any],
    output_dir: str,
    device: torch.device
):
    """Train GNN model with best configuration."""
    logger.info("Training GNN model with best configuration")
    
    # Move data to device
    data = data.to(device)
    
    # Create model
    model = TabGNNRegressor(
        in_channels=data.x.shape[1],
        hidden_channels=config["hidden_channels"],
        gnn_out_channels=config["hidden_channels"],
        mlp_hidden_channels=[config["hidden_channels"], config["hidden_channels"] // 2],
        num_layers=config["num_layers"],
        num_graph_types=1,  # Using combined graph
        model_type="gcn",
        dropout=config["dropout"]
    ).to(device)
    
    # Create optimizer
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config["learning_rate"],
        weight_decay=config["weight_decay"]
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
    epochs = 100
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
        
        from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
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
    
    # Save model
    os.makedirs(output_dir, exist_ok=True)
    model.save_checkpoint(
        checkpoint_dir=output_dir,
        filename="gnn_model_best.pt"
    )
    
    # Save metrics
    with open(os.path.join(output_dir, "gnn_metrics.json"), "w") as f:
        json.dump({m: float(v) for m, v in metrics.items()}, f, indent=4)
    
    # Save history
    with open(os.path.join(output_dir, "gnn_history.json"), "w") as f:
        json.dump({k: [float(v) for v in vals] for k, vals in history.items()}, f, indent=4)
    
    logger.info(f"GNN model trained and saved to {output_dir}")
    
    return model, metrics

def train_tabular_with_config(
    data,
    model_type: str,
    config: Dict[str, Any],
    output_dir: str
):
    """Train tabular model with best configuration."""
    logger.info(f"Training {model_type} model with best configuration")
    
    # Convert data to numpy
    X = data.x.cpu().numpy()
    y = data.y.cpu().numpy()
    
    # Get train, val, test indices
    train_idx = data.train_mask.cpu().numpy()
    val_idx = data.val_mask.cpu().numpy()
    test_idx = data.test_mask.cpu().numpy()
    
    # Split data
    X_train, y_train = X[train_idx], y[train_idx]
    X_val, y_val = X[val_idx], y[val_idx]
    X_test, y_test = X[test_idx], y[test_idx]
    
    # Create model
    if model_type == "lightgbm":
        model = LightGBMModel(
            random_state=42,
            **config
        )
    elif model_type == "catboost":
        model = CatBoostModel(
            random_state=42,
            **config
        )
    elif model_type == "xgboost":
        model = XGBoostModel(
            random_state=42,
            **config
        )
    elif model_type == "mlp":
        model = MLPModel(
            random_state=42,
            **config
        )
    elif model_type == "tabnet":
        model = TabNetModel(
            random_state=42,
            **config
        )
    else:
        raise ValueError(f"Unsupported model type: {model_type}")
    
    # Train model
    model.fit(X_train, y_train, eval_set=(X_val, y_val))
    
    # Evaluate model
    metrics = model.evaluate(X_test, y_test)
    
    # Save model
    os.makedirs(output_dir, exist_ok=True)
    model.save(os.path.join(output_dir, f"{model_type}_model_best.joblib"))
    
    # Save metrics
    with open(os.path.join(output_dir, f"{model_type}_metrics.json"), "w") as f:
        json.dump({m: float(v) for m, v in metrics.items()}, f, indent=4)
    
    logger.info(f"{model_type} model trained and saved to {output_dir}")
    
    return model, metrics

def main():
    """Main function."""
    # Parse arguments
    args = parse_args()
    
    # Set device
    if args.device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)
    logger.info(f"Using device: {device}")
    
    # Load best configurations
    with open(args.config_path, "r") as f:
        best_configs = json.load(f)
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Extract graph configuration
    graph_config = best_configs.get("graph", {})
    
    # Create data processor with best graph configuration
    important_features = [
        "POSIX_SEQ_WRITES",
        "POSIX_SIZE_READ_1K_10K",
        "POSIX_SIZE_READ_10K_100K",
        "POSIX_unique_bytes",
        "POSIX_SIZE_READ_0_100",
        "POSIX_SIZE_WRITE_100K_1M",
        "POSIX_write_only_bytes",
        "POSIX_MEM_NOT_ALIGNED",
        "POSIX_FILE_NOT_ALIGNED"
    ]
    
    similarity_threshold = graph_config.get("similarity_threshold", 0.05)
    similarity_metric = graph_config.get("similarity_metric", "cosine")
    max_edges_per_node = graph_config.get("max_edges_per_node", None)
    
    processor = IODataProcessor(
        data_path=args.data_path,
        important_features=important_features,
        similarity_thresholds={f: similarity_threshold for f in important_features},
        similarity_metric=similarity_metric,
        max_edges_per_node=max_edges_per_node
    )
    
    # Load and preprocess data
    processor.load_data()
    processor.preprocess_data()
    
    # Create PyG data
    data = processor.create_combined_pyg_data(target_column=args.target_column)
    
    # Split data
    data = processor.train_val_test_split(data)
    
    # Train GNN model with best configuration
    gnn_config = best_configs.get("gnn", {})
    if gnn_config:
        gnn_model, gnn_metrics = train_gnn_with_config(
            data=data,
            config=gnn_config,
            output_dir=os.path.join(args.output_dir, "gnn"),
            device=device
        )
    
    # Train tabular models with best configurations
    tabular_models = {}
    tabular_metrics = {}
    
    for model_type in ["lightgbm", "catboost", "xgboost", "mlp", "tabnet"]:
        config_key = f"tabular_{model_type}"
        if config_key in best_configs:
            model, metrics = train_tabular_with_config(
                data=data,
                model_type=model_type,
                config=best_configs[config_key],
                output_dir=os.path.join(args.output_dir, model_type)
            )
            tabular_models[model_type] = model
            tabular_metrics[model_type] = metrics
    
    # Create comparison report
    all_metrics = {
        **tabular_metrics,
        "gnn": gnn_metrics if "gnn" in locals() else {}
    }
    
    with open(os.path.join(args.output_dir, "all_metrics.json"), "w") as f:
        json.dump(all_metrics, f, indent=4)
    
    logger.info(f"All models trained and saved to {args.output_dir}")

if __name__ == "__main__":
    main()
