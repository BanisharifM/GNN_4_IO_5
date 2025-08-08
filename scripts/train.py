"""
Training script for TabGNN models with I/O performance data.

This script handles:
1. Loading and preprocessing I/O counter data
2. Constructing multiplex graphs based on important features
3. Training TabGNN models and traditional tabular models
4. Evaluating and comparing model performance
"""

import os
import argparse
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from torch_geometric.data import Data
from typing import List, Dict, Tuple, Optional, Union, Any
import logging
import sys
import json
from datetime import datetime
import yaml

# Add parent directory to path
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
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("train.log")
    ]
)
logger = logging.getLogger(__name__)

def parse_args():
    parser = argparse.ArgumentParser(description="Train TabGNN models for I/O performance prediction")
    
    parser.add_argument("--config_file", type=str, help="YAML config file containing all parameters")
    args, remaining_args = parser.parse_known_args()

    if args.config_file:
        with open(args.config_file, 'r') as f:
            config = yaml.safe_load(f)
        
        # Flatten YAML config into a list of CLI args
        for key, value in config.items():
            if isinstance(value, list):
                value = ' '.join(map(str, value))
            elif isinstance(value, bool):
                value = str(value).lower()
            else:
                value = str(value)
            remaining_args.extend([f'--{key}', value])

    # Parse the complete argument list (defaults + config overrides + CLI)
    final_parser = argparse.ArgumentParser(description="Train TabGNN models for I/O performance prediction")

    final_parser.add_argument("--data_path", type=str, required=True)
    final_parser.add_argument("--output_dir", type=str, default="./output")
    final_parser.add_argument("--target_column", type=str, required=True)
    final_parser.add_argument("--important_features", type=str, default=None)
    final_parser.add_argument("--similarity_threshold", type=float, default=0.05)
    final_parser.add_argument("--model_type", type=str, default="tabgnn",
                              choices=["tabgnn", "lightgbm", "catboost", "xgboost", "mlp", "tabnet", "combined"])
    final_parser.add_argument("--gnn_type", type=str, default="gcn", choices=["gcn", "gat"])
    final_parser.add_argument("--hidden_dim", type=int, default=64)
    final_parser.add_argument("--num_layers", type=int, default=2)
    final_parser.add_argument("--dropout", type=float, default=0.1)
    final_parser.add_argument("--batch_size", type=int, default=32)
    final_parser.add_argument("--epochs", type=int, default=100)
    final_parser.add_argument("--lr", type=float, default=0.001)
    final_parser.add_argument("--weight_decay", type=float, default=0.0001)
    final_parser.add_argument("--patience", type=int, default=10)
    final_parser.add_argument("--seed", type=int, default=42)
    final_parser.add_argument("--device", type=str, default=None)
    final_parser.add_argument("--precomputed_similarity_path", type=str, default=None)

    return final_parser.parse_args(remaining_args)

def set_seed(seed):
    """Set random seed for reproducibility."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def train_tabgnn(
    data: Data,
    args: argparse.Namespace,
    device: torch.device
) -> Tuple[TabGNNRegressor, Dict[str, List[float]]]:
    """
    Train TabGNN model.
    
    Args:
        data (Data): PyTorch Geometric data object
        args (argparse.Namespace): Command line arguments
        device (torch.device): Device to use
        
    Returns:
        Tuple[TabGNNRegressor, Dict[str, List[float]]]: Trained model and training history
    """
    # Move data to device
    data = data.to(device)

    # Prepare multiplex edge indices on the same device
    edge_indices = getattr(data, "edge_indices", None)
    if edge_indices is None:
        edge_indices = [data.edge_index]
    edge_indices = [ei.to(device) for ei in edge_indices]

    # Create model (use true number of channels)
    model = TabGNNRegressor(
        in_channels=data.x.shape[1],
        hidden_channels=args.hidden_dim,
        gnn_out_channels=args.hidden_dim,
        mlp_hidden_channels=[args.hidden_dim, args.hidden_dim // 2],
        num_layers=args.num_layers,
        num_graph_types=len(edge_indices),
        model_type=args.gnn_type,
        dropout=args.dropout
    ).to(device)
    
    # Create optimizer
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay
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
    best_val_loss = float('inf')
    best_model_state = None
    patience_counter = 0
    history = {
        'train_loss': [],
        'val_loss': [],
        'test_loss': []
    }
    
    for epoch in range(args.epochs):
        # Training
        model.train()
        optimizer.zero_grad()
        
        # Forward pass
        out = model(
            data.x,
            edge_indices,
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
            # Recreate the multiplex list on the active device
            edge_indices = getattr(data, "edge_indices", None)
            if edge_indices is None:
                edge_indices = [data.edge_index]
            edge_indices = [ei.to(device) for ei in edge_indices]

            out = model(
                data.x,
                edge_indices,
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
        logger.info(f"Epoch {epoch+1}/{args.epochs}: "
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
            if patience_counter >= args.patience:
                logger.info(f"Early stopping at epoch {epoch+1}")
                break
    
    # Load best model
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
    
    return model, history

def train_tabular_model(
    data: Data,
    args: argparse.Namespace,
    model_type: str
) -> Tuple[Any, Dict[str, float]]:
    """
    Train traditional tabular model.
    
    Args:
        data (Data): PyTorch Geometric data object
        args (argparse.Namespace): Command line arguments
        model_type (str): Type of tabular model
        
    Returns:
        Tuple[Any, Dict[str, float]]: Trained model and evaluation metrics
    """
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
    if model_type == 'lightgbm':
        model = LightGBMModel(
            random_state=args.seed,
            n_estimators=1000,
            learning_rate=0.05
        )
    elif model_type == 'catboost':
        model = CatBoostModel(
            random_state=args.seed,
            n_estimators=1000,
            learning_rate=0.05
        )
    elif model_type == 'xgboost':
        model = XGBoostModel(
            random_state=args.seed,
            n_estimators=1000,
            learning_rate=0.05
        )
    elif model_type == 'mlp':
        model = MLPModel(
            random_state=args.seed,
            hidden_layer_sizes=(args.hidden_dim, args.hidden_dim // 2),
            max_iter=args.epochs
        )
    elif model_type == 'tabnet':
        model = TabNetModel(
            random_state=args.seed,
            n_d=args.hidden_dim,
            n_a=args.hidden_dim,
            n_steps=3,
            max_epochs=args.epochs,
            patience=args.patience
        )
    else:
        raise ValueError(f"Unsupported model type: {model_type}")
    
    # Fit model
    model.fit(X_train, y_train)
    
    # Evaluate model
    metrics = model.evaluate(X_test, y_test)
    
    return model, metrics

def train_combined_model(
    data: Data,
    args: argparse.Namespace,
    device: torch.device
) -> Tuple[TabGNNTabularModel, Dict[str, float]]:
    """
    Train combined TabGNN + tabular model.
    
    Args:
        data (Data): PyTorch Geometric data object
        args (argparse.Namespace): Command line arguments
        device (torch.device): Device to use
        
    Returns:
        Tuple[TabGNNTabularModel, Dict[str, float]]: Trained model and evaluation metrics
    """
    # First train TabGNN model
    tabgnn_model, _ = train_tabgnn(data, args, device)
    
    # Create TabGNNTabularModel with LightGBM
    tabular_model = LightGBMModel(
        random_state=args.seed,
        n_estimators=1000,
        learning_rate=0.05
    )
    
    combined_model = TabGNNTabularModel(
        gnn_model=tabgnn_model,
        tabular_model=tabular_model,
        use_original_features=True
    )

    # Ensure tensors are on the same device as the trained GNN
    x_dev = data.x.to(device)
    y_dev = data.y.to(device)
    edge_indices = getattr(data, "edge_indices", None)
    if edge_indices is None:
        edge_indices = [data.edge_index]
    edge_indices = [ei.to(device) for ei in edge_indices]

    combined_model.fit(
        x_dev,
        edge_indices,
        y_dev,
        batch=None,
        train_mask=data.train_mask
    )

    metrics = combined_model.evaluate(
        X=x_dev,
        y=y_dev,
        edge_indices=edge_indices,
        batch=None,
        mask=data.test_mask
    )

    return combined_model, metrics

def plot_training_history(
    history: Dict[str, List[float]],
    output_path: str
):
    """
    Plot training history.
    
    Args:
        history (Dict[str, List[float]]): Training history
        output_path (str): Path to save plot
    """
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

def main():
    """Main function."""
    # Parse arguments
    args = parse_args()

    logger.info("Parsed arguments:\n" + json.dumps(vars(args), indent=4))
    
    # Parse important_features string to list
    # if args.important_features:
    #     args.important_features = args.important_features.strip().split()

    # Automatically use all features except the target if important_features is not specified
    df_preview = pd.read_csv(args.data_path, nrows=1)
    all_columns = list(df_preview.columns)
    if args.important_features:
        args.important_features = args.important_features.strip().split()
    else:
        args.important_features = [col for col in all_columns if col != args.target_column]

    # Set random seed
    set_seed(args.seed)
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Set device
    if args.device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)
    logger.info(f"Using device: {device}")
    
    # Create data processor
    data_processor = IODataProcessor(
        data_path=args.data_path,
        important_features=args.important_features,
        similarity_thresholds={f: args.similarity_threshold for f in args.important_features} if args.important_features else None,
        precomputed_similarity_path=args.precomputed_similarity_path
    )
    
    # Load and preprocess data
    data_processor.load_data()
    data_processor.preprocess_data()
    
    # Create combined PyG data
    data = data_processor.create_combined_pyg_data(target_column=args.target_column)
    
    # Split data into train, validation, and test sets
    data = data_processor.train_val_test_split(data, random_state=args.seed)
    
    # Save processed data
    processed_data_path = os.path.join(args.output_dir, "processed_data")
    os.makedirs(processed_data_path, exist_ok=True)
    data_processor.save_processed_data(processed_data_path, target_column=args.target_column)
    
    # Train model based on model type
    if args.model_type == 'tabgnn':
        model, history = train_tabgnn(data, args, device)
        
        # Save model
        model_path = os.path.join(args.output_dir, "tabgnn_model.pt")
        model.save_checkpoint(
            checkpoint_dir=args.output_dir,
            epoch=args.epochs,
            optimizer=None,
            train_loss=history['train_loss'][-1],
            val_loss=history['val_loss'][-1],
            filename="tabgnn_model.pt"
        )
        
        # Plot training history
        plot_path = os.path.join(args.output_dir, "training_history.png")
        plot_training_history(history, plot_path)
        
        # Evaluate model
        model.eval()
        with torch.no_grad():
            # Recreate the multiplex list on the active device
            edge_indices = getattr(data, "edge_indices", None)
            if edge_indices is None:
                edge_indices = [data.edge_index]
            edge_indices = [ei.to(device) for ei in edge_indices]

            out = model(
                data.x,
                edge_indices,
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

            # Save metrics
            metrics_path = os.path.join(args.output_dir, "metrics.json")
            with open(metrics_path, 'w') as f:
                json.dump(metrics, f, indent=4)

            logger.info(f"Final metrics: {metrics}")
            logger.info(f"Model and results saved to {args.output_dir}")

        
    elif args.model_type == 'combined':
        model, metrics = train_combined_model(data, args, device)

        # Save metrics
        metrics_path = os.path.join(args.output_dir, "metrics.json")
        with open(metrics_path, 'w') as f:
            json.dump(metrics, f, indent=4)

        logger.info(f"Final metrics: {metrics}")
        logger.info(f"Model and results saved to {args.output_dir}")
        
        # Save model
        gnn_path = os.path.join(args.output_dir, "tabgnn_part.pt")
        tabular_path = os.path.join(args.output_dir, "tabular_part.joblib")
        model.save(gnn_path, tabular_path)
        
    else:
        # Train traditional tabular model
        model, metrics = train_tabular_model(data, args, args.model_type)
        
        # Save metrics
        metrics_path = os.path.join(args.output_dir, "metrics.json")
        with open(metrics_path, 'w') as f:
            json.dump(metrics, f, indent=4)

        logger.info(f"Final metrics: {metrics}")
        logger.info(f"Model and results saved to {args.output_dir}")

        
        # Save model
        model_path = os.path.join(args.output_dir, f"{args.model_type}_model.joblib")
        model.save(model_path)
    
    pass

if __name__ == "__main__":
    main()
