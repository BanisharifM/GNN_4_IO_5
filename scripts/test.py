"""
Testing script for the GNN4_IO_4 implementation.

This script tests the data processing, graph construction, and model training
components of the GNN4_IO_4 solution using the provided sample data.
"""

import os
import sys
import argparse
import torch
import numpy as np
import pandas as pd
import logging
from sklearn.model_selection import train_test_split

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
        logging.FileHandler("test.log")
    ]
)
logger = logging.getLogger(__name__)

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Test GNN4_IO_4 implementation")
    
    # Data arguments
    parser.add_argument("--data_path", type=str, default="/home/ubuntu/upload/sample_train_100.csv", 
                        help="Path to input data CSV file")
    parser.add_argument("--output_dir", type=str, default="./test_output", 
                        help="Directory to save test outputs")
    
    # Test arguments
    parser.add_argument("--test_data_processing", action="store_true", 
                        help="Test data processing module")
    parser.add_argument("--test_graph_construction", action="store_true", 
                        help="Test graph construction module")
    parser.add_argument("--test_model_training", action="store_true", 
                        help="Test model training")
    parser.add_argument("--test_all", action="store_true", 
                        help="Run all tests")
    
    # Device arguments
    parser.add_argument("--device", type=str, default=None, 
                        help="Device to use (cuda or cpu, default: auto-detect)")
    
    return parser.parse_args()

def test_data_processing(data_path, output_dir):
    """
    Test data processing module.
    
    Args:
        data_path (str): Path to input data CSV file
        output_dir (str): Directory to save test outputs
    """
    logger.info("Testing data processing module...")
    
    # Create data processor
    data_processor = IODataProcessor(
        data_path=data_path,
        important_features=[
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
    )
    
    # Load and preprocess data
    data = data_processor.load_data()
    preprocessed_data = data_processor.preprocess_data()
    
    # Check if data is loaded and preprocessed correctly
    assert data is not None, "Failed to load data"
    assert preprocessed_data is not None, "Failed to preprocess data"
    
    # Save preprocessed data
    os.makedirs(output_dir, exist_ok=True)
    preprocessed_data.to_csv(os.path.join(output_dir, "preprocessed_data.csv"), index=False)
    
    logger.info(f"Data processing test passed. Data shape: {preprocessed_data.shape}")
    logger.info(f"Preprocessed data saved to {os.path.join(output_dir, 'preprocessed_data.csv')}")
    
    return preprocessed_data

def test_graph_construction(data_processor, output_dir):
    """
    Test graph construction module.
    
    Args:
        data_processor (IODataProcessor): Data processor instance
        output_dir (str): Directory to save test outputs
    """
    logger.info("Testing graph construction module...")
    
    # Construct multiplex graphs
    multiplex_edges = data_processor.construct_multiplex_graphs()
    
    # Check if graphs are constructed correctly
    assert multiplex_edges is not None, "Failed to construct multiplex graphs"
    assert len(multiplex_edges) > 0, "No graphs constructed"
    
    # Create PyG data objects
    pyg_data_dict = data_processor.create_pyg_data(target_column=None)
    
    # Check if PyG data objects are created correctly
    assert pyg_data_dict is not None, "Failed to create PyG data objects"
    assert len(pyg_data_dict) > 0, "No PyG data objects created"
    
    # Create combined PyG data
    combined_data = data_processor.create_combined_pyg_data(target_column=None)
    
    # Check if combined PyG data is created correctly
    assert combined_data is not None, "Failed to create combined PyG data"
    assert combined_data.x is not None, "Combined PyG data has no node features"
    assert combined_data.edge_index is not None, "Combined PyG data has no edges"
    
    # Split data into train, validation, and test sets
    split_data = data_processor.train_val_test_split(combined_data)
    
    # Check if data is split correctly
    assert split_data.train_mask is not None, "Failed to create train mask"
    assert split_data.val_mask is not None, "Failed to create validation mask"
    assert split_data.test_mask is not None, "Failed to create test mask"
    
    # Save graph statistics
    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, "graph_stats.txt"), "w") as f:
        f.write(f"Number of graphs: {len(multiplex_edges)}\n")
        for feature, (src, dst) in multiplex_edges.items():
            f.write(f"Graph for {feature}: {len(src)//2} edges\n")
        
        f.write(f"\nCombined graph: {combined_data.edge_index.shape[1]} edges\n")
        f.write(f"Number of nodes: {combined_data.x.shape[0]}\n")
        f.write(f"Node feature dimension: {combined_data.x.shape[1]}\n")
        
        f.write(f"\nTrain samples: {split_data.train_mask.sum().item()}\n")
        f.write(f"Validation samples: {split_data.val_mask.sum().item()}\n")
        f.write(f"Test samples: {split_data.test_mask.sum().item()}\n")
    
    logger.info(f"Graph construction test passed. Created {len(multiplex_edges)} graphs.")
    logger.info(f"Graph statistics saved to {os.path.join(output_dir, 'graph_stats.txt')}")
    
    return split_data

def test_model_training(data, output_dir, device):
    """
    Test model training.
    
    Args:
        data (Data): PyTorch Geometric data object
        output_dir (str): Directory to save test outputs
        device (torch.device): Device to use
    """
    logger.info("Testing model training...")
    
    # Move data to device
    data = data.to(device)
    
    # Create TabGNN model
    model = TabGNNRegressor(
        in_channels=data.x.shape[1],
        hidden_channels=32,
        gnn_out_channels=32,
        mlp_hidden_channels=[32, 16],
        num_layers=2,
        num_graph_types=1,
        model_type='gcn',
        dropout=0.1
    ).to(device)
    
    # Create optimizer
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=0.001,
        weight_decay=0.0001
    )
    
    # Train for a few epochs
    model.train()
    for epoch in range(5):
        optimizer.zero_grad()
        
        # Forward pass
        out = model(
            data.x, 
            [data.edge_index],
            batch=None
        )
        
        # Calculate loss (using all data for testing)
        loss = torch.nn.functional.mse_loss(
            out.squeeze(),
            data.y
        )
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        logger.info(f"Epoch {epoch+1}/5: Loss: {loss.item():.4f}")
    
    # Test LightGBM model
    X = data.x.cpu().numpy()
    y = data.y.cpu().numpy()
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Create and train LightGBM model
    lgb_model = LightGBMModel(random_state=42)
    lgb_model.fit(X_train, y_train)
    
    # Evaluate model
    metrics = lgb_model.evaluate(X_test, y_test)
    
    # Save metrics
    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, "model_metrics.txt"), "w") as f:
        f.write("LightGBM Model Metrics:\n")
        for metric, value in metrics.items():
            f.write(f"{metric}: {value}\n")
    
    logger.info(f"Model training test passed. LightGBM metrics: {metrics}")
    logger.info(f"Model metrics saved to {os.path.join(output_dir, 'model_metrics.txt')}")

def main():
    """Main function."""
    # Parse arguments
    args = parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Set device
    if args.device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)
    logger.info(f"Using device: {device}")
    
    # Run tests
    if args.test_all or args.test_data_processing:
        preprocessed_data = test_data_processing(args.data_path, args.output_dir)
        
        # Create data processor for subsequent tests
        data_processor = IODataProcessor(
            data_path=args.data_path,
            important_features=[
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
        )
        data_processor.data = preprocessed_data
    else:
        # Create data processor for subsequent tests
        data_processor = IODataProcessor(
            data_path=args.data_path,
            important_features=[
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
        )
        data_processor.load_data()
        data_processor.preprocess_data()
    
    if args.test_all or args.test_graph_construction:
        data = test_graph_construction(data_processor, args.output_dir)
    else:
        # Create data for model training test
        data = data_processor.create_combined_pyg_data(target_column=None)
        data = data_processor.train_val_test_split(data)
    
    if args.test_all or args.test_model_training:
        test_model_training(data, args.output_dir, device)
    
    logger.info("All tests completed successfully!")

if __name__ == "__main__":
    main()
