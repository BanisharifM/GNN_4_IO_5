"""
Ray Tune integration for GNN4_IO_4.

This module implements Ray Tune integration for hyperparameter optimization
of both graph construction parameters and model hyperparameters.
"""

import os
import torch
import numpy as np
import pandas as pd
import logging
from typing import Dict, List, Tuple, Optional, Union, Any
from ray import tune
from ray.tune.schedulers import ASHAScheduler
from ray.tune.search.optuna import OptunaSearch
from torch_geometric.data import Data

# Set up logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

class RayTuneOptimizer:
    """
    Ray Tune optimizer for hyperparameter tuning.
    """
    
    def __init__(
        self,
        data_processor,
        target_column: str,
        output_dir: str,
        num_samples: int = 10,
        max_num_epochs: int = 50,
        gpus_per_trial: float = 0.5,
        cpus_per_trial: int = 2
    ):
        """
        Initialize Ray Tune optimizer.
        
        Args:
            data_processor: Data processor for I/O counter data
            target_column: Target column for prediction
            output_dir: Directory to save outputs
            num_samples: Number of samples for hyperparameter search
            max_num_epochs: Maximum number of epochs per trial
            gpus_per_trial: GPUs per trial
            cpus_per_trial: CPUs per trial
        """
        self.data_processor = data_processor
        self.target_column = target_column
        self.output_dir = output_dir
        self.num_samples = num_samples
        self.max_num_epochs = max_num_epochs
        self.gpus_per_trial = gpus_per_trial
        self.cpus_per_trial = cpus_per_trial
        
        logger.info(f"Initialized Ray Tune optimizer with {num_samples} samples")
    
    def get_graph_construction_search_space(self) -> Dict[str, Any]:
        """
        Get search space for graph construction parameters.
        
        Returns:
            Dict[str, Any]: Search space for graph construction parameters
        """
        return {
            "similarity_threshold": tune.uniform(0.01, 0.5),
            "max_edges_per_node": tune.choice([None, 5, 10, 20, 50]),
            "similarity_metric": tune.choice(["cosine", "euclidean"])
        }
    
    def get_gnn_search_space(self) -> Dict[str, Any]:
        """
        Get search space for GNN hyperparameters.
        
        Returns:
            Dict[str, Any]: Search space for GNN hyperparameters
        """
        return {
            "hidden_channels": tune.choice([32, 64, 128, 256]),
            "num_layers": tune.choice([1, 2, 3, 4]),
            "dropout": tune.uniform(0.0, 0.5),
            "learning_rate": tune.loguniform(1e-4, 1e-2),
            "weight_decay": tune.loguniform(1e-5, 1e-3),
            "batch_size": tune.choice([16, 32, 64, 128])
        }
    
    def get_tabular_search_space(self, model_type: str) -> Dict[str, Any]:
        """
        Get search space for tabular model hyperparameters.
        
        Args:
            model_type: Type of tabular model
            
        Returns:
            Dict[str, Any]: Search space for tabular model hyperparameters
        """
        if model_type == "lightgbm":
            return {
                "n_estimators": tune.choice([100, 200, 500, 1000]),
                "learning_rate": tune.loguniform(0.01, 0.3),
                "max_depth": tune.choice([-1, 3, 5, 7, 9]),
                "num_leaves": tune.choice([31, 63, 127, 255]),
                "min_child_samples": tune.choice([5, 10, 20, 50]),
                "subsample": tune.uniform(0.5, 1.0),
                "colsample_bytree": tune.uniform(0.5, 1.0),
                "reg_alpha": tune.loguniform(1e-8, 1.0),
                "reg_lambda": tune.loguniform(1e-8, 1.0)
            }
        elif model_type == "catboost":
            return {
                "n_estimators": tune.choice([100, 200, 500, 1000]),
                "learning_rate": tune.loguniform(0.01, 0.3),
                "max_depth": tune.choice([3, 5, 7, 9]),
                "l2_leaf_reg": tune.loguniform(1.0, 10.0),
                "subsample": tune.uniform(0.5, 1.0),
                "colsample_bylevel": tune.uniform(0.5, 1.0)
            }
        elif model_type == "xgboost":
            return {
                "n_estimators": tune.choice([100, 200, 500, 1000]),
                "learning_rate": tune.loguniform(0.01, 0.3),
                "max_depth": tune.choice([3, 5, 7, 9]),
                "min_child_weight": tune.choice([1, 3, 5, 7]),
                "subsample": tune.uniform(0.5, 1.0),
                "colsample_bytree": tune.uniform(0.5, 1.0),
                "reg_alpha": tune.loguniform(1e-8, 1.0),
                "reg_lambda": tune.loguniform(1e-8, 1.0)
            }
        elif model_type == "mlp":
            return {
                "hidden_layer_sizes": tune.choice([
                    (64,), (128,), (256,),
                    (64, 32), (128, 64), (256, 128),
                    (64, 32, 16), (128, 64, 32), (256, 128, 64)
                ]),
                "activation": tune.choice(["relu", "tanh"]),
                "alpha": tune.loguniform(1e-5, 1e-3),
                "learning_rate_init": tune.loguniform(1e-4, 1e-2),
                "batch_size": tune.choice(["auto", 32, 64, 128, 256])
            }
        elif model_type == "tabnet":
            return {
                "n_d": tune.choice([8, 16, 32, 64]),
                "n_a": tune.choice([8, 16, 32, 64]),
                "n_steps": tune.choice([3, 5, 7, 9]),
                "gamma": tune.uniform(1.0, 2.0),
                "n_independent": tune.choice([1, 2, 3]),
                "n_shared": tune.choice([1, 2, 3]),
                "lambda_sparse": tune.loguniform(1e-6, 1e-3),
                "batch_size": tune.choice([256, 512, 1024, 2048])
            }
        else:
            raise ValueError(f"Unsupported model type: {model_type}")
    
    def train_gnn_with_config(self, config: Dict[str, Any], data: Data, checkpoint_dir: Optional[str] = None):
        """
        Train GNN model with given configuration.
        
        Args:
            config: Configuration for training
            data: PyTorch Geometric data object
            checkpoint_dir: Directory for checkpoints
            
        Returns:
            Dict[str, float]: Training metrics
        """
        import torch.nn as nn
        import torch.optim as optim
        from src.models.gnn import TabGNNRegressor
        
        # Set device
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
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
        
        # Load checkpoint if provided
        if checkpoint_dir:
            model_state, optimizer_state = torch.load(
                os.path.join(checkpoint_dir, "checkpoint")
            )
            model.load_state_dict(model_state)
        
        # Create optimizer
        optimizer = optim.Adam(
            model.parameters(),
            lr=config["learning_rate"],
            weight_decay=config["weight_decay"]
        )
        
        # Load optimizer state if provided
        if checkpoint_dir and optimizer_state:
            optimizer.load_state_dict(optimizer_state)
        
        # Training loop
        for epoch in range(self.max_num_epochs):
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
            train_loss = nn.functional.mse_loss(
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
                val_loss = nn.functional.mse_loss(
                    out[val_mask].squeeze(),
                    data.y[val_mask]
                )
            
            # Report metrics to Ray Tune
            tune.report(
                train_loss=train_loss.item(),
                val_loss=val_loss.item(),
                epoch=epoch
            )
            
            # Save checkpoint
            with tune.checkpoint_dir(epoch) as checkpoint_dir:
                torch.save(
                    (model.state_dict(), optimizer.state_dict()),
                    os.path.join(checkpoint_dir, "checkpoint")
                )
    
    def train_tabular_with_config(self, config: Dict[str, Any], data: Data, model_type: str):
        """
        Train tabular model with given configuration.
        
        Args:
            config: Configuration for training
            data: PyTorch Geometric data object
            model_type: Type of tabular model
            
        Returns:
            Dict[str, float]: Training metrics
        """
        from src.models.tabular import (
            LightGBMModel, CatBoostModel, XGBoostModel, 
            MLPModel, TabNetModel
        )
        
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
                n_estimators=config["n_estimators"],
                learning_rate=config["learning_rate"],
                max_depth=config["max_depth"],
                num_leaves=config["num_leaves"],
                min_child_samples=config["min_child_samples"],
                subsample=config["subsample"],
                colsample_bytree=config["colsample_bytree"],
                reg_alpha=config["reg_alpha"],
                reg_lambda=config["reg_lambda"]
            )
        elif model_type == "catboost":
            model = CatBoostModel(
                random_state=42,
                n_estimators=config["n_estimators"],
                learning_rate=config["learning_rate"],
                max_depth=config["max_depth"],
                l2_leaf_reg=config["l2_leaf_reg"],
                subsample=config["subsample"],
                colsample_bylevel=config["colsample_bylevel"]
            )
        elif model_type == "xgboost":
            model = XGBoostModel(
                random_state=42,
                n_estimators=config["n_estimators"],
                learning_rate=config["learning_rate"],
                max_depth=config["max_depth"],
                min_child_weight=config["min_child_weight"],
                subsample=config["subsample"],
                colsample_bytree=config["colsample_bytree"],
                reg_alpha=config["reg_alpha"],
                reg_lambda=config["reg_lambda"]
            )
        elif model_type == "mlp":
            model = MLPModel(
                random_state=42,
                hidden_layer_sizes=config["hidden_layer_sizes"],
                activation=config["activation"],
                alpha=config["alpha"],
                learning_rate_init=config["learning_rate_init"],
                batch_size=config["batch_size"]
            )
        elif model_type == "tabnet":
            model = TabNetModel(
                random_state=42,
                n_d=config["n_d"],
                n_a=config["n_a"],
                n_steps=config["n_steps"],
                gamma=config["gamma"],
                n_independent=config["n_independent"],
                n_shared=config["n_shared"],
                lambda_sparse=config["lambda_sparse"],
                batch_size=config["batch_size"]
            )
        else:
            raise ValueError(f"Unsupported model type: {model_type}")
        
        # Fit model
        model.fit(X_train, y_train, eval_set=(X_val, y_val))
        
        # Evaluate model
        val_pred = model.predict(X_val)
        val_mse = np.mean((val_pred - y_val) ** 2)
        val_rmse = np.sqrt(val_mse)
        
        # Report metrics to Ray Tune
        tune.report(val_loss=val_mse, val_rmse=val_rmse)
    
    def optimize_graph_construction(self):
        """
        Optimize graph construction parameters.
        
        Returns:
            Dict[str, Any]: Best configuration for graph construction
        """
        logger.info("Optimizing graph construction parameters")
        
        # Define search space
        search_space = self.get_graph_construction_search_space()
        
        # Define objective function
        def objective(config):
            # Update data processor with config
            self.data_processor.similarity_thresholds = {
                feature: config["similarity_threshold"] for feature in self.data_processor.important_features
            }
            self.data_processor.graph_constructor = None  # Reset graph constructor
            
            # Create graph constructor with config
            from src.data import MultiplexGraphConstructor
            self.data_processor.graph_constructor = MultiplexGraphConstructor(
                important_features=self.data_processor.important_features,
                similarity_thresholds=self.data_processor.similarity_thresholds,
                similarity_metric=config["similarity_metric"],
                max_edges_per_node=config["max_edges_per_node"]
            )
            
            # Create PyG data
            data = self.data_processor.create_combined_pyg_data(target_column=self.target_column)
            
            # Split data
            data = self.data_processor.train_val_test_split(data)
            
            # Train a simple model to evaluate graph quality
            from src.models.gnn import TabGNNRegressor
            
            # Set device
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            
            # Move data to device
            data = data.to(device)
            
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
            optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
            
            # Train for a few epochs
            for epoch in range(10):
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
            
            # Report final validation loss
            tune.report(val_loss=val_loss.item())
        
        # Create scheduler
        scheduler = ASHAScheduler(
            metric="val_loss",
            mode="min",
            max_t=10,
            grace_period=1,
            reduction_factor=2
        )
        
        # Create search algorithm
        search_alg = OptunaSearch(
            metric="val_loss",
            mode="min"
        )
        
        # Run optimization
        result = tune.run(
            objective,
            resources_per_trial={"cpu": self.cpus_per_trial, "gpu": self.gpus_per_trial},
            config=search_space,
            num_samples=self.num_samples,
            scheduler=scheduler,
            search_alg=search_alg,
            local_dir=os.path.join(self.output_dir, "ray_results"),
            name="graph_construction_optimization"
        )
        
        # Get best config
        best_config = result.get_best_config(metric="val_loss", mode="min")
        logger.info(f"Best graph construction config: {best_config}")
        
        return best_config
    
    def optimize_gnn(self, data: Data):
        """
        Optimize GNN hyperparameters.
        
        Args:
            data: PyTorch Geometric data object
            
        Returns:
            Dict[str, Any]: Best configuration for GNN
        """
        logger.info("Optimizing GNN hyperparameters")
        
        # Define search space
        search_space = self.get_gnn_search_space()
        
        # Create scheduler
        scheduler = ASHAScheduler(
            metric="val_loss",
            mode="min",
            max_t=self.max_num_epochs,
            grace_period=5,
            reduction_factor=2
        )
        
        # Create search algorithm
        search_alg = OptunaSearch(
            metric="val_loss",
            mode="min"
        )
        
        # Run optimization
        result = tune.run(
            tune.with_parameters(self.train_gnn_with_config, data=data),
            resources_per_trial={"cpu": self.cpus_per_trial, "gpu": self.gpus_per_trial},
            config=search_space,
            num_samples=self.num_samples,
            scheduler=scheduler,
            search_alg=search_alg,
            local_dir=os.path.join(self.output_dir, "ray_results"),
            name="gnn_optimization"
        )
        
        # Get best config
        best_config = result.get_best_config(metric="val_loss", mode="min")
        logger.info(f"Best GNN config: {best_config}")
        
        return best_config
    
    def optimize_tabular(self, data: Data, model_type: str):
        """
        Optimize tabular model hyperparameters.
        
        Args:
            data: PyTorch Geometric data object
            model_type: Type of tabular model
            
        Returns:
            Dict[str, Any]: Best configuration for tabular model
        """
        logger.info(f"Optimizing {model_type} hyperparameters")
        
        # Define search space
        search_space = self.get_tabular_search_space(model_type)
        
        # Create scheduler
        scheduler = ASHAScheduler(
            metric="val_loss",
            mode="min",
            max_t=1,  # Only one iteration for tabular models
            grace_period=1,
            reduction_factor=2
        )
        
        # Create search algorithm
        search_alg = OptunaSearch(
            metric="val_loss",
            mode="min"
        )
        
        # Run optimization
        result = tune.run(
            tune.with_parameters(self.train_tabular_with_config, data=data, model_type=model_type),
            resources_per_trial={"cpu": self.cpus_per_trial, "gpu": 0},
            config=search_space,
            num_samples=self.num_samples,
            scheduler=scheduler,
            search_alg=search_alg,
            local_dir=os.path.join(self.output_dir, "ray_results"),
            name=f"{model_type}_optimization"
        )
        
        # Get best config
        best_config = result.get_best_config(metric="val_loss", mode="min")
        logger.info(f"Best {model_type} config: {best_config}")
        
        return best_config
    
    def run_optimization(self):
        """
        Run full optimization pipeline.
        
        Returns:
            Dict[str, Dict[str, Any]]: Best configurations for all components
        """
        logger.info("Running full optimization pipeline")
        
        # Step 1: Optimize graph construction
        graph_config = self.optimize_graph_construction()
        
        # Update data processor with best graph config
        self.data_processor.similarity_thresholds = {
            feature: graph_config["similarity_threshold"] for feature in self.data_processor.important_features
        }
        self.data_processor.graph_constructor = None  # Reset graph constructor
        
        # Create graph constructor with best config
        from src.data import MultiplexGraphConstructor
        self.data_processor.graph_constructor = MultiplexGraphConstructor(
            important_features=self.data_processor.important_features,
            similarity_thresholds=self.data_processor.similarity_thresholds,
            similarity_metric=graph_config["similarity_metric"],
            max_edges_per_node=graph_config["max_edges_per_node"]
        )
        
        # Create PyG data with best graph config
        data = self.data_processor.create_combined_pyg_data(target_column=self.target_column)
        
        # Split data
        data = self.data_processor.train_val_test_split(data)
        
        # Step 2: Optimize GNN
        gnn_config = self.optimize_gnn(data)
        
        # Step 3: Optimize tabular models
        tabular_configs = {}
        for model_type in ["lightgbm", "catboost", "xgboost", "mlp", "tabnet"]:
            tabular_configs[model_type] = self.optimize_tabular(data, model_type)
        
        # Combine all configs
        best_configs = {
            "graph": graph_config,
            "gnn": gnn_config,
            **{f"tabular_{k}": v for k, v in tabular_configs.items()}
        }
        
        # Save best configs
        os.makedirs(self.output_dir, exist_ok=True)
        with open(os.path.join(self.output_dir, "best_configs.json"), "w") as f:
            import json
            json.dump(best_configs, f, indent=4)
        
        logger.info(f"Best configurations saved to {os.path.join(self.output_dir, 'best_configs.json')}")
        
        return best_configs
