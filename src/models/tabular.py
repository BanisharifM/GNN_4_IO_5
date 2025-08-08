"""
Tabular models implementation for GNN4_IO_4.

This module implements various tabular models from aiio-master:
- LightGBM
- CatBoost
- XGBoost
- MLP
- TabNet

Each model follows the TabularModelBase interface for consistency.
"""

import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Dict, Tuple, Optional, Union, Any
import logging
import joblib
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.preprocessing import StandardScaler
from src.models.base import TabularModel

# Set up logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

class TabularModelBase(TabularModel):
    """
    Base class for all tabular models.
    """
    
    def __init__(
        self, 
        model_type: str,
        random_state: int = 42
    ):
        """
        Initialize the base tabular model.
        
        Args:
            model_type (str): Type of model ('lightgbm', 'catboost', 'xgboost', 'mlp', 'tabnet')
            random_state (int): Random seed for reproducibility
        """
        self.model_type = model_type
        self.random_state = random_state
        self.model = None
        self.scaler = None
        self.feature_names = None
        
        logger.info(f"Initialized {model_type} model")
    
    def _to_numpy_mask(self, m):
        if m is None:
            return None
        if isinstance(m, torch.Tensor):
            m = m.detach().cpu().numpy()
        m = np.asarray(m).astype(bool)
        return m

    def _flatten1d(self, a):
        a = np.asarray(a)
        return a.reshape(-1)

    def fit(
        self,
        x,
        edge_indices,
        y,
        batch=None,
        train_mask=None,
        **kwargs
    ):
        """
        Fit the combined model on the (masked) training rows.
        """
        # 1) Extract embeddings on the full graph
        embeddings = self.extract_embeddings(x, edge_indices, batch)

        # 2) To numpy
        X_np = x.detach().cpu().numpy() if isinstance(x, torch.Tensor) else x
        y_np = y.detach().cpu().numpy() if isinstance(y, torch.Tensor) else y

        # 3) Combine features
        X_all = self.combine_features(X_np, embeddings)

        # 4) Apply train mask if provided
        if train_mask is not None:
            m = train_mask.detach().cpu().numpy() if isinstance(train_mask, torch.Tensor) else train_mask
            m = np.asarray(m, dtype=bool)
            X_fit = X_all[m]
            y_fit = y_np[m]
        else:
            X_fit = X_all
            y_fit = y_np

        # 5) Train underlying tabular estimator
        self.model.fit(X_fit, y_fit)
        logger.info(f"TabGNN tabular model fitted with {X_fit.shape[1]} features on {X_fit.shape[0]} rows")
        return self

    def predict(
        self, 
        X: np.ndarray
    ) -> np.ndarray:
        """
        Make predictions with the model.
        
        Args:
            X (np.ndarray): Input features
            
        Returns:
            np.ndarray: Predictions
        """
        raise NotImplementedError("Subclasses must implement predict method")
    
    def save(
        self, 
        path: str
    ) -> str:
        """
        Save the model to a file.
        
        Args:
            path (str): Path to save the model
            
        Returns:
            str: Path to the saved model
        """
        os.makedirs(os.path.dirname(path), exist_ok=True)
        
        # Save model and scaler
        model_data = {
            'model': self.model,
            'scaler': self.scaler,
            'feature_names': self.feature_names,
            'model_type': self.model_type,
            'random_state': self.random_state
        }
        
        joblib.dump(model_data, path)
        logger.info(f"Model saved to {path}")
        
        return path
    
    @classmethod
    def load(
        cls, 
        path: str
    ) -> 'TabularModelBase':
        """
        Load a model from a file.
        
        Args:
            path (str): Path to the saved model
            
        Returns:
            TabularModelBase: Loaded model
        """
        # Load model data
        model_data = joblib.load(path)
        
        # Create instance
        instance = cls(
            model_type=model_data['model_type'],
            random_state=model_data['random_state']
        )
        
        # Set attributes
        instance.model = model_data['model']
        instance.scaler = model_data['scaler']
        instance.feature_names = model_data['feature_names']
        
        logger.info(f"Model loaded from {path}")
        
        return instance

    def evaluate(
        self,
        X,
        y,
        edge_indices=None,
        batch=None,
        mask=None
    ) -> Dict[str, float]:
        """
        Dual-mode evaluation:
        * Plain tabular (edge_indices is None) → predict on X directly.
        * TabGNN + tabular (edge_indices provided) → extract embeddings, combine, then predict.
        """

        # ---------- Plain tabular ----------
        if edge_indices is None:
            X_np = X.detach().cpu().numpy() if isinstance(X, torch.Tensor) else X
            y_np = y.detach().cpu().numpy() if isinstance(y, torch.Tensor) else y
            y_pred = self.predict(X_np)
            y_pred = np.asarray(y_pred).reshape(-1)
            y_np = np.asarray(y_np).reshape(-1)

            mse  = mean_squared_error(y_np, y_pred)
            rmse = float(np.sqrt(mse))
            mae  = mean_absolute_error(y_np, y_pred)
            r2   = r2_score(y_np, y_pred)
            return {"mse": mse, "rmse": rmse, "mae": mae, "r2": r2}

        # ---------- TabGNN + tabular ----------
        embeddings = self.extract_embeddings(X, edge_indices, batch)

        X_np = X.detach().cpu().numpy() if isinstance(X, torch.Tensor) else X
        y_np = y.detach().cpu().numpy() if isinstance(y, torch.Tensor) else y

        X_all = self.combine_features(X_np, embeddings)

        if mask is not None:
            m = mask.detach().cpu().numpy() if isinstance(mask, torch.Tensor) else mask
            m = np.asarray(m, dtype=bool)
            X_eval = X_all[m]
            y_eval = y_np[m]
        else:
            X_eval = X_all
            y_eval = y_np

        y_pred = self.predict(X_eval)
        y_pred = np.asarray(y_pred).reshape(-1)
        y_eval = np.asarray(y_eval).reshape(-1)

        mse  = mean_squared_error(y_eval, y_pred)
        rmse = float(np.sqrt(mse))
        mae  = mean_absolute_error(y_eval, y_pred)
        r2   = r2_score(y_eval, y_pred)
        metrics = {"mse": mse, "rmse": rmse, "mae": mae, "r2": r2}
        logger.info(f"Evaluation metrics: {metrics}")
        return metrics

class LightGBMModel(TabularModelBase):
    """
    LightGBM model for tabular data.
    """
    
    def __init__(
        self, 
        random_state: int = 42,
        n_estimators: int = 1000,
        learning_rate: float = 0.05,
        max_depth: int = -1,
        num_leaves: int = 31,
        min_child_samples: int = 20,
        subsample: float = 0.8,
        colsample_bytree: float = 0.8,
        reg_alpha: float = 0.0,
        reg_lambda: float = 0.0
    ):
        """
        Initialize LightGBM model.
        
        Args:
            random_state (int): Random seed for reproducibility
            n_estimators (int): Number of boosting iterations
            learning_rate (float): Learning rate
            max_depth (int): Maximum tree depth
            num_leaves (int): Maximum number of leaves in a tree
            min_child_samples (int): Minimum number of samples in a leaf
            subsample (float): Subsample ratio of the training data
            colsample_bytree (float): Subsample ratio of columns when constructing each tree
            reg_alpha (float): L1 regularization term
            reg_lambda (float): L2 regularization term
        """
        super(LightGBMModel, self).__init__(model_type='lightgbm', random_state=random_state)
        
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.num_leaves = num_leaves
        self.min_child_samples = min_child_samples
        self.subsample = subsample
        self.colsample_bytree = colsample_bytree
        self.reg_alpha = reg_alpha
        self.reg_lambda = reg_lambda
    
    def fit(
        self, 
        X: np.ndarray, 
        y: np.ndarray,
        feature_names: Optional[List[str]] = None,
        scale_features: bool = True,
        **kwargs
    ) -> 'LightGBMModel':
        """
        Fit the LightGBM model to the data.
        
        Args:
            X (np.ndarray): Input features
            y (np.ndarray): Target values
            feature_names (List[str], optional): Names of features
            scale_features (bool): Whether to scale features
            **kwargs: Additional arguments for LightGBM
            
        Returns:
            LightGBMModel: Self
        """
        import lightgbm as lgb
        
        # Store feature names
        self.feature_names = feature_names
        
        # Scale features if needed
        if scale_features:
            self.scaler = StandardScaler()
            X = self.scaler.fit_transform(X)
        
        # Create LightGBM model
        self.model = lgb.LGBMRegressor(
            n_estimators=self.n_estimators,
            learning_rate=self.learning_rate,
            max_depth=self.max_depth,
            num_leaves=self.num_leaves,
            min_child_samples=self.min_child_samples,
            subsample=self.subsample,
            colsample_bytree=self.colsample_bytree,
            reg_alpha=self.reg_alpha,
            reg_lambda=self.reg_lambda,
            random_state=self.random_state,
            **kwargs
        )
        
        # Fit model
        if feature_names is not None:
            self.model.fit(X, y, feature_name=feature_names)
        else:
            self.model.fit(X, y)
        
        logger.info(f"LightGBM model fitted with {X.shape[1]} features")
        
        return self
    
    def predict(
        self, 
        X: np.ndarray
    ) -> np.ndarray:
        """
        Make predictions with the LightGBM model.
        
        Args:
            X (np.ndarray): Input features
            
        Returns:
            np.ndarray: Predictions
        """
        if self.model is None:
            raise ValueError("Model not fitted yet")
        
        # Scale features if needed
        if self.scaler is not None:
            X = self.scaler.transform(X)
        
        # Make predictions
        return self.model.predict(X)

class CatBoostModel(TabularModelBase):
    """
    CatBoost model for tabular data.
    """
    
    def __init__(
        self, 
        random_state: int = 42,
        n_estimators: int = 1000,
        learning_rate: float = 0.05,
        max_depth: int = 6,
        l2_leaf_reg: float = 3.0,
        subsample: float = 0.8,
        colsample_bylevel: float = 0.8
    ):
        """
        Initialize CatBoost model.
        
        Args:
            random_state (int): Random seed for reproducibility
            n_estimators (int): Number of boosting iterations
            learning_rate (float): Learning rate
            max_depth (int): Maximum tree depth
            l2_leaf_reg (float): L2 regularization coefficient
            subsample (float): Subsample ratio of the training data
            colsample_bylevel (float): Subsample ratio of columns for each level
        """
        super(CatBoostModel, self).__init__(model_type='catboost', random_state=random_state)
        
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.l2_leaf_reg = l2_leaf_reg
        self.subsample = subsample
        self.colsample_bylevel = colsample_bylevel
    
    def fit(
        self, 
        X: np.ndarray, 
        y: np.ndarray,
        feature_names: Optional[List[str]] = None,
        scale_features: bool = True,
        **kwargs
    ) -> 'CatBoostModel':
        """
        Fit the CatBoost model to the data.
        
        Args:
            X (np.ndarray): Input features
            y (np.ndarray): Target values
            feature_names (List[str], optional): Names of features
            scale_features (bool): Whether to scale features
            **kwargs: Additional arguments for CatBoost
            
        Returns:
            CatBoostModel: Self
        """
        from catboost import CatBoostRegressor
        
        # Store feature names
        self.feature_names = feature_names
        
        # Scale features if needed
        if scale_features:
            self.scaler = StandardScaler()
            X = self.scaler.fit_transform(X)
        
        # Create CatBoost model
        self.model = CatBoostRegressor(
            iterations=self.n_estimators,
            learning_rate=self.learning_rate,
            depth=self.max_depth,
            l2_leaf_reg=self.l2_leaf_reg,
            subsample=self.subsample,
            colsample_bylevel=self.colsample_bylevel,
            random_seed=self.random_state,
            verbose=False,
            **kwargs
        )
        
        # Fit model
        if feature_names is not None:
            self.model.fit(X, y, feature_names=feature_names)
        else:
            self.model.fit(X, y)
        
        logger.info(f"CatBoost model fitted with {X.shape[1]} features")
        
        return self
    
    def predict(
        self, 
        X: np.ndarray
    ) -> np.ndarray:
        """
        Make predictions with the CatBoost model.
        
        Args:
            X (np.ndarray): Input features
            
        Returns:
            np.ndarray: Predictions
        """
        if self.model is None:
            raise ValueError("Model not fitted yet")
        
        # Scale features if needed
        if self.scaler is not None:
            X = self.scaler.transform(X)
        
        # Make predictions
        return self.model.predict(X)

class XGBoostModel(TabularModelBase):
    """
    XGBoost model for tabular data.
    """
    
    def __init__(
        self, 
        random_state: int = 42,
        n_estimators: int = 1000,
        learning_rate: float = 0.05,
        max_depth: int = 6,
        min_child_weight: int = 1,
        subsample: float = 0.8,
        colsample_bytree: float = 0.8,
        reg_alpha: float = 0.0,
        reg_lambda: float = 1.0
    ):
        """
        Initialize XGBoost model.
        
        Args:
            random_state (int): Random seed for reproducibility
            n_estimators (int): Number of boosting iterations
            learning_rate (float): Learning rate
            max_depth (int): Maximum tree depth
            min_child_weight (int): Minimum sum of instance weight needed in a child
            subsample (float): Subsample ratio of the training data
            colsample_bytree (float): Subsample ratio of columns when constructing each tree
            reg_alpha (float): L1 regularization term
            reg_lambda (float): L2 regularization term
        """
        super(XGBoostModel, self).__init__(model_type='xgboost', random_state=random_state)
        
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.min_child_weight = min_child_weight
        self.subsample = subsample
        self.colsample_bytree = colsample_bytree
        self.reg_alpha = reg_alpha
        self.reg_lambda = reg_lambda
    
    def fit(
        self, 
        X: np.ndarray, 
        y: np.ndarray,
        feature_names: Optional[List[str]] = None,
        scale_features: bool = True,
        **kwargs
    ) -> 'XGBoostModel':
        """
        Fit the XGBoost model to the data.
        
        Args:
            X (np.ndarray): Input features
            y (np.ndarray): Target values
            feature_names (List[str], optional): Names of features
            scale_features (bool): Whether to scale features
            **kwargs: Additional arguments for XGBoost
            
        Returns:
            XGBoostModel: Self
        """
        import xgboost as xgb
        
        # Store feature names
        self.feature_names = feature_names
        
        # Scale features if needed
        if scale_features:
            self.scaler = StandardScaler()
            X = self.scaler.fit_transform(X)
        
        # Create XGBoost model
        self.model = xgb.XGBRegressor(
            n_estimators=self.n_estimators,
            learning_rate=self.learning_rate,
            max_depth=self.max_depth,
            min_child_weight=self.min_child_weight,
            subsample=self.subsample,
            colsample_bytree=self.colsample_bytree,
            reg_alpha=self.reg_alpha,
            reg_lambda=self.reg_lambda,
            random_state=self.random_state,
            **kwargs
        )
        
        # Fit model
        if feature_names is not None:
            self.model.fit(X, y, feature_names=feature_names)
        else:
            self.model.fit(X, y)
        
        logger.info(f"XGBoost model fitted with {X.shape[1]} features")
        
        return self
    
    def predict(
        self, 
        X: np.ndarray
    ) -> np.ndarray:
        """
        Make predictions with the XGBoost model.
        
        Args:
            X (np.ndarray): Input features
            
        Returns:
            np.ndarray: Predictions
        """
        if self.model is None:
            raise ValueError("Model not fitted yet")
        
        # Scale features if needed
        if self.scaler is not None:
            X = self.scaler.transform(X)
        
        # Make predictions
        return self.model.predict(X)

class MLPModel(TabularModelBase):
    """
    Multi-layer Perceptron model for tabular data.
    """
    
    def __init__(
        self, 
        random_state: int = 42,
        hidden_layer_sizes: Tuple[int, ...] = (100, 50),
        activation: str = 'relu',
        solver: str = 'adam',
        alpha: float = 0.0001,
        batch_size: Union[int, str] = 'auto',
        learning_rate: str = 'constant',
        learning_rate_init: float = 0.001,
        max_iter: int = 200,
        early_stopping: bool = True,
        validation_fraction: float = 0.1,
        n_iter_no_change: int = 10
    ):
        """
        Initialize MLP model.
        
        Args:
            random_state (int): Random seed for reproducibility
            hidden_layer_sizes (Tuple[int, ...]): Sizes of hidden layers
            activation (str): Activation function
            solver (str): Solver for weight optimization
            alpha (float): L2 penalty parameter
            batch_size (Union[int, str]): Size of minibatches
            learning_rate (str): Learning rate schedule
            learning_rate_init (float): Initial learning rate
            max_iter (int): Maximum number of iterations
            early_stopping (bool): Whether to use early stopping
            validation_fraction (float): Fraction of training data for validation
            n_iter_no_change (int): Number of iterations with no improvement to wait before early stopping
        """
        super(MLPModel, self).__init__(model_type='mlp', random_state=random_state)
        
        self.hidden_layer_sizes = hidden_layer_sizes
        self.activation = activation
        self.solver = solver
        self.alpha = alpha
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.learning_rate_init = learning_rate_init
        self.max_iter = max_iter
        self.early_stopping = early_stopping
        self.validation_fraction = validation_fraction
        self.n_iter_no_change = n_iter_no_change
    
    def fit(
        self, 
        X: np.ndarray, 
        y: np.ndarray,
        feature_names: Optional[List[str]] = None,
        scale_features: bool = True,
        **kwargs
    ) -> 'MLPModel':
        """
        Fit the MLP model to the data.
        
        Args:
            X (np.ndarray): Input features
            y (np.ndarray): Target values
            feature_names (List[str], optional): Names of features
            scale_features (bool): Whether to scale features
            **kwargs: Additional arguments for MLP
            
        Returns:
            MLPModel: Self
        """
        from sklearn.neural_network import MLPRegressor
        
        # Store feature names
        self.feature_names = feature_names
        
        # Scale features if needed
        if scale_features:
            self.scaler = StandardScaler()
            X = self.scaler.fit_transform(X)
        
        # Create MLP model
        self.model = MLPRegressor(
            hidden_layer_sizes=self.hidden_layer_sizes,
            activation=self.activation,
            solver=self.solver,
            alpha=self.alpha,
            batch_size=self.batch_size,
            learning_rate=self.learning_rate,
            learning_rate_init=self.learning_rate_init,
            max_iter=self.max_iter,
            random_state=self.random_state,
            early_stopping=self.early_stopping,
            validation_fraction=self.validation_fraction,
            n_iter_no_change=self.n_iter_no_change,
            **kwargs
        )
        
        # Fit model
        self.model.fit(X, y)
        
        logger.info(f"MLP model fitted with {X.shape[1]} features")
        
        return self
    
    def predict(
        self, 
        X: np.ndarray
    ) -> np.ndarray:
        """
        Make predictions with the MLP model.
        
        Args:
            X (np.ndarray): Input features
            
        Returns:
            np.ndarray: Predictions
        """
        if self.model is None:
            raise ValueError("Model not fitted yet")
        
        # Scale features if needed
        if self.scaler is not None:
            X = self.scaler.transform(X)
        
        # Make predictions
        return self.model.predict(X)

class TabNetModel(TabularModelBase):
    """
    TabNet model for tabular data.
    """
    
    def __init__(
        self, 
        random_state: int = 42,
        n_d: int = 8,
        n_a: int = 8,
        n_steps: int = 3,
        gamma: float = 1.3,
        n_independent: int = 2,
        n_shared: int = 2,
        lambda_sparse: float = 1e-3,
        optimizer_fn=None,
        optimizer_params=None,
        scheduler_fn=None,
        scheduler_params=None,
        mask_type: str = 'sparsemax',
        max_epochs: int = 200,
        batch_size: int = 1024,
        patience: int = 10
    ):
        """
        Initialize TabNet model.
        
        Args:
            random_state (int): Random seed for reproducibility
            n_d (int): Width of the decision prediction layer
            n_a (int): Width of the attention embedding for each mask
            n_steps (int): Number of steps in the architecture
            gamma (float): Coefficient for feature reusage in the masks
            n_independent (int): Number of independent GLU layers in each step
            n_shared (int): Number of shared GLU layers in each step
            lambda_sparse (float): Coefficient for the sparse regularization
            optimizer_fn: Optimizer function
            optimizer_params: Optimizer parameters
            scheduler_fn: Scheduler function
            scheduler_params: Scheduler parameters
            mask_type (str): Type of mask function
            max_epochs (int): Maximum number of epochs
            batch_size (int): Batch size
            patience (int): Patience for early stopping
        """
        super(TabNetModel, self).__init__(model_type='tabnet', random_state=random_state)
        
        self.n_d = n_d
        self.n_a = n_a
        self.n_steps = n_steps
        self.gamma = gamma
        self.n_independent = n_independent
        self.n_shared = n_shared
        self.lambda_sparse = lambda_sparse
        self.optimizer_fn = optimizer_fn or torch.optim.Adam
        self.optimizer_params = optimizer_params or {}
        self.scheduler_fn = scheduler_fn
        self.scheduler_params = scheduler_params or {}
        self.mask_type = mask_type
        self.max_epochs = max_epochs
        self.batch_size = batch_size
        self.patience = patience
    
    def fit(
        self, 
        X: np.ndarray, 
        y: np.ndarray,
        feature_names: Optional[List[str]] = None,
        scale_features: bool = True,
        **kwargs
    ) -> 'TabNetModel':
        """
        Fit the TabNet model to the data.
        
        Args:
            X (np.ndarray): Input features
            y (np.ndarray): Target values
            feature_names (List[str], optional): Names of features
            scale_features (bool): Whether to scale features
            **kwargs: Additional arguments for TabNet
            
        Returns:
            TabNetModel: Self
        """
        try:
            from pytorch_tabnet.tab_model import TabNetRegressor
        except ImportError:
            raise ImportError("pytorch-tabnet is not installed. Install it with 'pip install pytorch-tabnet'.")
        
        # Store feature names
        self.feature_names = feature_names
        
        # Scale features if needed
        if scale_features:
            self.scaler = StandardScaler()
            X = self.scaler.fit_transform(X)
        
        # Create TabNet model
        self.model = TabNetRegressor(
            n_d=self.n_d,
            n_a=self.n_a,
            n_steps=self.n_steps,
            gamma=self.gamma,
            n_independent=self.n_independent,
            n_shared=self.n_shared,
            lambda_sparse=self.lambda_sparse,
            optimizer_fn=self.optimizer_fn,
            optimizer_params=self.optimizer_params,
            scheduler_fn=self.scheduler_fn,
            scheduler_params=self.scheduler_params,
            mask_type=self.mask_type,
            seed=self.random_state,
            verbose=1
        )

       # Reshape y for single-target regression
        if y.ndim == 1:
            y = y.reshape(-1, 1) 
            
        # Fit model
        self.model.fit(
            X, y,
            max_epochs=self.max_epochs,
            batch_size=self.batch_size,
            patience=self.patience,
            **kwargs
        )
        
        logger.info(f"TabNet model fitted with {X.shape[1]} features")
        
        return self
    
    def predict(
        self, 
        X: np.ndarray
    ) -> np.ndarray:
        """
        Make predictions with the TabNet model.
        
        Args:
            X (np.ndarray): Input features
            
        Returns:
            np.ndarray: Predictions
        """
        if self.model is None:
            raise ValueError("Model not fitted yet")
        
        # Scale features if needed
        if self.scaler is not None:
            X = self.scaler.transform(X)
        
        # Make predictions
        return self.model.predict(X)

class TabGNNTabularModel:
    """
    Combined TabGNN and tabular model.
    """
    
    def __init__(
        self,
        gnn_model,
        tabular_model,
        use_original_features: bool = True
    ):
        """
        Initialize TabGNN tabular model.
        
        Args:
            gnn_model: GNN model for embedding extraction
            tabular_model: Tabular model for prediction
            use_original_features (bool): Whether to use original features along with embeddings
        """
        self.gnn_model = gnn_model
        self.tabular_model = tabular_model
        self.use_original_features = use_original_features
        
        logger.info(f"Initialized TabGNN tabular model with {tabular_model.model_type} base model")
    
    def extract_embeddings(
        self,
        x,
        edge_indices,
        batch=None
    ) -> np.ndarray:
        """
        Extract embeddings from GNN model.
        
        Args:
            x: Node features
            edge_indices: List of edge indices for each graph type
            batch: Batch assignment for nodes
            
        Returns:
            np.ndarray: Extracted embeddings
        """
        # Set model to evaluation mode
        self.gnn_model.eval()
        
        # Extract embeddings
        with torch.no_grad():
            embeddings = self.gnn_model.extract_embeddings(x, edge_indices, batch=batch)
            
            # Convert to numpy
            if isinstance(embeddings, torch.Tensor):
                embeddings = embeddings.cpu().numpy()
        
        return embeddings
    
    def combine_features(
        self,
        original_features,
        embeddings
    ) -> np.ndarray:
        """
        Combine original features with embeddings.
        
        Args:
            original_features: Original features
            embeddings: GNN embeddings
            
        Returns:
            np.ndarray: Combined features
        """
        # Convert to numpy if needed
        if isinstance(original_features, torch.Tensor):
            original_features = original_features.cpu().numpy()
        
        if isinstance(embeddings, torch.Tensor):
            embeddings = embeddings.cpu().numpy()
        
        # Combine features
        if self.use_original_features:
            combined_features = np.concatenate([original_features, embeddings], axis=1)
        else:
            combined_features = embeddings
        
        return combined_features
    
    def fit(
        self,
        x,
        edge_indices,
        y,
        batch=None,
        **kwargs
    ):
        """
        Fit the combined model.
        
        Args:
            x: Node features
            edge_indices: List of edge indices for each graph type
            y: Target values
            batch: Batch assignment for nodes
            **kwargs: Additional arguments for tabular model
            
        Returns:
            TabGNNTabularModel: Self
        """
        # Extract embeddings
        embeddings = self.extract_embeddings(x, edge_indices, batch)
        
        # Convert to numpy if needed
        if isinstance(x, torch.Tensor):
            x = x.cpu().numpy()
        
        if isinstance(y, torch.Tensor):
            y = y.cpu().numpy()
        
        # Combine features
        combined_features = self.combine_features(x, embeddings)
        
        # Fit tabular model
        self.tabular_model.fit(combined_features, y, **kwargs)
        
        logger.info(f"TabGNN tabular model fitted with {combined_features.shape[1]} features")
        
        return self
    
    def predict(
        self,
        x,
        edge_indices,
        batch=None
    ) -> np.ndarray:
        """
        Make predictions with the combined model.
        
        Args:
            x: Node features
            edge_indices: List of edge indices for each graph type
            batch: Batch assignment for nodes
            
        Returns:
            np.ndarray: Predictions
        """
        # Extract embeddings
        embeddings = self.extract_embeddings(x, edge_indices, batch)
        
        # Convert to numpy if needed
        if isinstance(x, torch.Tensor):
            x = x.cpu().numpy()
        
        # Combine features
        combined_features = self.combine_features(x, embeddings)
        
        # Make predictions
        return self.tabular_model.predict(combined_features)
    
    def evaluate(
        self,
        x,
        edge_indices,
        y,
        batch=None
    ) -> Dict[str, float]:
        """
        Evaluate the combined model.
        
        Args:
            x: Node features
            edge_indices: List of edge indices for each graph type
            y: Target values
            batch: Batch assignment for nodes
            
        Returns:
            Dict[str, float]: Dictionary of evaluation metrics
        """
        # Convert to numpy if needed
        if isinstance(y, torch.Tensor):
            y = y.cpu().numpy()
        
        # Make predictions
        y_pred = self.predict(x, edge_indices, batch)
        
        # Calculate metrics
        mse = mean_squared_error(y, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y, y_pred)
        r2 = r2_score(y, y_pred)
        
        # Return metrics
        metrics = {
            'mse': mse,
            'rmse': rmse,
            'mae': mae,
            'r2': r2
        }
        
        logger.info(f"Evaluation metrics: {metrics}")
        
        return metrics
    
    def save(
        self,
        gnn_path: str,
        tabular_path: str
    ) -> Tuple[str, str]:
        """
        Save the combined model.
        
        Args:
            gnn_path (str): Path to save the GNN model
            tabular_path (str): Path to save the tabular model
            
        Returns:
            Tuple[str, str]: Paths to the saved models
        """
        # Save GNN model
        self.gnn_model.save_checkpoint(
            checkpoint_dir=os.path.dirname(gnn_path),
            filename=os.path.basename(gnn_path)
        )
        
        # Save tabular model
        self.tabular_model.save(tabular_path)
        
        logger.info(f"TabGNN tabular model saved to {gnn_path} and {tabular_path}")
        
        return gnn_path, tabular_path
    
    @classmethod
    def load(
        cls,
        gnn_path: str,
        tabular_path: str,
        gnn_model_class,
        tabular_model_class,
        use_original_features: bool = True
    ) -> 'TabGNNTabularModel':
        """
        Load a combined model.
        
        Args:
            gnn_path (str): Path to the saved GNN model
            tabular_path (str): Path to the saved tabular model
            gnn_model_class: Class of the GNN model
            tabular_model_class: Class of the tabular model
            use_original_features (bool): Whether to use original features along with embeddings
            
        Returns:
            TabGNNTabularModel: Loaded model
        """
        # Load GNN model
        gnn_model = gnn_model_class.load_checkpoint(gnn_path)
        
        # Load tabular model
        tabular_model = tabular_model_class.load(tabular_path)
        
        # Create instance
        instance = cls(
            gnn_model=gnn_model,
            tabular_model=tabular_model,
            use_original_features=use_original_features
        )
        
        logger.info(f"TabGNN tabular model loaded from {gnn_path} and {tabular_path}")
        
        return instance
