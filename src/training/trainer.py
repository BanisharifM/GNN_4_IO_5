"""
Training pipeline for GAT model with automatic memory management
Includes RMSE tracking for AIIO comparison
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingWarmRestarts
import numpy as np
from typing import Dict, Optional, Union, Tuple, List
from tqdm import tqdm
import logging
import time
from pathlib import Path
import json
import wandb
from collections import defaultdict

logger = logging.getLogger(__name__)


class IOPerformanceTrainer:
    """
    Trainer for I/O performance GAT model with AIIO comparison
    """
    
    def __init__(
        self,
        model: nn.Module,
        device: torch.device = None,
        learning_rate: float = 0.001,
        weight_decay: float = 1e-5,
        optimizer_type: str = 'adamw',
        scheduler_type: str = 'plateau',
        gradient_clip: float = 1.0,
        mixed_precision: bool = False,
        accumulation_steps: int = 1,
        dtype: torch.dtype = torch.float64,
        save_dir: str = './checkpoints',
        use_wandb: bool = False,
        project_name: str = 'io_performance_gnn'
    ):
        """
        Args:
            model: GAT model
            device: Training device
            learning_rate: Initial learning rate
            weight_decay: L2 regularization
            optimizer_type: 'adam', 'adamw', 'sgd'
            scheduler_type: 'plateau', 'cosine', 'step'
            gradient_clip: Gradient clipping value
            mixed_precision: Use mixed precision training (for memory efficiency)
            accumulation_steps: Gradient accumulation steps
            dtype: Data type (float64 for precision)
            save_dir: Directory for checkpoints
            use_wandb: Use Weights & Biases for logging
            project_name: W&B project name
        """
        self.model = model
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.gradient_clip = gradient_clip
        self.mixed_precision = mixed_precision and dtype != torch.float64  # No mixed precision with float64
        self.accumulation_steps = accumulation_steps
        self.dtype = dtype
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        
        # Move model to device
        self.model = self.model.to(self.device)
        if dtype == torch.float64:
            self.model = self.model.double()
        
        # Setup optimizer
        self.optimizer = self._create_optimizer(optimizer_type)
        
        # Setup scheduler
        self.scheduler = self._create_scheduler(scheduler_type)
        
        # Mixed precision scaler
        self.scaler = torch.cuda.amp.GradScaler() if self.mixed_precision else None
        
        # Training history
        self.history = defaultdict(list)
        self.best_val_rmse = float('inf')
        self.best_epoch = 0
        
        # W&B logging
        self.use_wandb = use_wandb
        if use_wandb:
            wandb.init(project=project_name, config={
                'learning_rate': learning_rate,
                'weight_decay': weight_decay,
                'optimizer': optimizer_type,
                'scheduler': scheduler_type,
                'model_type': model.__class__.__name__
            })
            wandb.watch(model)
        
        # AIIO baseline RMSE for comparison
        self.aiio_baselines = {
            'XGBoost': 0.5634,
            'LightGBM': 0.2632,
            'CatBoost': 0.2686,
            'MLP': 0.5416,
            'TabNet': 0.3078,
            'Best': 0.1860  # Closest method from AIIO
        }
        
    def _create_optimizer(self, optimizer_type: str) -> optim.Optimizer:
        """Create optimizer"""
        params = self.model.parameters()
        
        if optimizer_type == 'adam':
            return optim.Adam(params, lr=self.learning_rate, weight_decay=self.weight_decay)
        elif optimizer_type == 'adamw':
            return optim.AdamW(params, lr=self.learning_rate, weight_decay=self.weight_decay)
        elif optimizer_type == 'sgd':
            return optim.SGD(params, lr=self.learning_rate, weight_decay=self.weight_decay, momentum=0.9)
        else:
            raise ValueError(f"Unknown optimizer type: {optimizer_type}")
    
    def _create_scheduler(self, scheduler_type: str):
        """Create learning rate scheduler"""
        if scheduler_type == 'plateau':
            return ReduceLROnPlateau(
                self.optimizer, mode='min', factor=0.5, patience=10, verbose=True
            )
        elif scheduler_type == 'cosine':
            return CosineAnnealingWarmRestarts(
                self.optimizer, T_0=10, T_mult=2, eta_min=1e-6
            )
        elif scheduler_type == 'step':
            return optim.lr_scheduler.StepLR(
                self.optimizer, step_size=30, gamma=0.5
            )
        else:
            return None
    
    def train_epoch_full_batch(
        self,
        data,
        train_mask: torch.Tensor
    ) -> Dict[str, float]:
        """
        Train one epoch with full-batch training
        """
        self.model.train()
        
        # Move data to device if not already
        if data.x.device != self.device:
            data = data.to(self.device)
        
        # Zero gradients
        self.optimizer.zero_grad()
        
        # Forward pass
        if self.mixed_precision:
            with torch.cuda.amp.autocast():
                out = self.model(data.x, data.edge_index, data.edge_attr)
        else:
            out = self.model(data.x, data.edge_index, data.edge_attr)
        
        # Calculate loss (only on training nodes)
        loss = self.calculate_rmse_loss(out[train_mask], data.y[train_mask])
        
        # Backward pass
        if self.mixed_precision:
            self.scaler.scale(loss).backward()
            if self.gradient_clip:
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.gradient_clip)
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            loss.backward()
            if self.gradient_clip:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.gradient_clip)
            self.optimizer.step()
        
        # Calculate metrics
        with torch.no_grad():
            rmse = self.calculate_rmse(out[train_mask], data.y[train_mask])
            mae = self.calculate_mae(out[train_mask], data.y[train_mask])
        
        return {
            'loss': loss.item(),
            'rmse': rmse.item(),
            'mae': mae.item()
        }
    
    def train_epoch_mini_batch(
        self,
        train_loader
    ) -> Dict[str, float]:
        """
        Train one epoch with mini-batch training
        """
        self.model.train()
        
        total_loss = 0
        total_rmse = 0
        total_mae = 0
        total_samples = 0
        
        for batch_idx, batch in enumerate(tqdm(train_loader, desc="Training")):
            batch = batch.to(self.device)
            
            # Forward pass
            if self.mixed_precision:
                with torch.cuda.amp.autocast():
                    out = self.model(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
            else:
                out = self.model(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
            
            # Calculate loss
            loss = self.calculate_rmse_loss(out, batch.y)
            
            # Normalize loss for gradient accumulation
            loss = loss / self.accumulation_steps
            
            # Backward pass
            if self.mixed_precision:
                self.scaler.scale(loss).backward()
            else:
                loss.backward()
            
            # Update weights
            if (batch_idx + 1) % self.accumulation_steps == 0:
                if self.mixed_precision:
                    if self.gradient_clip:
                        self.scaler.unscale_(self.optimizer)
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.gradient_clip)
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    if self.gradient_clip:
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.gradient_clip)
                    self.optimizer.step()
                
                self.optimizer.zero_grad()
            
            # Calculate metrics
            with torch.no_grad():
                batch_size = batch.y.size(0)
                total_loss += loss.item() * batch_size * self.accumulation_steps
                total_rmse += self.calculate_rmse(out, batch.y).item() * batch_size
                total_mae += self.calculate_mae(out, batch.y).item() * batch_size
                total_samples += batch_size
        
        return {
            'loss': total_loss / total_samples,
            'rmse': total_rmse / total_samples,
            'mae': total_mae / total_samples
        }
    
    @torch.no_grad()
    def evaluate(
        self,
        data_or_loader,
        mask: Optional[torch.Tensor] = None
    ) -> Dict[str, float]:
        """
        Evaluate model on validation/test set
        """
        self.model.eval()
        
        # Full-batch evaluation
        if hasattr(data_or_loader, 'x'):
            data = data_or_loader
            if data.x.device != self.device:
                data = data.to(self.device)
            
            out = self.model(data.x, data.edge_index, data.edge_attr)
            
            if mask is not None:
                out = out[mask]
                y = data.y[mask]
            else:
                y = data.y
            
            rmse = self.calculate_rmse(out, y)
            mae = self.calculate_mae(out, y)
            r2 = self.calculate_r2(out, y)
            
            return {
                'rmse': rmse.item(),
                'mae': mae.item(),
                'r2': r2.item()
            }
        
        # Mini-batch evaluation
        else:
            total_rmse = 0
            total_mae = 0
            total_r2 = 0
            total_samples = 0
            
            for batch in tqdm(data_or_loader, desc="Evaluating"):
                batch = batch.to(self.device)
                
                out = self.model(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
                
                batch_size = batch.y.size(0)
                total_rmse += self.calculate_rmse(out, batch.y).item() * batch_size
                total_mae += self.calculate_mae(out, batch.y).item() * batch_size
                total_r2 += self.calculate_r2(out, batch.y).item() * batch_size
                total_samples += batch_size
            
            return {
                'rmse': total_rmse / total_samples,
                'mae': total_mae / total_samples,
                'r2': total_r2 / total_samples
            }
    
    def train(
        self,
        train_data,
        val_data,
        test_data=None,
        epochs: int = 200,
        early_stopping_patience: int = 30,
        verbose: bool = True
    ) -> Dict[str, List[float]]:
        """
        Main training loop with early stopping
        """
        logger.info(f"Starting training for {epochs} epochs on {self.device}")
        logger.info(f"AIIO baselines - Best: {self.aiio_baselines['Best']}, "
                   f"LightGBM: {self.aiio_baselines['LightGBM']}")
        
        # Determine if we're doing full-batch or mini-batch
        is_full_batch = hasattr(train_data, 'x')
        
        # Early stopping counter
        patience_counter = 0
        
        for epoch in range(epochs):
            start_time = time.time()
            
            # Training
            if is_full_batch:
                train_metrics = self.train_epoch_full_batch(
                    train_data, train_data.train_mask
                )
            else:
                train_metrics = self.train_epoch_mini_batch(train_data['train'])
            
            # Validation
            if is_full_batch:
                val_metrics = self.evaluate(val_data, val_data.val_mask)
            else:
                val_metrics = self.evaluate(val_data['val'])
            
            # Update learning rate
            if self.scheduler:
                if isinstance(self.scheduler, ReduceLROnPlateau):
                    self.scheduler.step(val_metrics['rmse'])
                else:
                    self.scheduler.step()
            
            # Track history
            for key, value in train_metrics.items():
                self.history[f'train_{key}'].append(value)
            for key, value in val_metrics.items():
                self.history[f'val_{key}'].append(value)
            
            # Check for best model
            if val_metrics['rmse'] < self.best_val_rmse:
                self.best_val_rmse = val_metrics['rmse']
                self.best_epoch = epoch
                self.save_checkpoint(epoch, val_metrics['rmse'])
                patience_counter = 0
                
                # Compare with AIIO
                improvement_over_aiio = (self.aiio_baselines['Best'] - val_metrics['rmse']) / self.aiio_baselines['Best'] * 100
            else:
                patience_counter += 1
            
            # Logging
            if verbose and epoch % 10 == 0:
                elapsed = time.time() - start_time
                logger.info(
                    f"Epoch {epoch:3d} | "
                    f"Train RMSE: {train_metrics['rmse']:.4f} | "
                    f"Val RMSE: {val_metrics['rmse']:.4f} | "
                    f"Best: {self.best_val_rmse:.4f} | "
                    f"vs AIIO: {improvement_over_aiio:+.1f}% | "
                    f"Time: {elapsed:.1f}s"
                )
            
            # W&B logging
            if self.use_wandb:
                wandb.log({
                    'epoch': epoch,
                    'train/rmse': train_metrics['rmse'],
                    'train/mae': train_metrics['mae'],
                    'val/rmse': val_metrics['rmse'],
                    'val/mae': val_metrics['mae'],
                    'val/r2': val_metrics['r2'],
                    'lr': self.optimizer.param_groups[0]['lr']
                })
            
            # Early stopping
            if patience_counter >= early_stopping_patience:
                logger.info(f"Early stopping at epoch {epoch}")
                break
        
        # Load best model
        self.load_checkpoint()
        
        # Final test evaluation
        if test_data is not None:
            if is_full_batch:
                test_metrics = self.evaluate(test_data, test_data.test_mask)
            else:
                test_metrics = self.evaluate(test_data['test'])
            
            logger.info(f"\nFinal Test Results:")
            logger.info(f"  RMSE: {test_metrics['rmse']:.4f}")
            logger.info(f"  MAE: {test_metrics['mae']:.4f}")
            logger.info(f"  R²: {test_metrics['r2']:.4f}")
            
            # Compare with AIIO methods
            logger.info(f"\nComparison with AIIO:")
            for method, rmse in self.aiio_baselines.items():
                improvement = (rmse - test_metrics['rmse']) / rmse * 100
                logger.info(f"  vs {method}: {improvement:+.1f}% "
                          f"(AIIO: {rmse:.4f}, Ours: {test_metrics['rmse']:.4f})")
        
        return self.history
    
    def calculate_rmse_loss(
        self,
        pred: torch.Tensor,
        target: torch.Tensor
    ) -> torch.Tensor:
        """Calculate RMSE loss for training"""
        return torch.sqrt(torch.mean((pred.squeeze() - target.squeeze()) ** 2))
    
    def calculate_rmse(
        self,
        pred: torch.Tensor,
        target: torch.Tensor
    ) -> torch.Tensor:
        """Calculate RMSE metric"""
        return torch.sqrt(torch.mean((pred.squeeze() - target.squeeze()) ** 2))
    
    def calculate_mae(
        self,
        pred: torch.Tensor,
        target: torch.Tensor
    ) -> torch.Tensor:
        """Calculate MAE metric"""
        return torch.mean(torch.abs(pred.squeeze() - target.squeeze()))
    
    def calculate_r2(
        self,
        pred: torch.Tensor,
        target: torch.Tensor
    ) -> torch.Tensor:
        """Calculate R² score"""
        pred = pred.squeeze()
        target = target.squeeze()
        
        ss_res = torch.sum((target - pred) ** 2)
        ss_tot = torch.sum((target - target.mean()) ** 2)
        r2 = 1 - (ss_res / ss_tot)
        
        return r2
    
    def save_checkpoint(
        self,
        epoch: int,
        val_rmse: float
    ):
        """Save model checkpoint"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'val_rmse': val_rmse,
            'history': dict(self.history)
        }
        
        path = self.save_dir / f'best_model.pt'
        torch.save(checkpoint, path)
        logger.info(f"Saved checkpoint: {path}")
    
    def load_checkpoint(self):
        """Load model checkpoint"""
        path = self.save_dir / f'best_model.pt'
        if path.exists():
            checkpoint = torch.load(path, map_location=self.device)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            logger.info(f"Loaded checkpoint from epoch {checkpoint['epoch']} "
                       f"with val RMSE {checkpoint['val_rmse']:.4f}")