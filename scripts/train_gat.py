#!/usr/bin/env python
"""
Main training script for GAT model on I/O performance prediction
Compares with AIIO baselines and provides comprehensive evaluation
"""

import os
import sys
import argparse
import yaml
import torch
import numpy as np
import pandas as pd
from pathlib import Path
import logging
from datetime import datetime
import json
import warnings
warnings.filterwarnings('ignore')

# Add src to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data.dataset import IOPerformanceGraphDataset
from src.data.loaders import AdaptiveDataLoader
from src.models.gat import create_gat_model
from src.training.trainer import IOPerformanceTrainer
from src.training.metrics import PerformanceMetrics, MetricTracker
from src.interpretability.gradient_methods import BottleneckIdentifier

# Setup logging
def setup_logging(save_dir: Path, debug: bool = False):
    """Setup logging configuration"""
    log_level = logging.DEBUG if debug else logging.INFO
    
    # Create logs directory
    log_dir = save_dir / 'logs'
    log_dir.mkdir(parents=True, exist_ok=True)
    
    # Log file with timestamp
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_file = log_dir / f'training_{timestamp}.log'
    
    # Configure logging
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    
    return logging.getLogger(__name__)


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file"""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def set_seed(seed: int):
    """Set random seeds for reproducibility"""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def main(args):
    """Main training pipeline"""
    
    # Load configuration
    if args.config:
        config = load_config(args.config)
        
        # Override config with command-line arguments if provided
        
        # Always use command-line save-dir if provided
        if args.save_dir:
            if 'experiment' not in config:
                config['experiment'] = {}
            config['experiment']['save_dir'] = args.save_dir
        
        # Override other args if provided
        if args.epochs != 200:  # 200 is the default
            if 'training' not in config:
                config['training'] = {}
            config['training']['epochs'] = args.epochs
        
        if args.lr != 0.001:  # 0.001 is the default
            if 'training' not in config:
                config['training'] = {}
            config['training']['learning_rate'] = args.lr
        
        if args.seed != 42:  # 42 is the default
            if 'experiment' not in config:
                config['experiment'] = {}
            config['experiment']['seed'] = args.seed
        
        if args.run_name:
            if 'experiment' not in config:
                config['experiment'] = {}
            config['experiment']['run_name'] = args.run_name
        
        if args.use_wandb:
            if 'experiment' not in config:
                config['experiment'] = {}
            config['experiment']['use_wandb'] = args.use_wandb
            
        # Handle data paths from command line
        if 'data' not in config:
            config['data'] = {}
        config['data']['similarity_pt_path'] = args.similarity_pt
        config['data']['similarity_npz_path'] = args.similarity_npz
        config['data']['features_csv_path'] = args.features_csv
        
    else:
        # Default configuration when no config file is provided
        config = {
            'data': {
                'similarity_pt_path': args.similarity_pt,
                'similarity_npz_path': args.similarity_npz,
                'features_csv_path': args.features_csv,
                'root': './data/processed',
                'train_ratio': 0.6,
                'val_ratio': 0.2,
                'test_ratio': 0.2,
                'stratify': True
            },
            'model': {
                'type': 'standard',
                'hidden_channels': 256,
                'num_layers': 3,
                'heads': [8, 8, 1],
                'dropout': 0.2,
                'residual': True,
                'layer_norm': True,
                'feature_augmentation': True
            },
            'training': {
                'epochs': args.epochs,
                'learning_rate': args.lr,
                'weight_decay': 1e-5,
                'optimizer': 'adamw',
                'scheduler': 'plateau',
                'gradient_clip': 1.0,
                'early_stopping_patience': 30,
                'batch_size': 2048,
                'num_neighbors': [25, 10],
                'accumulation_steps': 1
            },
            'experiment': {
                'seed': args.seed,
                'save_dir': args.save_dir,
                'use_wandb': args.use_wandb,
                'project_name': 'io_performance_gnn',
                'run_name': args.run_name or f'gat_{datetime.now().strftime("%Y%m%d_%H%M%S")}'
            }
        }

    
    # Convert dtype string from config to torch dtype
    dtype_map = {
        'float32': torch.float32,
        'float64': torch.float64
    }

    # Get dtype from config, with fallback to command line arg or default
    if 'dtype' in config['model']:
        dtype = dtype_map[config['model']['dtype']]
    elif args.dtype:  # if you still want command line override
        dtype = dtype_map[args.dtype]
        config['model']['dtype'] = args.dtype
        config['training']['dtype'] = args.dtype
    else:
        dtype = torch.float64  # default
        config['model']['dtype'] = 'float64'
        config['training']['dtype'] = 'float64'

    # Setup directories
    save_dir = Path(config['experiment']['save_dir'])
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # Setup logging
    logger = setup_logging(save_dir, args.debug)
    logger.info("="*80)
    logger.info("GAT TRAINING FOR I/O PERFORMANCE PREDICTION")
    logger.info("="*80)
    
    # Set seed
    set_seed(config['experiment']['seed'])
    logger.info(f"Random seed: {config['experiment']['seed']}")
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() and not args.cpu else 'cpu')
    logger.info(f"Using device: {device}")
    
    # Save configuration
    config_save_path = save_dir / 'config.yaml'
    with open(config_save_path, 'w') as f:
        yaml.dump(config, f)
    logger.info(f"Configuration saved to {config_save_path}")
    
    # ============= Load Dataset =============
    logger.info("\n" + "="*40)
    logger.info("LOADING DATASET")
    logger.info("="*40)
    
    dataset = IOPerformanceGraphDataset(
        root=config['data']['root'],
        similarity_pt_path=config['data']['similarity_pt_path'],
        similarity_npz_path=config['data']['similarity_npz_path'],
        features_csv_path=config['data']['features_csv_path'],
        use_edge_weights=True,
        dtype=dtype,
        lazy_load=True
    )
    
    data = dataset[0]  # Get the graph
    logger.info(f"Dataset loaded: {data.num_nodes:,} nodes, {data.num_edges:,} edges")
    logger.info(f"Features: {data.x.shape}, Targets: {data.y.shape}")
    
    # Get train/val/test splits
    splits = dataset.get_splits(
        train_ratio=config['data']['train_ratio'],
        val_ratio=config['data']['val_ratio'],
        test_ratio=config['data']['test_ratio'],
        seed=config['experiment']['seed'],
        stratify=config['data']['stratify']
    )
    
    # Add masks to data
    data.train_mask = splits['train_mask']
    data.val_mask = splits['val_mask']
    data.test_mask = splits['test_mask']
    
    # ============= Setup Data Loader =============
    logger.info("\n" + "="*40)
    logger.info("SETTING UP DATA LOADER")
    logger.info("="*40)
    
    loader = AdaptiveDataLoader(
        dataset=data,
        device=device,
        force_mode=args.force_mode,  # None, 'full', or 'mini'
        batch_size=config['training']['batch_size'],
        num_neighbors=config['training']['num_neighbors'],
        num_workers=4,
        pin_memory=True
    )
    
    # Get appropriate loader
    data_loader = loader.get_loader(splits)
    
    # Check memory
    memory_stats = loader.get_memory_stats()
    logger.info(f"Memory stats: {memory_stats}")
    
    # ============= Create Model =============
    logger.info("\n" + "="*40)
    logger.info("CREATING MODEL")
    logger.info("="*40)
    
    model = create_gat_model(
        num_features=data.num_features,
        model_type=config['model']['type'],
        hidden_channels=config['model']['hidden_channels'],
        num_layers=config['model']['num_layers'],
        heads=config['model']['heads'],
        dropout=config['model']['dropout'],
        residual=config['model']['residual'],
        layer_norm=config['model']['layer_norm'],
        feature_augmentation=config['model']['feature_augmentation'],
        dtype=dtype
    )
    
    # Model summary
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Model: {model.__class__.__name__}")
    logger.info(f"Total parameters: {total_params:,}")
    logger.info(f"Trainable parameters: {trainable_params:,}")
    
    # ============= Setup Trainer =============
    logger.info("\n" + "="*40)
    logger.info("SETTING UP TRAINER")
    logger.info("="*40)
    
    trainer = IOPerformanceTrainer(
        model=model,
        device=device,
        learning_rate=config['training']['learning_rate'],
        weight_decay=config['training']['weight_decay'],
        optimizer_type=config['training']['optimizer'],
        scheduler_type=config['training']['scheduler'],
        gradient_clip=config['training']['gradient_clip'],
        accumulation_steps=config['training']['accumulation_steps'],
        dtype=dtype,
        save_dir=save_dir / 'checkpoints',
        use_wandb=config['experiment']['use_wandb'],
        project_name=config['experiment']['project_name']
    )
    
    # ============= Training =============
    logger.info("\n" + "="*40)
    logger.info("STARTING TRAINING")
    logger.info("="*40)
    logger.info(f"Training for {config['training']['epochs']} epochs")
    logger.info(f"Early stopping patience: {config['training']['early_stopping_patience']}")
    logger.info(f"AIIO Best Baseline: {trainer.aiio_baselines['Best']}")
    logger.info(f"AIIO LightGBM: {trainer.aiio_baselines['LightGBM']}")
    
    # Train model
    history = trainer.train(
        train_data=data_loader if isinstance(data_loader, dict) else data_loader,
        val_data=data_loader if isinstance(data_loader, dict) else data_loader,
        test_data=data_loader if isinstance(data_loader, dict) else data_loader,
        epochs=config['training']['epochs'],
        early_stopping_patience=config['training']['early_stopping_patience'],
        verbose=True
    )
    
    # ============= Final Evaluation =============
    logger.info("\n" + "="*40)
    logger.info("FINAL EVALUATION")
    logger.info("="*40)
    
    # Load best model
    trainer.load_checkpoint()
    
    # Evaluate on test set
    if isinstance(data_loader, dict):
        test_metrics = trainer.evaluate(data_loader['test'])
    else:
        test_metrics = trainer.evaluate(data_loader, data_loader.test_mask)
    
    # Compare with AIIO
    metrics_calculator = PerformanceMetrics()
    comparison = metrics_calculator.compare_with_aiio(test_metrics)
    metrics_calculator.print_comparison(test_metrics)
    
    # Save results
    results = {
        'test_metrics': test_metrics,
        'comparison_with_aiio': comparison,
        'best_val_rmse': trainer.best_val_rmse,
        'best_epoch': trainer.best_epoch,
        'training_history': history
    }
    
    results_path = save_dir / 'results.json'
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    logger.info(f"Results saved to {results_path}")
    
    # ============= Interpretability Analysis =============
    if args.interpret:
        logger.info("\n" + "="*40)
        logger.info("INTERPRETABILITY ANALYSIS")
        logger.info("="*40)
        
        # Get feature names
        feature_names = [
            'nprocs', 'POSIX_OPENS', 'LUSTRE_STRIPE_SIZE', 'LUSTRE_STRIPE_WIDTH',
            'POSIX_FILENOS', 'POSIX_MEM_ALIGNMENT', 'POSIX_FILE_ALIGNMENT',
            'POSIX_READS', 'POSIX_WRITES', 'POSIX_SEEKS', 'POSIX_STATS',
            'POSIX_BYTES_READ', 'POSIX_BYTES_WRITTEN', 'POSIX_CONSEC_READS',
            'POSIX_CONSEC_WRITES', 'POSIX_SEQ_READS', 'POSIX_SEQ_WRITES',
            'POSIX_RW_SWITCHES', 'POSIX_MEM_NOT_ALIGNED', 'POSIX_FILE_NOT_ALIGNED',
            'POSIX_SIZE_READ_0_100', 'POSIX_SIZE_READ_100_1K', 'POSIX_SIZE_READ_1K_10K',
            'POSIX_SIZE_READ_100K_1M', 'POSIX_SIZE_WRITE_0_100', 'POSIX_SIZE_WRITE_100_1K',
            'POSIX_SIZE_WRITE_1K_10K', 'POSIX_SIZE_WRITE_10K_100K', 'POSIX_SIZE_WRITE_100K_1M',
            'POSIX_STRIDE1_STRIDE', 'POSIX_STRIDE2_STRIDE', 'POSIX_STRIDE3_STRIDE',
            'POSIX_STRIDE4_STRIDE', 'POSIX_STRIDE1_COUNT', 'POSIX_STRIDE2_COUNT',
            'POSIX_STRIDE3_COUNT', 'POSIX_STRIDE4_COUNT', 'POSIX_ACCESS1_ACCESS',
            'POSIX_ACCESS2_ACCESS', 'POSIX_ACCESS3_ACCESS', 'POSIX_ACCESS4_ACCESS',
            'POSIX_ACCESS1_COUNT', 'POSIX_ACCESS2_COUNT', 'POSIX_ACCESS3_COUNT',
            'POSIX_ACCESS4_COUNT'
        ]
        
        # Initialize bottleneck identifier
        bottleneck_identifier = BottleneckIdentifier(
            model=model,
            feature_names=feature_names,
            device=device
        )
        
        # Analyze sample nodes with poor performance
        poor_performance_mask = data.y < data.y.median()
        poor_nodes = torch.where(poor_performance_mask & data.test_mask)[0][:5]
        
        logger.info(f"Analyzing {len(poor_nodes)} poor-performing nodes")
        
        bottleneck_results = {}
        for node_idx in poor_nodes:
            node_idx = node_idx.item()
            logger.info(f"\nAnalyzing node {node_idx}")
            
            # Comprehensive analysis
            analysis = bottleneck_identifier.comprehensive_analysis(data, node_idx)
            
            # Find consensus bottlenecks
            consensus = bottleneck_identifier.consensus_bottlenecks(analysis, min_methods=2)
            
            bottleneck_results[node_idx] = {
                'analysis': analysis,
                'consensus': consensus,
                'performance': data.y[node_idx].item()
            }
            
            # Log top bottlenecks
            logger.info(f"Top bottlenecks for node {node_idx}:")
            for feature, score in consensus[:5]:
                logger.info(f"  - {feature}: {score:.4f}")
        
        # Save bottleneck analysis
        bottleneck_path = save_dir / 'bottleneck_analysis.json'
        with open(bottleneck_path, 'w') as f:
            json.dump(bottleneck_results, f, indent=2, default=str)
        logger.info(f"Bottleneck analysis saved to {bottleneck_path}")
    
    # ============= Plot Training Curves =============
    if args.plot:
        logger.info("\n" + "="*40)
        logger.info("PLOTTING RESULTS")
        logger.info("="*40)
        
        metric_tracker = MetricTracker()
        for epoch in range(len(history.get('train_rmse', []))):
            train_metrics = {k.replace('train_', ''): history[f'train_{k}'][epoch] 
                           for k in ['rmse', 'mae', 'loss'] if f'train_{k}' in history}
            val_metrics = {k.replace('val_', ''): history[f'val_{k}'][epoch] 
                          for k in ['rmse', 'mae', 'r2'] if f'val_{k}' in history}
            metric_tracker.update(epoch, train_metrics, val_metrics)
        
        plot_path = save_dir / 'training_curves.png'
        metric_tracker.plot_metrics(save_path=str(plot_path), show_aiio=True)
        logger.info(f"Training curves saved to {plot_path}")
    
    logger.info("\n" + "="*80)
    logger.info("TRAINING COMPLETE!")
    logger.info("="*80)
    logger.info(f"Best validation RMSE: {trainer.best_val_rmse:.4f}")
    logger.info(f"Test RMSE: {test_metrics['rmse']:.4f}")
    
    # Final comparison with AIIO
    if test_metrics['rmse'] < trainer.aiio_baselines['Best']:
        improvement = (trainer.aiio_baselines['Best'] - test_metrics['rmse']) / trainer.aiio_baselines['Best'] * 100
        logger.info(f"🎉 BEAT AIIO's BEST METHOD by {improvement:.1f}%!")
    else:
        gap = (test_metrics['rmse'] - trainer.aiio_baselines['Best']) / trainer.aiio_baselines['Best'] * 100
        logger.info(f"Within {gap:.1f}% of AIIO's best method")
    
    return test_metrics['rmse']


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train GAT for I/O Performance Prediction')
    
    # Data arguments
    parser.add_argument('--similarity-pt', type=str, required=True,
                        help='Path to similarity graph PyTorch file')
    parser.add_argument('--similarity-npz', type=str, required=True,
                        help='Path to similarity graph NumPy file')
    parser.add_argument('--features-csv', type=str, required=True,
                        help='Path to normalized features CSV')
    
    # Training arguments
    parser.add_argument('--epochs', type=int, default=200,
                        help='Number of training epochs')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='Learning rate')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    
    # Model arguments
    parser.add_argument('--force-mode', type=str, choices=['full', 'mini'],
                        help='Force full-batch or mini-batch training')
    
    # Experiment arguments
    parser.add_argument('--config', type=str,
                        help='Path to configuration file')
    parser.add_argument('--save-dir', type=str, default='./experiments/gat_exp',
                        help='Directory to save results')
    parser.add_argument('--run-name', type=str,
                        help='Name for this run')
    parser.add_argument('--use-wandb', action='store_true',
                        help='Use Weights & Biases for logging')
    
    # Additional options
    parser.add_argument('--interpret', action='store_true',
                        help='Run interpretability analysis')
    parser.add_argument('--plot', action='store_true',
                        help='Plot training curves')
    parser.add_argument('--cpu', action='store_true',
                        help='Force CPU usage')
    parser.add_argument('--debug', action='store_true',
                        help='Enable debug logging')

    parser.add_argument('--dtype', type=str, choices=['float32', 'float64'], 
                        default='float64',
                        help='Data type for model and tensors (float32 or float64)')
    
    args = parser.parse_args()
    
    # Run training
    try:
        final_rmse = main(args)
        sys.exit(0)
    except Exception as e:
        logging.error(f"Training failed: {e}", exc_info=True)
        sys.exit(1)