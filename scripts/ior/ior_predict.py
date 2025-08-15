import pandas as pd
import torch
import torch.nn as nn
import numpy as np
import sys
import os

# Add the project directory to path
sys.path.append('/work/hdd/bdau/mbanisharifdehkordi/GNN_4_IO_5')

# Import your GAT model from src/models
from src.models.gat import IOPerformanceGAT, create_gat_model

# Load the normalized features
test_data = pd.read_csv('/work/hdd/bdau/mbanisharifdehkordi/GNN_4_IO_5/darshan_features_ior_normalized.csv')

# Extract features (excluding tag for prediction)
features = test_data.iloc[:, :-1].values  # All columns except 'tag' (45 features)
actual_tag = test_data['tag'].values[0]

# Convert to tensor
features_tensor = torch.FloatTensor(features)

# Load checkpoint
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
checkpoint = torch.load('/work/hdd/bdau/mbanisharifdehkordi/GNN_4_IO_5/data/1M/best_model.pt', 
                       map_location=device)

print("=" * 50)
print(f"Checkpoint loaded from epoch {checkpoint['epoch']}")
print(f"Validation RMSE: {checkpoint['val_rmse']:.4f}")

# Infer model architecture from checkpoint
state_dict = checkpoint['model_state_dict']

# Get hidden_channels from input_proj layer
hidden_channels = state_dict['input_proj.weight'].shape[0]  # 512

# Get number of layers from gat_layers (count gat_layers.X entries)
num_layers = max([int(k.split('.')[1]) for k in state_dict.keys() if k.startswith('gat_layers.')]) + 1  # 2

# Get heads configuration from gat_layers
heads = []
for i in range(num_layers):
    if f'gat_layers.{i}.gat_conv.att_src' in state_dict:
        n_heads = state_dict[f'gat_layers.{i}.gat_conv.att_src'].shape[1]
        heads.append(n_heads)

print(f"Detected model configuration:")
print(f"  hidden_channels: {hidden_channels}")
print(f"  num_layers: {num_layers}")
print(f"  heads: {heads}")

# Create model with the detected configuration
model = create_gat_model(
    num_features=45,  # 45 I/O features (49 with augmentation)
    model_type='standard',
    hidden_channels=hidden_channels,  # 512
    num_layers=num_layers,  # 2
    heads=heads,  # [8, 1]
    dropout=0.1,
    edge_dim=1,
    residual=True,
    layer_norm=True,
    feature_augmentation=True,  # This adds 4 statistical features
    pool_type='mean',
    dtype=torch.float32
)

# Load the model state
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()
model = model.to(device)

print("Model loaded successfully!")
print("=" * 50)

# For single sample prediction, we need to create a minimal graph structure
# Since GAT expects graph data, we'll create a single-node graph
with torch.no_grad():
    features_tensor = features_tensor.to(device)
    
    # Add batch dimension
    if len(features_tensor.shape) == 1:
        features_tensor = features_tensor.unsqueeze(0)  # [1, 45]
    
    # For feature augmentation (if enabled in training)
    if model.feature_augmentation:
        # Add statistical features (mean, std, min, max)
        feat_mean = features_tensor.mean(dim=1, keepdim=True)
        feat_std = features_tensor.std(dim=1, keepdim=True)
        feat_min = features_tensor.min(dim=1, keepdim=True)[0]
        feat_max = features_tensor.max(dim=1, keepdim=True)[0]
        
        features_augmented = torch.cat([
            features_tensor, feat_mean, feat_std, feat_min, feat_max
        ], dim=1)  # [1, 49]
    else:
        features_augmented = features_tensor
    
    # Create a dummy edge index for single node (self-loop)
    edge_index = torch.tensor([[0], [0]], dtype=torch.long).to(device)
    
    # Create batch tensor for pooling
    batch = torch.zeros(1, dtype=torch.long).to(device)
    
    # Forward pass through the model
    # Note: The model expects graph data, so we simulate it
    x = model.input_proj(features_augmented)  # Input projection
    
    # Pass through GAT layers
    for i, gat_layer in enumerate(model.gat_layers):
        # Store input for residual
        residual = x
        
        # GAT convolution
        x, _ = gat_layer(x, edge_index)
        
        # Residual connection
        if model.residual and i < len(model.residual_projs):
            residual = model.residual_projs[i](residual)
            x = x + residual
        
        # Layer norm
        if model.layer_norm and i < len(model.layer_norms):
            x = model.layer_norms[i](x)
        
        # Activation and dropout (except last layer)
        if i < model.num_layers - 1:
            x = torch.nn.functional.elu(x)
            x = torch.nn.functional.dropout(x, p=model.dropout, training=False)
    
    # Since we have a single node, pooling is just the node itself
    pooled = x
    
    # Final prediction
    predicted_tag = model.predictor(pooled)
    predicted_value = predicted_tag.squeeze().item()

# Convert back from log scale to actual bandwidth
predicted_bandwidth = 10**predicted_value - 1
actual_bandwidth = 10**actual_tag - 1

print("\n=== Prediction Results ===")
print(f"Predicted tag (log scale): {predicted_value:.4f}")
print(f"Actual tag (log scale): {actual_tag:.4f}")
print(f"Tag difference: {abs(predicted_value - actual_tag):.4f}")
print()
print(f"Predicted bandwidth: {predicted_bandwidth:.2f} MB/s")
print(f"Actual bandwidth: {actual_bandwidth:.2f} MB/s")
print(f"Absolute error: {abs(predicted_bandwidth - actual_bandwidth):.2f} MB/s")
print(f"Relative error: {abs(predicted_bandwidth - actual_bandwidth) / actual_bandwidth * 100:.1f}%")
print("=" * 50)