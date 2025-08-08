"""
README for GNN4_IO_4

A Graph Neural Network approach for I/O performance prediction, combining TabGNN with IODiagnose methods.
"""

# GNN4_IO_4

This project implements a Graph Neural Network (GNN) approach for I/O performance prediction by combining the TabGNN approach with IODiagnose methods. The implementation constructs multiplex graphs based on important I/O counter features and integrates GNN embeddings with traditional tabular models to improve prediction performance.

## Project Structure

```
GNN4_IO_4/
├── configs/           # Configuration files
├── data/              # Data directory
├── models/            # Saved models
├── scripts/           # Training and testing scripts
│   ├── train.py       # Training script
│   └── test.py        # Testing script
├── src/               # Source code
│   ├── data.py        # Data processing and graph construction
│   ├── models/        # Model implementations
│   │   ├── gnn.py     # GNN models
│   │   └── tabular.py # Tabular models
│   └── utils/         # Utility functions
└── README.md          # This file
```

## Features

- **Data Processing**: Preprocessing I/O counter data and constructing multiplex graphs based on important features.
- **Graph Construction**: Building graphs where samples are nodes and edges represent relations between similar samples based on feature similarity.
- **Multiplex Graphs**: Creating multiple graph layers based on different important features.
- **TabGNN Models**: Implementing GNN models that learn node embeddings from the graph structure.
- **Traditional Models**: Implementing various tabular models (LightGBM, CatBoost, XGBoost, MLP, TabNet).
- **Combined Approach**: Integrating GNN embeddings with traditional models for improved performance.

## Important Features for Graph Construction

Based on the AIIO paper, the following important I/O counter features are used for graph construction:

- POSIX_SEQ_WRITES_PERC
- POSIX_SIZE_READ_1K_10K_PERC
- POSIX_SIZE_READ_10K_100K_PERC
- POSIX_unique_bytes_perc
- POSIX_SIZE_READ_0_100_PERC
- POSIX_SIZE_WRITE_100K_1M_PERC
- POSIX_write_only_bytes_perc
- POSIX_MEM_NOT_ALIGNED_PERC
- POSIX_FILE_NOT_ALIGNED_PERC

## Usage

### Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/GNN4_IO_4.git
cd GNN4_IO_4

# Install dependencies
pip install -r requirements.txt
```

### Training

```bash
python scripts/train.py --data_path /path/to/data.csv --output_dir ./output --target_column TARGET_COLUMN --model_type tabgnn
```

Options:
- `--data_path`: Path to input data CSV file
- `--output_dir`: Directory to save outputs
- `--target_column`: Target column for prediction
- `--model_type`: Type of model to train (tabgnn, lightgbm, catboost, xgboost, mlp, tabnet, combined)
- `--gnn_type`: Type of GNN to use (gcn, gat)
- `--hidden_dim`: Hidden dimension size
- `--num_layers`: Number of GNN layers
- `--dropout`: Dropout rate
- `--batch_size`: Batch size
- `--epochs`: Number of epochs
- `--lr`: Learning rate
- `--weight_decay`: Weight decay
- `--patience`: Patience for early stopping
- `--seed`: Random seed
- `--device`: Device to use (cuda or cpu)

### Testing

```bash
python scripts/test.py --data_path /path/to/data.csv --output_dir ./test_output --test_all
```

Options:
- `--data_path`: Path to input data CSV file
- `--output_dir`: Directory to save test outputs
- `--test_data_processing`: Test data processing module
- `--test_graph_construction`: Test graph construction module
- `--test_model_training`: Test model training
- `--test_all`: Run all tests
- `--device`: Device to use (cuda or cpu)

## Implementation Details

### Graph Construction

The implementation constructs multiplex graphs based on important I/O counter features. For each feature, a graph is created where:

1. Nodes represent individual samples (rows) in the dataset.
2. Edges connect nodes that have similar values for the feature (within a specified threshold).

### TabGNN Approach

The TabGNN approach combines Graph Neural Networks with traditional tabular models:

1. GNN models learn node embeddings from the graph structure.
2. These embeddings capture relationships between similar samples.
3. The embeddings are combined with the original features.
4. The combined features are fed into traditional models for prediction.

### Model Integration

The implementation supports two integration approaches:

1. **TabGNN Model**: A single end-to-end model that processes the graph data and makes predictions.
2. **Combined Model**: A two-stage approach where GNN embeddings are extracted and then combined with original features for a traditional model.

## Requirements

- Python 3.8+
- PyTorch
- PyTorch Geometric
- LightGBM
- CatBoost
- XGBoost
- scikit-learn
- pandas
- numpy
- matplotlib

## References

- TabGNN: Multiplex Graph Neural Network for Tabular Data Prediction
- IODiagnose: Using Artificial Intelligence for Job-Level and Automatic I/O Performance Bottleneck Diagnosis
