# IO Bottleneck Analyzer

GNN-based I/O performance bottleneck identification tool for HPC applications.

## Quick Start

```bash
# Basic usage
python -m io_bottleneck_analyzer.run_analysis test_sample.csv

# With custom options
python -m io_bottleneck_analyzer.run_analysis test.csv \
    --model-path data/1M/best_model.pt \
    --data-dir data/1M \
    --output-dir results/
```

## Installation

```bash
# Required packages
pip install torch torch-geometric numpy pandas scipy scikit-learn
```

## Usage

### Command Line
```bash
python -m io_bottleneck_analyzer.run_analysis <test_file> [options]

Options:
  --model-path PATH      Trained model checkpoint
  --data-dir PATH        Directory with training data
  --output-dir PATH      Output directory (default: ./results)
  --device {cpu,cuda}    Computation device
  --k-neighbors INT      Neighbors for subgraph (default: 100)
  --verbose              Detailed output
  --quiet               Suppress output
```

### Python API
```python
from io_bottleneck_analyzer import BottleneckAnalyzer

# Initialize
analyzer = BottleneckAnalyzer(
    model_path='data/1M/best_model.pt',
    data_dir='data/1M'
)

# Analyze
results = analyzer.analyze('test.csv')
print(f"Predicted: {results['performance']['predicted_mbps']:.2f} MB/s")
```

## Input Format

Darshan log parsed.

## Output

JSON reports in output directory:
- `bottleneck_report_*.json` - Top bottlenecks with recommendations
- `bottleneck_report_full_scores_*.json` - Detailed scores

## Methods

- **Attention Analysis** - GAT attention weights
- **GNNExplainer** - Learned feature masks
- **Gradient Methods** - Integrated gradients
- **Consensus** - Z-score normalized combination

## Project Structure

```
io_bottleneck_analyzer/
├── config/          # Settings and feature definitions
├── core/            # Model loading, graph building
├── methods/         # Interpretability implementations
├── analysis/        # Orchestration and consensus
├── reporting/       # Report generation
└── run_analysis.py  # Main entry point
```

## License

Research use - See main project license