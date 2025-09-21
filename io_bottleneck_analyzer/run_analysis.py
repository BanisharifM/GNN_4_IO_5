#!/usr/bin/env python3
"""
Main entry point for IO Bottleneck Analyzer
"""
import argparse
import logging
import sys
from pathlib import Path

# Setup logging before imports
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger(__name__)

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from io_bottleneck_analyzer.config import (
    DEFAULT_MODEL_PATH,
    DEFAULT_DATA_DIR,
    MODEL_DEVICE,
    REPORT_OUTPUT_DIR,
    SAVE_FULL_SCORES,
    get_similarity_matrix_path,
    get_training_features_path
)
from io_bottleneck_analyzer.analysis import BottleneckAnalyzer
from io_bottleneck_analyzer.reporting import ReportGenerator


def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description='IO Bottleneck Analyzer - Identify I/O performance bottlenecks using GNN interpretability'
    )
    
    # Required arguments
    parser.add_argument(
        'test_file',
        type=str,
        help='Path to test CSV file containing I/O features'
    )
    
    # Optional arguments
    parser.add_argument(
        '--model-path',
        type=str,
        default=DEFAULT_MODEL_PATH,
        help=f'Path to trained model checkpoint (default: {DEFAULT_MODEL_PATH})'
    )
    
    parser.add_argument(
        '--data-dir',
        type=str,
        default=DEFAULT_DATA_DIR,
        help=f'Directory containing data files (default: {DEFAULT_DATA_DIR})'
    )
    
    parser.add_argument(
        '--output-dir',
        type=str,
        default=REPORT_OUTPUT_DIR,
        help=f'Output directory for reports (default: {REPORT_OUTPUT_DIR})'
    )
    
    parser.add_argument(
        '--similarity-matrix',
        type=str,
        default=None,
        help='Path to similarity matrix file (auto-detected if not specified)'
    )
    
    parser.add_argument(
        '--training-features',
        type=str,
        default=None,
        help='Path to training features CSV (auto-detected if not specified)'
    )
    
    parser.add_argument(
        '--device',
        type=str,
        choices=['cpu', 'cuda'],
        default=MODEL_DEVICE,
        help=f'Device to use for computation (default: {MODEL_DEVICE})'
    )
    
    parser.add_argument(
        '--k-neighbors',
        type=int,
        default=100,
        help='Number of neighbors for subgraph (default: 100)'
    )
    
    parser.add_argument(
        '--no-full-scores',
        action='store_true',
        help='Disable saving full scores report'
    )
    
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Enable verbose output'
    )
    
    parser.add_argument(
        '--quiet',
        action='store_true',
        help='Suppress all output except errors'
    )
    
    return parser.parse_args()


def setup_logging(verbose: bool, quiet: bool):
    """Configure logging based on verbosity settings"""
    if quiet:
        logging.getLogger().setLevel(logging.ERROR)
    elif verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    else:
        logging.getLogger().setLevel(logging.INFO)


def validate_paths(args):
    """Validate input paths exist"""
    test_path = Path(args.test_file)
    if not test_path.exists():
        logger.error(f"Test file not found: {test_path}")
        sys.exit(1)
    
    model_path = Path(args.model_path)
    if not model_path.exists():
        logger.error(f"Model checkpoint not found: {model_path}")
        sys.exit(1)
    
    return test_path, model_path


def auto_detect_data_files(args):
    """Auto-detect data files if not specified"""
    # Auto-detect similarity matrix
    if args.similarity_matrix is None:
        args.similarity_matrix = get_similarity_matrix_path(args.data_dir)
        if args.similarity_matrix:
            logger.info(f"Auto-detected similarity matrix: {args.similarity_matrix}")
        else:
            logger.warning("No similarity matrix found - will create single-node graph")
    
    # Auto-detect training features
    if args.training_features is None:
        args.training_features = get_training_features_path(args.data_dir)
        if args.training_features:
            logger.info(f"Auto-detected training features: {args.training_features}")
        else:
            logger.warning("No training features found - will create single-node graph")
    
    return args


def main():
    """Main execution function"""
    # Parse arguments
    args = parse_arguments()
    
    # Setup logging
    setup_logging(args.verbose, args.quiet)
    
    # Print header
    if not args.quiet:
        print("\n" + "="*70)
        print("IO BOTTLENECK ANALYZER")
        print("="*70)
        print(f"\nAnalyzing: {args.test_file}")
    
    # Validate paths
    test_path, model_path = validate_paths(args)
    
    # Auto-detect data files
    args = auto_detect_data_files(args)
    
    try:
        # Initialize analyzer
        logger.info("\nInitializing analyzer...")
        analyzer = BottleneckAnalyzer(
            model_path=str(model_path),
            data_dir=args.data_dir,
            similarity_matrix_path=args.similarity_matrix,
            training_features_path=args.training_features,
            device=args.device
        )
        
        # Run analysis
        logger.info("\nRunning analysis...")
        results = analyzer.analyze(str(test_path))
        
        # Generate report
        logger.info("\nGenerating report...")
        report_generator = ReportGenerator(output_dir=args.output_dir)
        report = report_generator.generate_report(
            results, 
            save_full_scores=not args.no_full_scores
        )
        
        # Print summary
        if not args.quiet:
            summary = report_generator.generate_summary(report)
            print("\n" + summary)
            print(f"\nReports saved to: {args.output_dir}/")
        
        return 0
        
    except Exception as e:
        logger.error(f"Analysis failed: {e}", exc_info=args.verbose)
        return 1


if __name__ == "__main__":
    sys.exit(main())