#!/usr/bin/env python
"""
Test script for Darshan log parsing and feature extraction
"""

import sys
import argparse
import logging
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))

from src.darshan import DarshanParser, FeatureExtractor

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description='Test Darshan parser')
    parser.add_argument('log_path', type=str, help='Path to Darshan log file')
    parser.add_argument('--darshan-parser', type=str, 
                       default='~/darshan-patched-install/bin/darshan-parser',
                       help='Path to darshan-parser executable')
    parser.add_argument('--output', type=str, help='Output CSV file')
    
    args = parser.parse_args()
    
    # Initialize parser with your Darshan installation
    darshan_parser = DarshanParser(args.darshan_parser)
    
    # Parse log
    logger.info(f"Parsing {args.log_path}")
    darshan_data = darshan_parser.parse_darshan_log(args.log_path)
    
    # Extract features
    extractor = FeatureExtractor()
    result = extractor.extract_from_darshan_log(darshan_data)
    
    # Print results
    print("\n" + "="*60)
    print("EXTRACTED FEATURES")
    print("="*60)
    
    print("\nJob Metadata:")
    for key, value in result['metadata'].items():
        print(f"  {key}: {value}")
    
    print("\nPerformance:")
    if 'performance' in result:
        print(f"  Log10 Performance: {result['performance']:.4f}")
        print(f"  Actual Performance: {result['performance_raw']:.2f} MB/s")
    
    print("\nNormalized Features (first 10):")
    for i, (name, value) in enumerate(zip(result['feature_names'][:10], 
                                         result['features'][:10])):
        print(f"  {name}: {value:.4f}")
    
    # Save if requested
    if args.output:
        import pandas as pd
        import numpy as np
        
        row = dict(zip(result['feature_names'], result['features']))
        row['tag'] = result.get('performance', np.nan)  # Use 'tag' as column name
        row['jobid'] = result['metadata']['jobid']
        
        df = pd.DataFrame([row])
        df.to_csv(args.output, index=False)
        print(f"\nSaved to {args.output}")


if __name__ == "__main__":
    main()