#!/usr/bin/env python3
"""
Darshan Log Parser for GNN-based I/O Optimization
Extracts 46 I/O counters from Darshan logs and applies AIIO normalization
"""

import subprocess
import pandas as pd
import numpy as np
import json
import os
import sys
import re
from pathlib import Path
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class DarshanParser:
    def __init__(self, config_file='configs/darshan_features.json'):
        """
        Initialize the Darshan parser with configuration file
        
        Args:
            config_file: Path to JSON configuration file containing feature names
        """
        self.config_file = config_file
        self.features = self.load_features_config()
        self.missing_features = []
        
    def load_features_config(self):
        """Load feature names from configuration file"""
        try:
            with open(self.config_file, 'r') as f:
                config = json.load(f)
                return config['features']
        except FileNotFoundError:
            logger.error(f"Configuration file {self.config_file} not found!")
            raise
        except json.JSONDecodeError as e:
            logger.error(f"Error parsing configuration file: {e}")
            raise
            
    def parse_darshan_log(self, darshan_log_path):
        """
        Parse Darshan binary log using darshan-parser
        
        Args:
            darshan_log_path: Path to Darshan binary log file
            
        Returns:
            Dictionary containing parsed data from different modules
        """
        if not os.path.exists(darshan_log_path):
            raise FileNotFoundError(f"Darshan log file not found: {darshan_log_path}")
            
        logger.info(f"Parsing Darshan log: {darshan_log_path}")
        
        # Run darshan-parser
        cmd = f"darshan-parser {darshan_log_path}"
        try:
            result = subprocess.run(cmd, shell=True, capture_output=True, text=True, check=True)
        except subprocess.CalledProcessError as e:
            logger.error(f"Error running darshan-parser: {e}")
            logger.error(f"stderr: {e.stderr}")
            raise
            
        # Parse the output
        parsed_data = self._parse_darshan_output(result.stdout)
        return parsed_data
    
    def _parse_darshan_output(self, output):
        """
        Parse the text output from darshan-parser
        
        Args:
            output: Text output from darshan-parser
            
        Returns:
            Dictionary with module data
        """
        data = {
            'header': {},
            'posix': {},
            'lustre': {},
            'stdio': {},
            'counters': {}
        }
        
        lines = output.split('\n')
        current_module = None
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
                
            # Parse header information
            if '# nprocs:' in line:
                data['header']['nprocs'] = int(line.split(':')[1].strip())
            elif '# run time:' in line:
                data['header']['runtime'] = float(line.split(':')[1].strip())
            elif '# start_time:' in line:
                data['header']['start_time'] = float(line.split(':')[1].strip())
            elif '# end_time:' in line:
                data['header']['end_time'] = float(line.split(':')[1].strip())
                
            # Detect module sections
            elif '# POSIX module data' in line:
                current_module = 'posix'
            elif '# LUSTRE module data' in line:
                current_module = 'lustre'
            elif '# STDIO module data' in line:
                current_module = 'stdio'
                
            # Parse counter data
            elif current_module and '\t' in line:
                parts = line.split('\t')
                if len(parts) >= 5:
                    # Format: rank, file_id, counter_name, value, file_name
                    try:
                        rank = int(parts[0])
                        counter_name = parts[3]
                        value = float(parts[4]) if parts[4] != '-1' else 0
                        
                        if counter_name not in data['counters']:
                            data['counters'][counter_name] = []
                        data['counters'][counter_name].append(value)
                    except (ValueError, IndexError):
                        continue
                        
        return data
    
    def extract_features(self, parsed_data):
        """
        Extract the 46 features from parsed Darshan data
        
        Args:
            parsed_data: Dictionary containing parsed Darshan data
            
        Returns:
            Dictionary with feature values
        """
        features_dict = {}
        self.missing_features = []
        
        # Get nprocs from header
        features_dict['nprocs'] = parsed_data['header'].get('nprocs', 1)
        
        # Process each feature
        for feature in self.features:
            if feature == 'nprocs':
                continue  # Already handled
            elif feature == 'tag':
                # Calculate performance (will be done in calculate_performance)
                continue
            elif feature.startswith('LUSTRE_'):
                # Handle LUSTRE features (likely missing)
                if feature in parsed_data['counters']:
                    values = parsed_data['counters'][feature]
                    features_dict[feature] = max(values) if values else 0
                else:
                    features_dict[feature] = 0
                    if feature not in self.missing_features:
                        self.missing_features.append(feature)
            else:
                # Handle POSIX and other features
                if feature in parsed_data['counters']:
                    values = parsed_data['counters'][feature]
                    # Aggregate across all ranks (use sum or max depending on counter type)
                    if 'BYTES' in feature or 'COUNT' in feature or 'READS' in feature or 'WRITES' in feature:
                        features_dict[feature] = sum(values) if values else 0
                    else:
                        features_dict[feature] = max(values) if values else 0
                else:
                    features_dict[feature] = 0
                    if feature not in self.missing_features:
                        self.missing_features.append(feature)
        
        # Log missing features
        if self.missing_features:
            logger.warning(f"Missing features (set to 0): {', '.join(self.missing_features)}")
            
        return features_dict
    
    def calculate_performance(self, parsed_data):
        """
        Calculate performance metric (tag) using AIIO formula
        Performance = total_bytes_transferred / time_of_slowest_process
        
        Args:
            parsed_data: Dictionary containing parsed Darshan data
            
        Returns:
            Performance in MiB/s
        """
        # Get total bytes transferred (read + written)
        bytes_read = sum(parsed_data['counters'].get('POSIX_BYTES_READ', [0]))
        bytes_written = sum(parsed_data['counters'].get('POSIX_BYTES_WRITTEN', [0]))
        total_bytes = bytes_read + bytes_written
        
        # Get runtime (time of slowest process)
        runtime = parsed_data['header'].get('runtime', 1.0)
        
        # Avoid division by zero
        if runtime == 0:
            runtime = 1.0
            
        # Calculate performance in MiB/s
        performance_mibs = (total_bytes / (1024 * 1024)) / runtime
        
        logger.info(f"Performance calculation: {total_bytes} bytes / {runtime} seconds = {performance_mibs:.2f} MiB/s")
        
        return performance_mibs
    
    def normalize_features(self, features_dict):
        """
        Apply AIIO normalization: log10(x + 1) to all features
        
        Args:
            features_dict: Dictionary with raw feature values
            
        Returns:
            Dictionary with normalized feature values
        """
        normalized = {}
        for key, value in features_dict.items():
            normalized[key] = np.log10(value + 1)
        return normalized
    
    def process_darshan_log(self, darshan_log_path, output_csv_path=None):
        """
        Complete pipeline to process a Darshan log file
        
        Args:
            darshan_log_path: Path to Darshan binary log
            output_csv_path: Path for output CSV (optional)
            
        Returns:
            DataFrame with normalized features
        """
        # Parse the Darshan log
        parsed_data = self.parse_darshan_log(darshan_log_path)
        
        # Extract features
        features = self.extract_features(parsed_data)
        
        # Calculate performance (tag)
        performance = self.calculate_performance(parsed_data)
        features['tag'] = performance
        
        # Normalize all features
        normalized_features = self.normalize_features(features)
        
        # Create DataFrame
        df = pd.DataFrame([normalized_features])
        
        # Ensure all columns are in the correct order
        column_order = self.features
        df = df[column_order]
        
        # Save to CSV if path provided
        if output_csv_path:
            df.to_csv(output_csv_path, index=False)
            logger.info(f"Saved normalized features to {output_csv_path}")
        
        return df
    
    def process_multiple_logs(self, log_directory, output_csv_path):
        """
        Process multiple Darshan logs from a directory
        
        Args:
            log_directory: Directory containing Darshan logs
            output_csv_path: Path for combined output CSV
            
        Returns:
            DataFrame with all normalized features
        """
        all_data = []
        log_files = list(Path(log_directory).glob('*.darshan'))
        
        logger.info(f"Found {len(log_files)} Darshan log files")
        
        for log_file in log_files:
            try:
                logger.info(f"Processing {log_file.name}")
                df = self.process_darshan_log(str(log_file))
                all_data.append(df)
            except Exception as e:
                logger.error(f"Error processing {log_file}: {e}")
                continue
        
        if all_data:
            combined_df = pd.concat(all_data, ignore_index=True)
            combined_df.to_csv(output_csv_path, index=False)
            logger.info(f"Saved {len(combined_df)} rows to {output_csv_path}")
            return combined_df
        else:
            logger.warning("No data processed successfully")
            return pd.DataFrame()


def main():
    """Main function for command-line usage"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Parse Darshan logs and extract normalized features')
    parser.add_argument('input', help='Path to Darshan log file or directory')
    parser.add_argument('output', help='Path for output CSV file')
    parser.add_argument('--config', default='configs/darshan_features.json',
                       help='Path to features configuration file')
    parser.add_argument('--batch', action='store_true',
                       help='Process all .darshan files in the input directory')
    
    args = parser.parse_args()
    
    # Initialize parser
    darshan_parser = DarshanParser(config_file=args.config)
    
    # Process logs
    if args.batch:
        darshan_parser.process_multiple_logs(args.input, args.output)
    else:
        darshan_parser.process_darshan_log(args.input, args.output)
    
    logger.info("Processing complete!")


if __name__ == "__main__":
    main()