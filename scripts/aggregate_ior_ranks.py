#!/usr/bin/env python
"""
Aggregate multiple rank logs from a single IOR job
Save as: scripts/aggregate_ior_ranks.py
"""

import json
import numpy as np
from pathlib import Path
import pandas as pd
import subprocess
import sys
import os

def parse_single_darshan_log(log_path):
    """Parse one Darshan log to JSON using the correct environment"""
    
    # Path to darshan-parser in ior_env
    parser_path = "/u/mbanisharifdehkordi/.conda/envs/ior_env/bin/darshan-parser"
    
    # Check if parser exists
    if not os.path.exists(parser_path):
        print(f"Error: darshan-parser not found at {parser_path}")
        return None
    
    # Check if log file exists
    if not os.path.exists(log_path):
        print(f"Error: Log file not found at {log_path}")
        return None
    
    # Run parser
    try:
        result = subprocess.run(
            [parser_path, '--json', log_path],
            capture_output=True,
            text=True,
            check=False
        )
        
        if result.returncode != 0:
            print(f"Error parsing {log_path}: {result.stderr}")
            return None
        
        if not result.stdout:
            print(f"Empty output from parser for {log_path}")
            return None
            
        return json.loads(result.stdout)
        
    except json.JSONDecodeError as e:
        print(f"Failed to parse JSON from {log_path}: {e}")
        print(f"Parser output: {result.stdout[:500]}...")  # Show first 500 chars
        return None
    except Exception as e:
        print(f"Unexpected error parsing {log_path}: {e}")
        return None

def extract_features_from_darshan_data(data):
    """Extract features from parsed Darshan JSON data"""
    
    features = {
        'POSIX_OPENS': 0,
        'POSIX_READS': 0,
        'POSIX_WRITES': 0,
        'POSIX_BYTES_READ': 0,
        'POSIX_BYTES_WRITTEN': 0,
        'POSIX_SEQ_READS': 0,
        'POSIX_SEQ_WRITES': 0,
        'POSIX_CONSEC_READS': 0,
        'POSIX_CONSEC_WRITES': 0,
        'POSIX_RW_SWITCHES': 0,
        'POSIX_FILE_NOT_ALIGNED': 0,
        'POSIX_MEM_NOT_ALIGNED': 0,
    }
    
    # Extract from POSIX module
    if 'records' in data and 'POSIX' in data['records']:
        posix_records = data['records']['POSIX']
        
        for file_id, record in posix_records.items():
            counters = record.get('counters', {})
            
            # Sum up counters
            for key in features.keys():
                if key in counters:
                    value = counters[key]
                    if value != -1:  # -1 means not monitored
                        features[key] += value
    
    return features

def aggregate_ior_job(rank_logs):
    """
    Aggregate metrics from multiple rank logs into single job metrics
    """
    
    # Initialize aggregated counters
    aggregated = {}
    all_features = []
    
    # Track timing
    start_times = []
    end_times = []
    
    # Process each rank
    for i, log_path in enumerate(rank_logs):
        print(f"Processing rank {i}: {Path(log_path).name}")
        
        # Parse the log
        data = parse_single_darshan_log(log_path)
        
        if data is None:
            print(f"  Skipping rank {i} due to parsing error")
            continue
        
        # Get timing info
        if 'header' in data:
            start_times.append(data['header'].get('start_time', 0))
            end_times.append(data['header'].get('end_time', 0))
        
        # Extract features
        rank_features = extract_features_from_darshan_data(data)
        all_features.append(rank_features)
    
    if not all_features:
        print("Error: No ranks could be parsed successfully")
        return None
    
    # Aggregate across ranks
    feature_names = all_features[0].keys()
    for key in feature_names:
        # Sum most counters
        if 'SIZE_' in key:
            # For size bins, take maximum
            aggregated[key] = max(f[key] for f in all_features)
        else:
            # For other counters, sum
            aggregated[key] = sum(f[key] for f in all_features)
    
    # Calculate performance
    if start_times and end_times:
        runtime = max(end_times) - min(start_times)
        total_bytes = aggregated['POSIX_BYTES_READ'] + aggregated['POSIX_BYTES_WRITTEN']
        performance_mbs = (total_bytes / (1024 * 1024)) / runtime if runtime > 0 else 0
    else:
        runtime = 0
        performance_mbs = 0
    
    # Add metadata
    aggregated['nprocs'] = len(rank_logs)
    aggregated['runtime'] = runtime
    aggregated['performance_raw_mbs'] = performance_mbs
    aggregated['performance'] = np.log10(performance_mbs + 1e-10)
    
    return aggregated

def process_existing_csv():
    """Process the existing ior_all_ranks_features.csv file"""
    
    # Load existing CSV
    df = pd.read_csv('ior_all_ranks_features.csv')
    print(f"Loaded {len(df)} ranks from CSV")
    
    # Since all 4 ranks have similar features, we can aggregate them
    # by taking mean or sum depending on the feature
    
    feature_cols = [col for col in df.columns if col not in ['performance', 'jobid']]
    
    # Create aggregated row
    aggregated = {}
    
    for col in feature_cols:
        if col == 'nprocs':
            # Sum the processes
            aggregated[col] = df[col].sum()
        elif 'BYTES' in col or 'READS' in col or 'WRITES' in col or 'OPENS' in col:
            # Sum I/O operations
            aggregated[col] = df[col].sum()
        else:
            # Take mean for other features
            aggregated[col] = df[col].mean()
    
    # Add performance (already in log scale)
    aggregated['performance'] = df['performance'].mean()
    aggregated['nprocs_actual'] = 4  # We know it was 4 ranks
    
    return aggregated

if __name__ == "__main__":
    
    print("Option 1: Process from existing CSV (simpler)")
    print("="*60)
    
    # Process the CSV you already have
    aggregated_from_csv = process_existing_csv()
    
    print("\nAggregated metrics from CSV:")
    print(f"  Total processes: {aggregated_from_csv.get('nprocs_actual', 4)}")
    print(f"  Performance (log10): {aggregated_from_csv['performance']:.4f}")
    print(f"  Performance (MB/s): {10**aggregated_from_csv['performance']:.2f}")
    
    # Save aggregated version
    df_agg = pd.DataFrame([aggregated_from_csv])
    df_agg.to_csv('ior_job_aggregated_from_csv.csv', index=False)
    print("\nSaved to: ior_job_aggregated_from_csv.csv")
    
    print("\n" + "="*60)
    print("Option 2: Process from Darshan logs (if you have access)")
    print("="*60)
    
    # Your 4 rank logs
    rank_logs = [
        "/u/mbanisharifdehkordi/ior-darshan-repo/logs/2025/8/14/mbanisha_ior_id311810-311810_8-14-76121-17188222232049623171_1.darshan",
        "/u/mbanisharifdehkordi/ior-darshan-repo/logs/2025/8/14/mbanisha_ior_id311811-311811_8-14-76121-12129391993060542129_1.darshan",
        "/u/mbanisharifdehkordi/ior-darshan-repo/logs/2025/8/14/mbanisha_ior_id311812-311812_8-14-76121-11600146204377663224_1.darshan",
        "/u/mbanisharifdehkordi/ior-darshan-repo/logs/2025/8/14/mbanisha_ior_id311813-311813_8-14-76121-7504921136711006361_1.darshan",
    ]
    
    # Check if we can access the logs
    if Path(rank_logs[0]).exists():
        print("Attempting to aggregate from Darshan logs...")
        aggregated = aggregate_ior_job(rank_logs)
        
        if aggregated:
            print("\nAggregated Job Metrics:")
            print(f"  Total Processes: {aggregated['nprocs']}")
            print(f"  Runtime: {aggregated['runtime']:.2f} seconds")
            print(f"  Performance (MB/s): {aggregated['performance_raw_mbs']:.2f}")
            print(f"  Performance (log10): {aggregated['performance']:.4f}")
            
            df = pd.DataFrame([aggregated])
            df.to_csv('ior_job_aggregated_from_logs.csv', index=False)
            print("\nSaved to: ior_job_aggregated_from_logs.csv")
    else:
        print("Cannot access Darshan logs from this environment")
        print("Using the CSV aggregation method instead")