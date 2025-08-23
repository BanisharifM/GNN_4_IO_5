#!/usr/bin/env python3
"""
Universal Darshan Feature Extractor for IOR Tests
Handles both READ and WRITE operations with automatic detection
"""

import subprocess
import pandas as pd
import numpy as np
import re
import sys
import os

def detect_operation_type(darshan_log_path):
    """
    Detect whether this is a read or write operation
    Returns: 'read', 'write', or 'mixed'
    """
    cmd = f"darshan-parser {darshan_log_path}"
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    
    reads = 0
    writes = 0
    
    for line in result.stdout.split('\n'):
        if 'POSIX_READS' in line and '/tmp/ior' in line:
            match = re.search(r'POSIX_READS\s+(\d+)', line)
            if match:
                reads = int(match.group(1))
        if 'POSIX_WRITES' in line and '/tmp/ior' in line:
            match = re.search(r'POSIX_WRITES\s+(\d+)', line)
            if match:
                writes = int(match.group(1))
    
    if reads > 0 and writes == 0:
        return 'read'
    elif writes > 0 and reads == 0:
        return 'write'
    elif reads > 0 and writes > 0:
        return 'mixed'
    else:
        return 'unknown'

def extract_bandwidth_from_darshan(darshan_log_path, operation='auto'):
    """
    Extract actual bandwidth from Darshan log for read or write operations
    """
    print(f"Extracting bandwidth from: {darshan_log_path}")
    
    # Auto-detect operation type if not specified
    if operation == 'auto':
        operation = detect_operation_type(darshan_log_path)
        print(f"Detected operation type: {operation}")
    
    # Parse the full Darshan output
    cmd = f"darshan-parser {darshan_log_path}"
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    
    if operation == 'read':
        # Method 1: Look for read performance metrics
        bytes_read = None
        read_time = None
        
        for line in result.stdout.split('\n'):
            if 'POSIX_BYTES_READ' in line and '/tmp/ior' in line:
                parts = line.split()
                for i, part in enumerate(parts):
                    if part == 'POSIX_BYTES_READ':
                        try:
                            bytes_read = float(parts[i+1])
                        except:
                            pass
            
            if 'POSIX_F_READ_TIME' in line and '/tmp/ior' in line:
                parts = line.split()
                for i, part in enumerate(parts):
                    if part == 'POSIX_F_READ_TIME':
                        try:
                            read_time = float(parts[i+1])
                        except:
                            pass
        
        if bytes_read and read_time and read_time > 0:
            bandwidth = (bytes_read / (1024 * 1024)) / read_time  # MB/s
            print(f"Calculated READ bandwidth: {bandwidth:.2f} MB/s")
            return bandwidth
            
    elif operation == 'write':
        # Method 1: Look for write performance metrics
        bytes_written = None
        write_time = None
        
        for line in result.stdout.split('\n'):
            if 'POSIX_BYTES_WRITTEN' in line and '/tmp/ior' in line:
                parts = line.split()
                for i, part in enumerate(parts):
                    if part == 'POSIX_BYTES_WRITTEN':
                        try:
                            bytes_written = float(parts[i+1])
                        except:
                            pass
            
            if 'POSIX_F_WRITE_TIME' in line and '/tmp/ior' in line:
                parts = line.split()
                for i, part in enumerate(parts):
                    if part == 'POSIX_F_WRITE_TIME':
                        try:
                            write_time = float(parts[i+1])
                        except:
                            pass
        
        if bytes_written and write_time and write_time > 0:
            bandwidth = (bytes_written / (1024 * 1024)) / write_time  # MB/s
            print(f"Calculated WRITE bandwidth: {bandwidth:.2f} MB/s")
            return bandwidth
    
    # Method 2: Try to get from agg_perf_by_slowest
    for line in result.stdout.split('\n'):
        if 'agg_perf_by_slowest' in line:
            match = re.search(r'agg_perf_by_slowest:\s*([\d.]+)', line)
            if match:
                bandwidth = float(match.group(1))
                print(f"Found bandwidth from agg_perf_by_slowest: {bandwidth} MB/s")
                return bandwidth
    
    print(f"WARNING: Could not extract {operation} bandwidth from Darshan log")
    return None

def extract_darshan_features_for_ior(darshan_log_path):
    """
    Extract POSIX features from Darshan log for any IOR configuration
    Works for both READ and WRITE operations
    """
    print(f"\nExtracting features from: {darshan_log_path}")
    
    # Parse Darshan log
    cmd = f"darshan-parser {darshan_log_path}"
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    
    if result.returncode != 0:
        print(f"Error parsing Darshan log: {result.stderr}")
        return None
    
    # Initialize all features to 0
    features = {
        'nprocs': 1,  # Default to single process
        'POSIX_OPENS': 0,
        'LUSTRE_STRIPE_SIZE': 0,
        'LUSTRE_STRIPE_WIDTH': 0,
        'POSIX_FILENOS': 0,
        'POSIX_MEM_ALIGNMENT': 0,
        'POSIX_FILE_ALIGNMENT': 0,
        'POSIX_READS': 0,
        'POSIX_WRITES': 0,
        'POSIX_SEEKS': 0,
        'POSIX_STATS': 0,
        'POSIX_BYTES_READ': 0,
        'POSIX_BYTES_WRITTEN': 0,
        'POSIX_CONSEC_READS': 0,
        'POSIX_CONSEC_WRITES': 0,
        'POSIX_SEQ_READS': 0,
        'POSIX_SEQ_WRITES': 0,
        'POSIX_RW_SWITCHES': 0,
        'POSIX_MEM_NOT_ALIGNED': 0,
        'POSIX_FILE_NOT_ALIGNED': 0,
        'POSIX_SIZE_READ_0_100': 0,
        'POSIX_SIZE_READ_100_1K': 0,
        'POSIX_SIZE_READ_1K_10K': 0,
        'POSIX_SIZE_READ_100K_1M': 0,
        'POSIX_SIZE_WRITE_0_100': 0,
        'POSIX_SIZE_WRITE_100_1K': 0,
        'POSIX_SIZE_WRITE_1K_10K': 0,
        'POSIX_SIZE_WRITE_10K_100K': 0,
        'POSIX_SIZE_WRITE_100K_1M': 0,
        'POSIX_STRIDE1_STRIDE': 0,
        'POSIX_STRIDE2_STRIDE': 0,
        'POSIX_STRIDE3_STRIDE': 0,
        'POSIX_STRIDE4_STRIDE': 0,
        'POSIX_STRIDE1_COUNT': 0,
        'POSIX_STRIDE2_COUNT': 0,
        'POSIX_STRIDE3_COUNT': 0,
        'POSIX_STRIDE4_COUNT': 0,
        'POSIX_ACCESS1_ACCESS': 0,
        'POSIX_ACCESS2_ACCESS': 0,
        'POSIX_ACCESS3_ACCESS': 0,
        'POSIX_ACCESS4_ACCESS': 0,
        'POSIX_ACCESS1_COUNT': 0,
        'POSIX_ACCESS2_COUNT': 0,
        'POSIX_ACCESS3_COUNT': 0,
        'POSIX_ACCESS4_COUNT': 0
    }
    
    # Parse output line by line
    lines = result.stdout.split('\n')
    
    # Track if we found IOR file data
    ior_file_found = False
    
    for line in lines:
        # Look for lines containing IOR test file data
        if '/tmp/ior' in line or 'ior_test' in line or 'ior_read' in line:
            parts = line.split()
            
            # Standard Darshan parser format: filename ... counter_name value
            for i, part in enumerate(parts):
                if part in features and i+1 < len(parts):
                    try:
                        value = float(parts[i+1])
                        features[part] = value
                        ior_file_found = True
                    except:
                        pass
    
    # Extract nprocs from job info if available
    for line in lines:
        if 'nprocs:' in line:
            match = re.search(r'nprocs:\s*(\d+)', line)
            if match:
                features['nprocs'] = int(match.group(1))
    
    # Validation
    if not ior_file_found:
        print("WARNING: No IOR file data found in Darshan log")
        print("The log might be corrupted or from a different application")
    else:
        print(f"\nExtracted {sum(1 for v in features.values() if v > 0)} non-zero features")
    
    return features

def normalize_and_save(raw_features, bandwidth_mbps, output_path, operation_type='auto'):
    """
    Apply AIIO normalization (log10(x+1)) and save to CSV
    """
    print(f"\n=== Normalizing Features ===")
    print(f"Operation: {operation_type.upper()}")
    print(f"Bandwidth: {bandwidth_mbps:.2f} MB/s")
    
    # Apply log10(x+1) normalization to all features
    normalized = {}
    for key, value in raw_features.items():
        normalized[key] = np.log10(value + 1)
    
    # Add performance tag (normalized bandwidth)
    normalized['tag'] = np.log10(bandwidth_mbps + 1)
    
    print(f"Normalized tag: {normalized['tag']:.4f} (log10({bandwidth_mbps + 1:.2f}))")
    
    # Define column order (must match training data format)
    column_order = [
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
        'POSIX_ACCESS4_COUNT', 'tag'
    ]
    
    # Create DataFrame and save
    df = pd.DataFrame([normalized])[column_order]
    df.to_csv(output_path, index=False)
    
    print(f"\n✅ Features saved to: {output_path}")
    print(f"Shape: {df.shape} (should be 1 row × 46 columns)")
    
    return df, normalized

def main():
    """Main function to process Darshan logs"""
    
    # Parse command line arguments
    if len(sys.argv) < 2:
        print("Usage: python extract_features.py <darshan_log> [test_name] [bandwidth_mb/s] [output_dir]")
        print("\nExamples:")
        print("  # Auto-detect bandwidth:")
        print("  python extract_features.py 11464139_read.darshan read_test")
        print("  ")
        print("  # Manually specify bandwidth:")
        print("  python extract_features.py 11464139_read.darshan read_test 703.27")
        print("  ")
        print("  # With custom output directory:")
        print("  python extract_features.py 11464139_write.darshan write_test 490.91 /path/to/output")
        sys.exit(1)
    
    darshan_log = sys.argv[1]
    
    # Determine test name from filename or argument
    if len(sys.argv) > 2:
        test_name = sys.argv[2]
    else:
        basename = os.path.basename(darshan_log)
        if 'read' in basename.lower():
            test_name = 'read_test'
        elif 'write' in basename.lower():
            test_name = 'write_test'
        else:
            test_name = 'test'
    
    # Check for manually specified bandwidth
    manual_bandwidth = None
    if len(sys.argv) > 3:
        try:
            manual_bandwidth = float(sys.argv[3])
            print(f"Using manually specified bandwidth: {manual_bandwidth} MB/s")
        except ValueError:
            # Not a number, might be output directory
            if not os.path.isdir(sys.argv[3]):
                print(f"Warning: {sys.argv[3]} is neither a valid bandwidth nor directory")
    
    # Output directory
    if len(sys.argv) > 4:
        output_dir = sys.argv[4]
    elif len(sys.argv) > 3 and os.path.isdir(sys.argv[3]):
        output_dir = sys.argv[3]
    else:
        output_dir = "/work/hdd/bdau/mbanisharifdehkordi/GNN_4_IO_5"
    
    print(f"\n{'='*70}")
    print(f"Processing Darshan Log: {test_name}")
    print(f"{'='*70}")
    
    # Check if file exists
    if not os.path.exists(darshan_log):
        print(f"ERROR: File not found: {darshan_log}")
        sys.exit(1)
    
    # Detect operation type
    operation_type = detect_operation_type(darshan_log)
    print(f"Operation type: {operation_type.upper()}")
    
    # Extract features from Darshan log
    raw_features = extract_darshan_features_for_ior(darshan_log)
    
    if raw_features is None:
        print("ERROR: Failed to extract features from Darshan log")
        sys.exit(1)
    
    # Show key metrics based on operation type
    print("\n=== Key Metrics ===")
    if operation_type == 'read':
        print(f"POSIX_READS: {raw_features['POSIX_READS']}")
        print(f"POSIX_BYTES_READ: {raw_features['POSIX_BYTES_READ']} ({raw_features['POSIX_BYTES_READ']/(1024*1024):.2f} MB)")
        print(f"POSIX_SIZE_READ_100_1K: {raw_features['POSIX_SIZE_READ_100_1K']}")
        print(f"POSIX_SIZE_READ_1K_10K: {raw_features['POSIX_SIZE_READ_1K_10K']}")
        print(f"POSIX_CONSEC_READS: {raw_features['POSIX_CONSEC_READS']}")
        print(f"POSIX_SEQ_READS: {raw_features['POSIX_SEQ_READS']}")
    elif operation_type == 'write':
        print(f"POSIX_WRITES: {raw_features['POSIX_WRITES']}")
        print(f"POSIX_BYTES_WRITTEN: {raw_features['POSIX_BYTES_WRITTEN']} ({raw_features['POSIX_BYTES_WRITTEN']/(1024*1024):.2f} MB)")
        print(f"POSIX_SIZE_WRITE_100_1K: {raw_features['POSIX_SIZE_WRITE_100_1K']}")
        print(f"POSIX_SIZE_WRITE_1K_10K: {raw_features['POSIX_SIZE_WRITE_1K_10K']}")
        print(f"POSIX_CONSEC_WRITES: {raw_features['POSIX_CONSEC_WRITES']}")
        print(f"POSIX_SEQ_WRITES: {raw_features['POSIX_SEQ_WRITES']}")
    
    # Use manual bandwidth if provided, otherwise extract from log
    if manual_bandwidth:
        bandwidth_mbps = manual_bandwidth
    else:
        # Extract actual bandwidth from the log
        bandwidth_mbps = extract_bandwidth_from_darshan(darshan_log, operation_type)
        
        if bandwidth_mbps is None:
            print(f"\nERROR: Could not extract {operation_type} bandwidth automatically.")
            print("Please enter the bandwidth manually (in MB/s):")
            print("From your IOR output:")
            if operation_type == 'read':
                print("  Look for: 'read      XXX.XX' in the IOR output")
            else:
                print("  Look for: 'write     XXX.XX' in the IOR output")
            
            try:
                bandwidth_mbps = float(input("Bandwidth (MB/s): "))
            except:
                print("Invalid input. Exiting.")
                sys.exit(1)
    
    # Determine IOR configuration based on features
    print(f"\n=== Detected IOR Configuration ===")
    if operation_type == 'read':
        if raw_features['POSIX_SIZE_READ_100_1K'] > 0:
            print("Transfer size: 100B-1KB range (likely -t 1k)")
        elif raw_features['POSIX_SIZE_READ_1K_10K'] > 0:
            print("Transfer size: 1KB-10KB range")
        elif raw_features['POSIX_SIZE_READ_100K_1M'] > 0:
            print("Transfer size: 100KB-1MB range")
        print(f"Total reads: {raw_features['POSIX_READS']}")
        print(f"Total bytes read: {raw_features['POSIX_BYTES_READ'] / (1024*1024):.2f} MB")
    else:
        if raw_features['POSIX_SIZE_WRITE_100_1K'] > 0:
            print("Transfer size: 100B-1KB range (likely -t 1k)")
        elif raw_features['POSIX_SIZE_WRITE_1K_10K'] > 0:
            print("Transfer size: 1KB-10KB range")
        elif raw_features['POSIX_SIZE_WRITE_100K_1M'] > 0:
            print("Transfer size: 100KB-1MB range")
        print(f"Total writes: {raw_features['POSIX_WRITES']}")
        print(f"Total bytes written: {raw_features['POSIX_BYTES_WRITTEN'] / (1024*1024):.2f} MB")
    
    print(f"Performance: {bandwidth_mbps:.2f} MB/s")
    
    # Save normalized features
    output_csv = os.path.join(output_dir, f"darshan_features_ior_normalized_{test_name}.csv")
    df, normalized = normalize_and_save(raw_features, bandwidth_mbps, output_csv, operation_type)
    
    print(f"\n{'='*70}")
    print(f"Processing complete for {test_name}")
    print(f"{'='*70}")

if __name__ == "__main__":
    main()