import csv
import os
import sys
import subprocess
import numpy as np
from pathlib import Path

def get_sample_headers(sample_file):
    """Extract headers from the sample training file"""
    with open(sample_file, 'r') as f:
        reader = csv.reader(f)
        headers = next(reader)
    return headers

def parse_darshan_file(darshan_file, temp_dir, verbose=True):
    """Parse a single Darshan file and extract all counters"""
    counters = {}
    
    # Parse total counters
    total_file = os.path.join(temp_dir, 'parsed_total.txt')
    cmd = f"darshan-parser --total {darshan_file} > {total_file} 2>/dev/null"
    ret = subprocess.call(cmd, shell=True)
    if ret != 0:
        return None
    
    if verbose:
        print(f"  Parsing POSIX counters:")
    
    with open(total_file, 'r') as f:
        for line in f:
            if line.startswith('total'):
                parts = line.strip().split(':', 1)
                if len(parts) == 2:
                    key = parts[0].strip()
                    value = parts[1].strip()
                    
                    # Log raw values for inspection
                    if verbose and 'POSIX' in key:
                        print(f"    RAW: {key} = {value}")
                    
                    # Check for invalid values
                    try:
                        float_val = float(value)
                        if np.isnan(float_val) or np.isinf(float_val):
                            print(f"    WARNING: Invalid value for {key}: {value}")
                            value = '0'
                        elif float_val < 0:
                            print(f"    WARNING: Negative value for {key}: {value}")
                    except ValueError:
                        print(f"    WARNING: Non-numeric value for {key}: {value}")
                        value = '0'
                    
                    # Remove 'total_' prefix for POSIX counters
                    if key.startswith('total_POSIX_'):
                        key = key.replace('total_', '', 1)
                    counters[key] = value
    
    # Parse performance data
    perf_file = os.path.join(temp_dir, 'parsed_perf.txt')
    cmd = f"darshan-parser --perf {darshan_file} > {perf_file} 2>/dev/null"
    subprocess.call(cmd, shell=True)
    
    if verbose:
        print(f"  Parsing performance data:")
    
    current_module = None
    with open(perf_file, 'r') as f:
        for line in f:
            # Track which module we're in
            if '# POSIX module data' in line:
                current_module = 'POSIX'
            elif '# MPI-IO module data' in line:
                current_module = 'MPIIO'
            elif '# STDIO module data' in line:
                current_module = 'STDIO'
            # Extract performance data
            elif 'agg_perf_by_slowest:' in line and current_module == 'POSIX':
                parts = line.split(':')
                if len(parts) > 1:
                    perf_str = parts[1].strip()
                    # Extract just the number (before "# MiB/s")
                    perf_value = perf_str.split('#')[0].strip()
                    if verbose:
                        print(f"    POSIX_PERF_MIBS = {perf_value}")
                    counters['POSIX_PERF_MIBS'] = perf_value
    
    # Parse Lustre data
    lustre_file = os.path.join(temp_dir, 'parsed_lustre.txt')
    cmd = f"darshan-parser {darshan_file} 2>/dev/null | grep '^LUSTRE' | cut -d$'\\t' -f 4-5 > {lustre_file}"
    subprocess.call(cmd, shell=True)

    lustre_stripe_widths = []
    lustre_stripe_sizes = []
    
    if verbose:
        print(f"  Parsing Lustre data:")

    with open(lustre_file, 'r') as f:
        lines = f.readlines()
        if not lines and verbose:
            print(f"    No Lustre data found in trace")
        
        for line in lines:
            if verbose:
                print(f"    RAW: {line.strip()}")
            
            parts = line.strip().split('\t')
            if len(parts) >= 2:
                try:
                    if parts[0] == 'LUSTRE_STRIPE_WIDTH':
                        val = int(parts[1])
                        if verbose:
                            print(f"      -> Found LUSTRE_STRIPE_WIDTH = {val}")
                        if val > 0:
                            lustre_stripe_widths.append(val)
                        else:
                            print(f"      -> WARNING: Invalid stripe width: {val}")
                    elif parts[0] == 'LUSTRE_STRIPE_SIZE':
                        val = int(parts[1])
                        if verbose:
                            print(f"      -> Found LUSTRE_STRIPE_SIZE = {val}")
                        if val > 0:
                            lustre_stripe_sizes.append(val)
                        else:
                            print(f"      -> WARNING: Invalid stripe size: {val}")
                except (ValueError, TypeError) as e:
                    print(f"      -> ERROR parsing Lustre value: {e}")
                    continue

    # Only set if we have valid data, otherwise leave unset (will become 0)
    if lustre_stripe_widths:
        avg_width = int(np.mean(lustre_stripe_widths))
        counters['LUSTRE_STRIPE_WIDTH'] = str(avg_width)
        if verbose:
            print(f"    Final LUSTRE_STRIPE_WIDTH: {avg_width} (from values: {lustre_stripe_widths})")
    else:
        if verbose:
            print(f"    No valid LUSTRE_STRIPE_WIDTH found (will be set to 0)")
    
    if lustre_stripe_sizes:
        avg_size = int(np.mean(lustre_stripe_sizes))
        counters['LUSTRE_STRIPE_SIZE'] = str(avg_size)
        if verbose:
            print(f"    Final LUSTRE_STRIPE_SIZE: {avg_size} (from values: {lustre_stripe_sizes})")
    else:
        if verbose:
            print(f"    No valid LUSTRE_STRIPE_SIZE found (will be set to 0)")
    
    # Get nprocs from the file
    with open(total_file, 'r') as f:
        for line in f:
            if line.startswith('# nprocs:'):
                nprocs = line.split(':')[1].strip()
                counters['nprocs'] = nprocs
                if verbose:
                    print(f"  Found nprocs: {nprocs}")
                break
    
    return counters

def normalize_value(value, header_name=""):
    """Apply log10(x+1) normalization"""
    try:
        numeric_val = float(value)
        if numeric_val < 0:
            print(f"    WARNING: Negative value for {header_name}: {numeric_val}, setting to 0")
            return 0.0
        if np.isnan(numeric_val):
            print(f"    WARNING: NaN value for {header_name}, setting to 0")
            return 0.0
        if np.isinf(numeric_val):
            print(f"    WARNING: Inf value for {header_name}, setting to 0")
            return 0.0
        normalized = np.log10(numeric_val + 1)
        return normalized
    except (ValueError, TypeError) as e:
        print(f"    WARNING: Cannot normalize {header_name}={value}: {e}, setting to 0")
        return 0.0

def process_darshan_logs(input_dir, output_csv, sample_csv, temp_dir, log_missing=True, verbose=True):
    """Process all Darshan logs and create output CSV matching sample format"""
    
    # Get headers from sample file
    headers = get_sample_headers(sample_csv)
    
    # Create temp directory
    os.makedirs(temp_dir, exist_ok=True)
    
    # Find all Darshan files
    darshan_files = []
    for root, dirs, files in os.walk(input_dir):
        for file in files:
            if file.endswith('.darshan'):
                darshan_files.append(os.path.join(root, file))
    
    if not darshan_files:
        print(f"No .darshan files found in {input_dir}")
        return
    
    print(f"Found {len(darshan_files)} Darshan files to process")
    
    # Track global missing counter statistics
    global_missing_counters = {}
    
    # Process each file and write to CSV
    with open(output_csv, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(headers)  # Write header
        
        for idx, darshan_file in enumerate(darshan_files):
            print(f"\nProcessing {idx+1}/{len(darshan_files)}: {os.path.basename(darshan_file)}")
            
            # Parse the Darshan file
            counters = parse_darshan_file(darshan_file, temp_dir, verbose=verbose)
            if counters is None:
                print(f"  Skipping due to parse error")
                continue
            
            # Track missing counters for this file
            missing_counters = []
            found_counters = []
            
            # Create row matching sample headers
            row = []
            for header in headers:
                if header == 'tag':
                    # tag is derived from POSIX_PERF_MIBS (normalized)
                    value = counters.get('POSIX_PERF_MIBS', '0')
                    if 'POSIX_PERF_MIBS' not in counters:
                        missing_counters.append('POSIX_PERF_MIBS (for tag)')
                    else:
                        normalized_val = normalize_value(value, "tag/POSIX_PERF_MIBS")
                        found_counters.append(f"tag={normalized_val:.4f}")
                    row.append(normalize_value(value, "tag"))
                elif header in counters:
                    # Direct match
                    normalized_val = normalize_value(counters[header], header)
                    found_counters.append(f"{header}={normalized_val:.4f}")
                    row.append(normalized_val)
                elif f"total_{header}" in counters:
                    # Try with total_ prefix
                    normalized_val = normalize_value(counters[f"total_{header}"], header)
                    found_counters.append(f"{header}={normalized_val:.4f}")
                    row.append(normalized_val)
                else:
                    # Missing value, use 0
                    missing_counters.append(header)
                    row.append(0.0)
                    
                    # Track globally
                    if header not in global_missing_counters:
                        global_missing_counters[header] = 0
                    global_missing_counters[header] += 1
            
            writer.writerow(row)
            
            # Log detailed information if requested
            if log_missing and missing_counters:
                print(f"  Missing {len(missing_counters)} counters (set to 0):")
                for counter in missing_counters[:10]:  # Show first 10
                    print(f"    - {counter}")
                if len(missing_counters) > 10:
                    print(f"    ... and {len(missing_counters)-10} more")
            
            print(f"  Found {len(found_counters)} counters successfully")
            print(f"  Processed successfully")
    
    print(f"\n{'='*60}")
    print(f"Output written to {output_csv}")
    
    # Summary of missing counters across all files
    if global_missing_counters:
        print(f"\nGlobal Missing Counter Summary:")
        print(f"  (Number shows how many files were missing each counter)")
        sorted_missing = sorted(global_missing_counters.items(), key=lambda x: x[1], reverse=True)
        for counter, count in sorted_missing[:20]:  # Show top 20
            print(f"    {counter}: missing in {count}/{len(darshan_files)} files")
    
    # Clean up temp directory
    subprocess.call(f"rm -rf {temp_dir}", shell=True)

def main():
    if len(sys.argv) < 4:
        print("Usage: python parser_clean.py <input_dir> <output_csv> <sample_csv> [temp_dir]")
        print("Example: python parser_clean.py ./darshan-logs ./output.csv ./sample_train_100.csv")
        sys.exit(1)
    
    input_dir = sys.argv[1]
    output_csv = sys.argv[2]
    sample_csv = sys.argv[3]
    temp_dir = sys.argv[4] if len(sys.argv) > 4 else f"/tmp/darshan_parse_{os.getpid()}"
    
    # Set verbose=False if you want less output
    process_darshan_logs(input_dir, output_csv, sample_csv, temp_dir, log_missing=True, verbose=True)

if __name__ == "__main__":
    main()