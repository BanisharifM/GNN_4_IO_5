import wandb
import os
import sys
import json
import pandas as pd
import numpy as np
from datetime import datetime

def parse_metrics_file(metrics_file):
    """Parse the metrics.txt file for key performance indicators"""
    metrics = {}
    if os.path.exists(metrics_file):
        with open(metrics_file, 'r') as f:
            content = f.read()
            # Extract aggregate performance
            if 'agg_perf_by_slowest' in content:
                for line in content.split('\n'):
                    if 'agg_perf_by_slowest' in line and 'MiB/s' in line:
                        try:
                            value = float(line.split(':')[-1].replace('MiB/s', '').strip())
                            metrics['agg_bandwidth_mib_s'] = value
                        except:
                            pass
            # Extract write counts
            if 'POSIX_WRITES:' in content:
                for line in content.split('\n'):
                    if 'POSIX_WRITES:' in line:
                        try:
                            value = int(line.split(':')[-1].strip())
                            metrics['posix_write_count'] = value
                        except:
                            pass
    return metrics

def parse_darshan_summary(parsed_file):
    """Extract detailed metrics from parsed Darshan log"""
    metrics = {}
    if os.path.exists(parsed_file):
        with open(parsed_file, 'r') as f:
            content = f.read()
            
            # Extract I/O sizes
            for line in content.split('\n'):
                if 'POSIX_SIZE_WRITE_0_100' in line:
                    parts = line.split()
                    if len(parts) >= 5:
                        metrics['small_writes_0_100'] = int(parts[4])
                elif 'POSIX_SIZE_WRITE_100K_1M' in line:
                    parts = line.split()
                    if len(parts) >= 5:
                        metrics['medium_writes_100K_1M'] = int(parts[4])
                elif 'POSIX_SIZE_WRITE_1M_4M' in line:
                    parts = line.split()
                    if len(parts) >= 5:
                        metrics['large_writes_1M_4M'] = int(parts[4])
                elif 'POSIX_FILE_NOT_ALIGNED' in line:
                    parts = line.split()
                    if len(parts) >= 5:
                        metrics['misaligned_accesses'] = int(parts[4])
    return metrics

def log_test_to_wandb(test_name, test_dir, config):
    """Log a single test's results to W&B"""
    
    # Parse metrics
    metrics_file = os.path.join(test_dir, 'metrics.txt')
    parsed_file = os.path.join(test_dir, f"{config['job_name']}_{config['slurm_job_id']}_parsed.txt")
    
    metrics = parse_metrics_file(metrics_file)
    detailed = parse_darshan_summary(parsed_file)
    metrics.update(detailed)
    
    # Add test configuration
    metrics.update({
        'test_name': test_name,
        'nodes': config['nodes'],
        'tasks': config['tasks'],
        'block_size_x': config['block_x'],
        'block_size_y': config['block_y'],
        'block_size_z': config['block_z'],
        'decomp_x': config['decomp_x'],
        'decomp_y': config['decomp_y'],
        'decomp_z': config['decomp_z'],
        'lustre_stripe_count': config['stripe_count'],
        'lustre_stripe_size': config['stripe_size'],
        'collective_io': config['collective_io'],
        'timestamp': datetime.now().isoformat()
    })
    
    # Log to W&B
    wandb.log(metrics)
    
    # Upload Darshan log as artifact
    darshan_file = os.path.join(test_dir, f"{config['job_name']}_{config['slurm_job_id']}.darshan")
    if os.path.exists(darshan_file):
        artifact = wandb.Artifact(f"darshan_{test_name}_{config['slurm_job_id']}", type="darshan_log")
        artifact.add_file(darshan_file)
        wandb.log_artifact(artifact)
    
    # Upload CSV if exists
    csv_file = os.path.join(test_dir, 'parsed.csv')
    if os.path.exists(csv_file):
        df = pd.read_csv(csv_file)
        wandb.log({f"{test_name}_data": wandb.Table(dataframe=df.head(100))})
    
    return metrics

if __name__ == "__main__":
    test_name = sys.argv[1]
    test_dir = sys.argv[2]
    config_json = sys.argv[3]
    
    config = json.loads(config_json)
    
    # Initialize W&B run if not already initialized
    if wandb.run is None:
        wandb.init(
            project=os.environ.get('WANDB_PROJECT', 'e2e-io-optimization'),
            entity=os.environ.get('WANDB_ENTITY'),
            job_type="performance_test",
            config=config,
            tags=["small_test", test_name, f"nodes_{config['nodes']}"],
            name=f"e2e_small_{config['slurm_job_id']}_{test_name}"
        )
    
    # Log the test
    metrics = log_test_to_wandb(test_name, test_dir, config)
    print(f"Logged {test_name}: {metrics}")
    
    # Don't finish W&B run here, let the main script handle it
