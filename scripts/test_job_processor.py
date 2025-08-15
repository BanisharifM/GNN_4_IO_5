#!/usr/bin/env python
"""
Test complete job processing pipeline with Phase 3
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
import logging
import json

# Setup
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.inference.job_processor import JobProcessor

def test_job_processor():
    """Test complete pipeline on IOR job"""
    
    print("="*70)
    print("Phase 3: Model Inference & Bottleneck Detection")
    print("="*70)
    
    # Paths to your model and data
    model_checkpoint = "/work/hdd/bdau/mbanisharifdehkordi/GNN_4_IO_5/data/1M/best_model.pt"
    model_config = "/work/hdd/bdau/mbanisharifdehkordi/GNN_4_IO_5/configs/gat_config6.yaml"
    similarity_path = "data/1M/similarity_output_0.75/similarity_graph_20250812_043913.pt"
    features_csv = "data/1M/aiio_sample_1000000_normalized.csv"
    
    # Initialize processor
    print("\nInitializing job processor...")
    processor = JobProcessor(
        model_checkpoint=model_checkpoint,
        model_config=model_config,
        similarity_path=similarity_path,
        features_csv_path=features_csv
    )
    
    # Load IOR job features
    print("\nLoading IOR job features...")
    ior_df = pd.read_csv('ior_all_ranks_features.csv')
    
    # Process each rank
    results_all = []
    
    for rank in range(len(ior_df)):
        print(f"\n{'='*70}")
        print(f"Processing Rank {rank}")
        print('='*70)
        
        job = ior_df.iloc[rank]
        
        # Get features (exclude performance and jobid)
        feature_cols = [col for col in ior_df.columns 
                       if col not in ['performance', 'jobid']]
        job_features = job[feature_cols].values
        
        actual_perf = job['performance']
        print(f"Actual performance: {actual_perf:.4f}")
        
        # Process the job
        results = processor.process_new_job(
            job_features=job_features,
            k_neighbors=50,
            return_details=True
        )
        
        # Add actual performance for comparison
        results['actual_performance'] = actual_perf
        results['rank'] = rank
        
        # Print results
        print(f"\nPredicted performance: {results['predicted_performance']:.4f}")
        print(f"Actual performance: {actual_perf:.4f}")
        print(f"Prediction error: {abs(results['predicted_performance'] - actual_perf):.4f}")
        
        print(f"\nNeighbor statistics:")
        if 'neighbor_stats' in results:
            stats = results['neighbor_stats']
            print(f"  Mean: {stats['mean']:.4f}")
            print(f"  Std:  {stats['std']:.4f}")
            print(f"  Min:  {stats['min']:.4f}")
            print(f"  Max:  {stats['max']:.4f}")
        
        # Print bottlenecks
        print("\nIdentified Bottlenecks (Consensus):")
        if 'bottlenecks' in results and 'consensus' in results['bottlenecks']:
            for i, (feature, score) in enumerate(results['bottlenecks']['consensus'][:10]):
                print(f"  {i+1:2d}. {feature:30s}: {score:.4f}")
        
        # Generate recommendations
        recommendations = processor.generate_recommendations(results['bottlenecks'])
        if recommendations:
            print("\nRecommendations:")
            for i, rec in enumerate(recommendations):
                print(f"  {i+1}. {rec}")
        
        results_all.append(results)
    
    # Save results
    output_file = "ior_job_analysis_results.json"
    with open(output_file, 'w') as f:
        json.dump(results_all, f, indent=2)
    print(f"\n{'='*70}")
    print(f"Results saved to {output_file}")
    
    # Summary across all ranks
    print(f"\n{'='*70}")
    print("Summary Across All Ranks")
    print('='*70)
    
    predictions = [r['predicted_performance'] for r in results_all]
    actuals = [r['actual_performance'] for r in results_all]
    errors = [abs(p - a) for p, a in zip(predictions, actuals)]
    
    print(f"Average predicted: {np.mean(predictions):.4f}")
    print(f"Average actual: {np.mean(actuals):.4f}")
    print(f"Average error: {np.mean(errors):.4f}")
    print(f"Max error: {np.max(errors):.4f}")
    
    # Common bottlenecks across all ranks
    from collections import Counter
    all_bottlenecks = []
    for r in results_all:
        if 'bottlenecks' in r and 'consensus' in r['bottlenecks']:
            for feature, score in r['bottlenecks']['consensus'][:5]:
                all_bottlenecks.append(feature)
    
    print("\nMost Common Bottlenecks Across All Ranks:")
    for feature, count in Counter(all_bottlenecks).most_common(5):
        print(f"  {feature}: appears in {count}/{len(results_all)} ranks")
    
    return results_all

def visualize_results(results_all):
    """Create visualization of results"""
    import matplotlib.pyplot as plt
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Plot 1: Predicted vs Actual
    ax = axes[0, 0]
    predictions = [r['predicted_performance'] for r in results_all]
    actuals = [r['actual_performance'] for r in results_all]
    ranks = [r['rank'] for r in results_all]
    
    ax.scatter(actuals, predictions, s=100, alpha=0.6)
    ax.plot([min(actuals), max(actuals)], [min(actuals), max(actuals)], 'r--', alpha=0.5)
    ax.set_xlabel('Actual Performance')
    ax.set_ylabel('Predicted Performance')
    ax.set_title('Model Predictions vs Actual')
    ax.grid(True, alpha=0.3)
    
    # Add rank labels
    for i, rank in enumerate(ranks):
        ax.annotate(f'R{rank}', (actuals[i], predictions[i]), fontsize=8)
    
    # Plot 2: Prediction Errors
    ax = axes[0, 1]
    errors = [abs(p - a) for p, a in zip(predictions, actuals)]
    ax.bar(range(len(errors)), errors)
    ax.set_xlabel('Rank')
    ax.set_ylabel('Absolute Error')
    ax.set_title('Prediction Errors by Rank')
    ax.grid(True, alpha=0.3)
    
    # Plot 3: Neighbor Performance Distribution
    ax = axes[1, 0]
    for r in results_all:
        if 'neighbor_stats' in r:
            stats = r['neighbor_stats']
            rank = r['rank']
            ax.errorbar(rank, stats['mean'], yerr=stats['std'], 
                       fmt='o', capsize=5, label=f'Rank {rank}')
            ax.scatter(rank, r['actual_performance'], color='red', s=100, marker='*')
    
    ax.set_xlabel('Rank')
    ax.set_ylabel('Performance')
    ax.set_title('Neighbor Performance vs Actual (red stars)')
    ax.grid(True, alpha=0.3)
    
    # Plot 4: Top Bottlenecks
    ax = axes[1, 1]
    from collections import Counter
    all_bottlenecks = []
    for r in results_all:
        if 'bottlenecks' in r and 'consensus' in r['bottlenecks']:
            for feature, score in r['bottlenecks']['consensus'][:3]:
                all_bottlenecks.append(feature)
    
    if all_bottlenecks:
        bottleneck_counts = Counter(all_bottlenecks)
        features = list(bottleneck_counts.keys())[:10]
        counts = [bottleneck_counts[f] for f in features]
        
        ax.barh(range(len(features)), counts)
        ax.set_yticks(range(len(features)))
        ax.set_yticklabels([f.replace('POSIX_', '') for f in features], fontsize=8)
        ax.set_xlabel('Frequency')
        ax.set_title('Most Common Bottlenecks')
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('ior_analysis_visualization.png', dpi=150, bbox_inches='tight')
    print("\nVisualization saved to ior_analysis_visualization.png")
    plt.show()

if __name__ == "__main__":
    try:
        # Run the test
        results = test_job_processor()
        
        # Create visualization
        print("\nGenerating visualization...")
        visualize_results(results)
        
        print("\n" + "="*70)
        print("✓ Phase 3 Complete!")
        print("="*70)
        print("\nThe system has successfully:")
        print("1. Loaded your trained GAT model")
        print("2. Processed the IOR job through the complete pipeline")
        print("3. Predicted performance")
        print("4. Identified I/O bottlenecks")
        print("5. Generated recommendations for optimization")
        
    except Exception as e:
        logger.error(f"Test failed: {e}")
        import traceback
        traceback.print_exc()