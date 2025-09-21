"""
Feature definitions and recommendations for IO bottleneck analysis
"""

# Complete list of POSIX I/O features
POSIX_FEATURES = [
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
    'POSIX_ACCESS4_COUNT'
]

# Feature recommendations for bottlenecks
BOTTLENECK_RECOMMENDATIONS = {
    # Write size issues
    'POSIX_SIZE_WRITE_100_1K': 'Increase write buffer size to at least 1MB for better performance',
    'POSIX_SIZE_WRITE_0_100': 'Avoid very small writes (<100B), batch operations to reduce overhead',
    'POSIX_SIZE_WRITE_1K_10K': 'Increase write size to 100KB or larger for improved throughput',
    'POSIX_SIZE_WRITE_10K_100K': 'Consider increasing write size to 1MB for optimal performance',
    
    # Read size issues
    'POSIX_SIZE_READ_0_100': 'Avoid very small reads (<100B), batch read operations',
    'POSIX_SIZE_READ_100_1K': 'Increase read buffer size to reduce I/O operations',
    'POSIX_SIZE_READ_1K_10K': 'Consider larger read sizes (>100KB) for better throughput',
    
    # Access pattern issues
    'POSIX_SEEKS': 'Reduce random access, use sequential I/O patterns when possible',
    'POSIX_RW_SWITCHES': 'Minimize switching between reads and writes, batch similar operations',
    'POSIX_FILE_NOT_ALIGNED': 'Align I/O operations to file system block boundaries (typically 4KB or 1MB)',
    'POSIX_MEM_NOT_ALIGNED': 'Align memory buffers for better DMA performance',
    
    # Write pattern issues
    'POSIX_CONSEC_WRITES': 'Already using consecutive writes, but check if size can be increased',
    'POSIX_SEQ_WRITES': 'Sequential write pattern is good, ensure write sizes are optimal',
    
    # Read pattern issues  
    'POSIX_CONSEC_READS': 'Consecutive reads detected, consider prefetching',
    'POSIX_SEQ_READS': 'Sequential read pattern is good, ensure read-ahead is enabled',
    
    # Data volume issues
    'POSIX_BYTES_WRITTEN': 'Large data volume - consider compression or data reduction techniques',
    'POSIX_BYTES_READ': 'Large read volume - consider caching frequently accessed data',
    
    # File system configuration
    'LUSTRE_STRIPE_SIZE': 'Adjust Lustre stripe size to match I/O pattern (typically 1MB-4MB)',
    'LUSTRE_STRIPE_WIDTH': 'Optimize Lustre stripe count based on file size and access pattern',
    
    # File operations
    'POSIX_OPENS': 'Reduce number of file open operations, reuse file handles when possible',
    'POSIX_STATS': 'Minimize stat operations, cache metadata when possible',
    
    # Stride pattern issues
    'POSIX_STRIDE1_STRIDE': 'Non-contiguous access pattern detected, consider reorganizing data layout',
    'POSIX_STRIDE2_STRIDE': 'Strided access pattern may cause poor performance',
    'POSIX_STRIDE3_STRIDE': 'Complex stride pattern detected, evaluate data access strategy',
    'POSIX_STRIDE4_STRIDE': 'Multiple stride patterns indicate fragmented access',
    
    # Process coordination
    'nprocs': 'Consider process count - too many processes may cause contention',
    
    # Default recommendation
    'default': 'Analyze I/O pattern for optimization opportunities'
}

def get_recommendation(feature_name: str) -> str:
    """Get recommendation for a specific bottleneck feature"""
    return BOTTLENECK_RECOMMENDATIONS.get(
        feature_name, 
        BOTTLENECK_RECOMMENDATIONS['default']
    )

# Feature groups for analysis
FEATURE_GROUPS = {
    'write_sizes': [
        'POSIX_SIZE_WRITE_0_100', 'POSIX_SIZE_WRITE_100_1K',
        'POSIX_SIZE_WRITE_1K_10K', 'POSIX_SIZE_WRITE_10K_100K',
        'POSIX_SIZE_WRITE_100K_1M'
    ],
    'read_sizes': [
        'POSIX_SIZE_READ_0_100', 'POSIX_SIZE_READ_100_1K',
        'POSIX_SIZE_READ_1K_10K', 'POSIX_SIZE_READ_100K_1M'
    ],
    'access_patterns': [
        'POSIX_SEQ_READS', 'POSIX_SEQ_WRITES',
        'POSIX_CONSEC_READS', 'POSIX_CONSEC_WRITES',
        'POSIX_SEEKS', 'POSIX_RW_SWITCHES'
    ],
    'alignment': [
        'POSIX_MEM_NOT_ALIGNED', 'POSIX_FILE_NOT_ALIGNED',
        'POSIX_MEM_ALIGNMENT', 'POSIX_FILE_ALIGNMENT'
    ],
    'lustre': [
        'LUSTRE_STRIPE_SIZE', 'LUSTRE_STRIPE_WIDTH'
    ],
    'stride_patterns': [
        'POSIX_STRIDE1_STRIDE', 'POSIX_STRIDE2_STRIDE',
        'POSIX_STRIDE3_STRIDE', 'POSIX_STRIDE4_STRIDE',
        'POSIX_STRIDE1_COUNT', 'POSIX_STRIDE2_COUNT',
        'POSIX_STRIDE3_COUNT', 'POSIX_STRIDE4_COUNT'
    ]
}