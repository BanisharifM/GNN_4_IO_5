# # Quick check for POSIX_FILENOS
# import pandas as pd
# import numpy as np

# # Load just the column names first
# df_header = pd.read_csv('/work/hdd/bdau/mbanisharifdehkordi/GNN_4_IO_5/data/1M/aiio_sample_1000000.csv', nrows=0)
# print(f"Total columns: {len(df_header.columns)}")
# print(f"POSIX_FILENOS present: {'POSIX_FILENOS' in df_header.columns}")

# # If present, check its values
# if 'POSIX_FILENOS' in df_header.columns:
#     # Read just this column
#     filenos = pd.read_csv('/work/hdd/bdau/mbanisharifdehkordi/GNN_4_IO_5/data/1M/aiio_sample_1000000.csv', 
#                           usecols=['POSIX_FILENOS'], 
#                           nrows=10000)
#     print(f"POSIX_FILENOS range: [{filenos.min().values[0]}, {filenos.max().values[0]}]")
    


import torch
print(f"GPU Count: {torch.cuda.device_count()}")
for i in range(torch.cuda.device_count()):
    print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
    print(f"  Memory: {torch.cuda.get_device_properties(i).total_memory / 1e9:.2f} GB")
    print(f"  Current Allocated: {torch.cuda.memory_allocated(i) / 1e9:.2f} GB")
    print(f"  Current Cached: {torch.cuda.memory_reserved(i) / 1e9:.2f} GB")