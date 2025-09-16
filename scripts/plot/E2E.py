import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from math import pi
import matplotlib.patches as mpatches

# Define the gradient data for all three cases
case1_decomp = {
    "nprocs": 0.02152669057250023,
    "POSIX_OPENS": 0.007182649336755276,
    "LUSTRE_STRIPE_SIZE": 0.09293287992477417,
    "LUSTRE_STRIPE_WIDTH": 0.009169748052954674,
    "POSIX_MEM_ALIGNMENT": 0.039302535355091095,
    "POSIX_FILE_ALIGNMENT": 0.003993287682533264,
    "POSIX_READS": 0.036800142377614975,
    "POSIX_WRITES": 0.017291652038693428,
    "POSIX_BYTES_READ": 0.1596270054578781,
    "POSIX_BYTES_WRITTEN": 0.10752660036087036,
    "POSIX_CONSEC_WRITES": 0.0021498806308954954,
    "POSIX_SEQ_READS": 0.03491266071796417,
    "POSIX_SEQ_WRITES": 0.016461603343486786,
    "POSIX_RW_SWITCHES": 0.04085277393460274,
    "POSIX_FILE_NOT_ALIGNED": 0.0386788435280323,
    "POSIX_SIZE_WRITE_100K_1M": 0.05530248582363129,
    "POSIX_STRIDE1_COUNT": 0.03588172420859337,
    "POSIX_ACCESS2_COUNT": 0.01981871947646141,
}

case2_checkpoint = {
    "nprocs": 0.0288534015417099,
    "POSIX_OPENS": 0.015736280009150505,
    "LUSTRE_STRIPE_SIZE": 0.09069731086492538,
    "LUSTRE_STRIPE_WIDTH": 0.0069286879152059555,
    "POSIX_MEM_ALIGNMENT": 0.029352447018027306,
    "POSIX_FILE_ALIGNMENT": 0.009587246924638748,
    "POSIX_READS": 0.004598875064402819,
    "POSIX_WRITES": 0.0269050020724535,
    "POSIX_BYTES_READ": 0.04125707596540451,
    "POSIX_BYTES_WRITTEN": 0.1131555438041687,
    "POSIX_CONSEC_WRITES": 0.04421637952327728,
    "POSIX_SEQ_READS": 0.002391262911260128,
    "POSIX_SEQ_WRITES": 0.04471535608172417,
    "POSIX_RW_SWITCHES": 0.006959717720746994,
    "POSIX_FILE_NOT_ALIGNED": 0.025559358298778534,
    "POSIX_SIZE_WRITE_100K_1M": 0.00893092155456543,
    "POSIX_STRIDE1_COUNT": 0.0901930034160614,
    "POSIX_ACCESS2_COUNT": 0.05785209685564041,
}

case3_combined = {
    "nprocs": 0.007534969598054886,
    "POSIX_OPENS": 0.021434742957353592,
    "LUSTRE_STRIPE_SIZE": 0.08575712144374847,
    "LUSTRE_STRIPE_WIDTH": 0.004397265613079071,
    "POSIX_MEM_ALIGNMENT": 0.038094520568847656,
    "POSIX_FILE_ALIGNMENT": 0.0025505400262773037,
    "POSIX_READS": 0.038111768662929535,
    "POSIX_WRITES": 0.014691520482301712,
    "POSIX_BYTES_READ": 0.15906549990177155,
    "POSIX_BYTES_WRITTEN": 0.08870361000299454,
    "POSIX_CONSEC_WRITES": 0.009971067309379578,
    "POSIX_SEQ_READS": 0.04346666485071182,
    "POSIX_SEQ_WRITES": 0.02383970282971859,
    "POSIX_RW_SWITCHES": 0.037021707743406296,
    "POSIX_FILE_NOT_ALIGNED": 0.037984833121299744,
    "POSIX_SIZE_WRITE_100K_1M": 0.06583236157894135,
    "POSIX_STRIDE1_COUNT": 0.047424789518117905,
    "POSIX_ACCESS2_COUNT": 0.019667252898216248,
}

# Get top 10 features for each case
def get_top_features(case_dict, n=10):
    sorted_items = sorted(case_dict.items(), key=lambda x: x[1], reverse=True)
    return dict(sorted_items[:n])

top10_case1 = get_top_features(case1_decomp)
top10_case2 = get_top_features(case2_checkpoint)
top10_case3 = get_top_features(case3_combined)

# Get union of all top features for consistent comparison
all_top_features = set(list(top10_case1.keys()) + list(top10_case2.keys()) + list(top10_case3.keys()))

# ============ Figure 1: Gradient Comparison Heatmap (Transposed) ============
fig1, ax1 = plt.subplots(figsize=(8, 10))

# Prepare data for heatmap (note: transposed)
heatmap_features = sorted(all_top_features)
case_names = ['Decomposition', 'Checkpoint', 'Combined']

# Create data matrix (features x cases)
heatmap_data = []
for feature in heatmap_features:
    row = [case1_decomp.get(feature, 0), case2_checkpoint.get(feature, 0), case3_combined.get(feature, 0)]
    heatmap_data.append(row)

# Create heatmap
heatmap_array = np.array(heatmap_data)
im = ax1.imshow(heatmap_array, cmap='YlOrRd', aspect='auto', vmin=0, vmax=0.16)

# Set ticks and labels
ax1.set_xticks(np.arange(len(case_names)))
ax1.set_yticks(np.arange(len(heatmap_features)))
ax1.set_xticklabels(case_names, fontsize=12)
ax1.set_yticklabels([f.replace('POSIX_', '').replace('LUSTRE_', 'L_')[:20] for f in heatmap_features], fontsize=10)

# Add colorbar
cbar = plt.colorbar(im, ax=ax1)
cbar.set_label('Gradient Score', fontsize=11)

# Add values in cells
for i in range(len(heatmap_features)):
    for j in range(len(case_names)):
        text = ax1.text(j, i, f'{heatmap_array[i, j]:.3f}',
                       ha="center", va="center", 
                       color="black" if heatmap_array[i, j] < 0.08 else "white",
                       fontsize=9)

ax1.set_title('Gradient Comparison Heatmap', fontsize=14, fontweight='bold', pad=15)
plt.tight_layout()
plt.savefig('e2e_heatmap.pdf', dpi=300, bbox_inches='tight')
plt.savefig('e2e_heatmap.png', dpi=300, bbox_inches='tight')
plt.show()

# ============ Figure 2: Feature Importance Radar Chart ============
fig2, ax2 = plt.subplots(figsize=(10, 10), subplot_kw=dict(projection='polar'))

# Select key features for radar chart (8 most important across all cases)
radar_features = ['POSIX_BYTES_READ', 'POSIX_BYTES_WRITTEN', 'LUSTRE_STRIPE_SIZE', 
                 'POSIX_FILE_NOT_ALIGNED', 'POSIX_SEQ_WRITES', 'POSIX_RW_SWITCHES',
                 'POSIX_STRIDE1_COUNT', 'POSIX_SIZE_WRITE_100K_1M']

# Prepare data
angles = [n / len(radar_features) * 2 * pi for n in range(len(radar_features))]
angles += angles[:1]

# Get values for each case
values_decomp = [case1_decomp.get(f, 0) for f in radar_features] + [case1_decomp.get(radar_features[0], 0)]
values_checkpoint = [case2_checkpoint.get(f, 0) for f in radar_features] + [case2_checkpoint.get(radar_features[0], 0)]
values_combined = [case3_combined.get(f, 0) for f in radar_features] + [case3_combined.get(radar_features[0], 0)]

# Plot
ax2.plot(angles, values_decomp, 'o-', linewidth=2, label='Decomposition', color='#1f77b4')
ax2.fill(angles, values_decomp, alpha=0.25, color='#1f77b4')
ax2.plot(angles, values_checkpoint, 'o-', linewidth=2, label='Checkpoint', color='#ff7f0e')
ax2.fill(angles, values_checkpoint, alpha=0.25, color='#ff7f0e')
ax2.plot(angles, values_combined, 'o-', linewidth=2, label='Combined', color='#2ca02c')
ax2.fill(angles, values_combined, alpha=0.25, color='#2ca02c')

# Fix axis
ax2.set_xticks(angles[:-1])
ax2.set_xticklabels([f.replace('POSIX_', '').replace('LUSTRE_', 'L_')[:12] for f in radar_features], fontsize=11)
ax2.set_ylim(0, 0.18)
ax2.set_title('Feature Importance Radar Chart', fontsize=14, fontweight='bold', pad=20)
ax2.legend(loc='upper right', bbox_to_anchor=(1.2, 1.1), fontsize=11)
ax2.grid(True)

plt.tight_layout()
plt.savefig('e2e_radar.pdf', dpi=300, bbox_inches='tight')
plt.savefig('e2e_radar.png', dpi=300, bbox_inches='tight')
plt.show()

# ============ Figure 3: Gradient Signature Comparison Table ============
fig3, ax3 = plt.subplots(figsize=(12, 6))
ax3.axis('tight')
ax3.axis('off')

# Prepare table data
key_features = ['POSIX_BYTES_WRITTEN', 'POSIX_BYTES_READ', 'POSIX_FILE_NOT_ALIGNED', 
                'POSIX_RW_SWITCHES', 'LUSTRE_STRIPE_SIZE', 'POSIX_SEQ_WRITES', 'POSIX_STRIDE1_COUNT']

table_data = []
for feature in key_features:
    row = [feature.replace('POSIX_', '').replace('LUSTRE_', ''),
           f"{case1_decomp.get(feature, 0):.4f}",
           f"{case2_checkpoint.get(feature, 0):.4f}",
           f"{case3_combined.get(feature, 0):.4f}"]
    table_data.append(row)

# Add performance row
table_data.append(['Performance (MB/s)', '3.92', '2.45', '2.81'])

# Create table
columns = ['Feature', 'Decomposition', 'Checkpoint', 'Combined']
table = ax3.table(cellText=table_data, colLabels=columns,
                  cellLoc='center', loc='center',
                  colWidths=[0.3, 0.15, 0.15, 0.15])

# Style the table
table.auto_set_font_size(False)
table.set_fontsize(11)
table.scale(1.2, 2.5)

# Color code cells based on values
for i in range(len(table_data)):
    for j in range(1, 4):  # Skip feature name column
        if i < len(table_data) - 1:  # Gradient values
            try:
                val = float(table_data[i][j])
                if val > 0.1:
                    table[(i+1, j)].set_facecolor('#ffcccc')
                elif val > 0.04:
                    table[(i+1, j)].set_facecolor('#ffffcc')
                else:
                    table[(i+1, j)].set_facecolor('#ccffcc')
            except:
                pass

# Header formatting
for j in range(4):
    table[(0, j)].set_facecolor('#4CAF50')
    table[(0, j)].set_text_props(weight='bold', color='white')

ax3.set_title('Gradient Signature Comparison Table', fontsize=14, fontweight='bold', pad=20)

# Add legend for table colors
legend_elements = [mpatches.Patch(color='#ffcccc', label='High (>0.1)'),
                  mpatches.Patch(color='#ffffcc', label='Medium (0.04-0.1)'),
                  mpatches.Patch(color='#ccffcc', label='Low (<0.04)')]
ax3.legend(handles=legend_elements, loc='upper right', bbox_to_anchor=(1.05, 0.95))

plt.tight_layout()
plt.savefig('e2e_table.pdf', dpi=300, bbox_inches='tight')
plt.savefig('e2e_table.png', dpi=300, bbox_inches='tight')
plt.show()

# Save data to CSV
df_comparison = pd.DataFrame({
    'Feature': list(all_top_features),
    'Decomposition': [case1_decomp.get(f, 0) for f in all_top_features],
    'Checkpoint': [case2_checkpoint.get(f, 0) for f in all_top_features],
    'Combined': [case3_combined.get(f, 0) for f in all_top_features]
})
df_comparison = df_comparison.sort_values('Feature')
df_comparison.to_csv('gradient_comparison.csv', index=False)
print("Data saved to gradient_comparison.csv")
print("Three separate figures saved: e2e_heatmap, e2e_radar, e2e_table (.pdf and .png)")