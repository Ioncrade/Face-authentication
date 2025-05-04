import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

# ---- Configuration ----
sns.set_style('whitegrid')
sns.set_palette(['#4C72B0', '#DD8452'])  # Custom two-color palette
plt.rcParams.update({
    'font.size': 12,
    'axes.titlesize': 14,
    'axes.labelsize': 12,
    'legend.fontsize': 11,
    'figure.titlesize': 16,
    'axes.edgecolor': 'gray',
    'axes.linewidth': 0.8
})

# ---- Data ----
metrics = {
    'FAR @ 1% FRR': {'Arcface': 5.2, 'CS|BA': 0.87},
    'FAR @ 0.1% FRR': {'Arcface': 12.7, 'CS|BA': 1.8},
    'FRR @ 1% FAR': {'Arcface': 7.4, 'CS|BA': 0.54},
    'FRR @ 0.1% FAR': {'Arcface': 15.3, 'CS|BA': 1.9},
    'Equal Error Rate (EER)': {'Arcface': 6.8, 'CS|BA': 0.29},
    'Accuracy (%)': {'Arcface': 91.2, 'CS|BA': 99.2},
    'AUC': {'Arcface': 0.957, 'CS|BA': 0.998}
}
df = pd.DataFrame(metrics).T

# ---- Figure Setup ----
fig, axes = plt.subplots(2, 2, figsize=(18, 14))
fig.patch.set_facecolor('white')

# 1. Error Rates
ax = axes[0, 0]
error_df = df.loc[['FAR @ 1% FRR', 'FAR @ 0.1% FRR', 'FRR @ 1% FAR', 'FRR @ 0.1% FAR', 'Equal Error Rate (EER)']]
error_df.plot(kind='bar', ax=ax)
ax.set_title('Error Rates Comparison')
ax.set_ylabel('Rate (%)')
ax.set_ylim(0, error_df.values.max() * 1.1)
ax.legend(title='Model', frameon=False)
for p in ax.patches:
    ax.annotate(f"{p.get_height():.2f}%", (p.get_x() + p.get_width() / 2, p.get_height()),
                ha='center', va='bottom', fontsize=10)

# 2. Accuracy
ax = axes[0, 1]
acc_df = df.loc[['Accuracy (%)']]
acc_df.plot(kind='bar', ax=ax)
ax.set_title('Accuracy at Optimal Threshold')
ax.set_ylabel('Accuracy (%)')
ax.set_ylim(80, 100)
for p in ax.patches:
    ax.annotate(f"{p.get_height():.1f}%", (p.get_x() + p.get_width() / 2, p.get_height()),
                ha='center', va='bottom', fontsize=10)
ax.get_legend().remove()

# 3. AUC
ax = axes[1, 0]
auc_df = df.loc[['AUC']]
auc_df.plot(kind='bar', ax=ax)
ax.set_title('Area Under ROC Curve (AUC)')
ax.set_ylabel('AUC Score')
ax.set_ylim(0.9, 1.0)
for p in ax.patches:
    ax.annotate(f"{p.get_height():.3f}", (p.get_x() + p.get_width() / 2, p.get_height()),
                ha='center', va='bottom', fontsize=10)
ax.get_legend().remove()

# 4. ROC Curves
ax = axes[1, 1]
fpr = np.linspace(0, 1, 200)
# Approximate ROC from AUC
def roc_from_auc(auc, fpr):
    # simple quadratic approx for visualization
    return fpr ** (1 - auc)

ax.plot(fpr, roc_from_auc(df.loc['AUC', 'Arcface'], fpr), label=f"Arcface (AUC={df.loc['AUC','Arcface']:.3f})")
ax.plot(fpr, roc_from_auc(df.loc['AUC', 'CS|BA'], fpr), label=f"CS|BA (AUC={df.loc['AUC','CS|BA']:.3f})")
ax.plot([0,1],[0,1],'--', color='gray', linewidth=1)
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
ax.set_xlabel('False Positive Rate (FPR)')
ax.set_ylabel('True Positive Rate (TPR)')
ax.set_title('ROC Curve Comparison')
ax.legend(frameon=False)

# ---- Overall Title & Summary ----
fig.suptitle('Arcface vs CS|BA: Face Recognition Performance', fontsize=18, fontweight='bold')
summary = (
    "CS|BA consistently outperforms Arcface across all metrics,\n"
    "with an EER 23× lower (0.29% vs 6.8%) and near-perfect ROC (AUC=0.998).\n"
    "Accuracy difference: 99.2% vs 91.2%."
)
plt.figtext(0.5, 0.02, summary, ha='center', fontsize=12, bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

plt.tight_layout(rect=[0, 0.04, 1, 0.96])

# Save the figure with high resolution
plt.savefig('face_recognition_comparison.png', dpi=300, bbox_inches='tight')

# Display the figure
plt.show()
