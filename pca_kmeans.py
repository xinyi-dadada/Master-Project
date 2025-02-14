import os
import glob
import scipy
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

def apply_PCA(df_new):
    fft_for_pca = df_new['fft_result']
    min_length = min(fft_for_pca.apply(len))
    fft_for_pca = np.array(fft_for_pca)
    truncated_arrays = np.array([arr[:min_length] for arr in fft_for_pca])
    pca = PCA(n_components=2)
    components = pca.fit_transform(truncated_arrays)
    return components

data = pd.read_parquet('/home/Shared/xinyi/blob1/thesis/radar_112/all_part_tasks.parquet')
components_pca_2 = apply_PCA(data)
df_pca = pd.DataFrame(components_pca_2, columns=['PC1', 'PC2'])
# Generate labels from 0 to 10, repeated 41 times to get exactly 451 entries
labels = np.tile(np.arange(11), 41)
df_pca['label'] = pd.Series(labels).apply(lambda x: f'task {x}')
df_pca['label_num'] = labels

plt.figure(figsize=(10, 8))
sns.scatterplot(
    x='PC1',
    y='PC2',
    hue='label',
    palette='Paired',
    data=df_pca,
    s=50,
    edgecolor=None,
    alpha=0.7
)
# Add labels and title
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.title('PCA Result: 2D Scatter Plot')
plt.legend(bbox_to_anchor=(1.2, 1), loc='upper right', borderaxespad=0)
plt.grid(True)
plt.xlim(right=0)
plt.savefig('/home/Shared/xinyi/blob1/thesis/figure/result_pca/pca2d.png')
plt.close()
