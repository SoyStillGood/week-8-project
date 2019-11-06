import pandas as pd
import matplotlib.pyplot as plt
import os
import seaborn as sns
import numpy as np

wine_df = pd.read_csv('C:/Users/zachz/Desktop/hwk4_data/wine.data',
                 sep=',',
                 header=0)

print(wine_df.columns)

os.makedirs('plots/seaborn_heatmap', exist_ok=True)

sns.set()

fig, ax = plt.subplots(figsize=(14,14))
sns.heatmap(wine_df.corr(), annot=True, ax=ax, cmap='prism', fmt='.2f', annot_kws={"size": 15}, linewidths=.05)
ax.set_xticklabels(wine_df.columns, rotation=45)
ax.set_yticklabels(wine_df.columns, rotation=45)
fig.subplots_adjust(top=.75)
plt.savefig('plots/seaborn_heatmap/wine_heatmap.png')
# plt.tight_layout()
plt.close()

wine_df.hist(bins=14, color='steelblue', edgecolor='black', linewidth=1.0,
           xlabelsize=6, ylabelsize=6, grid=False)
# plt.tight_layout(rect=(0, 0, 1.2, 1.2))
plt.savefig('plots/seaborn_heatmap/wine_histogram.png')